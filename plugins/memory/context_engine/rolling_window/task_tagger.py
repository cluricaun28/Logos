"""Task-Aware Context Pruning — Task Detection and Marker Injection.

Detects task boundaries in conversation messages and injects visible markers
that survive compression. Markers are part of message content, not metadata,
so they work with any context engine without schema changes.

Marker Format:
    📋 [task: <name> | label: <display>] ← task start (injected on first assistant message)
    📋 [task: <name>] ← legacy format (backward compatible)
    [phase: <phase>] ← optional phase tag
    📋 [task-end: <name> | status=<open|closed|soft_closed> | phase=<phase> | duration=<time>]

Why in content, not metadata:
1. Survives compression without special handling
2. Visible to user for verification/correction
3. Searchable in Perpetual Memory via content search
4. No database schema changes needed
5. Works with existing rolling window engine
"""

from __future__ import annotations

import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Marker constants — change these to customize marker format
MARKER_START_PREFIX = "📋 [task:"
MARKER_END_PREFIX = "📋 [task-end:"
PHASE_PREFIX = "[phase:"

# Precompiled regex patterns for parsing markers (module-level, compiled once)
# Handles both new format with pipe-separated label and legacy format without it.
_TASK_START_RE = re.compile(
    re.escape(MARKER_START_PREFIX) + r"\s*(.+?)\s*\]"
)

# Conclusive language patterns for closure detection (case-insensitive).
# Only checks assistant message content — no peeking at future messages.
CLOSURE_PATTERNS = [
    r"resolved\s*(?:✅|✓|\\^)",
    r"(?:both|all)\s+open\s+items?\s+(?:are\s+)?(?:resolved|done|complete)",
    r"✅\s*(?:both|all)\s+",
    r"(?:fixed|implemented|completed|deployed)\s+(?:and\s+)?(?:verified|tested|confirmed)",
    # Natural completion phrases — "here you go", "that's it", "all set"
    r"(?:here\s+you\s+go|that'?s?\s+(?:it|all)|all\s+set)",
    # Standalone done/complete (with optional trailing emoji)
    r"(?:done|complete)\b.*?(?:✅|✓)?",
    # Wrapped-up / finished phrases
    r"(?:wrapped\s+up|finished\s+up|that'?s?\s+everything)",
]

# Precompiled closure patterns
_CLOSURE_RE = [re.compile(p, re.IGNORECASE) for p in CLOSURE_PATTERNS]

# Sequential daily task naming — counter keyed by date string (YYYY-MM-DD).
_daily_counter: Dict[str, int] = {}

# Task name detection pattern (precompiled) — look for action verbs + object
_TASK_NAME_RE = re.compile(
    r"(?:fix|implement|add|remove|update|create|review|check|research)\s+"
    + r"((?:\S+\s+){0,4}\S+)",
    re.IGNORECASE,
)

# Outcome extraction patterns (precompiled)
_OUTCOME_CLEAN_RE = re.compile(r"[📋✅✓^👍\[\]]+")
_OUTCOME_MARKER_RE = re.compile(r"📋 \[task-end:[^\]]*\]")
_OUTCOME_PHRASE_RE = re.compile(
    r"(?:resolved|fixed|implemented|completed|deployed)\s+([a-z][^.]{5,80})",
    re.IGNORECASE,
)

# Phase detection patterns (tool call types indicate phase).
# "proposal" intentionally has an empty set — it's the default when no tools are called.
PHASE_TOOL_MAP = {
    "research": {"read_file", "search_files", "terminal", "web_search", "perpetual_search"},
    "proposal": set(),  # No tool calls — assistant presenting options only
    "execution": {"write_file", "patch", "terminal"},
    "validation": {"terminal"},  # Tests, verification commands
}


class TaskTagger:
    """Detect task boundaries and inject visible markers into assistant messages.

    Stateless analyzer that takes a message list and returns an annotated version
    with task start/end markers injected as prefix text on assistant messages.
    Idempotent — running twice produces same result as running once.
    """

    @classmethod
    def annotate(cls, messages: List[Dict[str, Any]], pm_context: str = None) -> List[Dict[str, Any]]:
        """Analyze message list and inject task markers where missing.

        Args:
            messages: List of OpenAI-format messages with 'role' and 'content'.
            pm_context: Optional Perpetual Memory context string (open task summaries).
                       Currently unused — reserved for future closure detection enhancement.

        Returns:
            New message list with task markers injected into assistant messages.
            Original messages are not modified.
        """
        if not messages or len(messages) < 2:
            return messages

        # Deep copy to avoid mutating originals
        annotated = [msg.copy() for msg in messages]

        # Track open tasks: task_name -> (start_idx, last_assistant_idx, tools_seen, display_label)
        open_tasks: Dict[str, Tuple[int, int, set, Optional[str]]] = {}

        # Current active task name (for closure detection)
        current_task: Optional[str] = None

        for i, msg in enumerate(annotated):
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "") or ""

            # Skip if already fully tagged (has both start and end markers)
            if cls._has_start_marker(content) and cls._has_end_marker(content):
                task_name = cls._extract_task_name_from_start(content)
                if task_name:
                    open_tasks.pop(task_name, None)  # Closed task, remove from tracking
                continue

            # Skip if already has start marker but no end (open task in progress)
            if cls._has_start_marker(content):
                task_name = cls._extract_task_name_from_start(content)
                if task_name:
                    current_task = task_name
                    if task_name not in open_tasks:
                        # Extract label from marker if present
                        raw_match = _TASK_START_RE.search(content)
                        label = None
                        if raw_match:
                            raw_fields = raw_match.group(1).strip()
                            parts = [p.strip() for p in raw_fields.split("|")]
                            for part in parts[1:]:  # Skip task name portion
                                if part.lower().startswith("label:"):
                                    label = part[len("label:"):].strip()
                        open_tasks[task_name] = (i, i, set(), label)
                    else:
                        # Update last assistant index for this task
                        start_idx, _, tools_seen, label = open_tasks[task_name]
                        open_tasks[task_name] = (start_idx, i, tools_seen, label)
                continue

            # Detect new task start — be conservative about what counts as a shift.
            # Multiple tasks can run concurrently; don't close old ones on shifts.
            is_shift = current_task and cls._is_topic_shift(annotated, i, msg)

            # Start a new task if we have none, or there's an explicit topic shift
            if not current_task or is_shift:
                name, label = cls._detect_task_name(annotated, end_idx=i)
                if name and name not in open_tasks:
                    if label:
                        content = f"{MARKER_START_PREFIX} {name} | label: {label}]\n\n{content}"
                    else:
                        content = f"{MARKER_START_PREFIX} {name}]\n\n{content}"
                    annotated[i]["content"] = content
                    current_task = name
                    open_tasks[name] = (i, i, set(), label)

            # If we have a current task, update its last-assistant index
            if current_task and current_task in open_tasks:
                start_idx, _, tools_seen, label = open_tasks[current_task]
                open_tasks[current_task] = (start_idx, i, tools_seen, label)

            # Detect closure — check only this assistant message's language.
            # No peeking at future messages that may not exist yet.
            if current_task and cls._is_closure(msg):
                start_idx, last_asst_idx, tools_seen, _ = open_tasks.get(current_task, (i, i, set(), None))
                phase = cls._detect_phase(annotated, tools_seen, start_idx=start_idx, end_idx=i)
                duration = cls._calculate_duration(start_idx, i, messages)

                end_marker = (
                    f"\n\n{MARKER_END_PREFIX} {current_task} "
                    f"| status=closed | phase={phase} | duration={duration}]"
                )
                annotated[i]["content"] = content + end_marker
                open_tasks.pop(current_task, None)
                current_task = None

            # Soft-close: if next user message doesn't reference any open task keywords
            # and is not an acknowledgment/continuation, close the old task.
            elif open_tasks:
                next_user_idx = cls._find_next_user_index(annotated, i + 1)
                if next_user_idx is not None:
                    next_user_content = (annotated[next_user_idx].get("content", "") or "").lower()
                    words_in_next = set(next_user_content.split())

                    # Check keyword overlap with ALL open tasks using their display labels
                    any_overlap = False
                    for task_name, (_, _, _, label) in list(open_tasks.items()):
                        keywords = cls._extract_keywords_from_label(label) if label else set()
                        if not keywords:
                            continue
                        if words_in_next & keywords:
                            any_overlap = True
                            break

                    # No keyword overlap and message is substantive (>= 4 words) → soft-close all open tasks
                    if not any_overlap and len(words_in_next) >= 4:
                        for task_name, (start_idx, last_asst_idx, tools_seen, _) in list(open_tasks.items()):
                            phase = cls._detect_phase(annotated, tools_seen, start_idx=start_idx, end_idx=i)
                            duration = cls._calculate_duration(start_idx, i, messages)

                            end_marker = (
                                f"\n\n{MARKER_END_PREFIX} {task_name} "
                                f"| status=soft_closed | phase={phase} | duration={duration}]"
                            )
                            annotated[i]["content"] = content + end_marker

                        open_tasks.clear()
                        current_task = None

        return annotated

    @classmethod
    def _has_start_marker(cls, content: str) -> bool:
        """Check if content has a task start marker."""
        return MARKER_START_PREFIX in content

    @classmethod
    def _has_end_marker(cls, content: str) -> bool:
        """Check if content has a task end marker."""
        return MARKER_END_PREFIX in content

    @classmethod
    def _extract_task_name_from_start(cls, content: str) -> Optional[str]:
        """Extract task name from start marker.

        Handles both new format (with pipe-separated label) and legacy format.
        Returns just the date-task-N portion (e.g., '2026-04-30-task-1').
        """
        match = _TASK_START_RE.search(content)
        if not match:
            return None
        raw = match.group(1).strip()
        # Split on pipe to get just the task name portion (before any label)
        return raw.split("|")[0].strip()

    @classmethod
    def _is_topic_shift(
        cls, all_messages: List[Dict[str, Any]], current_idx: int, new_msg: Dict[str, Any]
    ) -> bool:
        """Detect if new assistant message represents a genuine topic shift.

        Uses keyword overlap between the current task's display label and the last user message.
        If zero overlap AND the user message has >= 4 words → topic shift detected.
        Short messages (< 4 words) are treated as acknowledgments, not shifts.
        """
        # Look at recent user messages before current position (no slicing)
        recent_users = [
            all_messages[j]
            for j in range(max(0, current_idx - 4), current_idx)
            if all_messages[j].get("role") == "user" and all_messages[j].get("content")
        ]

        if not recent_users:
            return False

        last_user = (recent_users[-1].get("content", "") or "").lower()

        # Short user messages (acknowledgments, questions) don't indicate topic shift
        if len(last_user.split()) < 4:
            return False

        # Find current task name and label from the most recent start marker
        current_task_name = None
        current_label = None
        for j in range(current_idx - 1, -1, -1):
            m = all_messages[j]
            if m.get("role") == "assistant":
                mc = (m.get("content", "") or "")
                if MARKER_START_PREFIX in mc:
                    current_task_name = cls._extract_task_name_from_start(mc)
                    # Try to extract label from marker
                    raw_match = _TASK_START_RE.search(mc)
                    if raw_match:
                        raw_fields = raw_match.group(1).strip()
                        parts = [p.strip() for p in raw_fields.split("|")]
                        for part in parts[1:]:
                            if part.lower().startswith("label:"):
                                current_label = part[len("label:"):].strip()
                    break

        if not current_task_name:
            return False

        # Prefer label keywords (semantic) over task name keywords (date-based)
        if current_label:
            task_keywords = cls._extract_keywords_from_label(current_label)
        else:
            task_keywords = cls._extract_keywords_from_task_name(current_task_name)

        if not task_keywords:
            return False

        user_words = set(last_user.split())

        # Zero overlap → topic shift detected
        return len(task_keywords & user_words) == 0

    @classmethod
    def _detect_task_name(cls, prior_messages: List[Dict[str, Any]], end_idx: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
        """Detect task name and display label from recent conversation context.

        Returns a tuple of (task_name, display_label):
          - task_name: sequential daily format like '2026-04-30-task-1'
          - display_label: human-readable phrase extracted from user message (max 5 words)

        Uses module-level _daily_counter keyed by date string for sequential numbering.
        Falls back to None, None if no suitable label can be extracted.
        """
        # Search wider window — tasks can start several turns ago.
        if end_idx is not None:
            search_start = max(0, end_idx - 10)
            search_end = end_idx
        else:
            search_start = max(0, len(prior_messages) - 10)
            search_end = len(prior_messages)

        # Extract display label from last user message
        display_label = None
        for idx in range(search_end - 1, search_start - 1, -1):
            msg = prior_messages[idx]
            if msg.get("role") == "user":
                content = msg.get("content", "") or ""

                # Skip very short messages (likely acknowledgments)
                if len(content.split()) < 3:
                    continue

                # Extract key phrase — look for action verbs + object.
                task_match = _TASK_NAME_RE.search(content)
                if task_match:
                    raw = task_match.group(1).strip()
                    # Remove leading articles/determiners
                    for prefix in ("the ", "a ", "an "):
                        if raw.startswith(prefix):
                            raw = raw[len(prefix):]
                    # Split to words, drop trailing prepositions
                    words = raw.split()
                    while words and words[-1] in (
                        "in", "on", "at", "to", "for", "with", "of",
                    ):
                        words.pop()
                    if 2 <= len(words) <= 5:
                        display_label = "-".join(words)
                        break

                # Fallback: use first meaningful phrase (up to 5 words), skip stop-word leads
                words = content.split()[:6]
                meaningful = [w for w in words if w.lower() not in {
                    "yes", "no", "ok", "please", "thanks", "the", "a", "an"
                }]
                # Drop trailing prepositions from fallback too
                while meaningful and meaningful[-1].lower() in (
                    "in", "on", "at", "to", "for", "with"
                ):
                    meaningful.pop()
                if len(meaningful) >= 3:
                    display_label = "-".join(meaningful[:5])
                    break

        # Generate sequential daily task name
        today = datetime.date.today().isoformat()  # YYYY-MM-DD
        _daily_counter[today] = _daily_counter.get(today, 0) + 1
        task_name = f"{today}-task-{_daily_counter[today]}"

        return task_name, display_label

    @classmethod
    def _extract_keywords_from_task_name(cls, task_name: str) -> Set[str]:
        """Extract meaningful keywords from a task name for overlap detection.

        Handles both new format ('2026-04-30-task-1') and legacy format ('fix-schema-versioning').
        For new format, returns empty set (date-based names don't carry semantic keywords).
        For legacy format, splits on hyphens and filters out date-like tokens.
        """
        parts = task_name.lower().split("-")

        # Filter out purely numeric/date parts for keyword comparison
        keywords = set()
        for part in parts:
            if not part.isdigit():
                keywords.add(part)

        return keywords

    @classmethod
    def _extract_keywords_from_label(cls, label: str) -> Set[str]:
        """Extract meaningful keywords from a display label for overlap detection.

        Labels are hyphen-separated phrases like 'fix-schema-versioning' or 'database-connection'.
        Splits on hyphens and filters out common stop words.
        """
        if not label:
            return set()

        stop_words = {"the", "a", "an", "to", "for", "with", "in", "on", "at", "of"}
        return {w.lower().strip(".,!?\"'") for w in label.split("-") if w.lower() not in stop_words}

    @classmethod
    def _find_next_user_index(cls, messages: List[Dict[str, Any]], start_from: int) -> Optional[int]:
        """Find the index of the next user message starting from a given position."""
        for i in range(start_from, len(messages)):
            if messages[i].get("role") == "user":
                return i
        return None

    @classmethod
    def _is_closure(cls, assistant_msg: Dict[str, Any]) -> bool:
        """Check if assistant message indicates task closure.

        Closure requires conclusive language in the assistant message only.
        No peeking at future messages — those may not exist yet during annotation.
        """
        content = (assistant_msg.get("content", "") or "").lower()
        return any(p.search(content) for p in _CLOSURE_RE)

    @classmethod
    def _detect_phase(cls, messages: List[Dict[str, Any]], tools_seen: Optional[Set[str]] = None, *, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
        """Detect current phase based on tool calls in task messages.

        Phases: research → proposal → execution → validation → conclusion

        Args:
            messages: Full message list (no slicing needed by caller).
            tools_seen: Pre-extracted tool names, or None to extract from range.
            start_idx: Inclusive start index into messages.
            end_idx: Exclusive end index. If None, uses len(messages).
        """
        if end_idx is None:
            end_idx = len(messages)

        task_range = range(start_idx, end_idx)

        if not tools_seen:
            tools_seen = set()
            for idx in task_range:
                m = messages[idx]
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        if isinstance(tc, dict) and "function" in tc:
                            func = tc["function"]
                            if isinstance(func, dict):
                                tools_seen.add(func.get("name", ""))
                            elif isinstance(func, str):
                                tools_seen.add(func)

        # Phase detection logic — check most specific first
        has_validation_tools = bool(tools_seen & PHASE_TOOL_MAP["validation"])
        has_execution_tools = bool(tools_seen & PHASE_TOOL_MAP["execution"])
        has_research_tools = bool(tools_seen & PHASE_TOOL_MAP["research"])

        # Check for validation language in assistant messages (tests, verification)
        has_validation_lang = any(
            "test" in str(messages[idx].get("content", "")).lower() or
            "verif" in str(messages[idx].get("content", "")).lower() or
            "confirm" in str(messages[idx].get("content", "")).lower()
            for idx in task_range if messages[idx].get("role") == "assistant"
        )

        if has_validation_tools and has_validation_lang:
            return "validation"
        elif has_execution_tools:
            return "execution"
        elif has_research_tools and not has_execution_tools:
            return "research"
        else:
            return "proposal"  # Default for discussion-only tasks

    @classmethod
    def _extract_tools_from_messages(cls, messages: List[Dict[str, Any]]) -> set:
        """Extract tool call names from assistant messages."""
        tools = set()
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and "function" in tc:
                        func = tc["function"]
                        if isinstance(func, dict):
                            tools.add(func.get("name", ""))
                        elif isinstance(func, str):
                            tools.add(func)
        return tools

    @classmethod
    def _calculate_duration(cls, start_idx: int, end_idx: int, messages: List[Dict[str, Any]]) -> str:
        """Calculate approximate duration based on message count.

        Rough heuristic: ~5 minutes per message pair (user + assistant).
        More accurate timing would require timestamps in messages.
        """
        message_pairs = max(1, (end_idx - start_idx) // 2)
        minutes = message_pairs * 5

        if minutes < 60:
            return f"{minutes}min"
        else:
            hours = minutes // 60
            remaining_min = minutes % 60
            return f"{hours}h{remaining_min}m" if remaining_min else f"{hours}h"


# -- Convenience functions for use from context engine ----------------------

def annotate_tasks(messages: List[Dict[str, Any]], pm_context: str = None) -> List[Dict[str, Any]]:
    """Top-level function for hooking into run_agent.py."""
    return TaskTagger.annotate(messages, pm_context=pm_context)


def categorize_tasks(
    messages: List[Dict[str, Any]]
) -> Tuple[Set[int], Set[int]]:
    """Split message indices into closed-task and open-task groups.

    Two-pass algorithm:
      Pass 1: Scan for task boundaries (start/end markers) to build a map of
              which message indices belong to closed tasks.
      Pass 2: Categorize each index based on its task membership.

    User messages always go to open_ — they drive the conversation forward and
    provide context for why subsequent work was done, even if the task is closed.

    Returns:
        (closed_indices, open_indices) tuple of sets.
    """
    # Pass 1: Find all closed task ranges (start_idx -> end_idx)
    closed_ranges: List[Tuple[int, int]] = []
    current_start: Optional[int] = None

    for i, msg in enumerate(messages):
        content = msg.get("content", "") or ""

        if MARKER_START_PREFIX in content:
            current_start = i

        if MARKER_END_PREFIX in content and ("status=closed" in content or "status=soft_closed" in content):
            if current_start is not None:
                closed_ranges.append((current_start, i))
            current_start = None

    # Build set of indices that belong to closed tasks (non-user messages only)
    closed_indices: Set[int] = set()
    for start_idx, end_idx in closed_ranges:
        for idx in range(start_idx, end_idx + 1):
            if messages[idx].get("role") != "user":
                closed_indices.add(idx)

    # Pass 2: Categorize each index
    open_indices: Set[int] = set()
    for i in range(len(messages)):
        msg = messages[i]
        # User messages always go to open — they provide conversation context
        if msg.get("role") == "user":
            open_indices.add(i)
            continue

        if i not in closed_indices:
            open_indices.add(i)

    return closed_indices, open_indices


def build_task_summary(closed_tasks: List[Dict[str, Any]]) -> str:
    """Build concise summary of archived closed tasks.

    Returns empty string if no closed tasks.
    """
    if not closed_tasks:
        return ""

    lines = ["## Archived Tasks (searchable in Perpetual Memory)"]

    # Group by task name
    task_groups: Dict[str, List[Dict[str, Any]]] = {}
    current_task = None

    for msg in closed_tasks:
        content = msg.get("content", "") or ""

        if MARKER_START_PREFIX in content:
            current_task = TaskTagger._extract_task_name_from_start(content)
            if current_task:
                task_groups.setdefault(current_task, [])

        if current_task:
            task_groups[current_task].append(msg)

    # Extract outcome for each task
    for task_name, messages in task_groups.items():
        outcome = _extract_outcome(messages)
        lines.append(f"- {task_name} ✅ — {outcome}")

    lines.append("\nFull details available via perpetual_search or recent_messages.")
    return "\n".join(lines)


def _extract_outcome(messages: List[Dict[str, Any]]) -> str:
    """Extract one-line outcome from task messages."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "") or ""

            # Remove task-end markers first — they corrupt the outcome text
            content = _OUTCOME_MARKER_RE.sub('', content)

            # Remove emojis and remaining markers
            clean_content = _OUTCOME_CLEAN_RE.sub(' ', content)
            clean_content = ' '.join(clean_content.split())

            # Look for key outcome phrases (avoiding emoji-only matches)
            outcome_match = _OUTCOME_PHRASE_RE.search(clean_content.lower())
            if outcome_match:
                return outcome_match.group(1).strip()[:100]

            # Fallback: use last meaningful sentence (at least 3 words)
            sentences = [s.strip() for s in re.split(r'[.!?]+', clean_content)
                        if len(s.split()) >= 3]
            if sentences:
                return sentences[-1][:100]

    return "Completed"

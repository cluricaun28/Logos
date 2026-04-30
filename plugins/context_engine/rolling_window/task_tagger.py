"""Task-Aware Context Pruning — Task Detection and Marker Injection.

Detects task boundaries in conversation messages and injects visible markers
that survive compression. Markers are part of message content, not metadata,
so they work with any context engine without schema changes.

Marker Format:
    📋 [task: <name>] ← task start (injected on first assistant message)
    [phase: <phase>] ← optional phase tag
    📋 [task-end: <name> | status=<open|closed> | phase=<phase> | duration=<time>]

Why in content, not metadata:
1. Survives compression without special handling
2. Visible to user for verification/correction
3. Searchable in Perpetual Memory via content search
4. No database schema changes needed
5. Works with existing rolling window engine
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Marker constants — change these to customize marker format
MARKER_START_PREFIX = "📋 [task:"
MARKER_END_PREFIX = "📋 [task-end:"
PHASE_PREFIX = "[phase:"

# Precompiled regex patterns for parsing markers (module-level, compiled once)
_TASK_START_RE = re.compile(re.escape(MARKER_START_PREFIX) + r"\s*:\s*([^\]]+)\]")

# Conclusive language patterns for closure detection (case-insensitive).
# Only checks assistant message content — no peeking at future messages.
CLOSURE_PATTERNS = [
    r"resolved\s*(?:✅|✓|\^)",
    r"(?:both|all)\s+open\s+items?\s+(?:are\s+)?(?:resolved|done|complete)",
    r"✅\s*(?:both|all)\s+",
    r"(?:fixed|implemented|completed|deployed)\s+(?:and\s+)?(?:verified|tested|confirmed)",
]

# Precompiled closure patterns
_CLOSURE_RE = [re.compile(p, re.IGNORECASE) for p in CLOSURE_PATTERNS]

# Topic shift detection patterns (precompiled)
_TOPIC_SHIFT_PATTERNS = [
    re.compile(
        r"(?:new\s+topic|different\s+thing|separate\s+issue|another\s+task)"
        + r"|(?:switch\s+(?:to|over)|let's?\s+(?:move|switch))"
        + r"|(?:by\s+the\s+way|on\s+a\s+different\s+note)",
        re.IGNORECASE,
    ),
]

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
    def annotate(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze message list and inject task markers where missing.

        Args:
            messages: List of OpenAI-format messages with 'role' and 'content'.

        Returns:
            New message list with task markers injected into assistant messages.
            Original messages are not modified.
        """
        if not messages or len(messages) < 2:
            return messages

        # Deep copy to avoid mutating originals
        annotated = [msg.copy() for msg in messages]

        # Track open tasks: task_name -> (start_idx, last_assistant_idx, tools_seen)
        open_tasks: Dict[str, Tuple[int, int, set]] = {}

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
                        open_tasks[task_name] = (i, i, set())
                    else:
                        # Update last assistant index for this task
                        start_idx, _, tools_seen = open_tasks[task_name]
                        open_tasks[task_name] = (start_idx, i, tools_seen)
                continue

            # Detect new task start — be conservative about what counts as a shift.
            # Multiple tasks can run concurrently; don't close old ones on shifts.
            is_shift = current_task and cls._is_topic_shift(annotated, i, msg)

            # Start a new task if we have none, or there's an explicit topic shift
            if not current_task or is_shift:
                task_name = cls._detect_task_name(annotated, end_idx=i)
                if task_name and task_name not in open_tasks:
                    content = f"{MARKER_START_PREFIX} {task_name}]\n\n{content}"
                    annotated[i]["content"] = content
                    current_task = task_name
                    open_tasks[task_name] = (i, i, set())

            # If we have a current task, update its last-assistant index
            if current_task and current_task in open_tasks:
                start_idx, _, tools_seen = open_tasks[current_task]
                open_tasks[current_task] = (start_idx, i, tools_seen)

            # Detect closure — check only this assistant message's language.
            # No peeking at future messages that may not exist yet.
            if current_task and cls._is_closure(msg):
                start_idx, last_asst_idx, tools_seen = open_tasks.get(current_task, (i, i, set()))
                phase = cls._detect_phase(annotated, tools_seen, start_idx=start_idx, end_idx=i)
                duration = cls._calculate_duration(start_idx, i, messages)

                end_marker = (
                    f"\n\n{MARKER_END_PREFIX} {current_task} "
                    f"| status=closed | phase={phase} | duration={duration}]"
                )
                annotated[i]["content"] = content + end_marker
                open_tasks.pop(current_task, None)
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
        """Extract task name from start marker."""
        match = _TASK_START_RE.search(content)
        return match.group(1).strip() if match else None

    @classmethod
    def _is_topic_shift(
        cls, all_messages: List[Dict[str, Any]], current_idx: int, new_msg: Dict[str, Any]
    ) -> bool:
        """Detect if new assistant message represents a genuine topic shift.

        Conservative — only explicit "new topic" phrases trigger shifts.
        Action verbs like "fix X" are normal mid-task and should NOT count as shifts.
        Multiple concurrent tasks are allowed; casual chat during tasks is fine.
        """
        # Look at recent user messages before current position (no slicing)
        recent_users = [
            all_messages[j]
            for j in range(max(0, current_idx - 4), current_idx)
            if all_messages[j].get("role") == "user" and all_messages[j].get("content")
        ]

        if not recent_users:
            return False

        last_user = recent_users[-1].get("content", "") or ""

        # Short user messages (acknowledgments, questions) don't indicate topic shift
        if len(last_user.split()) < 4:
            return False

        # Only explicit "new topic" language — NOT action verbs.
        # Action verbs are normal mid-task work and should not trigger shifts.
        return any(p.search(last_user) for p in _TOPIC_SHIFT_PATTERNS)

    @classmethod
    def _detect_task_name(cls, prior_messages: List[Dict[str, Any]], end_idx: Optional[int] = None) -> Optional[str]:
        """Detect task name from recent conversation context.

        Strategy: Extract key phrase from last user message that describes the task.
        Stops at sentence boundaries or conjunctions to avoid greedy capture.
        Searches up to 10 prior messages (not just 5) for better recall on longer threads.
        Falls back to topic clustering if available.

        Args:
            prior_messages: Full message list (no slicing needed by caller).
            end_idx: Exclusive upper bound — only consider messages before this index.
                If None, considers all messages in the list.
        """
        # Search wider window — tasks can start several turns ago.
        # Use index range instead of slicing to avoid O(n²) memory allocation.
        if end_idx is not None:
            search_start = max(0, end_idx - 10)
            search_end = end_idx
        else:
            search_start = max(0, len(prior_messages) - 10)
            search_end = len(prior_messages)

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
                        "in",
                        "on",
                        "at",
                        "to",
                        "for",
                        "with",
                        "of",
                    ):
                        words.pop()
                    if 2 <= len(words) <= 4:
                        return "-".join(words)

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
                    return "-".join(meaningful[:4])

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

def annotate_tasks(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Top-level function for hooking into run_agent.py."""
    return TaskTagger.annotate(messages)


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

        if MARKER_END_PREFIX in content and "status=closed" in content:
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

"""Rolling Window Context Engine.

Replaces LLM-based summarization with a deterministic rolling window architecture.
Instead of summarizing old turns, archive them to Perpetual Memory and rely on
retrieval for historical recall. Designed for models with large native context windows (131k+ tokens).

Architecture:
- Context = working RAM (short-term, rolling window)
- Perpetual Memory/Wiki = long-term storage (infinite recall via SQLite/FTS5)
- [ACTIVE TASKS] anchor survives pruning to prevent mid-session drift

Pruning Rules (task-aware mode):
1. Strip raw assistant tool calls entirely (verbose JSON bloat)
2. Truncate role:"tool" results to first/last 3 lines
3. Categorize messages by task status (closed vs open) via task markers
4. Drop closed tasks first (searchable in Perpetual Memory)
5. If still over budget, drop oldest open tasks from unprotected middle zone
6. Enforce hard token budget with aggressive truncation as last resort

Fallback (no task markers):
1-2. Same strip/truncate steps
3. Drop oldest messages when window_size is exceeded
4. Enforce hard token budget with aggressive truncation as last resort
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


class RollingWindowContextEngine(ContextEngine):
    """Rolling window context engine — deterministic pruning without LLM summarization.

    Inherits from ContextEngine ABC — all abstract methods must be implemented.

    Task-aware mode (default): Uses task markers injected by annotate_tasks() to
    selectively archive closed tasks while keeping active work in context. Falls
    back to original algorithm when no markers are present.
    """

    @property
    def name(self) -> str:
        return "rolling_window"

    # Token state (inherited from ContextEngine ABC, set in __init__ or update_from_response)
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    archive_count: int = 0

    protect_first_n: int = 3
    protect_last_n: int = 30

    def __init__(self, **kwargs):
        """Initialize RollingWindowContextEngine with configurable parameters.

        Configurable via plugin config (passed as kwargs from the loader):

        - window_size: int — number of turns to keep in the rolling window (default: 20)
        - max_tokens: int — hard token ceiling for the context window (default: 131072)
        - task_aware: bool — enable task-aware pruning using task markers (default: True)
        - threshold_percent: float — trigger archiving when usage exceeds this ratio of context_length (default: 0.75)
        - archive_target: float — target usage ratio after archiving (default: 0.65)
        - hard_ceiling_percent: float — absolute maximum usage before nuclear truncation (default: 0.95)
        - effective_window_ratio: float — multiplier applied to reported context_length to compute the
          effective working window. E.g. 0.8 means only use 80% of the model's stated context length,
          leaving a safety margin for system prompts and tool schemas (default: 1.0)
        """
        super().__init__()
        # Set engine-specific config from kwargs or defaults
        self.window_size: int = kwargs.get("window_size", 20)
        self.max_tokens: int = kwargs.get("max_tokens", 131072)
        self.task_aware: bool = kwargs.get("task_aware", True)
        self.threshold_percent: float = kwargs.get("threshold_percent", 0.75)
        self.archive_target: float = kwargs.get("archive_target", 0.65)
        self.hard_ceiling_percent: float = kwargs.get("hard_ceiling_percent", 0.95)
        self.effective_window_ratio: float = kwargs.get("effective_window_ratio", 1.0)

    def annotate_tasks(self, messages: List[Dict[str, Any]], pm_context: str = None) -> List[Dict[str, Any]]:
        """Inject task markers into assistant messages for task-aware archiving.

        Called by run_agent.py before archive() to tag message boundaries.
        Idempotent — safe to call multiple times on same messages.

        Args:
            messages: Conversation history to annotate.
            pm_context: Optional Perpetual Memory context string (open task summaries).
        """
        if not self.task_aware:
            return messages
        try:
            from .task_tagger import annotate_tasks
            return annotate_tasks(messages, pm_context=pm_context)
        except Exception as e:
            logger.debug("Task annotation failed (non-fatal): %s", e)
            return messages

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        if isinstance(usage, dict):
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
            self.last_completion_tokens = int(usage.get("completion_tokens", 0))
            self.last_total_tokens = int(usage.get("total_tokens", 0))

    def should_archive(self, prompt_tokens: int = None) -> bool:
        """Return True if archiving should fire this turn."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        # Guard against zero threshold (update_model hasn't been called yet)
        if self.threshold_tokens <= 0:
            return False
        return tokens > self.threshold_tokens

    def _log_archive_stats(
        self, before: List[Dict[str, Any]], after: List[Dict[str, Any]], pruner_name: str
    ) -> None:
        """Log telemetry about what the archive operation did."""
        from .task_pruner import count_tokens

        before_tokens = count_tokens(before)
        after_tokens = count_tokens(after)
        dropped = before_tokens - after_tokens

        # Role breakdown of what was kept
        role_counts = {}
        for msg in after:
            r = msg.get("role", "unknown")
            role_counts[r] = role_counts.get(r, 0) + 1

        # Check if protected zones were touched (should never happen)
        before_protected_ids = {id(m) for m in before[:self.protect_first_n]} | \
                               {id(m) for m in before[-self.protect_last_n:]}
        after_ids = {id(m) for m in after}
        protected_kept = len(before_protected_ids & after_ids)
        protected_total = len(before_protected_ids)

        logger.info(
            "ARCHIVE [%s]: %d→%d msgs, %d→%d tokens (dropped %d). "
            "Roles kept: %s. Protected: %d/%d retained.",
            pruner_name,
            len(before), len(after),
            before_tokens, after_tokens, dropped,
            dict(sorted(role_counts.items())),
            protected_kept, protected_total,
        )

    def archive(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        **kwargs,  # focus_topic and any future params are ignored (deterministic pruning)
    ) -> List[Dict[str, Any]]:
        """Archive old messages and return the new message list.

        Rolling window approach — deterministic pruning without LLM summarization.
        Designed for models with large native context windows (131k+ tokens).

        Escalating pressure model:
          - 75-80% full → light touch (closed tasks only)
          - 80-90% full → moderate (closed + oldest open from middle zone)
          - 90-95% full → aggressive (drop more open tasks, tighten target)
          - 95%+ full → nuclear (aggressive truncation as last resort)

        When task_aware=True and task markers are present:
          1. Strip raw assistant tool calls entirely (verbose JSON bloat)
          2. Truncate role:"tool" results to first/last 3 lines
          3. Categorize messages by task status (closed vs open) via task markers
          4. Drop closed tasks first (searchable in Perpetual Memory)
          5. If still over budget, drop oldest open tasks from unprotected middle zone
          6. Enforce hard token budget with aggressive truncation as last resort

        When no task markers or task_aware=False:
          Falls back to original "drop middle" algorithm.
        """
        if not messages:
            return messages

        # Calculate pressure ratio — how full is the window?
        tokens = current_tokens or self.last_prompt_tokens
        pressure = tokens / self.max_tokens if self.max_tokens else 0.0

        # Escalating target: the closer to full, the more we drop.
        # At threshold (75%) → aim for archive_target (65%).
        # Near capacity (95%+) → aim lower (~40%) for safety margin.
        # Linear interpolation between these two points.
        low_pressure = self.threshold_percent
        high_pressure = 0.95
        if pressure <= low_pressure:
            target_ratio = self.archive_target
        elif pressure >= high_pressure:
            target_ratio = max(0.3, self.archive_target * 0.6)
        else:
            # Linear interpolation
            t = (pressure - low_pressure) / (high_pressure - low_pressure)
            target_ratio = self.archive_target * (1.0 - 0.4 * t)

        target_tokens = int(self.max_tokens * target_ratio)

        # Try task-aware pruner first (when enabled)
        if self.task_aware:
            try:
                from .task_pruner import TaskAwarePruner
                result = TaskAwarePruner.archive(
                    messages,
                    max_tokens=self.max_tokens,
                    target_tokens=target_tokens,
                    protect_first_n=self.protect_first_n,
                    protect_last_n=self.protect_last_n,
                    window_size=self.window_size,
                    task_aware=True,
                )
                self._log_archive_stats(messages, result, "task-aware")
                self.archive_count += 1
                return result
            except Exception as e:
                # Graceful degradation — if pruner fails for any reason,
                # fall back to original algorithm. Never crash the agent loop.
                logger.warning(
                    "Task-aware pruner failed (%s), falling back to original algorithm", e
                )

        # Original "drop middle" algorithm (fallback)
        result = self._original_archive(messages, target_tokens=target_tokens)
        self._log_archive_stats(messages, result, "original")
        return result

    def _original_archive(
        self, messages: List[Dict[str, Any]], target_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """Fallback wrapper — delegates to shared original_archive function."""
        from .task_pruner import original_archive

        result = original_archive(
            messages,
            max_tokens=self.max_tokens,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
            window_size=self.window_size,
            target_tokens=target_tokens,
        )
        self.archive_count += 1
        return result

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins."""
        pass

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Called at real session boundaries. NOT called per-turn."""
        pass

    def on_session_reset(self) -> None:
        """Called on /new or /reset."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.archive_count = 0

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this engine provides. Default returns empty list."""
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent. Only called for engine-specific tools."""
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging."""
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "archive_count": self.archive_count,
            "task_aware": self.task_aware,
            "threshold_percent": self.threshold_percent,
            "archive_target": self.archive_target,
            "hard_ceiling_percent": self.hard_ceiling_percent,
            "effective_window_ratio": self.effective_window_ratio,
        }

    def update_model(self, model: str, context_length: int, **kwargs) -> None:
        """Called when the user switches models."""
        # Apply effective_window_ratio if configured — scales reported context_length
        # to leave safety margin for system prompts, tool schemas, etc.
        self.context_length = int(context_length * self.effective_window_ratio)
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    def get_recent_context(
        self,
        messages: List[Dict[str, Any]],
        pm_callback=None,
    ) -> Optional[str]:
        """Query recent conversation history from Perpetual Memory.

        Extracts open task names from current messages and queries PM for whether
        those tasks were completed in prior sessions. Provides ground truth about
        conversation flow beyond regex heuristics.

        Args:
            messages: Current message list.
            pm_callback: Optional callable(task_names: List[str]) -> str that queries
                Perpetual Memory and returns formatted context. If None, returns None.

        Returns:
            Formatted context string, or None if unavailable.
        """
        if not pm_callback:
            return None

        # Extract open task names from messages (tasks with start marker but no closed end)
        open_task_names = self._extract_open_task_names(messages)
        if not open_task_names:
            return None

        try:
            result = pm_callback(open_task_names)
            if result and result.strip():
                return f"## Perpetual Memory Context (recent sessions)\n{result.strip()}"
        except Exception as e:
            logger.debug("PM query failed (non-fatal): %s", e)

        return None

    @staticmethod
    def _extract_open_task_names(messages: List[Dict[str, Any]]) -> List[str]:
        """Extract names of tasks that are currently open (started but not closed)."""
        import re
        # Match full marker content up to closing bracket
        start_re = re.compile(r"📋 \[task:\s*(.+?)\]")
        end_re = re.compile(r"📋 \[task-end:\s*(.+?)\]")

        started = set()
        closed = set()

        for msg in messages:
            content = msg.get("content", "") or ""
            for m in start_re.finditer(content):
                # Task name is the first field before any pipe separator
                name = m.group(1).split("|")[0].strip()
                started.add(name)
            for m in end_re.finditer(content):
                # Only count as closed if status=closed or status=soft_closed
                marker_content = m.group(1)
                if "status=closed" in marker_content or "status=soft_closed" in marker_content:
                    name = marker_content.split("|")[0].strip()
                    closed.add(name)

        return list(started - closed)


# -- Plugin registration (required for discovery) --------------------------

def register(collector):
    """Register this engine with the Hermes plugin system."""
    collector.register_context_engine(RollingWindowContextEngine())

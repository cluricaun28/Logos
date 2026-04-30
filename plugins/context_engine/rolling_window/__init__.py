"""Rolling Window Context Engine.

Replaces LLM-based compression with a deterministic rolling window architecture.
Instead of summarizing old turns, drop them entirely and rely on Perpetual Memory
for historical retrieval. Designed for models with large native context windows (131k+ tokens).

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
from typing import Any, Dict, List

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
    compression_count: int = 0

    protect_first_n: int = 3
    protect_last_n: int = 30

    def __init__(self, **kwargs):
        super().__init__()
        # Set engine-specific config from kwargs or defaults
        self.window_size: int = kwargs.get("window_size", 20)
        self.max_tokens: int = kwargs.get("max_tokens", 131072)
        self.task_aware: bool = kwargs.get("task_aware", True)
        self.threshold_percent: float = kwargs.get("threshold_percent", 0.75)
        self.compression_target: float = kwargs.get("compression_target", 0.65)
        self.hard_ceiling_percent: float = kwargs.get("hard_ceiling_percent", 0.95)

    def annotate_tasks(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inject task markers into assistant messages for task-aware compression.

        Called by run_agent.py before compress() to tag message boundaries.
        Idempotent — safe to call multiple times on same messages.
        """
        if not self.task_aware:
            return messages
        try:
            from .task_tagger import annotate_tasks
            return annotate_tasks(messages)
        except Exception as e:
            logger.debug("Task annotation failed (non-fatal): %s", e)
            return messages

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        if isinstance(usage, dict):
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
            self.last_completion_tokens = int(usage.get("completion_tokens", 0))
            self.last_total_tokens = int(usage.get("total_tokens", 0))

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        # Guard against zero threshold (update_model hasn't been called yet)
        if self.threshold_tokens <= 0:
            return False
        return tokens > self.threshold_tokens

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        **kwargs,  # focus_topic and any future params are ignored (deterministic pruning)
    ) -> List[Dict[str, Any]]:
        """Compact the message list and return the new message list.

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
        # At threshold (75%) → aim for compression_target (65%).
        # Near capacity (95%+) → aim lower (~40%) for safety margin.
        # Linear interpolation between these two points.
        low_pressure = self.threshold_percent
        high_pressure = 0.95
        if pressure <= low_pressure:
            target_ratio = self.compression_target
        elif pressure >= high_pressure:
            target_ratio = max(0.3, self.compression_target * 0.6)
        else:
            # Linear interpolation
            t = (pressure - low_pressure) / (high_pressure - low_pressure)
            target_ratio = self.compression_target * (1.0 - 0.4 * t)

        target_tokens = int(self.max_tokens * target_ratio)

        # Try task-aware pruner first (when enabled)
        if self.task_aware:
            try:
                from .task_pruner import TaskAwarePruner
                result = TaskAwarePruner.compress(
                    messages,
                    max_tokens=self.max_tokens,
                    target_tokens=target_tokens,
                    protect_first_n=self.protect_first_n,
                    protect_last_n=self.protect_last_n,
                    window_size=self.window_size,
                    task_aware=True,
                )
                self.compression_count += 1
                return result
            except Exception as e:
                # Graceful degradation — if pruner fails for any reason,
                # fall back to original algorithm. Never crash the agent loop.
                logger.warning(
                    "Task-aware pruner failed (%s), falling back to original algorithm", e
                )

        # Original "drop middle" algorithm (fallback)
        return self._original_compress(messages, target_tokens=target_tokens)

    def _original_compress(
        self, messages: List[Dict[str, Any]], target_tokens: int = None
    ) -> List[Dict[str, Any]]:
        """Fallback wrapper — delegates to shared original_compress function."""
        from .task_pruner import original_compress

        result = original_compress(
            messages,
            max_tokens=self.max_tokens,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
            window_size=self.window_size,
            target_tokens=target_tokens,
        )
        self.compression_count += 1
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
        self.compression_count = 0

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
            "compression_count": self.compression_count,
            "task_aware": self.task_aware,
            "threshold_percent": self.threshold_percent,
            "compression_target": self.compression_target,
            "hard_ceiling_percent": self.hard_ceiling_percent,
        }

    def update_model(self, model: str, context_length: int, **kwargs) -> None:
        """Called when the user switches models."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


# -- Plugin registration (required for discovery) --------------------------

def register(collector):
    """Register this engine with the Hermes plugin system."""
    collector.register_context_engine(RollingWindowContextEngine())

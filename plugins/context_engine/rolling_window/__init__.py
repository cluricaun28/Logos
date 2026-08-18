"""Rolling Window Context Engine with Task-Aware Pruning.

Replaces LLM-based compression with a deterministic rolling window architecture.
Instead of summarizing old turns, drop them entirely and rely on Perpetual Memory
for historical retrieval. Designed for models with large native context windows (131k+ tokens).

Architecture:
- Context = working RAM (short-term, rolling window)
- Perpetual Memory/Wiki = long-term storage (infinite recall via SQLite/FTS5)
- [ACTIVE TASKS] anchor survives pruning to prevent mid-session drift

Pruning Rules:
1. Strip raw assistant tool calls entirely (verbose JSON bloat)
2. Truncate role:"tool" results to first/last 3 lines
3. Task-aware scoring: preserve turns from active/incomplete tasks
4. Drop lowest-scoring messages when window_size is exceeded
5. Enforce hard token budget with aggressive truncation as last resort

Task Markers (injected into system prompt):
- [TASK_START: id] — New task initiated
- [TASK_COMPLETE: id] — Task finished successfully  
- [TASK_DEFERRED: id] — Task postponed for later
- [TASK_CANCELLED: id] — Task abandoned/cancelled
- [TASK_SWITCH: from -> to] — Context switching between tasks
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from typing import Any

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


class RollingWindowContextEngine(ContextEngine):
    """Rolling window context engine with task-aware pruning.

    Inherits from ContextEngine ABC — all abstract methods must be implemented.
    
    Task-aware pruning preserves turns belonging to active, incomplete tasks
    while aggressively dropping completed or low-value turns. This prevents
    mid-session drift when the model is in the middle of multi-step work.
    """

    @property
    def name(self) -> str:
        return "rolling_window"

    # Token state — initialised in __init__ (not class-level defaults)
    last_prompt_tokens: int
    last_completion_tokens: int
    last_total_tokens: int
    threshold_tokens: int
    context_length: int
    archive_count: int

    # Compaction parameters (override defaults as needed)
    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    def __init__(self, **kwargs):
        super().__init__()

        # Initialise token-tracking instance attributes
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.archive_count = 0
        # Set engine-specific config from kwargs or defaults
        self.window_size: int = kwargs.get("window_size", 20)
        self.max_tokens: int = kwargs.get("max_tokens", 131072)
        
        # Update protection parameters from kwargs
        self.protect_first_n: int = kwargs.get("protect_first_n", self.protect_first_n)
        self.protect_last_n: int = kwargs.get("protect_last_n", self.protect_last_n)
        
        # Lazy-import task-aware components (avoid circular imports)
        try:
            # Try relative import first (when loaded as package)
            from .task_aware_pruner import TaskAwarePruner
            from .task_marker_injector import TaskMarkerInjector
        except ImportError:
            try:
                # Fallback: use importlib to load modules without polluting sys.path
                plugin_dir = os.path.dirname(os.path.abspath(__file__))

                spec_pruner = importlib.util.spec_from_file_location(
                    "task_aware_pruner", os.path.join(plugin_dir, "task_aware_pruner.py")
                )
                mod_pruner = importlib.util.module_from_spec(spec_pruner)
                spec_pruner.loader.exec_module(mod_pruner)
                TaskAwarePruner = mod_pruner.TaskAwarePruner

                spec_injector = importlib.util.spec_from_file_location(
                    "task_marker_injector", os.path.join(plugin_dir, "task_marker_injector.py")
                )
                mod_injector = importlib.util.module_from_spec(spec_injector)
                spec_injector.loader.exec_module(mod_injector)
                TaskMarkerInjector = mod_injector.TaskMarkerInjector
            except ImportError as e:
                logger.warning("Task-aware pruning unavailable: %s", e)
                self._pruner = None
                self._injector = None
                return
        
        self._pruner = TaskAwarePruner(
            window_size=self.window_size,
            protect_first_n=self.protect_first_n,
            protect_last_n=self.protect_last_n,
        )
        self._injector = TaskMarkerInjector()

    def update_from_response(self, usage: dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        if isinstance(usage, dict):
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
            self.last_completion_tokens = int(usage.get("completion_tokens", 0))
            self.last_total_tokens = int(usage.get("total_tokens", 0))

    def should_archive(self, prompt_tokens: int = None) -> bool:
        """Return True if archiving should fire this turn."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens > self.threshold_tokens

    def archive(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,  # ignored (deterministic pruning)
    ) -> list[dict[str, Any]]:
        """Compact the message list and return the new message list.

        Rolling window approach — deterministic pruning without LLM summarization.
        Designed for models with large native context windows (131k+ tokens).

        Algorithm:
          1. Strip raw assistant tool calls entirely (verbose JSON bloat)
          2. Truncate role:"tool" results to first/last 3 lines
          3. Task-aware scoring + selection (preserves active task context)
          4. Enforce hard token budget with aggressive truncation as last resort
        """
        if not messages:
            return messages

        self.archive_count += 1

        # Step 1: Strip raw assistant tool calls (verbose JSON bloat)
        stripped = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Keep the message but remove tool_calls to save space
                stripped.append({**msg, "tool_calls": None})
            else:
                stripped.append(msg)

        # Step 2: Truncate role:"tool" results to first/last 3 lines
        truncated = []
        for msg in stripped:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.split("\n")) > 6:
                    lines = content.split("\n")
                    # Keep first 3 and last 3 lines
                    truncated_content = "\n".join(lines[:3]) + "\n...[truncated]...\n" + "\n".join(lines[-3:])
                    msg = {**msg, "content": truncated_content}
            truncated.append(msg)

        # Step 3: Task-aware pruning (preserves active task context)
        if self._pruner is not None and len(truncated) > self.window_size:
            try:
                keep_indices = self._pruner.select_turns_to_keep(
                    truncated, target_count=self.window_size
                )
                pruned = [truncated[i] for i in keep_indices]
                
                # Log pruning stats if debug enabled
                if logger.isEnabledFor(logging.DEBUG):
                    report = self._pruner.get_pruning_report(truncated, keep_indices)
                    logger.debug("Task-aware prune: %s", report)
            except Exception as e:
                logger.warning("Task-aware pruning failed, falling back to simple window: %s", e)
                pruned = self._fallback_window_prune(truncated)
        elif len(truncated) > self.window_size:
            # Fallback if task-aware components unavailable
            pruned = self._fallback_window_prune(truncated)
        else:
            pruned = truncated

        # Step 4: Enforce hard token budget with aggressive truncation as last resort
        if current_tokens and current_tokens > self.max_tokens:
            # Aggressive truncation — reduce to first/last N messages only
            keep_first = max(1, len(pruned) // 4)
            keep_last = max(1, len(pruned) // 4)
            pruned = pruned[:keep_first] + pruned[-keep_last:]

        return pruned
    
    def _fallback_window_prune(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Simple window-based pruning fallback when task-aware components fail."""
        if len(messages) <= self.window_size:
            return messages
        
        protected_first = min(self.protect_first_n, len(messages))
        protected_last = min(self.protect_last_n, len(messages))

        keep_count = max(protected_first + protected_last, self.window_size)
        if len(messages) > keep_count:
            # Drop oldest messages beyond window_size
            return messages[-keep_count:]
        
        return messages

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        """Quick rough check before the API call. Default returns False."""
        return False

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins."""
        pass

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Called at real session boundaries. NOT called per-turn."""
        pass

    def on_session_reset(self) -> None:
        """Called on /new or /reset."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.archive_count = 0

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas this engine provides. Default returns empty list."""
        return []

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent. Only called for engine-specific tools."""
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def get_status(self) -> dict[str, Any]:
        """Return status dict for display/logging."""
        status = {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "archive_count": self.archive_count,
        }
        
        # Add task-aware status if available
        if self._injector is not None:
            active_tasks = self._injector.get_active_task_ids()
            if active_tasks:
                status["active_tasks"] = active_tasks
        
        return status

    def update_model(self, model: str, context_length: int, **kwargs) -> None:
        """Called when the user switches models."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


# -- Plugin registration (required for discovery) --------------------------

def register(collector, config=None):
    """Register this engine with the Hermes plugin system.

    C9-A: accept the config dict from the loader so the user's
    context.rolling_window section (window_size, max_tokens, thresholds)
    actually reaches the engine instead of silently using defaults.
    """
    collector.register_context_engine(RollingWindowContextEngine(**(config or {})))

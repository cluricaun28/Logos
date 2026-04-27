"""Rolling Window Context Engine.

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
3. Drop oldest messages when window_size is exceeded
4. Enforce hard token budget with aggressive truncation as last resort
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


class RollingWindowContextEngine(ContextEngine):
    """Rolling window context engine — deterministic pruning without LLM summarization.

    Inherits from ContextEngine ABC — all abstract methods must be implemented.
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

    # Compaction parameters (override defaults as needed)
    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    def __init__(self, **kwargs):
        super().__init__()
        # Set engine-specific config from kwargs or defaults
        self.window_size: int = kwargs.get("window_size", 20)
        self.max_tokens: int = kwargs.get("max_tokens", 131072)

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        if isinstance(usage, dict):
            self.last_prompt_tokens = int(usage.get("prompt_tokens", 0))
            self.last_completion_tokens = int(usage.get("completion_tokens", 0))
            self.last_total_tokens = int(usage.get("total_tokens", 0))

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Return True if compaction should fire this turn."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
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

        Algorithm:
          1. Strip raw assistant tool calls entirely (verbose JSON bloat)
          2. Truncate role:"tool" results to first/last 3 lines
          3. Drop oldest messages when window_size is exceeded
          4. Enforce hard token budget with aggressive truncation as last resort
        """
        if not messages:
            return messages

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

        # Step 3: Drop oldest messages when window_size is exceeded
        if len(truncated) > self.window_size:
            # Protect first_n and last_n messages
            protected_first = min(self.protect_first_n, len(truncated))
            protected_last = min(self.protect_last_n, len(truncated))

            # Calculate how many to drop from the middle
            keep_count = max(protected_first + protected_last, self.window_size)
            if len(truncated) > keep_count:
                # Drop oldest messages beyond window_size
                truncated = truncated[-keep_count:]

        # Step 4: Enforce hard token budget with aggressive truncation as last resort
        if current_tokens and current_tokens > self.max_tokens:
            # Aggressive truncation — reduce to first/last N messages only
            keep_first = max(1, len(truncated) // 4)
            keep_last = max(1, len(truncated) // 4)
            truncated = truncated[:keep_first] + truncated[-keep_last:]

        return truncated

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick rough check before the API call. Default returns False."""
        return False

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
        }

    def update_model(self, model: str, context_length: int, **kwargs) -> None:
        """Called when the user switches models."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


# -- Plugin registration (required for discovery) --------------------------

def register(collector):
    """Register this engine with the Hermes plugin system."""
    collector.register_context_engine(RollingWindowContextEngine())

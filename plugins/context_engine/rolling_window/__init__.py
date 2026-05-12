"""Incremental Tail-Off Context Engine.

Replaces the old rolling_window engine. Instead of aggressive archival
to a fixed message count, this engine:

  1. Fires when prompt tokens exceed threshold (config-driven)
  2. Strips tool_calls and truncates verbose tool results (cheap pre-pass)
  3. Drops oldest messages one at a time until token count falls
     below archive_target (config-driven fraction of context_length)
  4. Enforces a hard ceiling as last resort (drops to 50/50 split)

No LLM summarization. No task tracking. No semantic vectors.
Perpetual Memory saves everything verbatim — the model can retrieve.

Config-driven parameters (all from config.yaml context.rolling_window):
  - threshold_percent:    fire when prompt_tokens exceed this fraction (default 0.75)
  - archive_target:       prune down to this fraction (default 0.65)
  - hard_ceiling_percent: absolute max before emergency cut (default 0.85)
  - context_length:       from model metadata or vLLM --max-model-len
"""

from __future__ import annotations

import logging
from typing import Any

from agent.context_engine import ContextEngine
from agent.model_metadata import estimate_messages_tokens_rough

logger = logging.getLogger(__name__)


class RollingWindowContextEngine(ContextEngine):
    """Incremental tail-off context engine.

    Keeps the context window as full as possible. When threshold is
    exceeded, drops oldest messages until we're back under archive_target.
    No task tracking, no semantic vectors, no LLM summarization.
    """

    @property
    def name(self) -> str:
        return "rolling_window"

    # Token state -- initialised in __init__ (not class-level defaults)
    last_prompt_tokens: int
    last_completion_tokens: int
    last_total_tokens: int
    threshold_tokens: int
    context_length: int
    archive_count: int

    def __init__(self, **kwargs):
        super().__init__()

        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.threshold_tokens = 0
        self.context_length = 0
        self.archive_count = 0

        # Config-driven parameters with defaults matching current config.yaml
        self.threshold_percent: float = kwargs.get("threshold_percent", 0.75)
        self.archive_target: float = kwargs.get("archive_target", 0.65)
        self.hard_ceiling_percent: float = kwargs.get("hard_ceiling_percent", 0.85)
        self.protect_first_n: int = kwargs.get("protect_first_n", 3)
        self.protect_last_n: int = kwargs.get("protect_last_n", 6)

        logger.info(
            "RollingWindowContextEngine (tail-off) initialized: "
            "threshold=%.0f%%, archive_target=%.0f%%, hard_ceiling=%.0f%%",
            self.threshold_percent * 100,
            self.archive_target * 100,
            self.hard_ceiling_percent * 100,
        )

    # -- ContextEngine ABC --------------------------------------------------

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
        focus_topic: str = None,
    ) -> list[dict[str, Any]]:
        """Incremental tail-off archive.

        Strategy:
          1. Strip assistant tool_calls (verbose JSON bloat)
          2. Truncate verbose tool results to first/last 3 lines
          3. Incrementally drop oldest unprotected messages until
             estimated tokens fall below archive_target
          4. Hard ceiling: if still over hard_ceiling_percent, do 50/50 split
        """
        if not messages:
            return messages

        target_tokens = int(self.context_length * self.archive_target)
        hard_ceiling_tokens = int(self.context_length * self.hard_ceiling_percent)
        before = len(messages)

        # Step 1: Strip assistant tool_calls
        pruned = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                pruned.append({**msg, "tool_calls": None})
            else:
                pruned.append(msg)

        # Step 2: Truncate verbose tool results
        truncated = []
        for msg in pruned:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.split("\n")) > 6:
                    lines = content.split("\n")
                    truncated_content = (
                        "\n".join(lines[:3])
                        + "\n...[truncated]...\n"
                        + "\n".join(lines[-3:])
                    )
                    msg = {**msg, "content": truncated_content}
            truncated.append(msg)

        # Step 3: Incremental tail-off -- drop oldest until under target
        # Protect first N (system prompt area) and last N (current conversation)
        protected_first = min(self.protect_first_n, len(truncated))
        drop_idx = 1  # start dropping after system prompt at index 0

        # Use gateway's actual token count if available; otherwise estimate
        if current_tokens is not None:
            estimated = current_tokens
        else:
            estimated = estimate_messages_tokens_rough(truncated)

        while True:
            protected_last_end = max(
                protected_first, len(truncated) - self.protect_last_n
            )
            if (
                drop_idx >= protected_last_end
                or estimated <= target_tokens
                or len(truncated) <= protected_first + self.protect_last_n
            ):
                break

            # Remove message at drop_idx (oldest unprotected)
            truncated.pop(drop_idx)
            # After first drop, use rough estimate on the reduced set
            estimated = estimate_messages_tokens_rough(truncated)
            # Don't advance drop_idx -- next message is now at same index

        # Step 4: Hard ceiling safety net
        step4_cut = False
        if estimated > hard_ceiling_tokens and len(truncated) > protected_first + self.protect_last_n:
            keep_first = max(1, len(truncated) // 4)
            keep_last = max(1, len(truncated) // 4)
            truncated = truncated[:keep_first] + truncated[-keep_last:]
            step4_cut = True

        self.archive_count += 1

        logger.info(
            "RollingWindow tail-off archive: %d->%d msgs, est_tokens=%d, "
            "target=%d, ceiling=%d, hard_cut=%s",
            before, len(truncated), estimated,
            target_tokens, hard_ceiling_tokens, step4_cut,
        )

        return truncated

    # -- Session lifecycle ---------------------------------------------------

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
        """Called on /new or /reset. Reset per-session state."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.archive_count = 0

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas this engine provides. Default returns empty list."""
        return []

    def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent. Only called for engine-specific tools."""
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Status --------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
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
        }

    def update_model(self, model: str, context_length: int, **kwargs) -> None:
        """Called when the user switches models."""
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)


# -- Plugin registration (required for discovery) --------------------------

def register(collector, config=None):
    """Register this engine with the Hermes plugin system."""
    collector.register_context_engine(RollingWindowContextEngine(**(config or {})))

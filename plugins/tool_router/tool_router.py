"""Tool Router — Selective Injection System (repo copy).

Reduces context bloat by splitting tools into two tiers:
  - ESSENTIAL: Full JSON schema injected on every API call
  - DEFERRED: Compact index in system prompt; schema promoted on first call
    (promotion is implemented in run_agent.py's dispatch path) and the model
    reads the RL page for parameter details when needed.

Essential set re-audited 2026-08-21 from 90 days of session JSONL
(91 sessions, 14,006 lines, owner's fleet):
  - promoted to essential: process (130 calls / 33 sessions),
    vision_analyze (148 calls / 9 sessions)
  - demoted to deferred:   skills_list (2 sessions), clarify (3 sessions),
    PM extended suite: query_messages, get_messages, smart_retrieve,
    source_analyze, topic_flow, context_depth

Import order (model_tools._get_tool_router): repo copy first, safe-harbor
copy (~/.hermes/plugins/tool_router/) as fallback. The repo copy is the
version-controlled source of truth; the safe-harbor copy remains for
per-user overrides.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# PM/RL tools that stay full-schema (injected by the memory plugin path,
# listed here so is_deferred() agrees with the memory plugin's CORE set).
PM_CORE_TOOLS: set[str] = {
    "perpetual_search",
    "reference_library_search",
    "session_search",
    "recent_messages",
}

# PM/RL tools demoted to the deferred tier (2026-08-21 audit). The memory
# plugin stops force-injecting these; they ride the deferred index and are
# promoted on first call.
PM_EXTENDED_TOOLS: set[str] = {
    "query_messages",
    "get_messages",
    "smart_retrieve",
    "source_analyze",
    "topic_flow",
    "context_depth",
}


@dataclass
class ToolRouterConfig:
    """Configuration for the selective injection tool router."""

    # Essential tools — full schema injected on every API call.
    # Re-audited 2026-08-21 (90-day session usage, see module docstring).
    essential_tools: set[str] = field(default_factory=lambda: {
        # File operations (read_file 299c/74s, patch 410c/70s,
        # write_file 132c/50s, search_files 76c/26s)
        "read_file",
        "write_file",
        "patch",
        "search_files",

        # Execution (terminal 3256c/87s, execute_code 439c/42s,
        # process 130c/33s — promoted from deferred)
        "terminal",
        "execute_code",
        "process",

        # Memory & Perpetual Context recall
        # (memory 131c/43s; PM core set is PM_CORE_TOOLS above)
        "memory",
        "session_search",
        "perpetual_search",
        "reference_library_search",
        "recent_messages",

        # Web search (conditionally available when SearXNG/Firecrawl running)
        # (web_search 60c/21s, web_extract 31c/18s)
        "web_search",
        "web_extract",

        # Skills management (skill_view 48c/26s, skill_manage 60c/17s;
        # skills_list demoted — 2 sessions in 90d)
        "skill_view",
        "skill_manage",

        # Core agent workflow (todo 70c/29s, cronjob 30c/18s,
        # delegate_task 19c/10s)
        "todo",
        "cronjob",
        "delegate_task",

        # Media (vision_analyze 148c/9s — promoted from deferred)
        "vision_analyze",
    })

    # Deferred tools — compact index only; schema promoted on first call;
    # model reads the RL page for parameter details when needed.
    deferred_tools: set[str] = field(default_factory=lambda: {
        # Browser suite (low usage: navigate 5c/3s, console 5c/2s)
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_type",
        "browser_back",
        "browser_press",
        "browser_scroll",
        "browser_console",
        "browser_get_images",
        "browser_vision",

        # Communication & media (rare usage)
        "send_message",
        "text_to_speech",
        "clarify",
        "skills_list",
        "image_generate",

        # Feishu/Lark tools (niche, rarely used)
        "feishu_doc_read",
        "feishu_drive_add_comment",
        "feishu_drive_list_comments",
        "feishu_drive_list_comment_replies",
        "feishu_drive_reply_comment",

        # PM/RL extended suite (2026-08-21 tiering; see PM_EXTENDED_TOOLS)
        *PM_EXTENDED_TOOLS,
    })

    # RL tools directory path
    rl_tools_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.hermes/reference-library/tools")
    )

    # Failure tracking for transient injection fallback
    failure_threshold: int = 3
    success_reset_count: int = 2


# ---------------------------------------------------------------------------
# Tool Router — Lazy Singleton
# ---------------------------------------------------------------------------

class ToolRouter:
    """Selective injection router for Logos tools.

    Splits tools into essential (full schema) and deferred (index +
    promotion-on-first-call). Thread-safe, configurable, with failure
    tracking fallback.
    """

    def __init__(self, config: ToolRouterConfig | None = None):
        self._config = config or ToolRouterConfig()
        self._lock = threading.RLock()

        # Failure tracking: tool_name -> consecutive failures
        self._failure_counts: dict[str, int] = {}
        self._success_counts: dict[str, int] = {}

        # Cache for deferred tool index (regenerate when tools change)
        self._cached_index: tuple[tuple[frozenset[str], ...], str] | None = None

    @property
    def config(self) -> ToolRouterConfig:
        return self._config

    def is_essential(self, tool_name: str) -> bool:
        """Check if a tool should have its full schema injected."""
        return tool_name in self._config.essential_tools

    def is_deferred(self, tool_name: str) -> bool:
        """Check if a tool should use the deferred tier (index + promotion).

        PM core tools are essential even though they ride the memory-plugin
        injection path rather than this router's essential set.
        """
        if tool_name in PM_CORE_TOOLS:
            return False
        return (
            tool_name in self._config.deferred_tools
            or not self.is_essential(tool_name)
        )

    def get_essential_definitions(
        self,
        all_definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter tool definitions to only essential tools.

        PM core tools are included: they are part of the essential tier even
        though the memory plugin also injects them (dedup happens at the
        injection site, keyed on function name).
        """
        names = self._config.essential_tools | PM_CORE_TOOLS
        return [
            d for d in all_definitions
            if d["function"]["name"] in names
        ]

    def get_deferred_definitions(
        self,
        all_definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter tool definitions to only deferred tools."""
        return [
            d for d in all_definitions
            if self.is_deferred(d["function"]["name"])
        ]

    def get_deferred_index(
        self,
        injected_names: set[str] | frozenset[str] | None = None,
        extra_deferred: set[str] | frozenset[str] | None = None,
    ) -> str:
        """Build compact markdown index of deferred tools for system prompt.

        Only lists tools that are genuinely in the deferred tier AND not
        already fully injected (e.g. via a force_toolsets override) — no
        double listing.

        ``extra_deferred``: tools that were demoted this session (formerly
        full-schema, now index-only). Listed under a separate heading so the
        model knows they exist and self-re-list on next call.

        Example output:
        ## Deferred Tools (RL Lookup Required)

        When you need a tool listed below, read its full schema from the
        Reference Library **before** calling it:

        ### Browser Suite
        - '''browser_navigate''' — Navigate to URL, initialize browser session
        """
        excluded = frozenset(injected_names or ())
        demoted_key = frozenset(extra_deferred or ())
        cache_key = (excluded, demoted_key)
        if self._cached_index is not None and self._cached_index[0] == cache_key:
            return self._cached_index[1]

        with self._lock:
            if self._cached_index is not None and self._cached_index[0] == cache_key:
                return self._cached_index[1]

            lines = []
            lines.append("## Deferred Tools (RL Lookup Required)")
            lines.append("")
            lines.append(
                "Tools listed below are deferred: their schemas are loaded on "
                "first use. When you need one, read its full schema from the "
                "Reference Library **before** calling it:"
            )
            lines.append("")
            lines.append(
                '```python\nread_file("~/.hermes/reference-library/tools/{tool_name}.md")\n```'
            )
            lines.append("")

            # Group deferred tools by category
            categories = {
                "Browser Suite": [
                    "browser_navigate", "browser_snapshot", "browser_click",
                    "browser_type", "browser_back", "browser_press",
                    "browser_scroll", "browser_console", "browser_get_images",
                    "browser_vision"
                ],
                "Communication & Media": [
                    "send_message", "text_to_speech", "vision_analyze",
                    "clarify", "skills_list", "image_generate"
                ],
                "Memory & Recall (extended)": [
                    "query_messages", "get_messages", "smart_retrieve",
                    "source_analyze", "topic_flow", "context_depth"
                ],
                "Feishu/Lark": [
                    "feishu_doc_read", "feishu_drive_add_comment",
                    "feishu_drive_list_comments",
                    "feishu_drive_list_comment_replies",
                    "feishu_drive_reply_comment"
                ],
            }

            # Short descriptions for each tool (optimized for model parsing)
            descriptions = {
                "browser_navigate": "Navigate to URL, initialize browser session",
                "browser_snapshot": "Get accessibility tree with interactive element refs",
                "browser_click": "Click element by ref ID from snapshot",
                "browser_type": "Type text into input field by ref ID",
                "browser_back": "Go back in browser history",
                "browser_press": "Press keyboard key (Enter, Tab, Escape)",
                "browser_scroll": "Scroll page up or down",
                "browser_console": "Get console output / evaluate JavaScript expression",
                "browser_get_images": "List all images with URLs and alt text",
                "browser_vision": "Screenshot + AI vision analysis of current page",
                "send_message": "Send message to connected platform (Telegram, Discord, etc)",
                "text_to_speech": "Convert text to speech audio (voice messages)",
                "vision_analyze": "Analyze image with AI vision (URL or local file)",
                "clarify": "Ask the user a multiple-choice or open-ended question",
                "skills_list": "List available skills (name + description)",
                "image_generate": "Generate an image via the configured local image model",
                "query_messages": "Master message query: filters, time ranges, stats, direct ID lookup",
                "get_messages": "SQL LIKE pattern search over raw message content",
                "smart_retrieve": "Adaptive retrieval: auto/recent/topic/decision_trace/file_history",
                "source_analyze": "Source intelligence for web_search results (alignment, omissions, bias)",
                "topic_flow": "View/add session topic clusters; drift detection",
                "context_depth": "Get/set perpetual-memory recall depth level",
                "feishu_doc_read": "Read Feishu/Lark document content as plain text",
                "feishu_drive_add_comment": "Add whole-document comment on Feishu doc",
                "feishu_drive_list_comments": "List comments on Feishu document",
                "feishu_drive_list_comment_replies": "List replies in a comment thread",
                "feishu_drive_reply_comment": "Reply to comment thread on Feishu doc",
                "process": "Manage background processes (list/poll/wait/kill)",
            }

            listed_any = False
            for category, tools in categories.items():
                # Only show category if at least one tool is actually deferred,
                # not already injected, and known to the router.
                active_tools = [
                    t for t in tools
                    if self.is_deferred(t) and t not in excluded
                ]
                if not active_tools:
                    continue

                lines.append(f"### {category}")
                for tool_name in active_tools:
                    desc = descriptions.get(tool_name, "No description")
                    # Use ''' delimiters for easy model parsing/searching
                    lines.append(f"- '''{tool_name}''' — {desc}")
                lines.append("")
                listed_any = True

            if not listed_any and not extra_deferred:
                # Nothing deferred (all tools force-injected) — emit nothing.
                self._cached_index = (cache_key, "")
                return ""

            # Recently demoted tools — self-re-list on next call. Skip any
            # already listed in the static deferred categories (no dupes);
            # this section is for tools that aren't in the static index.
            _static_deferred = set(self._config.deferred_tools)
            demoted = [
                t for t in (extra_deferred or ())
                if t not in excluded
                and t not in self._config.essential_tools
                and t not in _static_deferred
            ]
            if demoted:
                lines.append("### Recently Demoted (self-re-list on next call)")
                for tool_name in demoted:
                    desc = descriptions.get(tool_name, "Previously available tool — read the RL page before calling")
                    lines.append(f"- '''{tool_name}''' — {desc}")
                lines.append("")

            lines.append(
                "**CRITICAL:** Never guess deferred tool parameters. "
                "Always read the RL page first. The tool's schema is loaded "
                "automatically on first successful call."
            )
            lines.append(
                f"RL path: `~/.hermes/reference-library/tools/{{tool_name}}.md`"
            )

            index = "\n".join(lines)
            self._cached_index = (cache_key, index)
            return index

    def clear_cache(self):
        """Clear cached index (call after tools change)."""
        with self._lock:
            self._cached_index = None

    # -----------------------------------------------------------------------
    # Failure tracking — transient injection fallback
    # -----------------------------------------------------------------------

    def track_failure(self, tool_name: str, error_msg: str = "") -> bool:
        """Track a tool failure. Returns True if threshold reached."""
        with self._lock:
            # Reset success count on first failure in burst
            if self._failure_counts.get(tool_name, 0) == 0:
                self._success_counts[tool_name] = 0

            self._failure_counts[tool_name] = self._failure_counts.get(tool_name, 0) + 1
            return self._failure_counts[tool_name] >= self._config.failure_threshold

    def track_success(self, tool_name: str):
        """Track a tool success. Resets failure count."""
        with self._lock:
            self._failure_counts[tool_name] = 0
            self._success_counts[tool_name] = self._success_counts.get(tool_name, 0) + 1

    def should_inject_fallback(self, tool_name: str) -> bool:
        """Check if transient context injection is needed for a failing tool."""
        with self._lock:
            failures = self._failure_counts.get(tool_name, 0)
            successes_after = self._success_counts.get(tool_name, 0)
            return (failures >= self._config.failure_threshold and
                    successes_after < self._config.success_reset_count)

    def get_transient_context(self, tool_name: str) -> str | None:
        """Get fallback context for a failing deferred tool.

        Reads the RL page if available, returns formatted schema + guidance.
        """
        rl_path = Path(self._config.rl_tools_dir) / f"{tool_name}.md"
        try:
            content = rl_path.read_text(encoding="utf-8")
            return f"[Transient context for '{tool_name}' — read full schema above]\n\n{content[:3000]}"
        except (FileNotFoundError, OSError):
            logger.warning("No RL page found for failing tool '%s'", tool_name)
            return None

    def get_tool_stats(self) -> dict[str, dict[str, int]]:
        """Get failure/success stats for all tracked tools."""
        with self._lock:
            result = {}
            for name in set(list(self._failure_counts.keys()) + list(self._success_counts.keys())):
                result[name] = {
                    "failures": self._failure_counts.get(name, 0),
                    "successes": self._success_counts.get(name, 0),
                }
            return result

    def reset_tool_state(self, tool_name: str):
        """Reset all tracking state for a tool."""
        with self._lock:
            self._failure_counts.pop(tool_name, None)
            self._success_counts.pop(tool_name, None)


# ---------------------------------------------------------------------------
# Lazy Singleton Accessor
# ---------------------------------------------------------------------------

_router_instance: ToolRouter | None = None
_router_lock = threading.Lock()


def get_tool_router(config: ToolRouterConfig | None = None) -> ToolRouter | None:
    """Get the ToolRouter singleton. Returns None if unavailable."""
    global _router_instance

    if _router_instance is not None:
        return _router_instance

    with _router_lock:
        if _router_instance is not None:
            return _router_instance

        try:
            _router_instance = ToolRouter(config=config)
            logger.info("ToolRouter initialized — selective injection active")
            return _router_instance
        except Exception as e:
            logger.warning("Failed to initialize ToolRouter: %s", e)
            return None


def reset_tool_router():
    """Reset the singleton (for testing or reconfiguration)."""
    global _router_instance
    with _router_lock:
        _router_instance = None

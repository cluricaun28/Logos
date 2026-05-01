"""Task-Aware Context Pruning — Selective Archival Logic.

Replaces blind "drop middle" archival with selective archival of completed
tasks while preserving active work. Falls back to original algorithm when no
task markers are present.

Algorithm:
  1. Strip tool calls & truncate results (existing)
  2. Categorize messages by task status (closed vs open) — via task_tagger.categorize_tasks()
  3. Drop closed tasks first (searchable in Perpetual Memory)
  4. If still over budget, drop oldest open tasks from unprotected middle zone
  5. Inject task summary for archived closed tasks

Design notes:
- Uses index-based tracking instead of id() comparisons — survives dict copies
- Gracefully degrades if model_metadata import fails (falls back to char estimate)
- Single source of truth for categorization lives in task_tagger module
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

logger = logging.getLogger(__name__)


# Lazy token estimator — gracefully degrades if model_metadata is unavailable.
def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate token count for a message list with graceful fallback."""
    try:
        from agent.model_metadata import estimate_messages_tokens_rough
        return estimate_messages_tokens_rough(messages)
    except (ImportError, AttributeError, ModuleNotFoundError):
        # Fallback: rough char-based estimate (4 chars ≈ 1 token)
        total_chars = sum(len(str(msg)) for msg in messages)
        return (total_chars + 3) // 4


# Public alias — used by __init__.py _log_archive_stats()
count_tokens = _estimate_tokens


def _strip_and_truncate(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Strip raw assistant tool calls and truncate role:'tool' results.

    Shared utility used by both TaskAwarePruner.archive() and original_archive().
    Returns a new list — does not mutate input messages.
    """
    result = []
    for msg in messages:
        # Strip tool_calls from assistant messages (verbose JSON bloat)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            msg = {**msg, "tool_calls": None}

        # Truncate role:'tool' results to first/last 3 lines
        if msg.get("role") == "tool":
            content = msg.get("content", "") or ""
            if isinstance(content, str) and len(content.split("\n")) > 6:
                lines = content.split("\n")
                truncated_content = (
                    "\n".join(lines[:3]) + "\n...[truncated]...\n" + "\n".join(lines[-3:])
                )
                msg = {**msg, "content": truncated_content}

        result.append(msg)
    return result


class TaskAwarePruner:
    """Selective archival based on task boundaries.

    When task markers are present, archives closed tasks first and keeps open
    tasks alive. Falls back to the original rolling window algorithm when no
    markers exist or when task_aware is disabled.
    """

    @staticmethod
    def archive(
        messages: List[Dict[str, Any]],
        max_tokens: int = 131072,
        target_tokens: int = None,
        protect_first_n: int = 3,
        protect_last_n: int = 30,
        window_size: int = 20,
        task_aware: bool = True,
    ) -> List[Dict[str, Any]]:
        """Task-aware archival with fallback to original algorithm.

        Args:
            messages: Full message list to archive (already annotated by tagger).
            max_tokens: Hard token budget for the context window.
            target_tokens: Target token count after pruning. If None, defaults to 70% of max_tokens.
                Set lower for more aggressive pruning under pressure.
            protect_first_n: First N messages never dropped (system prompt, etc.).
            protect_last_n: Last N messages never dropped (recent conversation).
            window_size: Target message count after archival.
            task_aware: If False, skip task-aware logic entirely.

        Returns:
            Archived message list.
        """
        from .task_tagger import MARKER_START_PREFIX, categorize_tasks, build_task_summary

        # Step 1: Strip tool calls & truncate results (existing logic)
        stripped = _strip_and_truncate(messages)

        # Step 2: Check if any task markers exist and task_aware is enabled
        has_markers = any(
            MARKER_START_PREFIX in (m.get("content") or "") for m in stripped
        )

        if not task_aware or not has_markers:
            logger.debug(
                "Task-aware pruner: no markers found (%d messages), using original algorithm",
                len(messages)
            )
            return original_archive(
                stripped, max_tokens, protect_first_n, protect_last_n, window_size, target_tokens
            )

        # Step 3: Categorize tasks — returns index sets, not message lists.
        # Index-based avoids fragile object-identity comparison after dict copies.
        closed_indices, open_indices = categorize_tasks(stripped)

        if not closed_indices:
            logger.debug(
                "Task-aware pruner: no closed tasks (%d messages), using original algorithm",
                len(messages)
            )
            return original_archive(
                stripped, max_tokens, protect_first_n, protect_last_n, window_size, target_tokens
            )

        logger.info(
            "Task-aware pruner: %d closed task indices, %d open task indices (total: %d)",
            len(closed_indices), len(open_indices), len(stripped)
        )

        # Step 4: Build remaining list using index sets directly.
        protected_head_indices = set(range(min(protect_first_n, len(stripped))))
        protected_tail_start = max(0, len(stripped) - protect_last_n)
        protected_tail_indices = set(range(protected_tail_start, len(stripped)))
        protected_indices = protected_head_indices | protected_tail_indices

        # Keep: all protected messages + open task messages outside protection
        keep_indices: Set[int] = set()
        keep_indices.update(protected_indices)
        for idx in open_indices:
            if idx not in protected_indices:
                keep_indices.add(idx)

        remaining = [stripped[i] for i in sorted(keep_indices)]

        # Step 5: Inject summary of archived closed tasks as a system message.
        # Extract actual messages from indices for the summary builder.
        closed_messages = [stripped[i] for i in sorted(closed_indices)]
        summary = build_task_summary(closed_messages)
        if summary:
            summary_msg = {
                "role": "system",
                "content": summary,
                "_archived_summary": True,  # Marker for debugging/inspection
            }
            # Insert after system prompt (index 1) or at the beginning
            insert_at = 1 if remaining and remaining[0].get("role") == "system" else 0
            remaining.insert(insert_at, summary_msg)
            logger.info(
                "Task-aware pruner: injected archived tasks summary (%d chars)", len(summary)
            )

        # Step 6: If still over budget, drop oldest from unprotected middle zone.
        # Pass `remaining` (not `stripped`) — indices must match the message list.
        if target_tokens is None:
            target_tokens = int(max_tokens * 0.7)  # Default: 70% of max for headroom
        current_tokens = _estimate_tokens(remaining)

        if current_tokens > target_tokens:
            logger.info(
                "Task-aware pruner: still over budget (%d > %d tokens), dropping oldest open tasks",
                current_tokens, target_tokens
            )
            remaining = TaskAwarePruner._drop_oldest_from_middle(
                remaining,
                target_tokens, protect_first_n, protect_last_n
            )

        logger.info(
            "Task-aware pruner: archived %d -> %d messages (%d closed tasks archived)",
            len(stripped), len(remaining), len(closed_indices)
        )

        # Post-archival validation: did we actually hit our target?
        final_tokens = _estimate_tokens(remaining)
        if final_tokens > target_tokens:
            logger.warning(
                "Task-aware pruner: failed to reach target (%d > %d tokens). "
                "Protected zones may be preventing further reduction.",
                final_tokens, target_tokens
            )

        return remaining

    @staticmethod
    def _drop_oldest_from_middle(
        messages: List[Dict[str, Any]],
        target_tokens: int,
        protect_first_n: int,
        protect_last_n: int,
    ) -> List[Dict[str, Any]]:
        """Drop oldest non-protected messages until under target token budget.

        Iteratively removes one message at a time from the unprotected middle zone,
        starting from the oldest, until we're under budget or no more droppable
        messages remain.

        Uses precomputed per-message token estimates for O(n) performance instead of
        re-estimating the full list each iteration (which would be O(n²)).
        """
        # Precompute per-message token estimates — avoids O(n²) re-scanning
        msg_tokens = [_estimate_tokens([m]) for m in messages]
        current_total = sum(msg_tokens)

        if current_total <= target_tokens:
            return messages

        # Protected zones (by index into messages list)
        protected_head = set(range(min(protect_first_n, len(messages))))
        protected_tail_start = max(0, len(messages) - protect_last_n)
        protected_tail = set(range(protected_tail_start, len(messages)))
        protected = protected_head | protected_tail

        # Build droppable indices (unprotected, non-system), sorted oldest-first.
        # System messages are never dropped — this includes the system prompt with
        # RL/PM/tool/skill hooks AND injected archived-task summaries.
        unprotected_end = max(protect_first_n, len(messages) - protect_last_n)
        droppable = []
        for idx in range(protect_first_n, unprotected_end):
            if idx not in protected and messages[idx].get("role") != "system":
                # Extra safety: never drop critical injected context.
                # These are appended after archival by run_agent.py and contain
                # active task state that the model needs to continue work.
                content = messages[idx].get("content", "") or ""
                if "_archived_summary" in str(messages[idx]) or "Archived Tasks" in content:
                    continue
                if "[Your active task list was preserved" in content:
                    # Todo snapshot — contains pending/in_progress action items
                    continue
                if content.startswith("## Context Bridge") or "<context-bridge>" in content:
                    # Perpetual context bridge — active tasks, files, knowledge gaps
                    continue
                droppable.append(idx)

        # Drop oldest first until under budget
        drop_set: Set[int] = set()
        for idx in droppable:
            if current_total - msg_tokens[idx] <= target_tokens:
                break
            drop_set.add(idx)
            current_total -= msg_tokens[idx]
            logger.debug(
                "Task-aware pruner: dropped message %d (%d tokens), remaining: %d",
                idx, msg_tokens[idx], current_total
            )

        return [messages[i] for i in range(len(messages)) if i not in drop_set]


def original_archive(
    messages: List[Dict[str, Any]],
    max_tokens: int = 131072,
    protect_first_n: int = 3,
    protect_last_n: int = 30,
    window_size: int = 20,
    target_tokens: int = None,
) -> List[Dict[str, Any]]:
    """Original rolling window archival — shared fallback algorithm.

    Single source of truth used by both TaskAwarePruner (when no markers found)
    and RollingWindowContextEngine (when task-aware pruner fails entirely).

    Args:
        target_tokens: If set, use this as the budget ceiling instead of max_tokens.
            Allows pressure-based scaling from the caller.
    """
    if target_tokens is None:
        target_tokens = int(max_tokens * 0.7)  # Default: 70% for headroom

    # Step 1-2: Strip tool calls & truncate results (shared utility)
    truncated = _strip_and_truncate(messages)

    # Step 3: Drop oldest messages when window_size is exceeded.
    # CRITICAL: Must preserve the head (system prompt + hooks) AND tail (recent context).
    # The old code did `truncated[-keep_count:]` which dropped the system prompt entirely.
    if len(truncated) > window_size:
        protected_first = min(protect_first_n, len(truncated))
        protected_last = min(protect_last_n, len(truncated))

        keep_count = max(protected_first + protected_last, window_size)
        if len(truncated) > keep_count:
            # Keep head (system prompt + hooks) and fill rest from tail.
            # Never drop the first `protected_first` messages — they contain the
            # system prompt with RL/PM/tool/skill indexes that the model needs
            # to function after archival.
            head = truncated[:protected_first]
            remaining_slots = keep_count - protected_first
            tail = truncated[-remaining_slots:] if remaining_slots > 0 else []
            truncated = head + tail

    # Step 4: Enforce token budget with aggressive truncation as last resort.
    # Keeps first quarter (includes system prompt + hooks) and last quarter.
    current_tokens = _estimate_tokens(truncated)
    if current_tokens and current_tokens > target_tokens:
        keep_first = max(protect_first_n, len(truncated) // 4)  # Never drop below protect_first_n
        keep_last = max(1, len(truncated) // 4)
        truncated = truncated[:keep_first] + truncated[-keep_last:]

    return truncated

#!/usr/bin/env python3
"""
Async (background) delegation registry — Logos adaptation.

Backs ``delegate_task(background=true)``: the parent agent dispatches a
subagent that runs on a module-level daemon executor and returns a handle
immediately, so the user and model keep working while the child runs.

When the child finishes, a completion event is pushed onto the SHARED
``process_registry.completion_queue`` with ``type="async_delegation"``. The
CLI and gateway poll that queue while idle and forge a fresh turn from each
event. We deliberately reuse that rail rather than reaching into a running
agent loop:

  - completions surface as a NEW turn when idle, never spliced between a tool
    result and an assistant message. That keeps strict message-role alternation
    legal and the prompt cache intact.
  - we inherit the queue's de-dup and drain wiring for free.

LOGOS-SPECIFIC DIFFERENCES FROM UPSTREAM:

  1. **Richer completion events** — carry ``pm_session_id``,
     ``distillation_eligible``, and ``rl_entries_created`` for automatic
     Perpetual Memory persistence and RL distillation flagging.
  2. **PM persistence hook** — ``_persist_to_pm()`` writes the subagent's
     conversation turns to Perpetual Memory with a distinct session prefix.
  3. **Thread-safe PM writes** — the PM DB is opened lazily per-worker to
     avoid cross-thread SQLite locking issues.

This module owns ONLY the async lifecycle. The actual child build + run is
delegated back to ``delegate_tool._run_single_child`` via an injected
runner, so all credential leasing, heartbeat, timeout, and result-shaping
logic stays in one place.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
# A persistent daemon executor (NOT a `with ThreadPoolExecutor()` block, which
# would join on exit and defeat the whole point of async). Workers are daemon
# threads so a hard process exit doesn't hang on an in-flight child.
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = threading.Lock()
_executor_max_workers: int = 0

_records_lock = threading.Lock()
# delegation_id -> record dict. Kept for the lifetime of the run plus a short
# tail after completion so `list_async_delegations()` can show recent results.
_records: Dict[str, Dict[str, Any]] = {}

_DEFAULT_MAX_ASYNC_CHILDREN = 3
# How many completed records to retain for status queries before pruning.
_MAX_RETAINED_COMPLETED = 50


def _get_executor(max_workers: int) -> ThreadPoolExecutor:
    """Lazily create (or grow) the shared daemon executor.

    We never shrink — ThreadPoolExecutor can't resize — but if the configured
    cap grows between calls we rebuild a larger pool. Existing in-flight
    futures keep running on the old pool until it's garbage collected.
    """
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is None or max_workers > _executor_max_workers:
            # Daemon threads: thread_name_prefix aids debugging in stack dumps.
            _executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="logos-async-delegate",
            )
            _executor_max_workers = max_workers
        return _executor


def active_count() -> int:
    """Number of async delegations currently running."""
    with _records_lock:
        return sum(1 for r in _records.values() if r.get("status") == "running")


def _new_delegation_id() -> str:
    return f"deleg_{uuid.uuid4().hex[:8]}"


def _prune_completed_locked() -> None:
    """Drop the oldest completed records beyond the retention cap.

    Caller must hold ``_records_lock``.
    """
    completed = [
        (rid, r)
        for rid, r in _records.items()
        if r.get("status") != "running"
    ]
    if len(completed) <= _MAX_RETAINED_COMPLETED:
        return
    # Oldest-first by completion time (fall back to dispatch time).
    completed.sort(key=lambda kv: kv[1].get("completed_at") or kv[1].get("dispatched_at") or 0)
    for rid, _ in completed[: len(completed) - _MAX_RETAINED_COMPLETED]:
        _records.pop(rid, None)


# ---------------------------------------------------------------------------
# Logos-specific: Perpetual Memory persistence
# ---------------------------------------------------------------------------

def _persist_to_pm(
    delegation_id: str,
    conversation_messages: List[Dict[str, Any]],
    *,
    goal: str,
    parent_session_id: str = "",
    dispatch_time: float,
) -> Optional[str]:
    """Persist a subagent's conversation to Perpetual Memory.

    Returns the new PM session_id on success, None on failure. Best-effort:
    a failure here must not crash the worker or lose the completion event.

    Each async subagent gets its own PM session with a ``deleg_`` prefix so
    they're queryable independently and traceable back to the delegation.
    """
    try:
        # Lazy import — avoid pulling in PM DB on module load.
        from agent.perpetual_context_db import PerpetualContextDB

        pm_db = PerpetualContextDB.get_instance()
        pm_session_id = f"deleg_{delegation_id}"

        # Build a metadata payload linking this session to the delegation.
        metadata = {
            "delegation_id": delegation_id,
            "parent_session_id": parent_session_id,
            "goal": goal[:500],  # cap to avoid bloating metadata
            "dispatch_time": dispatch_time,
            "is_async_delegation": True,
        }

        # Write messages to PM — best effort, non-blocking.
        for msg in conversation_messages:
            if not isinstance(msg, dict):
                continue
            pm_db.write_message(
                session_id=pm_session_id,
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
                metadata=metadata,
                timestamp=msg.get("timestamp", time.time()),
                token_count=msg.get("token_count", 0),
            )

        logger.info(
            "Persisted async delegation %s to PM as session %s (%d messages)",
            delegation_id, pm_session_id, len(conversation_messages),
        )
        return pm_session_id

    except Exception as exc:
        # Non-fatal — log and continue. The completion event still fires.
        logger.debug(
            "Failed to persist async delegation %s to PM: %s",
            delegation_id, exc,
        )
        return None


def _is_distillation_eligible(summary: Optional[str], api_calls: int) -> bool:
    """Heuristic: flag results that likely contain high-signal knowledge.

    Returns True when the summary is substantive enough to potentially
    warrant Reference Library distillation. Called at completion time.
    """
    if not summary:
        return False
    # Heuristic: substantial text from a research-heavy delegation.
    # Adjust thresholds as the system learns.
    if len(summary) < 200:
        return False
    if api_calls < 3:
        return False
    return True


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch_async_delegation(
    *,
    goal: str,
    context: Optional[str],
    toolsets: Optional[List[str]],
    role: str,
    model: Optional[str],
    session_key: str,
    runner: Callable[[], Dict[str, Any]],
    interrupt_fn: Optional[Callable[[], None]] = None,
    max_async_children: int = _DEFAULT_MAX_ASYNC_CHILDREN,
    parent_session_id: str = "",
) -> Dict[str, Any]:
    """Spawn ``runner`` on the daemon executor and return a handle immediately.

    Parameters
    ----------
    goal, context, toolsets, role, model
        The dispatch-time task spec, captured verbatim for the rich
        completion block.
    session_key
        The gateway session_key captured on the parent thread BEFORE
        dispatch. Used to route the completion back to the originating
        session.
    runner
        Zero-arg callable that builds + runs the child and returns the same
        result dict ``_run_single_child`` produces. Runs on the worker thread.
    interrupt_fn
        Optional callable to signal the child to stop (used on shutdown /
        explicit cancel).
    max_async_children
        Concurrency cap. When at capacity the dispatch is REJECTED so a
        runaway model can't pile up unbounded background work.
    parent_session_id
        The parent agent's session ID (for PM traceability).

    Returns
    -------
    dict
        ``{"status": "dispatched", "delegation_id": ...}`` on success, or
        ``{"status": "rejected", "error": ...}`` when at capacity.
    """
    delegation_id = _new_delegation_id()
    dispatched_at = time.time()
    record: Dict[str, Any] = {
        "delegation_id": delegation_id,
        "goal": goal,
        "context": context,
        "toolsets": list(toolsets) if toolsets else None,
        "role": role,
        "model": model,
        "session_key": session_key,
        "parent_session_id": parent_session_id,
        "status": "running",
        "dispatched_at": dispatched_at,
        "completed_at": None,
        "interrupt_fn": interrupt_fn,
    }
    # Capacity check and record insert under ONE lock hold — checking
    # active_count() separately would let two concurrent dispatches both
    # pass the check and exceed the cap (TOCTOU fix from upstream 1c00cb6e0).
    with _records_lock:
        running = sum(
            1 for r in _records.values() if r.get("status") == "running"
        )
        if running >= max_async_children:
            return {
                "status": "rejected",
                "error": (
                    f"Async delegation capacity reached ({max_async_children} "
                    f"running). Wait for one to finish (its result will re-enter "
                    f"the chat), or run this task synchronously "
                    f"(background=false). Raise delegation.max_async_children in "
                    f"config.yaml to allow more concurrent background subagents."
                ),
            }
        _records[delegation_id] = record

    executor = _get_executor(max_async_children)

    def _worker() -> None:
        result: Dict[str, Any] = {}
        status = "error"
        try:
            result = runner() or {}
            status = result.get("status") or "completed"
        except Exception as exc:  # noqa: BLE001 — must never crash the worker
            logger.exception("Async delegation %s crashed", delegation_id)
            result = {
                "status": "error",
                "summary": None,
                "error": f"{type(exc).__name__}: {exc}",
                "api_calls": 0,
                "duration_seconds": round(time.time() - dispatched_at, 2),
            }
            status = "error"
        finally:
            _finalize(delegation_id, result, status, dispatched_at)

    try:
        executor.submit(_worker)
    except Exception as exc:  # pragma: no cover — pool submit failure is rare
        with _records_lock:
            _records.pop(delegation_id, None)
        return {
            "status": "rejected",
            "error": f"Failed to schedule async delegation: {exc}",
        }

    logger.info(
        "Dispatched async delegation %s (session_key=%s): %s",
        delegation_id, session_key or "<cli>", (goal or "")[:80],
    )
    return {"status": "dispatched", "delegation_id": delegation_id}


def _finalize(
    delegation_id: str,
    result: Dict[str, Any],
    status: str,
    dispatched_at: float,
) -> None:
    """Mark a record complete, persist to PM, and push the completion event."""
    with _records_lock:
        record = _records.get(delegation_id)
        if record is None:
            return
        record["status"] = status
        record["completed_at"] = time.time()
        record["interrupt_fn"] = None  # drop the closure; child is done
        event_record = dict(record)
        _prune_completed_locked()

    # Logos: persist the subagent's conversation to Perpetual Memory.
    conversation = result.get("messages")
    if conversation and isinstance(conversation, list):
        pm_session_id = _persist_to_pm(
            delegation_id,
            conversation,
            goal=record.get("goal", ""),
            parent_session_id=record.get("parent_session_id", ""),
            dispatch_time=dispatched_at,
        )
        if pm_session_id:
            event_record["pm_session_id"] = pm_session_id

    # Logos: flag for distillation eligibility.
    event_record["distillation_eligible"] = _is_distillation_eligible(
        result.get("summary"), result.get("api_calls", 0)
    )

    _push_completion_event(event_record, result, status)


def _push_completion_event(
    record: Dict[str, Any], result: Dict[str, Any], status: str
) -> None:
    """Push a type='async_delegation' event onto the shared completion queue.

    Best-effort: a failure here must not crash the worker, but it WOULD mean a
    silently-lost result, so we log loudly.
    """
    try:
        from tools.process_registry import process_registry
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s finished but process_registry import failed; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )
        return

    summary = result.get("summary")
    error = result.get("error")
    dispatched_at = record.get("dispatched_at") or time.time()
    completed_at = record.get("completed_at") or time.time()

    evt = {
        "type": "async_delegation",
        "delegation_id": record.get("delegation_id"),
        # session_key routes the completion back to the originating gateway
        # session; empty string => CLI (single-session) path.
        "session_key": record.get("session_key", ""),
        "goal": record.get("goal", ""),
        "context": record.get("context"),
        "toolsets": record.get("toolsets"),
        "role": record.get("role"),
        "model": result.get("model") or record.get("model"),
        "status": status,
        "summary": summary,
        "error": error,
        "api_calls": result.get("api_calls", 0),
        "duration_seconds": result.get(
            "duration_seconds", round(completed_at - dispatched_at, 2)
        ),
        "dispatched_at": dispatched_at,
        "completed_at": completed_at,
        "exit_reason": result.get("exit_reason"),
        # --- Logos-specific fields ---
        "pm_session_id": record.get("pm_session_id"),
        "distillation_eligible": record.get("distillation_eligible", False),
        "rl_entries_created": result.get("rl_entries_created", 0),
        "parent_session_id": record.get("parent_session_id", ""),
    }
    try:
        process_registry.completion_queue.put(evt)
    except Exception as exc:  # pragma: no cover
        logger.error(
            "Async delegation %s: failed to enqueue completion event; "
            "result lost: %s",
            record.get("delegation_id"), exc,
        )


# ---------------------------------------------------------------------------
# Status & lifecycle
# ---------------------------------------------------------------------------

def list_async_delegations() -> List[Dict[str, Any]]:
    """Snapshot of async delegations (running + recently completed).

    Safe to call from any thread. Excludes the non-serialisable interrupt_fn.
    """
    with _records_lock:
        return [
            {k: v for k, v in r.items() if k != "interrupt_fn"}
            for r in _records.values()
        ]


def interrupt_all(reason: str = "shutdown") -> int:
    """Signal every running async delegation to stop. Returns how many.

    Used on ``/stop`` and gateway shutdown so a dangling background subagent
    can't keep burning tokens with no one listening. The child still emits a
    completion event (status='interrupted') via the normal finalize path.
    """
    count = 0
    with _records_lock:
        targets = [
            r for r in _records.values() if r.get("status") == "running"
        ]
    for r in targets:
        fn = r.get("interrupt_fn")
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception as exc:
                logger.debug(
                    "interrupt_all: %s interrupt failed: %s",
                    r.get("delegation_id"), exc,
                )
    if count:
        logger.info("Interrupted %d async delegation(s) (%s)", count, reason)
    return count


def _reset_for_tests() -> None:
    """Test-only: clear all state and tear down the executor."""
    global _executor, _executor_max_workers
    with _executor_lock:
        if _executor is not None:
            _executor.shutdown(wait=False)
        _executor = None
        _executor_max_workers = 0
    with _records_lock:
        _records.clear()

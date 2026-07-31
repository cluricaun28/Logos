"""Async/sync bridging helpers — shim for Logos fork.

Upstream extracted this from the monolithic run_agent.py.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from typing import Any, Coroutine, Optional

_DEFAULT_LOGGER = logging.getLogger(__name__)


def safe_schedule_threadsafe(
    coro: Coroutine[Any, Any, Any],
    loop: Optional[asyncio.AbstractEventLoop],
    *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop",
    log_level: int = logging.DEBUG,
) -> Optional[Future]:
    """Schedule ``coro`` on ``loop`` from a sync context, leak-safe."""
    log = logger if logger is not None else _DEFAULT_LOGGER
    if loop is None:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: loop is None", log_message)
        return None
    try:
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except RuntimeError:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: %s", log_message, asyncio.current_task())
        return None

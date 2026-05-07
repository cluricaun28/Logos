"""Shared utilities for the perpetual_context plugin."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

def _safe_except(
    exc: BaseException,
    msg: str = "",
    level: str = "error",
) -> None:
    """Handle an exception in a standard way.

    * Re-raises ``KeyboardInterrupt`` and ``SystemExit`` immediately.
    * Logs everything else at the requested level (*debug*, *warning*, *error*).

    Args:
        exc:   The caught exception.
        msg:   Optional context string appended to the log message.
        level: Log level name -- ``"debug"``, ``"warning"``, or ``"error"``.
    """
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc

    context = f": {msg}" if msg else ""
    detail = f"{type(exc).__name__}{context}"

    if level == "debug":
        logger.debug(detail, exc_info=False)
    elif level == "warning":
        logger.warning(detail, exc_info=False)
    else:
        logger.error(detail, exc_info=True)

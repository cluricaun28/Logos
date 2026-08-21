"""Resolve HERMES_HOME for standalone skill scripts.

Skill scripts may run outside the Logos process (e.g. system Python,
nix env, CI) where ``logos_constants`` is not importable.  This module
provides the same ``get_logos_home()`` and ``display_logos_home()``
contracts as ``logos_constants`` without requiring it on ``sys.path``.

When ``logos_constants`` IS available it is used directly so that any
future enhancements (profile resolution, Docker detection, etc.) are
picked up automatically.  The fallback path replicates the core logic
from ``logos_constants.py`` using only the stdlib.

All scripts under ``google-workspace/scripts/`` should import from here
instead of duplicating the ``HERMES_HOME = Path(os.getenv(...))`` pattern.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from logos_constants import display_logos_home as display_logos_home
    from logos_constants import get_logos_home as get_logos_home
except (ModuleNotFoundError, ImportError):

    def _default_home() -> Path:
        """Default home: ~/.hermes if it exists (legacy), else ~/.logos."""
        legacy = Path.home() / ".hermes"
        if legacy.exists():
            return legacy
        return Path.home() / ".logos"

    def get_logos_home() -> Path:
        """Return the Logos home directory.

        Mirrors ``logos_constants.get_logos_home()``:
        1. ``$LOGOS_HOME`` env var
        2. ``$HERMES_HOME`` env var (legacy)
        3. ``~/.logos`` if it exists
        4. ``~/.hermes`` (legacy fallback)
        """
        val = (os.environ.get("LOGOS_HOME", "").strip()
               or os.environ.get("HERMES_HOME", "").strip())
        return Path(val) if val else _default_home()

    def display_logos_home() -> str:
        """Return a user-friendly ``~/``-shortened display string.

        Mirrors ``logos_constants.display_logos_home()``."""
        home = get_logos_home()
        try:
            return "~/" + str(home.relative_to(Path.home()))
        except ValueError:
            return str(home)

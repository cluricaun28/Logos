"""Lazy dependency installer — minimal shim for Logos fork.

Upstream uses this to install optional backends at runtime.
Our fork has deps pre-installed, so this is a pass-through.
"""
from __future__ import annotations


class FeatureUnavailable(Exception):
    """Raised when a lazy dependency cannot be installed."""
    pass


def ensure(feature_name: str, prompt: bool = True) -> None:
    """Ensure dependencies for a feature are available.

    In our fork, all deps are pre-installed. This is a no-op.
    """
    # If the dep is truly missing, the actual import will fail
    # with a clear error message. No need for runtime pip install.
    pass

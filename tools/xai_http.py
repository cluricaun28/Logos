"""Shared helpers for direct xAI HTTP integrations."""

from __future__ import annotations


def hermes_xai_user_agent() -> str:
    """Return a stable Logos-specific User-Agent for xAI HTTP calls."""
    try:
        from logos_cli import __version__
    except Exception:
        __version__ = "unknown"
    return f"Logos-Agent/{__version__}"

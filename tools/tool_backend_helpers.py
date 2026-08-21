"""Shared helpers for tool backend selection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from utils import is_truthy_value


_DEFAULT_BROWSER_PROVIDER = "local"


def managed_nous_tools_enabled() -> bool:
    """Return True when the user has an active paid Nous subscription.

    The Tool Gateway is available to any Nous subscriber who is NOT on
    the free tier.  We intentionally catch all exceptions and return
    False — never block the agent startup path.
    """
    try:
        from logos_cli.auth import get_nous_auth_status

        status = get_nous_auth_status()
        if not status.get("logged_in"):
            return False

        from logos_cli.models import check_nous_free_tier

        if check_nous_free_tier():
            return False  # free-tier users don't get gateway access
        return True
    except Exception:
        return False


def normalize_browser_cloud_provider(value: object | None) -> str:
    """Return a normalized browser provider key."""
    provider = str(value or _DEFAULT_BROWSER_PROVIDER).strip().lower()
    return provider or _DEFAULT_BROWSER_PROVIDER


def resolve_openai_audio_api_key() -> str:
    """Prefer the voice-tools key, but fall back to the normal OpenAI key."""
    return (
        os.getenv("VOICE_TOOLS_OPENAI_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    ).strip()


def prefers_gateway(config_section: str) -> bool:
    """Return True when the user opted into the Tool Gateway for this tool.

    Reads ``<section>.use_gateway`` from config.yaml.  Never raises.
    """
    try:
        from logos_cli.config import load_config
        section = (load_config() or {}).get(config_section)
        if isinstance(section, dict):
            return is_truthy_value(section.get("use_gateway"), default=False)
    except Exception:
        pass
    return False


def fal_key_is_configured() -> bool:
    """Return True when FAL_KEY is set to a non-whitespace value.

    Consults both ``os.environ`` and ``~/.hermes/.env`` (via
    ``logos_cli.config.get_env_value`` when available) so tool-side
    checks and CLI setup-time checks agree.  A whitespace-only value
    is treated as unset everywhere.
    """
    value = os.getenv("FAL_KEY")
    if value is None:
        # Fall back to the .env file for CLI paths that may run before
        # dotenv is loaded into os.environ.
        try:
            from logos_cli.config import get_env_value

            value = get_env_value("FAL_KEY")
        except Exception:
            value = None
    return bool(value and value.strip())

"""Unit tests for tools/tool_backend_helpers.py.

Tests cover:
- managed_nous_tools_enabled() subscription-based gate
- normalize_browser_cloud_provider() coercion
- resolve_openai_audio_api_key() priority chain
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tools.tool_backend_helpers import (
    managed_nous_tools_enabled,
    normalize_browser_cloud_provider,
    prefers_gateway,
    resolve_openai_audio_api_key,
)


def _raise_import():
    raise ImportError("simulated missing module")


# ---------------------------------------------------------------------------
# managed_nous_tools_enabled
# ---------------------------------------------------------------------------
class TestManagedNousToolsEnabled:
    """Subscription-based gate: True for paid Nous subscribers."""

    def test_disabled_when_not_logged_in(self, monkeypatch):
        monkeypatch.setattr(
            "logos_cli.auth.get_nous_auth_status",
            lambda: {},
        )
        assert managed_nous_tools_enabled() is False

    def test_disabled_for_free_tier(self, monkeypatch):
        monkeypatch.setattr(
            "logos_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": True},
        )
        monkeypatch.setattr(
            "logos_cli.models.check_nous_free_tier",
            lambda: True,
        )
        assert managed_nous_tools_enabled() is False

    def test_enabled_for_paid_subscriber(self, monkeypatch):
        monkeypatch.setattr(
            "logos_cli.auth.get_nous_auth_status",
            lambda: {"logged_in": True},
        )
        monkeypatch.setattr(
            "logos_cli.models.check_nous_free_tier",
            lambda: False,
        )
        assert managed_nous_tools_enabled() is True

    def test_returns_false_on_exception(self, monkeypatch):
        """Should never crash — returns False on any exception."""
        monkeypatch.setattr(
            "logos_cli.auth.get_nous_auth_status",
            _raise_import,
        )
        assert managed_nous_tools_enabled() is False


# ---------------------------------------------------------------------------
# normalize_browser_cloud_provider
# ---------------------------------------------------------------------------
class TestNormalizeBrowserCloudProvider:
    """Coerce arbitrary input to a lowercase browser provider key."""

    def test_none_returns_default(self):
        assert normalize_browser_cloud_provider(None) == "local"

    def test_empty_string_returns_default(self):
        assert normalize_browser_cloud_provider("") == "local"

    def test_whitespace_only_returns_default(self):
        assert normalize_browser_cloud_provider("   ") == "local"

    def test_known_provider_normalized(self):
        assert normalize_browser_cloud_provider("BrowserBase") == "browserbase"

    def test_strips_whitespace(self):
        assert normalize_browser_cloud_provider("  Local  ") == "local"

    def test_integer_coerced(self):
        result = normalize_browser_cloud_provider(42)
        assert isinstance(result, str)
        assert result == "42"


# ---------------------------------------------------------------------------
# prefers_gateway
# ---------------------------------------------------------------------------
class TestPrefersGateway:
    """Honor bool-ish config values for tool gateway routing."""

    def test_returns_false_for_quoted_false(self, monkeypatch):
        monkeypatch.setattr(
            "logos_cli.config.load_config",
            lambda: {"web": {"use_gateway": "false"}},
        )
        assert prefers_gateway("web") is False

    def test_returns_true_for_quoted_true(self, monkeypatch):
        monkeypatch.setattr(
            "logos_cli.config.load_config",
            lambda: {"web": {"use_gateway": "true"}},
        )
        assert prefers_gateway("web") is True


# ---------------------------------------------------------------------------
# resolve_openai_audio_api_key
# ---------------------------------------------------------------------------
class TestResolveOpenaiAudioApiKey:
    """Priority: VOICE_TOOLS_OPENAI_KEY > OPENAI_API_KEY."""

    def test_voice_key_preferred(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "voice-key")
        monkeypatch.setenv("OPENAI_API_KEY", "general-key")
        assert resolve_openai_audio_api_key() == "voice-key"

    def test_falls_back_to_openai_key(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "general-key")
        assert resolve_openai_audio_api_key() == "general-key"

    def test_empty_voice_key_falls_back(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "")
        monkeypatch.setenv("OPENAI_API_KEY", "general-key")
        assert resolve_openai_audio_api_key() == "general-key"

    def test_no_keys_returns_empty(self, monkeypatch):
        monkeypatch.delenv("VOICE_TOOLS_OPENAI_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_openai_audio_api_key() == ""

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("VOICE_TOOLS_OPENAI_KEY", "  voice-key  ")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert resolve_openai_audio_api_key() == "voice-key"

"""Tests for logos_constants module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

import logos_constants
from logos_constants import (
    get_default_hermes_root,
    get_hermes_home,
    get_logos_home,
    get_logos_root,
    is_container,
)


class TestGetDefaultHermesRoot:
    """Tests for get_logos_root() — Docker/custom deployment awareness."""

    def test_no_home_env_returns_logos_default(self, tmp_path, monkeypatch):
        """No home env vars + no ~/.hermes on disk → ~/.logos (new-install
        default)."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOGOS_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_logos_root() == tmp_path / ".logos"

    def test_no_home_env_legacy_hermes_dir_on_disk(self, tmp_path, monkeypatch):
        """No home env vars but ~/.hermes exists (legacy install) → ~/.hermes
        keeps working with zero migration."""
        legacy = tmp_path / ".hermes"
        legacy.mkdir()
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOGOS_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_logos_root() == legacy

    def test_hermes_home_is_native(self, tmp_path, monkeypatch):
        """When HERMES_HOME = ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        native.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(native))
        assert get_logos_root() == native

    def test_hermes_home_is_profile(self, tmp_path, monkeypatch):
        """When HERMES_HOME is a profile under ~/.hermes, returns ~/.hermes."""
        native = tmp_path / ".hermes"
        profile = native / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_logos_root() == native

    def test_hermes_home_is_docker(self, tmp_path, monkeypatch):
        """When HERMES_HOME points outside ~/.hermes (Docker), returns HERMES_HOME."""
        docker_home = tmp_path / "opt" / "data"
        docker_home.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(docker_home))
        assert get_logos_root() == docker_home

    def test_hermes_home_is_custom_path(self, tmp_path, monkeypatch):
        """Any HERMES_HOME outside ~/.hermes is treated as the root."""
        custom = tmp_path / "my-hermes-data"
        custom.mkdir()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(custom))
        assert get_logos_root() == custom

    def test_docker_profile_active(self, tmp_path, monkeypatch):
        """When a Docker profile is active (HERMES_HOME=<root>/profiles/<name>),
        returns the Docker root, not the profile dir."""
        docker_root = tmp_path / "opt" / "data"
        profile = docker_root / "profiles" / "coder"
        profile.mkdir(parents=True)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(profile))
        assert get_logos_root() == docker_root


class TestGetLogosHome:
    """Tests for get_logos_home() — resolution order:
    1. $LOGOS_HOME env var
    2. $HERMES_HOME env var (legacy, still honored)
    3. ~/.logos if it exists (new-install default)
    4. ~/.hermes (legacy fallback)
    """

    def test_logos_home_env_wins_over_hermes_home(self, tmp_path, monkeypatch):
        custom = tmp_path / "logos-custom"
        monkeypatch.setenv("LOGOS_HOME", str(custom))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        assert get_logos_home() == custom

    def test_hermes_home_env_honored_as_legacy(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LOGOS_HOME", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
        assert get_logos_home() == tmp_path / ".hermes"

    def test_no_env_logos_dir_on_disk(self, tmp_path, monkeypatch):
        new_home = tmp_path / ".logos"
        new_home.mkdir()
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOGOS_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_logos_home() == new_home

    def test_no_env_fresh_machine_defaults_to_logos(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOGOS_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_logos_home() == tmp_path / ".logos"

    def test_legacy_alias_matches_new_name(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("LOGOS_HOME", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert get_hermes_home() == get_logos_home()
        assert get_default_hermes_root() == get_logos_root()


class TestIsContainer:
    """Tests for is_container() — Docker/Podman detection."""

    def _reset_cache(self, monkeypatch):
        """Reset the cached detection result before each test."""
        monkeypatch.setattr(logos_constants, "_container_detected", None)

    def test_detects_dockerenv(self, monkeypatch, tmp_path):
        """/.dockerenv triggers container detection."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/.dockerenv")
        assert is_container() is True

    def test_detects_containerenv(self, monkeypatch, tmp_path):
        """/run/.containerenv triggers container detection (Podman)."""
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: p == "/run/.containerenv")
        assert is_container() is True

    def test_detects_cgroup_docker(self, monkeypatch, tmp_path):
        """/proc/1/cgroup containing 'docker' triggers detection."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/docker/abc123\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is True

    def test_negative_case(self, monkeypatch, tmp_path):
        """Returns False on a regular Linux host."""
        import builtins
        self._reset_cache(monkeypatch)
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/\n")
        _real_open = builtins.open
        monkeypatch.setattr("builtins.open", lambda p, *a, **kw: _real_open(str(cgroup_file), *a, **kw) if p == "/proc/1/cgroup" else _real_open(p, *a, **kw))
        assert is_container() is False

    def test_caches_result(self, monkeypatch):
        """Second call uses cached value without re-probing."""
        monkeypatch.setattr(logos_constants, "_container_detected", True)
        assert is_container() is True
        # Even if we make os.path.exists return False, cached value wins
        monkeypatch.setattr(os.path, "exists", lambda p: False)
        assert is_container() is True

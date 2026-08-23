"""Phase D5 — pin the HERMES_* legacy env fallback for the LOGOS_* dual-read shims.

The Hermes→Logos rebrand renamed every ``HERMES_*`` env var to ``LOGOS_*``.
Fleet ``.env`` files still carry the old names, so ``logos_constants`` exposes
single documented shims (logos_env / logos_env_set / logos_env_raw /
logos_env_delete) that read ``LOGOS_<name>`` first and fall back to
``HERMES_<name>``.  These tests are the fallback coverage — do not delete the
HERMES_* references here.
"""

import os

import pytest

import logos_constants


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("LOGOS_TESTVAR", "HERMES_TESTVAR"):
        monkeypatch.delenv(var, raising=False)
    yield
    for var in ("LOGOS_TESTVAR", "HERMES_TESTVAR"):
        monkeypatch.delenv(var, raising=False)


# ─── logos_env ───────────────────────────────────────────────────────────────


def test_logos_env_reads_logos_primary(monkeypatch):
    monkeypatch.setenv("LOGOS_TESTVAR", "new")
    assert logos_constants.logos_env("TESTVAR") == "new"


def test_logos_env_falls_back_to_hermes_legacy(monkeypatch):
    monkeypatch.setenv("HERMES_TESTVAR", "legacy")
    assert logos_constants.logos_env("TESTVAR") == "legacy"


def test_logos_env_logos_wins_when_both_set(monkeypatch):
    monkeypatch.setenv("LOGOS_TESTVAR", "new")
    monkeypatch.setenv("HERMES_TESTVAR", "legacy")
    assert logos_constants.logos_env("TESTVAR") == "new"


def test_logos_env_empty_logos_falls_through(monkeypatch):
    # Empty LOGOS_* is treated as unset — falls through to legacy.
    monkeypatch.setenv("LOGOS_TESTVAR", "")
    monkeypatch.setenv("HERMES_TESTVAR", "legacy")
    assert logos_constants.logos_env("TESTVAR") == "legacy"


def test_logos_env_default_when_neither_set():
    assert logos_constants.logos_env("TESTVAR") is None
    assert logos_constants.logos_env("TESTVAR", "dflt") == "dflt"


# ─── logos_env_set ───────────────────────────────────────────────────────────


def test_logos_env_set_detects_legacy_only(monkeypatch):
    monkeypatch.setenv("HERMES_TESTVAR", "x")
    assert logos_constants.logos_env_set("TESTVAR") is True


def test_logos_env_set_detects_new_only(monkeypatch):
    monkeypatch.setenv("LOGOS_TESTVAR", "x")
    assert logos_constants.logos_env_set("TESTVAR") is True


def test_logos_env_set_false_when_absent():
    assert logos_constants.logos_env_set("TESTVAR") is False


# ─── logos_env_raw ───────────────────────────────────────────────────────────


def test_logos_env_raw_reads_logos_primary(monkeypatch):
    monkeypatch.setenv("LOGOS_TESTVAR", "new")
    monkeypatch.setenv("HERMES_TESTVAR", "legacy")
    assert logos_constants.logos_env_raw("TESTVAR") == "new"


def test_logos_env_raw_falls_back_to_hermes_legacy(monkeypatch):
    monkeypatch.setenv("HERMES_TESTVAR", "legacy")
    assert logos_constants.logos_env_raw("TESTVAR") == "legacy"


def test_logos_env_raw_raises_keyerror_when_absent():
    with pytest.raises(KeyError):
        logos_constants.logos_env_raw("TESTVAR")


# ─── logos_env_delete ────────────────────────────────────────────────────────


def test_logos_env_delete_clears_both_names(monkeypatch):
    monkeypatch.setenv("LOGOS_TESTVAR", "new")
    monkeypatch.setenv("HERMES_TESTVAR", "legacy")
    logos_constants.logos_env_delete("TESTVAR")
    assert "LOGOS_TESTVAR" not in os.environ
    assert "HERMES_TESTVAR" not in os.environ


def test_logos_env_delete_noop_when_absent():
    logos_constants.logos_env_delete("TESTVAR")  # must not raise

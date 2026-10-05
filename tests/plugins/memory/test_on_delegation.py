"""Tests for PerpetualContextProvider.on_delegation (10/1 sandbox WIP).

The delegation hook records ONE compact [DELEGATION] system row per
subagent completion in the PARENT session, so delegated tasks are
recallable (``perpetual_search "DELEGATION"``) and traceable to the child
session via metadata — with zero embedding cost and no live-context
injection. The hook must degrade, never crash: a failed write can never
break the delegation-completion path.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from plugins.memory.perpetual_context import PerpetualContextProvider


class _StubFactory:
    """Component bring-up is out of scope for this unit test."""

    def ensure_all(self) -> None:
        pass


@pytest.fixture()
def provider(tmp_path):
    p = PerpetualContextProvider()
    p._get_factory = lambda: _StubFactory()  # type: ignore[method-assign]
    p.initialize(
        "parent-session-1",
        config={"perpetual_context": {"db_path": str(tmp_path / "pm.db")}},
    )
    yield p
    try:
        p.shutdown()
    except Exception:
        pass


def _delegation_rows(p: PerpetualContextProvider) -> list[dict[str, Any]]:
    rows = p._db.get_messages(session_id="parent-session-1", role="system", limit=100)
    return [m for m in rows if str(m.get("content", "")).startswith("[DELEGATION]")]


def _meta(m: dict[str, Any]) -> dict[str, Any]:
    meta = m.get("metadata")
    if isinstance(meta, str):
        meta = json.loads(meta)
    return meta or {}


class TestOnDelegation:
    def test_records_one_system_row_with_parent_child_links(self, provider):
        provider.on_delegation(
            "Research the X topic", "Found three sources.",
            child_session_id="child-9", status="done",
        )
        rows = _delegation_rows(provider)
        assert len(rows) == 1
        m = rows[0]
        content = m["content"]
        assert "status=done" in content
        assert "child_session=child-9" in content
        assert 'goal="Research the X topic"' in content
        assert 'summary="Found three sources."' in content
        meta = _meta(m)
        assert meta.get("delegation") == "event"
        assert meta.get("parent_session_id") == "parent-session-1"
        assert meta.get("child_session_id") == "child-9"

    def test_truncates_long_goal_and_summary(self, provider):
        goal, summary = "g" * 400, "s" * 600
        provider.on_delegation(goal, summary, child_session_id="c")
        (m,) = _delegation_rows(provider)
        assert f'goal="{"g" * 300}\u2026"' in m["content"]
        assert f'summary="{"s" * 500}\u2026"' in m["content"]
        assert "g" * 301 not in m["content"]
        assert "s" * 501 not in m["content"]

    def test_default_status_is_done(self, provider):
        provider.on_delegation("t", "r")
        (m,) = _delegation_rows(provider)
        assert m["content"].startswith("[DELEGATION] status=done ")

    def test_newlines_flattened(self, provider):
        provider.on_delegation("line one\nline two", "a\nb")
        (m,) = _delegation_rows(provider)
        assert "\n" not in m["content"]
        assert 'goal="line one line two"' in m["content"]

    def test_noop_when_uninitialized(self):
        p = PerpetualContextProvider()
        p._get_factory = lambda: _StubFactory()  # type: ignore[method-assign]
        p.on_delegation("t", "r")  # must not raise

    def test_never_raises_on_db_failure(self, provider, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("db down")

        monkeypatch.setattr(provider._db, "add_message", boom)
        provider.on_delegation("t", "r")  # must not raise

    def test_row_not_embedded(self, provider, tmp_path):
        provider.on_delegation("t", "r", child_session_id="c")
        con = sqlite3.connect(tmp_path / "pm.db")
        try:
            cols = {row[1] for row in con.execute("PRAGMA table_info(messages)")}
            if "embedding" in cols:
                (emb,) = con.execute(
                    "SELECT embedding FROM messages WHERE content LIKE '[DELEGATION]%'"
                ).fetchone()
                assert not emb, "system delegation rows must skip embedding"
        finally:
            con.close()

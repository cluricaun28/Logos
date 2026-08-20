"""Regression tests: pcdb v3 preconnect schema check.

Bug (2026-08-20, prod restart): a gateway booting against a pre-v3 database
rebuilt the messages table *after* the main connection had cached the old
schema. The connection then kept rejecting role='tool' inserts
("CHECK constraint failed") until the next restart, so PM tool-result
persistence stayed dead even though the DB file was correctly migrated.

Fix under test: `_preconnect_v3_check()` runs the 12-step rebuild on a
throwaway connection BEFORE the main connection opens, so every connection
in the process sees the final 4-role CHECK on first use.
"""

import sqlite3

import pytest

from agent.perpetual_context_db import PerpetualContextDB

V1_MESSAGES_DDL = """
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
        content TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        created_at REAL DEFAULT 0
    )
"""


def _make_v1_db(path: str) -> None:
    """Create a pre-v3 database: old 3-role CHECK on messages."""
    conn = sqlite3.connect(path)
    conn.execute(V1_MESSAGES_DDL)
    # Seed a row that must survive the rebuild.
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) "
        "VALUES ('s1', 'user', 'seed', 1.0)"
    )
    conn.commit()
    conn.close()


@pytest.fixture()
def v1_db(tmp_path):
    p = str(tmp_path / "pm_v1.db")
    _make_v1_db(p)
    return p


def test_v1_db_tool_insert_works_immediately_after_init(v1_db):
    """The exact failure mode: tool insert must succeed right after init,
    on the instance's own connection — no restart in between."""
    db = PerpetualContextDB(db_path=v1_db)
    assert db.initialize() is True
    try:
        mid = db.add_message("s1", "tool", "tool result text", {"k": "v"})
        assert mid is not None, "add_message(role='tool') returned None — insert failed"
        row = db._conn.execute(
            "SELECT role, content FROM messages WHERE id = ?", (mid,)
        ).fetchone()
        assert row == ("tool", "tool result text")
    finally:
        db.shutdown()


def test_v1_rebuild_preserves_existing_rows(v1_db):
    """The 12-step rebuild must not drop pre-existing messages."""
    db = PerpetualContextDB(db_path=v1_db)
    assert db.initialize() is True
    try:
        n = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert n == 1, "seed row lost during rebuild"
        sql = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()[0]
        assert "'tool'" in sql, "messages CHECK not upgraded to 4-role"
    finally:
        db.shutdown()


def test_v3_db_skips_rebuild(v1_db):
    """Already-migrated DB: second init is a no-op (no rebuild, no data loss)."""
    db = PerpetualContextDB(db_path=v1_db)
    assert db.initialize() is True
    db.shutdown()

    db2 = PerpetualContextDB(db_path=v1_db)
    assert db2.initialize() is True
    try:
        n = db2._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert n == 1
        # And tool inserts still work on the second init.
        mid = db2.add_message("s1", "tool", "second init tool row")
        assert mid is not None
    finally:
        db2.shutdown()


def test_fresh_db_still_works(tmp_path):
    """No pre-existing DB at all — brand-new init path unaffected."""
    p = str(tmp_path / "pm_fresh.db")
    db = PerpetualContextDB(db_path=p)
    assert db.initialize() is True
    try:
        mid = db.add_message("s1", "tool", "fresh db tool row")
        assert mid is not None
    finally:
        db.shutdown()

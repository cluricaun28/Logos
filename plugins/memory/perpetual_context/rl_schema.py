"""RL Index Schema — table and trigger creation.

Extracted from RLIndex._create_tables() for SRP compliance.
"""

from __future__ import annotations

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)


def create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables, triggers, and indexes for the RL index.

    Called from RLIndex.initialize() after connection is established.
    """
    # Core file metadata
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rl_files ("
        "  file_path TEXT PRIMARY KEY,"
        "  title TEXT NOT NULL DEFAULT '',"
        "  frontmatter TEXT NOT NULL DEFAULT '{}',"
        "  body TEXT NOT NULL DEFAULT '',"
        "  category TEXT NOT NULL DEFAULT 'other',"
        "  mtime REAL NOT NULL DEFAULT 0,"
        "  size INTEGER NOT NULL DEFAULT 0,"
        "  indexed_at REAL NOT NULL DEFAULT 0"
        ")"
    )

    # FTS5 virtual table — external content mode backed by rl_files
    _create_fts5_table(conn)

    # Embedding table
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rl_embeddings ("
        "  file_path TEXT PRIMARY KEY,"
        "  embedding BLOB NOT NULL,"
        "  mtime REAL NOT NULL DEFAULT 0,"
        "  created_at REAL NOT NULL DEFAULT 0"
        ")"
    )

    # Indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rl_files_category"
        " ON rl_files(category)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rl_embeddings_mtime"
        " ON rl_embeddings(mtime)"
    )

    # Schema version tracking
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rl_schema ("
        "  version INTEGER PRIMARY KEY,"
        "  updated_at REAL NOT NULL DEFAULT 0"
        ")"
    )
    cursor = conn.execute(
        "SELECT version FROM rl_schema WHERE version = 1"
    )
    if cursor.fetchone() is None:
        conn.execute(
            "INSERT INTO rl_schema(version, updated_at)"
            " VALUES(1, ?)",
            (time.time(),),
        )

    conn.commit()


def _create_fts5_table(conn: sqlite3.Connection) -> None:
    """Create the FTS5 virtual table and sync triggers (idempotent).

    The table is external-content mode backed by rl_files. NEVER drop an
    existing table here: an unconditional drop+recreate leaves the index
    empty until the next full build, which silently degraded hybrid
    search to semantic-only after every process start. Drift between the
    index and rl_files is detected and repaired by
    RLIndex._self_heal_fts() instead.
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type = 'table' AND name = 'rl_index_fts'"
    )
    if cursor.fetchone() is None:
        conn.execute(
            "CREATE VIRTUAL TABLE rl_index_fts USING fts5("
            "  file_path, title, body, category,"
            "  content='rl_files',"
            "  content_rowid='rowid',"
            "  tokenize='unicode61'"
            "  )"
        )

    # Triggers to keep FTS5 in sync with rl_files (IF NOT EXISTS — they
    # reference the virtual table and must survive re-initialization)
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS rl_files_ai AFTER INSERT ON rl_files BEGIN"
        "  INSERT INTO rl_index_fts("
        "    rowid, file_path, title, body, category)"
        "  VALUES ("
        "    new.rowid, new.file_path, new.title,"
        "    new.body, new.category);"
        "END"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS rl_files_ad AFTER DELETE ON rl_files BEGIN"
        "  DELETE FROM rl_index_fts WHERE rowid = old.rowid;"
        "END"
    )
    conn.execute(
        "CREATE TRIGGER IF NOT EXISTS rl_files_au AFTER UPDATE ON rl_files BEGIN"
        "  DELETE FROM rl_index_fts WHERE rowid = old.rowid;"
        "  INSERT INTO rl_index_fts("
        "    rowid, file_path, title, body, category)"
        "  VALUES ("
        "    new.rowid, new.file_path, new.title,"
        "    new.body, new.category);"
        "END"
    )

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
    """Create or recreate the FTS5 virtual table and sync triggers."""
    # Drop old table and triggers
    try:
        conn.execute("DROP TABLE IF EXISTS rl_index_fts")
    except (sqlite3.Error, AttributeError) as e:
        logger.debug("Error dropping old rl_index_fts: %s", e)

    conn.execute(
        "CREATE VIRTUAL TABLE rl_index_fts USING fts5("
        "  file_path, title, body, category,"
        "  content='rl_files',"
        "  content_rowid='rowid',"
        "  tokenize='unicode61'"
        ")"
    )

    # Triggers to keep FTS5 in sync with rl_files
    for trig in ("rl_files_ai", "rl_files_ad", "rl_files_au"):
        try:
            conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
        except (sqlite3.Error, AttributeError) as e:
            logger.debug("Error dropping trigger %s: %s", trig, e)

    conn.execute(
        "CREATE TRIGGER rl_files_ai AFTER INSERT ON rl_files BEGIN"
        "  INSERT INTO rl_index_fts("
        "    rowid, file_path, title, body, category)"
        "  VALUES ("
        "    new.rowid, new.file_path, new.title,"
        "    new.body, new.category);"
        "END"
    )
    conn.execute(
        "CREATE TRIGGER rl_files_ad AFTER DELETE ON rl_files BEGIN"
        "  DELETE FROM rl_index_fts WHERE rowid = old.rowid;"
        "END"
    )
    conn.execute(
        "CREATE TRIGGER rl_files_au AFTER UPDATE ON rl_files BEGIN"
        "  DELETE FROM rl_index_fts WHERE rowid = old.rowid;"
        "  INSERT INTO rl_index_fts("
        "    rowid, file_path, title, body, category)"
        "  VALUES ("
        "    new.rowid, new.file_path, new.title,"
        "    new.body, new.category);"
        "END"
    )

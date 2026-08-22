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

    # ── Schema v2: search chunks (full-body coverage) ──────────────────
    # Pre-v2, rl_files.body was truncated to RL_EMBED_MAX_CONTENT_LEN
    # (8000 chars) at extract time, so content past that mark was invisible
    # to both FTS5 and the (first-8K) file-level embedding. v2 stores the
    # full body in rl_files and splits it into overlapping chunks that get
    # their own FTS5 index and per-chunk embeddings. All v2 tables are
    # ADDITIVE: pre-v2 code ignores them, and v2 code falls back to the
    # legacy single-row paths when rl_chunks is empty (unmigrated DB).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rl_chunks ("
        "  rowid INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  file_path TEXT NOT NULL,"
        "  chunk_idx INTEGER NOT NULL,"
        "  body TEXT NOT NULL,"
        "  UNIQUE(file_path, chunk_idx)"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rl_chunks_file_path"
        " ON rl_chunks(file_path)"
    )

    # Per-chunk embeddings (separate from rl_embeddings so pre-v2 code —
    # which reads rl_embeddings as one row per file — is never disturbed).
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rl_chunk_embeddings ("
        "  chunk_id INTEGER PRIMARY KEY,"
        "  file_path TEXT NOT NULL,"
        "  embedding BLOB NOT NULL,"
        "  mtime REAL NOT NULL DEFAULT 0,"
        "  created_at REAL NOT NULL DEFAULT 0"
        ")"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_rl_chunk_embeddings_file"
        " ON rl_chunk_embeddings(file_path)"
    )

    _create_chunks_fts5_table(conn)

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
    # Version 2: search-chunk tables (see block above). Bump-only — v2 adds
    # tables but changes no existing one, so an existing v1 DB is already
    # compatible once the tables exist.
    cursor = conn.execute(
        "SELECT version FROM rl_schema WHERE version = 2"
    )
    if cursor.fetchone() is None:
        conn.execute(
            "INSERT INTO rl_schema(version, updated_at)"
            " VALUES(2, ?)",
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


def _create_chunks_fts5_table(conn: sqlite3.Connection) -> None:
    """Create the chunk FTS5 virtual table (idempotent, schema v2).

    External-content mode backed by rl_chunks, mirroring the legacy
    _create_fts5_table pattern. No triggers: chunk rows are managed
    explicitly by rl_builder (delete FTS rows, delete content rows,
    insert new pairs) so the builder stays the single writer. Drift is
    detected by RLIndex._self_heal_fts().
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master"
        " WHERE type = 'table' AND name = 'rl_chunks_fts'"
    )
    if cursor.fetchone() is None:
        conn.execute(
            "CREATE VIRTUAL TABLE rl_chunks_fts USING fts5("
            "  file_path, chunk_idx, body,"
            "  content='rl_chunks',"
            "  content_rowid='rowid',"
            "  tokenize='unicode61'"
            "  )"
        )

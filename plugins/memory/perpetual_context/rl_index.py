"""Reference Library Index — Pre-computed FTS5 + embedding search over RL files.

Design rationale:
  - The Reference Library is a curated knowledge base (32K+ files) that lives on disk
    as markdown. Searching it by globbing and word-counting is O(n) file I/O per query.
  - This module materializes an inverted index (FTS5) and pre-computed embeddings
    into a dedicated SQLite database (~50MB for 32K files).
  - At query time: FTS5 keyword search + cosine similarity hybrid scoring.
    Total latency target: < 200ms on local hardware.

Architecture:
  - Separate DB (rl_index.db) from perpetual_context.db — clean separation of concerns.
  - FTS5 virtual table (rl_index_fts) for keyword/BM25 search.
  - Embedding table (rl_embeddings) for semantic search (384-dim float32 vectors).
  - Metadata table (rl_files) for file properties (mtime, category, etc.).
  - Embedding model: all-MiniLM-L6-v2 (reuses EmbeddingEngine from perpetual_context_db).

Usage:
    from plugins.memory.perpetual_context.rl_index import RLIndex

    rl = RLIndex()
    rl.initialize()

    # One-shot build (call once, or when files change significantly)
    rl.build_index()

    # Search
    results = rl.search("dog training methods", top_k=5)

    # Incremental update (call when a file changes)
    rl.update_file("~/.hermes/reference-library/entities/patrick-daley.md")

    # Remove a file from the index
    rl.remove_file("entities/patrick-daley.md")
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import struct
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RL_INDEX_DB_PATH = os.path.expanduser("~/.hermes/rl_index.db")
RL_BASE_DIR = os.path.expanduser("~/.hermes/reference-library")

# Embedding config — matches PerpetualContextDB
RL_EMBED_DIM = 384
RL_EMBED_MAX_CONTENT_LEN = 8000  # Max chars to embed per file

# Hybrid search weights — same as PM for consistency
RL_SEMANTIC_WEIGHT = 0.4
RL_FTS5_WEIGHT = 0.6
RL_COSINE_THRESHOLD = 0.05

# Batch sizes
EMBED_BATCH_SIZE = 64
PROGRESS_LOG_INTERVAL = 512

# Frontmatter parsing
YAML_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


# ---------------------------------------------------------------------------
# Simple YAML frontmatter parser (no external dependency)
# ---------------------------------------------------------------------------

def _parse_yaml_frontmatter(text: str) -> dict:
    """Parse simple YAML frontmatter.

    Handles flat key: value pairs.
    Does NOT handle complex YAML — sufficient for our frontmatter format.
    """
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                result[key] = value
    return result


def _extract_file_info(file_path: Path) -> Optional[dict]:
    """Read a markdown file and extract title, frontmatter, body, category.

    Returns dict with keys: file_path, title, frontmatter (str),
    body, category, mtime, size. Returns None if the file can't be read.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug("Failed to read %s: %s", file_path.name, e)
        return None

    # Determine relative path from RL base
    try:
        rel_path = str(file_path.relative_to(RL_BASE_DIR))
    except ValueError:
        rel_path = str(file_path)

    # Determine category from directory structure
    parts = rel_path.split("/")
    category = parts[0] if parts else "other"
    if len(parts) >= 2 and parts[0] == "entities":
        sub = parts[1].split("/")[0]
        if sub == "britannica":
            category = "britannica"
        elif sub == "aquinas-library":
            category = "aquinas"

    # Extract frontmatter
    frontmatter = "{}"
    body = content
    match = YAML_FRONTMATTER_RE.search(content)
    if match:
        fm_text = match.group(1)
        fm_data = _parse_yaml_frontmatter(fm_text)
        frontmatter = json.dumps(fm_data)

        # Extract body (after frontmatter)
        body = content[match.end():]

        # Title from frontmatter or H1
        title = (
            fm_data.get("title")
            or fm_data.get("name")
            or fm_data.get("topic", "")
        )
        if not title:
            h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1).strip()
        if not title:
            title = file_path.stem.replace("-", " ").title()
    else:
        # No frontmatter — try H1
        h1_match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
        title = (
            h1_match.group(1).strip()
            if h1_match
            else file_path.stem.replace("-", " ").title()
        )

    # Strip markdown formatting for cleaner indexing/embedding
    body_clean = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    body_clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body_clean)
    body_clean = re.sub(r"[*_~]{1,2}", "", body_clean)
    body_clean = re.sub(r"#{1,6}\s*", " ", body_clean)
    body_clean = re.sub(r"---+", "", body_clean)
    body_clean = re.sub(r"\[!\[.*?\]\(.*?\)\]", "", body_clean)
    body_clean = re.sub(r"\|.*\|", "", body_clean)
    body_clean = re.sub(r"\n{3,}", "\n\n", body_clean)
    body_clean = body_clean.strip()

    mtime = file_path.stat().st_mtime
    size = len(content)

    return {
        "file_path": rel_path,
        "title": title,
        "frontmatter": frontmatter,
        "body": body_clean[:RL_EMBED_MAX_CONTENT_LEN],
        "category": category,
        "mtime": mtime,
        "size": size,
    }


# ---------------------------------------------------------------------------
# Embedding helper — wraps the shared EmbeddingEngine
# ---------------------------------------------------------------------------

def _get_embedding_engine() -> Any:
    """Lazy import and get the shared EmbeddingEngine singleton."""
    # Deferred import to avoid loading perpetual_context_db on module import
    from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415
    return EmbeddingEngine.get()


def _embed_batch(
    texts: List[str], batch_size: int = EMBED_BATCH_SIZE
) -> List[Optional[List[float]]]:
    """Embed a batch of texts using the shared EmbeddingEngine.

    Returns list of vectors (same length as input).
    None for individual failed embeddings — never discards
    the entire batch because one item failed.
    """
    engine = _get_embedding_engine()
    model = engine._load_model()
    if model is None:
        return [None] * len(texts)

    results: List[Optional[List[float]]] = []
    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch_truncated = [t[:RL_EMBED_MAX_CONTENT_LEN] for t in batch]
            try:
                vectors = model.encode(
                    batch_truncated,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    batch_size=batch_size,
                )
                # Handle both single and batch returns
                if hasattr(vectors, "__len__") and not isinstance(vectors, str):
                    for v in vectors:
                        results.append(v.tolist())
                else:
                    results.append(vectors.tolist())
            except Exception as e:
                logger.debug(
                    "Batch embedding failed for %d items: %s", len(batch), e
                )
                # Mark individual items as None, don't discard the whole batch
                results.extend([None] * len(batch))
    except Exception as e:
        logger.debug("Embedding batch outer failure: %s", e)
        results.extend([None] * len(texts))

    return results


# ---------------------------------------------------------------------------
# RLIndex — Main class
# ---------------------------------------------------------------------------

class RLIndex:
    """Search index for the Reference Library.

    Manages a dedicated SQLite database with FTS5 + pre-computed embeddings.
    Thread-safe for reads; writes should be serialized.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or RL_INDEX_DB_PATH
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize the database connection and create tables if needed."""
        try:
            with self._lock:
                self._conn = sqlite3.connect(
                    self._db_path, timeout=30, check_same_thread=False
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA busy_timeout=5000")
                self._create_tables()
                self._initialized = True
                return True
        except Exception as e:
            logger.error("Failed to initialize RLIndex: %s", e)
            self._conn = None
            return False

    def shutdown(self) -> None:
        """Close the database connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.commit()
                except Exception as e:
                    logger.debug("Error during RLIndex shutdown commit: %s", e)
                try:
                    self._conn.close()
                except Exception as e:
                    logger.debug("Error closing RLIndex connection: %s", e)
                self._conn = None
            self._initialized = False

    # -----------------------------------------------------------------------
    # Schema
    # -----------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create all tables and triggers for the RL index."""
        conn = self._conn
        if not conn:
            raise RuntimeError("RLIndex not initialized")

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
        try:
            conn.execute("DROP TABLE IF EXISTS rl_index_fts")
        except Exception as e:
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
            except Exception as e:
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

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------

    def build_index(self, rl_base: Optional[str] = None) -> dict:
        """Build the full RL index from scratch.

        Walks the RL directory tree, extracts file info, computes embeddings,
        and populates all tables. Idempotent — safe to call multiple times.

        Returns dict with build stats.
        """
        rl_base = rl_base or RL_BASE_DIR
        base_path = Path(rl_base)

        if not base_path.exists():
            raise FileNotFoundError(
                f"RL base directory not found: {rl_base}"
            )

        # Collect all .md files
        md_files = sorted(base_path.rglob("*.md"))
        total = len(md_files)
        logger.info(
            "RLIndex build: found %d .md files in %s", total, rl_base
        )

        # Clear existing index
        with self._lock:
            self._conn.execute("DELETE FROM rl_embeddings")
            self._conn.execute("DELETE FROM rl_files")

        start = time.time()
        now = time.time()

        stats: Dict[str, Any] = {
            "files_processed": 0,
            "files_indexed": 0,
            "files_embedded": 0,
            "files_failed": 0,
            "elapsed_seconds": 0,
            "categories": {},
        }

        # Phase 1: Extract file info
        file_infos: List[dict] = []
        for md_file in md_files:
            info = _extract_file_info(md_file)
            if info is None:
                stats["files_failed"] += 1
                continue

            info["indexed_at"] = now
            file_infos.append(info)

            cat = info["category"]
            stats["categories"][cat] = (
                stats["categories"].get(cat, 0) + 1
            )
            stats["files_processed"] += 1

        # Build lookup dict — O(1) instead of O(n) linear scan
        info_by_path: Dict[str, dict] = {
            fi["file_path"]: fi for fi in file_infos
        }

        # Phase 2: Bulk insert file metadata
        # Pattern: drop triggers, drop FTS5, bulk insert into rl_files,
        # recreate FTS5, explicitly populate from rl_files, re-add triggers.
        # This avoids the known SQLite issue where INSERT OR REPLACE triggers
        # corrupt the external-content FTS5 index during bulk loads.
        with self._lock:
            for trig in ("rl_files_ai", "rl_files_ad", "rl_files_au"):
                try:
                    self._conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
                except Exception as e:
                    logger.debug("Error dropping trigger %s: %s", trig, e)

            self._conn.execute("DROP TABLE IF EXISTS rl_index_fts")
            self._conn.execute("DELETE FROM rl_files")

            self._conn.executemany(
                "INSERT INTO rl_files"
                " (file_path, title, frontmatter, body,"
                "  category, mtime, size, indexed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        fi["file_path"], fi["title"], fi["frontmatter"],
                        fi["body"], fi["category"], fi["mtime"],
                        fi["size"], fi["indexed_at"],
                    )
                    for fi in file_infos
                ],
            )

            # Recreate FTS5 virtual table
            self._conn.execute(
                "CREATE VIRTUAL TABLE rl_index_fts USING fts5("
                "  file_path, title, body, category,"
                "  content='rl_files',"
                "  content_rowid='rowid',"
                "  tokenize='unicode61'"
                ")"
            )

            # Explicitly populate FTS5 from rl_files (not via triggers)
            self._conn.execute(
                "INSERT INTO rl_index_fts(rowid, file_path, title, body, "
                "category) "
                "SELECT rowid, file_path, title, body, category "
                "FROM rl_files"
            )

            # Recreate triggers for future incremental updates
            for trig in (
                "CREATE TRIGGER rl_files_ai AFTER INSERT ON rl_files BEGIN "
                "  INSERT INTO rl_index_fts(rowid, file_path, title, body, "
                "category) VALUES (new.rowid, new.file_path, new.title, "
                "new.body, new.category); END",
                "CREATE TRIGGER rl_files_ad AFTER DELETE ON rl_files BEGIN "
                "  DELETE FROM rl_index_fts WHERE rowid = old.rowid; END",
                "CREATE TRIGGER rl_files_au AFTER UPDATE ON rl_files BEGIN "
                "  DELETE FROM rl_index_fts WHERE rowid = old.rowid; "
                "  INSERT INTO rl_index_fts(rowid, file_path, title, body, "
                "category) VALUES (new.rowid, new.file_path, new.title, "
                "new.body, new.category); END",
            ):
                self._conn.execute(trig)

            self._conn.commit()

        stats["files_indexed"] = len(file_infos)
        logger.info(
            "RLIndex build: inserted %d files, rebuilt FTS5 index",
            len(file_infos),
        )

        # Phase 3: Compute embeddings in batches
        texts_by_path = [
            (fi["file_path"], fi["body"]) for fi in file_infos
        ]

        for i in range(0, len(texts_by_path), EMBED_BATCH_SIZE):
            batch = texts_by_path[i: i + EMBED_BATCH_SIZE]
            paths, texts = zip(*batch)

            vectors = _embed_batch(list(texts), batch_size=EMBED_BATCH_SIZE)

            embed_rows: List[tuple] = []
            for path, vector in zip(paths, vectors):
                if vector is not None:
                    blob = struct.pack(f"{RL_EMBED_DIM}f", *vector)
                    info = info_by_path.get(path)
                    mtime = info["mtime"] if info else 0
                    embed_rows.append((path, blob, mtime, now))
                    stats["files_embedded"] += 1

            if embed_rows:
                with self._lock:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO rl_embeddings"
                        " (file_path, embedding, mtime, created_at)"
                        " VALUES (?, ?, ?, ?)",
                        embed_rows,
                    )
                    self._conn.commit()

            if (i + EMBED_BATCH_SIZE) % PROGRESS_LOG_INTERVAL == 0 or (
                i + EMBED_BATCH_SIZE >= len(texts_by_path)
            ):
                elapsed = time.time() - start
                rate = (
                    len(texts_by_path) / elapsed if elapsed > 0 else 0
                )
                logger.info(
                    "RLIndex build: embeddings %d/%d (%.1fs, %.0f/s)",
                    min(i + EMBED_BATCH_SIZE, len(texts_by_path)),
                    len(texts_by_path),
                    elapsed,
                    rate,
                )

        stats["elapsed_seconds"] = round(time.time() - start, 1)
        logger.info(
            "RLIndex build complete: %d files, %d embeddings, %.1fs",
            stats["files_indexed"],
            stats["files_embedded"],
            stats["elapsed_seconds"],
        )

        return stats

    # -----------------------------------------------------------------------
    # Incremental updates
    # -----------------------------------------------------------------------

    def update_file(self, file_path_str: str) -> bool:
        """Update a single file in the index.

        Args:
            file_path_str: Absolute or relative path to the .md file.

        Returns:
            True if the file was updated, False if not found or failed.
        """
        path = Path(file_path_str)
        if not path.exists():
            return False

        info = _extract_file_info(path)
        if info is None:
            return False

        info["indexed_at"] = time.time()

        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO rl_files"
                    " (file_path, title, frontmatter, body,"
                    "  category, mtime, size, indexed_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        info["file_path"], info["title"],
                        info["frontmatter"], info["body"],
                        info["category"], info["mtime"],
                        info["size"], info["indexed_at"],
                    ),
                )

                # Update embedding
                vector = _get_embedding_engine().embed(info["body"])
                if vector is not None:
                    blob = struct.pack(f"{RL_EMBED_DIM}f", *vector)
                    self._conn.execute(
                        "INSERT OR REPLACE INTO rl_embeddings"
                        " (file_path, embedding, mtime, created_at)"
                        " VALUES (?, ?, ?, ?)",
                        (
                            info["file_path"], blob,
                            info["mtime"], time.time(),
                        ),
                    )

                self._conn.commit()
                return True
            except Exception as e:
                logger.error(
                    "Failed to update RL file %s: %s",
                    info.get("file_path", path),
                    e,
                )
                return False

    def remove_file(self, file_path_str: str) -> bool:
        """Remove a file from the index.

        Args:
            file_path_str: Relative path (e.g., 'entities/patrick-daley.md')
                or absolute path to the .md file.
        """
        path = Path(file_path_str)

        # Compute relative path — works whether input is absolute or relative
        try:
            rel_path = str(path.relative_to(RL_BASE_DIR))
        except ValueError:
            # Already relative or from different base — use as-is
            rel_path = str(path)

        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM rl_files WHERE file_path = ?",
                    (rel_path,),
                )
                self._conn.execute(
                    "DELETE FROM rl_embeddings WHERE file_path = ?",
                    (rel_path,),
                )
                self._conn.commit()
                return True
            except Exception as e:
                logger.error(
                    "Failed to remove RL file %s: %s", rel_path, e
                )
                return False

    def reindex_stale(self) -> int:
        """Re-index files whose mtime has changed since last indexing.

        Returns the number of files updated.
        """
        stale: List[str] = []
        deleted: List[str] = []

        with self._lock:
            cursor = self._conn.execute(
                "SELECT file_path, mtime FROM rl_files"
            )
            for row in cursor.fetchall():
                fp, recorded_mtime = row
                full_path = os.path.join(RL_BASE_DIR, fp)
                try:
                    actual_mtime = os.path.getmtime(full_path)
                    if actual_mtime != recorded_mtime:
                        stale.append(fp)
                except FileNotFoundError:
                    deleted.append(fp)

            # Remove deleted files
            for fp in deleted:
                self._conn.execute(
                    "DELETE FROM rl_files WHERE file_path = ?", (fp,)
                )
                self._conn.execute(
                    "DELETE FROM rl_embeddings"
                    " WHERE file_path = ?", (fp,)
                )

            if deleted:
                self._conn.commit()

        # Update stale files (outside the lock to avoid holding it too long)
        count = 0
        for fp in stale:
            full_path = os.path.join(RL_BASE_DIR, fp)
            if self.update_file(full_path):
                count += 1

        if stale:
            logger.info(
                "RLIndex reindex_stale: updated %d/%d, "
                "removed %d deleted",
                count, len(stale), len(deleted),
            )

        return count

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
        semantic_weight: float = RL_SEMANTIC_WEIGHT,
        fts5_weight: float = RL_FTS5_WEIGHT,
    ) -> List[Dict[str, Any]]:
        """Hybrid search: FTS5 keyword + cosine similarity.

        Args:
            query: Search query text.
            top_k: Max results to return.
            category: Optional category filter.
            semantic_weight: Weight for cosine similarity.
            fts5_weight: Weight for BM25 rank.

        Returns:
            List of result dicts with keys: file_path, title,
            category, snippet, score, fts5_score, semantic_score.
        """
        if not self._initialized or not self._conn:
            return []

        try:
            fts_results = self._fts_search(
                query, top_k=top_k * 2, category=category
            )
            semantic_results = self._semantic_search(
                query, top_k=top_k * 2, category=category
            )

            # Hybrid fusion
            scored: Dict[str, float] = {}
            result_cache: Dict[str, Dict[str, Any]] = {}

            # Score FTS5 results
            if fts_results:
                min_rank = min(m["_rank"] for m in fts_results)
                max_rank = max(m["_rank"] for m in fts_results)
                rank_range = (
                    max_rank - min_rank
                    if max_rank != min_rank
                    else 1.0
                )

                for i, msg in enumerate(fts_results):
                    fp = msg["file_path"]
                    normalized = 1.0 - (
                        (msg["_rank"] - min_rank) / rank_range
                    )
                    score = (
                        fts5_weight * normalized
                        + (1.0 / (i + 1)) * fts5_weight * 0.1
                    )
                    scored[fp] = scored.get(fp, 0) + score
                    msg["_fts5_score"] = round(score, 4)
                    result_cache[fp] = msg

            # Score semantic results
            for i, msg in enumerate(semantic_results):
                fp = msg["file_path"]
                sim_score = msg["_similarity"]
                score = (
                    semantic_weight * sim_score
                    + (1.0 / (i + 1)) * semantic_weight * 0.1
                )
                scored[fp] = scored.get(fp, 0) + score
                msg["_semantic_score"] = round(score, 4)
                if fp not in result_cache:
                    result_cache[fp] = msg
                else:
                    result_cache[fp]["_semantic_score"] = msg.get(
                        "_semantic_score", 0
                    )

            # Sort and return top_k
            sorted_results = sorted(
                scored.items(), key=lambda x: -x[1]
            )[:top_k]

            # Batch-fetch snippets — avoid N+1 queries
            file_paths = [fp for fp, _ in sorted_results]
            snippets = self._extract_snippets_batch(file_paths, query)

            final: List[Dict[str, Any]] = []
            for fp, total_score in sorted_results:
                r = result_cache.get(fp, {})
                final.append({
                    "file_path": r.get("file_path", fp),
                    "title": r.get("title", ""),
                    "category": r.get("category", ""),
                    "snippet": snippets.get(fp, ""),
                    "score": round(total_score, 4),
                    "fts5_score": r.get("_fts5_score", 0),
                    "semantic_score": r.get("_semantic_score", 0),
                })

            return final

        except Exception as e:
            logger.error("RLIndex search failed: %s", e)
            return []

    def _fts_search(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """FTS5 keyword search."""
        if not self._conn:
            return []

        try:
            # FTS5 phrase query: escape single quotes and wrap in double quotes
            # to force phrase matching. The double quotes prevent FTS5 from
            # treating the query as a column name reference.
            escaped = query.replace("'", "''")
            fts_query = f'"{escaped}"'

            where = "rl_index_fts MATCH ?"
            params: list = [fts_query]

            if category:
                where += " AND category = ?"
                params.append(category)

            sql = (
                "SELECT r.rowid, r.file_path, r.title, r.category, rank"
                " FROM rl_index_fts r"
                f" WHERE {where}"
                " ORDER BY rank LIMIT ?"
            )
            params.append(top_k)

            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

            results: List[Dict[str, Any]] = []
            for row in rows:
                results.append({
                    "file_path": row[1],
                    "title": row[2],
                    "category": row[3],
                    "_rank": float(row[4]),
                })

            return results

        except Exception as e:
            logger.error("FTS5 search failed: %s", e)
            return []

    def _semantic_search(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search via cosine similarity against stored embeddings."""
        if not self._conn:
            return []

        try:
            engine = _get_embedding_engine()
            query_vector = engine.embed(query)
            if query_vector is None:
                logger.debug(
                    "RL semantic search skipped — "
                    "embedding model unavailable"
                )
                return []

            if category:
                sql = (
                    "SELECT rf.file_path, rf.title, rf.category,"
                    " re.embedding"
                    " FROM rl_embeddings re"
                    " JOIN rl_files rf"
                    " ON rf.file_path = re.file_path"
                    " WHERE rf.category = ?"
                    " AND LENGTH(re.embedding) >= ?"
                )
                params: list = [category, RL_EMBED_DIM * 4]
            else:
                sql = (
                    "SELECT rf.file_path, rf.title, rf.category,"
                    " re.embedding"
                    " FROM rl_embeddings re"
                    " JOIN rl_files rf"
                    " ON rf.file_path = re.file_path"
                    " WHERE LENGTH(re.embedding) >= ?"
                )
                params = [RL_EMBED_DIM * 4]

            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()

            if not rows:
                return []

            # Compute cosine similarity for each
            scored: List[tuple] = []
            for row in rows:
                blob = row[3]
                if not blob or len(blob) < RL_EMBED_DIM * 4:
                    continue
                try:
                    vector = list(
                        struct.unpack(
                            f"{RL_EMBED_DIM}f",
                            blob[: RL_EMBED_DIM * 4],
                        )
                    )
                except struct.error:
                    continue

                sim = self._cosine_similarity(query_vector, vector)
                if sim < RL_COSINE_THRESHOLD:
                    continue

                scored.append((row[0], row[1], row[2], sim))

            scored.sort(key=lambda x: -x[3])

            results: List[Dict[str, Any]] = []
            for fp, title, cat, sim in scored[:top_k]:
                results.append({
                    "file_path": fp,
                    "title": title,
                    "category": cat,
                    "_similarity": round(sim, 4),
                })

            return results

        except Exception as e:
            logger.error("RL semantic search failed: %s", e)
            return []

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _extract_snippets_batch(
        self, file_paths: List[str], query: str
    ) -> Dict[str, str]:
        """Batch-extract snippets for multiple files in one query.

        Returns dict mapping file_path -> snippet text.
        """
        if not self._conn or not file_paths:
            return {}

        try:
            placeholders = ",".join("?" for _ in file_paths)
            cursor = self._conn.execute(
                f"SELECT file_path, body FROM rl_files"
                f" WHERE file_path IN ({placeholders})",
                file_paths,
            )

            snippets: Dict[str, str] = {}
            query_lower = query.lower()
            max_len = 300

            for row in cursor.fetchall():
                fp, body = row
                idx = body.lower().find(query_lower)

                if idx >= 0:
                    start = max(0, idx - 100)
                    end = min(len(body), idx + max_len)
                else:
                    start = 0
                    end = min(len(body), max_len)

                snippet = body[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(body):
                    snippet = snippet + "..."

                snippets[fp] = snippet

            return snippets

        except Exception as e:
            logger.debug("Snippet extraction failed: %s", e)
            return {}

    # -----------------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return index statistics."""
        if not self._conn:
            return {"error": "not initialized"}

        try:
            file_count = self._conn.execute(
                "SELECT COUNT(*) FROM rl_files"
            ).fetchone()[0]
            embed_count = self._conn.execute(
                "SELECT COUNT(*) FROM rl_embeddings"
            ).fetchone()[0]
            categories: Dict[str, int] = {}
            for row in self._conn.execute(
                "SELECT category, COUNT(*) FROM rl_files"
                " GROUP BY category"
            ):
                categories[row[0]] = row[1]
            total_size = self._conn.execute(
                "SELECT COALESCE(SUM(size), 0) FROM rl_files"
            ).fetchone()[0]

            return {
                "files_indexed": file_count,
                "files_embedded": embed_count,
                "categories": categories,
                "total_bytes": total_size,
            }
        except Exception as e:
            logger.error("RLIndex stats failed: %s", e)
            return {"error": str(e)}

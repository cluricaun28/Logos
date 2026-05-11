"""RL Index Builder — build, update, remove, reindex stale.

Extracted from RLIndex for SRP compliance. Receives all dependencies
(conn, lock, embedding engine/model) as parameters. Stateless functions.
"""

from __future__ import annotations

import logging
import os
import struct
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Re-export constants needed by callers
RL_EMBED_DIM = 384
RL_EMBED_MAX_CONTENT_LEN = 8000
EMBED_BATCH_SIZE = 64
PROGRESS_LOG_INTERVAL = 512
RL_BASE_DIR = os.path.expanduser("~/.hermes/reference-library")


def build_index(
    *,
    conn: Any,
    lock: Any,
    rl_base: str,
    file_info_extractor,
    embed_batch_fn,
) -> dict:
    """Build the full RL index from scratch. Returns dict with build stats."""
    base_path = Path(rl_base)
    if not base_path.exists():
        raise FileNotFoundError(f"RL base directory not found: {rl_base}")

    md_files = sorted(base_path.rglob("*.md"))
    total = len(md_files)
    logger.info("RLIndex build: found %d .md files in %s", total, rl_base)

    with lock:
        conn.execute("DELETE FROM rl_embeddings")
        conn.execute("DELETE FROM rl_files")

    start = time.time()
    now = time.time()

    stats: dict[str, Any] = {
        "files_processed": 0,
        "files_indexed": 0,
        "files_embedded": 0,
        "files_failed": 0,
        "elapsed_seconds": 0,
        "categories": {},
    }

    # Phase 1: Extract file info
    file_infos: list[dict] = []
    for md_file in md_files:
        info = file_info_extractor(md_file)
        if info is None:
            stats["files_failed"] += 1
            continue
        info["indexed_at"] = now
        file_infos.append(info)
        cat = info["category"]
        stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
        stats["files_processed"] += 1

    info_by_path: dict[str, dict] = {
        fi["file_path"]: fi for fi in file_infos
    }

    # Phase 2: Bulk insert — drop triggers, drop FTS5, bulk insert, recreate
    with lock:
        for trig in ("rl_files_ai", "rl_files_ad", "rl_files_au"):
            try:
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            except (sqlite3.Error, AttributeError) as e:
                logger.debug("Error dropping trigger %s: %s", trig, e)

        conn.execute("DROP TABLE IF EXISTS rl_index_fts")
        conn.execute("DELETE FROM rl_files")

        conn.executemany(
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

        conn.execute(
            "CREATE VIRTUAL TABLE rl_index_fts USING fts5("
            "  file_path, title, body, category,"
            "  content='rl_files',"
            "  content_rowid='rowid',"
            "  tokenize='unicode61'"
            ")"
        )

        conn.execute(
            "INSERT INTO rl_index_fts(rowid, file_path, title, body, "
            "category) "
            "SELECT rowid, file_path, title, body, category "
            "FROM rl_files"
        )

        for trig_sql in (
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
            conn.execute(trig_sql)

        conn.commit()

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
        vectors = embed_batch_fn(list(texts), batch_size=EMBED_BATCH_SIZE)

        embed_rows: list[tuple] = []
        for path, vector in zip(paths, vectors):
            if vector is not None:
                blob = struct.pack(f"{RL_EMBED_DIM}f", *vector)
                info = info_by_path.get(path)
                mtime = info["mtime"] if info else 0
                embed_rows.append((path, blob, mtime, now))
                stats["files_embedded"] += 1

        if embed_rows:
            with lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO rl_embeddings"
                    " (file_path, embedding, mtime, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    embed_rows,
                )
                conn.commit()

        if (i + EMBED_BATCH_SIZE) % PROGRESS_LOG_INTERVAL == 0 or (
            i + EMBED_BATCH_SIZE >= len(texts_by_path)
        ):
            elapsed = time.time() - start
            rate = len(texts_by_path) / elapsed if elapsed > 0 else 0
            logger.info(
                "RLIndex build: embeddings %d/%d (%.1fs, %.0f/s)",
                min(i + EMBED_BATCH_SIZE, len(texts_by_path)),
                len(texts_by_path),
                elapsed, rate,
            )

    stats["elapsed_seconds"] = round(time.time() - start, 1)
    logger.info(
        "RLIndex build complete: %d files, %d embeddings, %.1fs",
        stats["files_indexed"], stats["files_embedded"],
        stats["elapsed_seconds"],
    )
    return stats


def update_file(
    *,
    conn: Any,
    lock: Any,
    file_path_str: str,
    file_info_extractor,
    embed_fn,
    embed_dim: int = RL_EMBED_DIM,
) -> bool:
    """Update a single file in the index. Returns True if updated."""
    path = Path(file_path_str)
    if not path.exists():
        return False

    info = file_info_extractor(path)
    if info is None:
        return False

    info["indexed_at"] = time.time()

    with lock:
        try:
            conn.execute(
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

            vector = embed_fn(info["body"])
            if vector is not None:
                blob = struct.pack(f"{embed_dim}f", *vector)
                conn.execute(
                    "INSERT OR REPLACE INTO rl_embeddings"
                    " (file_path, embedding, mtime, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (info["file_path"], blob, info["mtime"], time.time()),
                )

            conn.commit()
            return True
        except (sqlite3.Error, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to update RL file %s: %s",
                info.get("file_path", path), e,
            )
            return False


def remove_file(
    *,
    conn: Any,
    lock: Any,
    file_path_str: str,
    rl_base: str = RL_BASE_DIR,
) -> bool:
    """Remove a file from the index. Returns True if removed."""
    path = Path(file_path_str)

    try:
        rel_path = str(path.relative_to(rl_base))
    except ValueError:
        rel_path = str(path)

    with lock:
        try:
            conn.execute(
                "DELETE FROM rl_files WHERE file_path = ?", (rel_path,)
            )
            conn.execute(
                "DELETE FROM rl_embeddings WHERE file_path = ?", (rel_path,)
            )
            conn.commit()
            return True
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Failed to remove RL file %s: %s", rel_path, e)
            return False


def reindex_stale(
    *,
    conn: Any,
    lock: Any,
    file_info_extractor,
    embed_batch_fn,
    update_file_fn,
    rl_base: str = RL_BASE_DIR,
    embed_dim: int = RL_EMBED_DIM,
) -> int:
    """Re-index files whose mtime has changed. Returns count updated."""
    stale: list[str] = []
    deleted: list[str] = []

    with lock:
        cursor = conn.execute(
            "SELECT file_path, mtime FROM rl_files"
        )
        for row in cursor.fetchall():
            fp, recorded_mtime = row
            full_path = os.path.join(rl_base, fp)
            try:
                actual_mtime = os.path.getmtime(full_path)
                if actual_mtime != recorded_mtime:
                    stale.append(fp)
            except FileNotFoundError:
                deleted.append(fp)

        for fp in deleted:
            conn.execute("DELETE FROM rl_files WHERE file_path = ?", (fp,))
            conn.execute(
                "DELETE FROM rl_embeddings WHERE file_path = ?", (fp,)
            )

        if deleted:
            conn.commit()

    if not stale:
        return 0

    file_infos: list[dict] = []
    now = time.time()
    for fp in stale:
        full_path = os.path.join(rl_base, fp)
        info = file_info_extractor(Path(full_path))
        if info is not None:
            info["indexed_at"] = now
            file_infos.append(info)

    if not file_infos:
        if stale:
            logger.info(
                "RLIndex reindex_stale: 0 updated, %d failed, "
                "removed %d deleted",
                len(stale), len(deleted),
            )
        return 0

    count = 0

    with lock:
        for info in file_infos:
            try:
                conn.execute(
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
            except (sqlite3.Error, KeyError, AttributeError) as e:
                logger.error(
                    "Failed to update RL file %s: %s",
                    info.get("file_path"), e,
                )
                continue

        # Batch embeddings in one model call
        texts = [info["body"] for info in file_infos]
        vectors = embed_batch_fn(texts, batch_size=EMBED_BATCH_SIZE)

        embed_rows: list[tuple] = []
        for info, vector in zip(file_infos, vectors):
            if vector is not None:
                blob = struct.pack(f"{embed_dim}f", *vector)
                embed_rows.append((
                    info["file_path"], blob, info["mtime"], now,
                ))
                count += 1

        if embed_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO rl_embeddings"
                " (file_path, embedding, mtime, created_at)"
                " VALUES (?, ?, ?, ?)",
                embed_rows,
            )

        conn.commit()

    logger.info(
        "RLIndex reindex_stale: updated %d/%d, removed %d deleted",
        count, len(stale), len(deleted),
    )
    return count

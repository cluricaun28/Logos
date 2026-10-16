"""RL Index Builder — build, update, remove, reindex stale.

Extracted from RLIndex for SRP compliance. Receives all dependencies
(conn, lock, embedding engine/model) as parameters. Stateless functions.
"""

from __future__ import annotations

import logging
import os
import sqlite3
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

# ── Schema v2 chunking ─────────────────────────────────────────────────
# Chunk size is tuned for the embedding model, not just FTS:
# all-MiniLM-L6-v2 truncates at 256 tokens (~1100-1200 chars of English),
# so a 1000-char chunk is the largest unit that a single chunk embedding
# can represent in full. Overlap keeps phrases straddling a boundary
# searchable and lets word-boundary back-off lose little coverage.
RL_CHUNK_SIZE = 1000
RL_CHUNK_OVERLAP = 150


def _chunk_text(
    text: str,
    size: int = RL_CHUNK_SIZE,
    overlap: int = RL_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks on whitespace boundaries.

    Guarantees:
      - every character of the input appears in at least one chunk
        (consecutive chunks overlap by `overlap` chars);
      - at most one chunk for text of length <= size;
      - no empty chunks; never raises on odd input.

    Breaks prefer the last whitespace inside the final `overlap` window of
    a chunk, so words are not split mid-token. A break window with no
    whitespace (very long unbroken run) falls back to a hard cut — forward
    progress is guaranteed either way (stride = size - overlap > 0).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while True:
        end = min(start + size, n)
        if end < n:
            ws = text.rfind(" ", end - overlap, end)
            if ws <= start:
                ws = text.rfind("\n", end - overlap, end)
            if ws > start:
                end = ws
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
        if start <= (end - size):  # defensive: guarantee forward progress
            start = end - overlap + 1
    return chunks


def _insert_chunk_rows(conn: Any, file_infos: list[dict]) -> None:
    """Bulk-insert rl_chunks rows for a set of file infos and rebuild the
    chunk FTS5 table from them.

    Called under the caller's lock (build_index / reindex_stale). The FTS
    table is dropped and recreated — mirroring the legacy rl_index_fts
    pattern — because after a bulk content rewrite the cheapest correct
    state is a fresh index. Drift from a crash mid-build is repaired by
    RLIndex._self_heal_fts().
    """
    chunk_rows: list[tuple] = []
    for fi in file_infos:
        for idx, chunk in enumerate(_chunk_text(fi["body"])):
            chunk_rows.append((fi["file_path"], idx, chunk))

    conn.execute("DELETE FROM rl_chunks")
    conn.executemany(
        "INSERT INTO rl_chunks(file_path, chunk_idx, body) VALUES (?,?,?)",
        chunk_rows,
    )

    conn.execute("DROP TABLE IF EXISTS rl_chunks_fts")
    conn.execute(
        "CREATE VIRTUAL TABLE rl_chunks_fts USING fts5("
        "  file_path, chunk_idx, body,"
        "  content='rl_chunks',"
        "  content_rowid='rowid',"
        "  tokenize='unicode61'"
        ")"
    )
    conn.execute(
        "INSERT INTO rl_chunks_fts(rowid, file_path, chunk_idx, body) "
        "SELECT rowid, file_path, chunk_idx, body FROM rl_chunks"
    )


def _replace_file_chunks(conn: Any, file_path: str, body: str) -> list[int]:
    """Replace the rl_chunks rows (and their FTS rows) for one file.

    Returns the new chunk rowids in chunk_idx order. Caller must hold the
    lock and be inside an open transaction.
    """
    old_rowids = [
        r[0]
        for r in conn.execute(
            "SELECT rowid FROM rl_chunks WHERE file_path = ?", (file_path,)
        )
    ]
    if old_rowids:
        conn.executemany(
            "DELETE FROM rl_chunks_fts WHERE rowid = ?",
            [(rid,) for rid in old_rowids],
        )
        conn.execute(
            "DELETE FROM rl_chunks WHERE file_path = ?", (file_path,)
        )
        conn.execute(
            "DELETE FROM rl_chunk_embeddings WHERE file_path = ?",
            (file_path,),
        )

    chunks = _chunk_text(body)
    for idx, chunk in enumerate(chunks):
        conn.execute(
            "INSERT INTO rl_chunks(file_path, chunk_idx, body) VALUES (?,?,?)",
            (file_path, idx, chunk),
        )
    new_rowids = [
        r[0]
        for r in conn.execute(
            "SELECT rowid FROM rl_chunks WHERE file_path = ?"
            " ORDER BY chunk_idx",
            (file_path,),
        )
    ]
    conn.executemany(
        "INSERT INTO rl_chunks_fts(rowid, file_path, chunk_idx, body)"
        " VALUES (?,?,?,?)",
        [(rid, file_path, i, chunk) for i, (rid, chunk) in enumerate(
            zip(new_rowids, chunks)
        )],
    )
    return new_rowids


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
        # Schema v2: clear chunk tables too. Deleting rows (rather than
        # dropping tables) keeps AUTOINCREMENT rowids monotonic, so the
        # external-content FTS rowid mapping never collides with a live
        # reader's snapshot mid-build.
        conn.execute("DELETE FROM rl_chunk_embeddings")
        conn.execute("DELETE FROM rl_chunks")

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

        # Schema v2: chunk rows for full-body FTS + chunk embeddings.
        # rl_files now carries the FULL body (see _extract_file_info), so
        # the legacy file-level FTS above is also full-text — pre-v2
        # readers of a v2-built index gain keyword coverage immediately.
        _insert_chunk_rows(conn, file_infos)

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
    stats["embeddings_ok"] = stats["files_embedded"] > 0 or stats["files_indexed"] == 0
    if not stats["embeddings_ok"]:
        logger.error(
            "RLIndex build: 0 embeddings produced for %d files — index is "
            "FTS-ONLY (semantic search degraded). Embedding engine was "
            "unavailable (GPU OOM? missing model?). Set HERMES_EMBED_DEVICE "
            "or free a GPU and rebuild.",
            stats["files_indexed"],
        )

    # Phase 3b (schema v2): chunk embeddings — semantic coverage for the
    # full body, not just the first RL_EMBED_MAX_CONTENT_LEN chars.
    stats["chunks_indexed"] = 0
    stats["chunks_embedded"] = 0
    try:
        with lock:
            chunk_rows = conn.execute(
                "SELECT rowid, file_path, body FROM rl_chunks"
            ).fetchall()
    except (sqlite3.Error, AttributeError) as e:
        logger.debug("Chunk rows unreadable (pre-v2 DB?): %s", e)
        chunk_rows = []
    stats["chunks_indexed"] = len(chunk_rows)

    for i in range(0, len(chunk_rows), EMBED_BATCH_SIZE):
        batch = chunk_rows[i: i + EMBED_BATCH_SIZE]
        cids = [r[0] for r in batch]
        cpaths = [r[1] for r in batch]
        ctexts = [r[2] for r in batch]
        vectors = embed_batch_fn(ctexts, batch_size=EMBED_BATCH_SIZE)

        embed_rows: list[tuple] = []
        for cid, cpath, vector in zip(cids, cpaths, vectors):
            if vector is not None:
                blob = struct.pack(f"{RL_EMBED_DIM}f", *vector)
                info = info_by_path.get(cpath)
                mtime = info["mtime"] if info else 0
                embed_rows.append((cid, cpath, blob, mtime, now))

        if embed_rows:
            with lock:
                conn.executemany(
                    "INSERT OR REPLACE INTO rl_chunk_embeddings"
                    " (chunk_id, file_path, embedding, mtime, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    embed_rows,
                )
                conn.commit()
            stats["chunks_embedded"] += len(embed_rows)

    logger.info(
        "RLIndex build complete: %d files, %d embeddings, %d chunks "
        "(%d chunk embeddings), %.1fs",
        stats["files_indexed"], stats["files_embedded"],
        stats["chunks_indexed"], stats["chunks_embedded"],
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

            # Schema v2: refresh this file's chunks + chunk embeddings.
            # Each chunk embed is a single encode call (chunks are small
            # by construction); a failed chunk embed degrades to FTS-only
            # for that chunk, same as the file-level path.
            chunk_rowids = _replace_file_chunks(
                conn, info["file_path"], info["body"]
            )
            for rid, chunk in zip(chunk_rowids, _chunk_text(info["body"])):
                cv = embed_fn(chunk)
                if cv is not None:
                    cblob = struct.pack(f"{embed_dim}f", *cv)
                    conn.execute(
                        "INSERT OR REPLACE INTO rl_chunk_embeddings"
                        " (chunk_id, file_path, embedding, mtime,"
                        "  created_at) VALUES (?, ?, ?, ?, ?)",
                        (rid, info["file_path"], cblob,
                         info["mtime"], time.time()),
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
            # Schema v2: drop chunk FTS rows before the content rows they
            # reference (external-content FTS must not keep dead rowids).
            old_rowids = [
                r[0]
                for r in conn.execute(
                    "SELECT rowid FROM rl_chunks WHERE file_path = ?",
                    (rel_path,),
                )
            ]
            if old_rowids:
                conn.executemany(
                    "DELETE FROM rl_chunks_fts WHERE rowid = ?",
                    [(rid,) for rid in old_rowids],
                )
                conn.execute(
                    "DELETE FROM rl_chunks WHERE file_path = ?", (rel_path,)
                )
                conn.execute(
                    "DELETE FROM rl_chunk_embeddings WHERE file_path = ?",
                    (rel_path,),
                )

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
    """Re-index files whose mtime has changed, and discover new files.

    Returns count updated. New-file discovery (2026-10-15, measured on the
    fleet box): previously only rows already tracked in rl_files were
    scanned, so files added to disk stayed invisible to search until a full
    build_index() ran — the first 10 book-catalog pages (RL
    library/corpus-catalog) plus 7 other pages were missing from the index
    and ranked absent from top-20 on their own topical queries.
    """
    stale: list[str] = []
    deleted: list[str] = []

    with lock:
        cursor = conn.execute(
            "SELECT file_path, mtime FROM rl_files"
        )
        rows = cursor.fetchall()
        for fp, recorded_mtime in rows:
            full_path = os.path.join(rl_base, fp)
            try:
                actual_mtime = os.path.getmtime(full_path)
                if actual_mtime != recorded_mtime:
                    stale.append(fp)
            except FileNotFoundError:
                deleted.append(fp)

        # Discovery: .md files on disk that were never tracked. They join
        # the stale set; the extractor below reads them fresh from disk.
        tracked = {fp for fp, _mt in rows}
        for dirpath, _dirnames, filenames in os.walk(rl_base):
            for fn in filenames:
                if not fn.endswith(".md"):
                    continue
                rel = os.path.relpath(os.path.join(dirpath, fn), rl_base)
                if rel not in tracked:
                    stale.append(rel)

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
        updated: list[dict] = []
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
                updated.append(info)
            except (sqlite3.Error, KeyError, AttributeError) as e:
                logger.error(
                    "Failed to update RL file %s: %s",
                    info.get("file_path"), e,
                )
                continue

        # Batch embeddings in one model call
        texts = [info["body"] for info in updated]
        vectors = embed_batch_fn(texts, batch_size=EMBED_BATCH_SIZE)

        embed_rows: list[tuple] = []
        for info, vector in zip(updated, vectors):
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

        # Schema v2: re-chunk the updated files and batch-embed the new
        # chunks. Chunk FTS rows are replaced per file (not rebuilt
        # wholesale) — reindex_stale may touch a small slice of a large
        # index, and a full FTS drop/recreate per sync would be wasteful.
        chunk_pairs: list[tuple] = []  # (rowid, file_path, mtime, chunk_text)
        for info in updated:
            rowids = _replace_file_chunks(
                conn, info["file_path"], info["body"]
            )
            chunk_pairs.extend(
                (rid, info["file_path"], info["mtime"], chunk)
                for rid, chunk in zip(rowids, _chunk_text(info["body"]))
            )

        chunk_rows: list[tuple] = []
        for i in range(0, len(chunk_pairs), EMBED_BATCH_SIZE):
            batch = chunk_pairs[i: i + EMBED_BATCH_SIZE]
            cids = [p[0] for p in batch]
            cpaths = [p[1] for p in batch]
            cmtime = [p[2] for p in batch]
            ctexts = [p[3] for p in batch]
            cvectors = embed_batch_fn(ctexts, batch_size=EMBED_BATCH_SIZE)
            chunk_rows.extend(
                (cid, cpath, struct.pack(f"{embed_dim}f", *v), fmtime, now)
                for cid, cpath, fmtime, v in zip(
                    cids, cpaths, cmtime, cvectors
                )
                if v is not None
            )

        if chunk_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO rl_chunk_embeddings"
                " (chunk_id, file_path, embedding, mtime, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                chunk_rows,
            )

        conn.commit()

    if count == 0 and file_infos:
        logger.warning(
            "RLIndex reindex_stale: %d file(s) re-indexed but 0 embedded — "
            "embedding engine unavailable (GPU OOM?). New/changed files are "
            "FTS-only until the next successful embed.",
            len(file_infos),
        )
    else:
        logger.info(
            "RLIndex reindex_stale: updated %d/%d, removed %d deleted",
            count, len(stale), len(deleted),
        )
    return count

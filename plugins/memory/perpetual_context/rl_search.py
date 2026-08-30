"""RL Index Search — FTS5, semantic, hybrid fusion, snippet extraction.

Extracted from RLIndex for SRP compliance. Stateless functions that receive
the connection and embedding engine as parameters.
"""

from __future__ import annotations

import logging
import math
import struct
from typing import Any

logger = logging.getLogger(__name__)

# Re-export constants
RL_SEMANTIC_WEIGHT = 0.4
RL_FTS5_WEIGHT = 0.6
RL_COSINE_THRESHOLD = 0.05
RL_EMBED_DIM = 384


def search(
    *,
    conn: Any,
    query: str,
    top_k: int = 5,
    category: str | None = None,
    semantic_weight: float = RL_SEMANTIC_WEIGHT,
    fts5_weight: float = RL_FTS5_WEIGHT,
    embedding_engine: Any | None = None,
) -> list[dict[str, Any]]:
    """Hybrid search: FTS5 keyword + cosine similarity."""
    if not conn:
        return []

    try:
        # Candidate pool: at least 50 per branch. With top_k*2=10, a relevant
        # page ranking #15 in either branch never received a score — the
        # dominant failure mode in the 2026-08 search battery (specific
        # pages drowned out by common-term matches in the small pool).
        pool = max(top_k * 2, 50)
        fts_results = _fts_search(
            conn=conn, query=query, top_k=pool, category=category,
        )
        semantic_results = _semantic_search(
            conn=conn, query=query, top_k=pool, category=category,
            embedding_engine=embedding_engine,
        )

        scored: dict[str, float] = {}
        result_cache: dict[str, dict[str, Any]] = {}

        # Score FTS5 results
        if fts_results:
            min_rank = min(m["_rank"] for m in fts_results)
            max_rank = max(m["_rank"] for m in fts_results)
            rank_range = (
                max_rank - min_rank if max_rank != min_rank else 1.0
            )

            for i, msg in enumerate(fts_results):
                fp = msg["file_path"]
                normalized = 1.0 - ((msg["_rank"] - min_rank) / rank_range)
                score = (
                    fts5_weight * normalized
                    + (1.0 / (i + 1)) * fts5_weight * 0.1
                )
                scored[fp] = scored.get(fp, 0) + score
                msg["_fts5_score"] = round(score, 4)
                result_cache[fp] = msg

        # Score semantic results. Min-max normalize similarity across the
        # pool: raw cosine typically tops out near 0.5, which would starve
        # the semantic branch vs the min-max-normalized FTS branch (0..1).
        if semantic_results:
            sims = [m["_similarity"] for m in semantic_results]
            smin = min(sims)
            smax = max(sims)
            srange = (smax - smin) if smax != smin else 1.0
        for i, msg in enumerate(semantic_results):
            fp = msg["file_path"]
            sim_score = (
                (msg["_similarity"] - smin) / srange if semantic_results else 0.0
            )
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

        sorted_results = sorted(
            scored.items(), key=lambda x: -x[1]
        )[:top_k]

        file_paths = [fp for fp, _ in sorted_results]
        snippets = _extract_snippets_batch(conn=conn, file_paths=file_paths, query=query)

        final: list[dict[str, Any]] = []
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

    except Exception as e:  # noqa: S110 — degradation wrapper, must never fail RL search
        logger.error("RLIndex search failed: %s", e)
        return []


def _fts_search(
    *,
    conn: Any,
    query: str,
    top_k: int = 10,
    category: str | None = None,
) -> list[dict[str, Any]]:
    """FTS5 keyword search."""
    if not conn:
        return []

    try:
        # OR-of-quoted-terms: each word is a literal phrase, so queries like
        # "NAS ZFS storage" match docs containing ANY of the words.
        # (Wrapping the whole query in one phrase required the exact sequence
        # to appear verbatim and returned ~0 hits for normal queries,
        # silently defeating the FTS half of hybrid search.)
        terms = [t for t in query.replace("'", "''").split() if t]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)

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

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append({
                "file_path": row[1],
                "title": row[2],
                "category": row[3],
                "_rank": float(row[4]),
            })

        return results

    except Exception as e:  # noqa: S110 — degradation wrapper, must never fail FTS5 search
        logger.error("FTS5 search failed: %s", e)
        return []


def _semantic_search(
    *,
    conn: Any,
    query: str,
    top_k: int = 10,
    category: str | None = None,
    embedding_engine: Any | None = None,
) -> list[dict[str, Any]]:
    """Semantic search via cosine similarity against stored embeddings."""
    if not conn:
        return []

    try:
        if embedding_engine is None:
            logger.debug("RL semantic search skipped — no embedding engine")
            return []

        query_vector = embedding_engine.embed(query)
        if query_vector is None:
            logger.debug(
                "RL semantic search skipped — embedding model unavailable"
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

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # Vectorized cosine scoring. numpy is a hard dependency of the
        # embedding engine (sentence-transformers); if it's somehow missing,
        # fall back to the pure-Python loop below.
        scored: list[tuple] = []
        try:
            import numpy as np

            dim = RL_EMBED_DIM
            valid_rows: list[tuple] = []
            vecs: list[list[float]] = []
            for row in rows:
                blob = row[3]
                if not blob or len(blob) < dim * 4:
                    continue
                vecs.append(list(struct.unpack(f"{dim}f", blob[: dim * 4])))
                valid_rows.append((row[0], row[1], row[2]))
            if vecs:
                mat = np.asarray(vecs, dtype=np.float32)
                qv = np.asarray(query_vector, dtype=np.float32)
                norms = np.linalg.norm(mat, axis=1)
                qnorm = np.linalg.norm(qv)
                if qnorm > 0:
                    sims = (mat @ qv) / (norms * qnorm)
                    for idx, sim in enumerate(sims):
                        if sim >= RL_COSINE_THRESHOLD:
                            scored.append(
                                (valid_rows[idx][0], valid_rows[idx][1],
                                 valid_rows[idx][2], float(sim))
                            )
        except ImportError:
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

                sim = _cosine_similarity(query_vector, vector)
                if sim < RL_COSINE_THRESHOLD:
                    continue

                scored.append((row[0], row[1], row[2], sim))

        scored.sort(key=lambda x: -x[3])

        results: list[dict[str, Any]] = []
        for fp, title, cat, sim in scored[:top_k]:
            results.append({
                "file_path": fp,
                "title": title,
                "category": cat,
                "_similarity": round(sim, 4),
            })

        return results

    except Exception as e:  # noqa: S110 — degradation wrapper, must never fail semantic search
        logger.error("Semantic search failed: %s", e)
        return []


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_snippets_batch(
    *,
    conn: Any,
    file_paths: list[str],
    query: str,
) -> dict[str, str]:
    """Batch-extract snippets for multiple files in one query."""
    if not conn or not file_paths:
        return {}

    try:
        placeholders = ",".join("?" for _ in file_paths)
        cursor = conn.execute(
            f"SELECT file_path, body FROM rl_files"
            f" WHERE file_path IN ({placeholders})",
            file_paths,
        )

        snippets: dict[str, str] = {}
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

    except Exception as e:  # noqa: S110 — degradation wrapper, must never fail snippet extraction
        logger.debug("Snippet extraction failed: %s", e)
        return {}


def get_stats(*, conn: Any) -> dict:
    """Return index statistics."""
    if not conn:
        return {"error": "not initialized"}

    try:
        file_count = conn.execute(
            "SELECT COUNT(*) FROM rl_files"
        ).fetchone()[0]
        embed_count = conn.execute(
            "SELECT COUNT(*) FROM rl_embeddings"
        ).fetchone()[0]
        categories: dict[str, int] = {}
        for row in conn.execute(
            "SELECT category, COUNT(*) FROM rl_files"
            " GROUP BY category"
        ):
            categories[row[0]] = row[1]
        total_size = conn.execute(
            "SELECT COALESCE(SUM(size), 0) FROM rl_files"
        ).fetchone()[0]

        return {
            "files_indexed": file_count,
            "files_embedded": embed_count,
            "categories": categories,
            "total_bytes": total_size,
        }
    except Exception as e:  # noqa: S110 — degradation wrapper, must never fail stats fetch
        logger.error("RLIndex stats failed: %s", e)
        return {"error": str(e)}

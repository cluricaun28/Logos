"""Search operations for Perpetual Context.

Handles FTS5 keyword search, semantic vector search, hybrid fusion,
and cross-session recall.

Search architecture:
  - FTS5: Actual BM25 ranking via SQLite FTS5 virtual table with safe query escaping
  - Semantic: sqlite-vec vec0 KNN (primary) with FAISS fallback, then full-table Python cosine
  - Hybrid: Reciprocal Rank Fusion of FTS5 rank and vector cosine similarity
  - Recall: Cross-session pointer injection with hard character cap
  - Recency: Time-based multiplicative boost after RRF (no decay, recent messages gain score)
"""

from __future__ import annotations

import json
import logging
import re as _re
import sqlite3
import time as _time
from typing import Any

logger = logging.getLogger(__name__)

# Local copies of constants from perpetual_context_db —
# these must match the canonical definitions in that module.
EMBED_DIM = 384
COSINE_SIMILARITY_THRESHOLD = 0.1
SEMANTIC_WEIGHT = 0.4
FTS5_WEIGHT = 0.6
RECALL_TOP_K_MULTIPLIER = 3
RECALL_MIN_SCORE = 0.15
RECALL_SNIPPET_MAX_LEN = 80
RECALL_OUTPUT_MAX_CHARS = 200

# Recency weighting — boost recent messages in hybrid search results.
# After RRF fusion, recent messages get a multiplicative score boost so they
# rank higher when semantic/text relevance is equal. No decay — old messages
# don't lose score, recent ones gain it. This preserves data integrity while
# favoring temporally relevant context.
RECENT_BOOST_DAYS = 7.0  # Window for strong recency boost (days)
RECENT_BOOST_FACTOR = 1.5  # Max multiplier within the boost window
MODERATE_BOOST_DAYS = 30.0  # Extended window for moderate boost
MODERATE_BOOST_FACTOR = 1.2  # Multiplier within extended window

# FTS5 query operators that must be escaped to prevent injection
_FTS5_SPECIAL_CHARS = _re.compile(r"[%*\+\-&|>()~]")
_FTS5_ESCAPE_MAP = str.maketrans(
    {
        "%": "%25",
        "*": "%2A",
        "+": "%2B",
        "-": "%2D",
        "&": "%26",
        "|": "%7C",
        ">": "%3E",
        "<": "%3C",
        "(": "%28",
        ")": "%29",
        "~": "%7E",
    }
)


def _escape_fts5_query(query: str) -> str:
    """Escape special FTS5 operators in a search query.

    FTS5 MATCH expressions cannot be parameterized (the query is part of
    the SQL string, not a bound parameter). This function escapes all
    special characters so arbitrary user input can be safely interpolated.

    Args:
        query: Raw search query string.

    Returns:
        Safe FTS5-compatible query string.
    """
    if not query:
        return ""
    return query.translate(_FTS5_ESCAPE_MAP)


class _SearchEngine:
    """Mixin: FTS5, semantic, and hybrid search with recall formatting.

    Expects the host class to provide:
      - self._conn, self._lock, self._initialized, self.time_column
    """

    def fts_search(self, query: str, session_id: str = None, top_k: int = 10) -> list[dict[str, Any]]:
        """Full-text search using SQLite FTS5 with BM25 ranking.

        Uses actual FTS5 MATCH (not LIKE) with proper query escaping to prevent
        injection. Returns results ranked by BM25 relevance score.

        Args:
            query: Search query string.
            session_id: Optional session filter.
            top_k: Maximum results to return.

        Returns:
            List of matching message dicts with _rank field (BM25 score).
        """
        if not self._initialized or not self._conn:
            return []

        try:
            time_col = self.time_column
            escaped = _escape_fts5_query(query.strip())
            if not escaped:
                return []

            # Use FTS5 MATCH with proper BM25 ranking.
            # The escaped query is safe to interpolate (all special chars percent-encoded).
            base_sql = f"""
                SELECT m.id, m.session_id, m.role, m.content,
                       COALESCE(m.metadata, '{{}}') as metadata,
                       m.{time_col},
                       -bm25(messages_fts) as rank
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                WHERE messages_fts MATCH ?
            """
            params: list = [escaped]

            if session_id:
                base_sql += " AND m.session_id = ?"
                params.append(session_id)

            base_sql += " ORDER BY rank LIMIT ?"
            params.append(top_k)

            cursor = self._conn.execute(base_sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                msg = {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5],
                    "_rank": round(float(row[6]), 4),
                }
                results.append(msg)

            return results

        except sqlite3.OperationalError as e:
            # FTS5 table may not exist yet on fresh DB — degrade gracefully
            logger.debug("FTS5 search unavailable (%s), falling back to LIKE", e)
            return self._fts_search_fallback(query, session_id, top_k)
        except Exception as e:
            logger.error("FTS search failed: %s", e)
            return self._fts_search_fallback(query, session_id, top_k)

    def _fts_search_fallback(self, query: str, session_id: str = None, top_k: int = 10) -> list[dict[str, Any]]:
        """LIKE-based fallback if FTS5 virtual table is not available."""
        if not self._initialized or not self._conn:
            return []

        try:
            time_col = self.time_column
            like_pattern = f"%{query.strip()}%"
            query_words = set(query.lower().split())

            escaped_words = [w.replace("'", "''") for w in list(query_words)[:5]]
            base_sql = f"""
                SELECT m.id, m.session_id, m.role, m.content,
                       COALESCE(m.metadata, '{{}}') as metadata,
                       m.{time_col},
                       -({"+ ".join([f"CASE WHEN LOWER(m.content) LIKE '%{w}%' THEN 1 ELSE 0 END" for w in escaped_words])}) as rank
                FROM messages m
                WHERE m.content LIKE ?
            """
            params: list = [like_pattern]

            if session_id:
                base_sql += " AND m.session_id = ?"
                params.append(session_id)

            base_sql += " ORDER BY rank LIMIT ?"
            params.append(top_k)

            cursor = self._conn.execute(base_sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                msg = {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5],
                    "_rank": float(row[6]),
                }
                results.append(msg)

            return results

        except Exception as e:
            logger.error("FTS fallback search also failed: %s", e)
            return []

    def semantic_search(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 10,
        exclude_session_id: str = None,
    ) -> list[dict[str, Any]]:
        """Semantic search using sqlite-vec vec0 KNN (primary) with graceful degradation.

        Three-tier fallback:
          1. sqlite-vec vec0 KNN query — single SQL, atomic with DB
          2. FAISS index — legacy fallback for backwards compatibility
          3. Full-table Python cosine scan — always available, slow for large DBs

        Graceful degradation: if embedding model is unavailable or no messages
        have embeddings yet, returns empty list without error.

        Args:
            query: Search query text to embed.
            session_id: Optional session filter.
            top_k: Maximum results to return.
            exclude_session_id: Optional session ID to skip.

        Returns:
            List of message dicts with '_similarity' score field (0-1, higher = more similar).
        """
        if not self._initialized or not self._conn:
            return []

        try:
            from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415

            engine = EmbeddingEngine.get()
            query_vector = engine.embed(query)
            if query_vector is None:
                logger.debug("Semantic search skipped — embedding model unavailable")
                return []

            # Tier 1: sqlite-vec vec0 KNN (primary)
            try:
                sqlite_vec_results = self._semantic_search_sqlite_vec(query_vector, session_id, exclude_session_id, top_k, engine)
                if sqlite_vec_results:
                    return sqlite_vec_results
            except Exception as e:
                logger.debug("sqlite-vec search unavailable (%s), falling back to FAISS", e)

            # Tier 2: FAISS index (legacy fallback)
            try:
                from agent.pcdb_vector_index import VectorIndex  # noqa: PLC0415

                vi = VectorIndex(EMBED_DIM)
                if not vi.load():
                    vi.build_from_db(self._conn)
                    vi.save()

                faiss_results = vi.search(query_vector, top_k=top_k * 2)
                if faiss_results:
                    return self._apply_semantic_filters(faiss_results, session_id, exclude_session_id, top_k, engine)
            except (ImportError, ModuleNotFoundError) as e:
                logger.debug("FAISS search unavailable (%s), falling back to DB scan", e)

            # Tier 3: full-table scan with Python cosine (always available)
            return self._semantic_search_fallback(query_vector, session_id, exclude_session_id, top_k, engine)

        except (ImportError, ModuleNotFoundError) as e:
            logger.error("Semantic search failed: %s", e)
            return []

    def _semantic_search_sqlite_vec(
        self,
        query_vector: list[float],
        session_id: str = None,
        exclude_session_id: str = None,
        top_k: int = 10,
        engine: Any = None,
    ) -> list[dict[str, Any]]:
        """Tier 1 semantic search: query vec0 table for KNN results.

        Uses sqlite-vec's vec0 virtual table for vector similarity search.
        Single SQL query, atomic with DB transaction.

        Args:
            query_vector: Pre-computed query embedding vector.
            session_id: Optional session filter.
            exclude_session_id: Optional session ID to skip.
            top_k: Maximum results to return.
            engine: EmbeddingEngine instance for distance-to-similarity conversion.

        Returns:
            List of message dicts with '_similarity' score field, or empty list.
        """
        import sqlite_vec as _sqlite_vec  # noqa: PLC0415

        # Load sqlite-vec extension into the connection (idempotent)
        self._conn.enable_load_extension(True)
        _sqlite_vec.load(self._conn)

        # Build KNN query — fetch extra to allow filtering
        fetch_count = top_k * 3
        vec_query = f"""
            SELECT msg_id, distance
            FROM chunks_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT {fetch_count}
        """
        query_blob = _sqlite_vec.serialize_float32(query_vector)
        vec_results = self._conn.execute(vec_query, (query_blob,)).fetchall()

        if not vec_results:
            return []

        # Convert distance to similarity (sqlite-vec uses L2 distance)
        scored: list[tuple[int, float, dict[str, Any]]] = []
        for msg_id, distance in vec_results:
            # Convert L2 distance to cosine similarity approximation
            # distance 0 = identical, distance ~2 = orthogonal
            sim = max(0.0, 1.0 - (distance / 2.0))
            if sim < COSINE_SIMILARITY_THRESHOLD:
                continue
            scored.append((msg_id, sim, {"_similarity": round(sim, 4)}))

        if not scored:
            return []

        # Apply session filters via DB lookup
        ids_to_fetch = [msg_id for msg_id, _, _ in scored]
        time_col = self.time_column
        placeholders = ",".join("?" for _ in ids_to_fetch)
        where_parts = []
        params: list = list(ids_to_fetch)

        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if exclude_session_id:
            where_parts.append("session_id != ?")
            params.append(exclude_session_id)

        where_clause = ""
        if where_parts:
            where_clause = f" AND {' AND '.join(where_parts)}"

        sql = f"SELECT id, session_id, role, content, metadata, {time_col} FROM messages WHERE id IN ({placeholders}){where_clause}"
        cursor = self._conn.execute(sql, params)

        # Build lookup of filtered messages
        msg_lookup = {}
        for row in cursor.fetchall():
            msg_lookup[row[0]] = {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
            }

        # Assemble results — only include messages that passed session filter
        results = []
        for msg_id, sim, extra in scored[:top_k]:
            if msg_id in msg_lookup:
                msg = dict(msg_lookup[msg_id])
                msg["_similarity"] = extra.get("_similarity", round(sim, 4))
                results.append(msg)

        return results

    def _apply_semantic_filters(
        self,
        faiss_results: list[tuple[int, float]],
        session_id: str = None,
        exclude_session_id: str = None,
        top_k: int = 10,
        engine: Any = None,
    ) -> list[dict[str, Any]]:
        """Apply session filters to FAISS results and fetch full message data."""
        # Build session filter query
        ids_to_fetch = []
        filtered_scores: dict[int, float] = {}

        for msg_id, sim in faiss_results:
            if sim < COSINE_SIMILARITY_THRESHOLD:
                continue
            ids_to_fetch.append(msg_id)
            filtered_scores[msg_id] = sim

        if not ids_to_fetch:
            return []

        time_col = self.time_column
        placeholders = ",".join("?" for _ in ids_to_fetch)
        sql = f"SELECT id, session_id, role, content, metadata, {time_col} FROM messages WHERE id IN ({placeholders})"
        params: list = list(ids_to_fetch)

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if exclude_session_id:
            sql += " AND session_id != ?"
            params.append(exclude_session_id)

        cursor = self._conn.execute(sql, params)
        results = []
        for row in cursor.fetchall():
            msg_id = row[0]
            sim = filtered_scores.get(msg_id, 0)
            if sim < COSINE_SIMILARITY_THRESHOLD:
                continue
            msg = {
                "id": msg_id,
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
                "_similarity": round(sim, 4),
            }
            results.append(msg)

        results.sort(key=lambda m: -m["_similarity"])
        return results[:top_k]

    def _semantic_search_fallback(
        self,
        query_vector: list[float],
        session_id: str = None,
        exclude_session_id: str = None,
        top_k: int = 10,
        engine: Any = None,
    ) -> list[dict[str, Any]]:
        """Full-table Python cosine fallback when FAISS is unavailable."""
        from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415

        where_parts = ["embedding IS NOT NULL AND LENGTH(embedding) >= ?"]
        params: list = [EMBED_DIM * 4]

        if session_id:
            where_parts.append("session_id = ?")
            params.append(session_id)
        if exclude_session_id:
            where_parts.append("session_id != ?")
            params.append(exclude_session_id)

        time_col = self.time_column
        cursor = self._conn.execute(
            f"SELECT id, session_id, role, content, metadata, {time_col}, embedding FROM messages WHERE {' AND '.join(where_parts)}",
            params,
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        scored: list[tuple[int, float, dict[str, Any]]] = []
        for row in rows:
            vector = engine.deserialize(row[6])
            if vector is None:
                continue
            sim = EmbeddingEngine.cosine_similarity(query_vector, vector)
            if sim < COSINE_SIMILARITY_THRESHOLD:
                continue
            msg = {
                "id": row[0],
                "session_id": row[1],
                "role": row[2],
                "content": row[3],
                "metadata": json.loads(row[4]) if row[4] else {},
                "created_at": row[5],
            }
            scored.append((row[0], sim, msg))

        scored.sort(key=lambda x: -x[1])
        results = []
        for _msg_id, sim, msg in scored[:top_k]:
            result = dict(msg)
            result["_similarity"] = round(sim, 4)
            results.append(result)

        return results

    def hybrid_search(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 5,
        exclude_session_id: str = None,
        semantic_weight: float = SEMANTIC_WEIGHT,
        fts5_weight: float = FTS5_WEIGHT,
    ) -> list[dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion of FTS5 BM25 + semantic cosine.

        RRF is the standard approach used by modern search engines:
        score(d) = sum(1 / (k + rank_i)) for each ranking.
        This avoids the normalization issues of weighted scoring with
        different scales (BM25 rank vs cosine similarity).

        Args:
            query: Search query string.
            session_id: Optional session filter.
            top_k: Maximum results to return.
            exclude_session_id: Optional session ID to skip.
            semantic_weight: Weight multiplier for semantic ranking.
            fts5_weight: Weight multiplier for FTS5 ranking.

        Returns:
            List of matching message dicts with '_score' field (RRF hybrid score).
        """
        # Fetch extra results from each ranking for fusion
        fts_results = self.fts_search(query, session_id=session_id, top_k=top_k * 3)
        semantic_results = self.semantic_search(
            query,
            session_id=session_id,
            top_k=top_k * 3,
            exclude_session_id=exclude_session_id,
        )

        # Reciprocal Rank Fusion with weighting
        k = 60  # Standard RRF constant
        scores: dict[int, float] = {}
        msg_cache: dict[int, dict[str, Any]] = {}

        for i, msg in enumerate(fts_results):
            rid = msg["id"]
            rrf_score = fts5_weight / (k + i + 1)
            scores[rid] = scores.get(rid, 0) + rrf_score
            msg_cache[rid] = msg

        for i, msg in enumerate(semantic_results):
            rid = msg["id"]
            rrf_score = semantic_weight / (k + i + 1)
            scores[rid] = scores.get(rid, 0) + rrf_score
            if rid not in msg_cache:
                msg_cache[rid] = msg

        # Recency weighting — boost recent messages after RRF fusion.
        # Recent messages get higher scores so they rank above equally-relevant
        # but temporally distant matches. No decay — old messages keep their
        # base score, recent ones gain a multiplier.
        now = _time.time()
        for msg_id, base_score in scores.items():
            msg = msg_cache.get(msg_id)
            if msg is None:
                continue
            created_at = msg.get("created_at")
            if created_at is None:
                continue
            age_days = (now - created_at) / 86400.0
            if age_days < RECENT_BOOST_DAYS:
                scores[msg_id] = base_score * RECENT_BOOST_FACTOR
            elif age_days < MODERATE_BOOST_DAYS:
                scores[msg_id] = base_score * MODERATE_BOOST_FACTOR

        sorted_msgs = sorted(scores.items(), key=lambda x: -x[1])[:top_k]

        if not sorted_msgs:
            return []

        # Fetch any missing messages from DB
        msg_ids = [mid for mid, _ in sorted_msgs]
        cached_ids = set(msg_cache.keys())
        missing_ids = [rid for rid in msg_ids if rid not in cached_ids]

        db_lookup: dict[int, dict[str, Any]] = {}
        if missing_ids:
            placeholders = ",".join("?" for _ in missing_ids)
            cursor = self._conn.execute(
                f"SELECT id, session_id, role, content, metadata, {self.time_column} FROM messages WHERE id IN ({placeholders})",
                missing_ids,
            )
            for row in cursor.fetchall():
                db_lookup[row[0]] = {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5],
                }

        results = []
        for msg_id, score in sorted_msgs:
            if msg_id in msg_cache:
                result = {k2: v2 for k2, v2 in msg_cache[msg_id].items() if not k2.startswith("_")}
            elif msg_id in db_lookup:
                result = dict(db_lookup[msg_id])
            else:
                continue
            result["_score"] = round(score, 4)
            results.append(result)

        return results

    def recall_past_discussions(
        self,
        query: str,
        exclude_session_id: str,
        top_k: int = 3,
        min_score: float = RECALL_MIN_SCORE,
        max_chars: int = RECALL_OUTPUT_MAX_CHARS,
    ) -> str:
        """Recall relevant past discussions from sessions OTHER than the current one.

        Searches all historical sessions (excluding the given session_id) for
        messages related to the query using hybrid semantic + keyword search.
        Returns a compact pointer string suitable for injection into the user
        message — not full content, just enough to remind the model that prior
        context exists and where to find it.

        Args:
            query: Search query text (typically the current user message).
            exclude_session_id: Session ID to exclude (the active conversation).
            top_k: Maximum number of results to consider before formatting.
            min_score: Minimum hybrid score threshold to include a result.
            max_chars: Hard cap on total output characters.

        Returns:
            Formatted pointer string, or empty string if nothing relevant found.
        """
        if not self._initialized or not self._conn:
            return ""

        try:
            hybrid_results = self.hybrid_search(
                query,
                session_id=None,
                top_k=top_k * RECALL_TOP_K_MULTIPLIER,
                exclude_session_id=exclude_session_id,
            )
            if not hybrid_results:
                return ""

            qualified = [(msg["id"], msg["_score"]) for msg in hybrid_results[:top_k] if msg.get("_score", 0) >= min_score]
            if not qualified:
                return ""

            msg_lookup = {msg["id"]: msg for msg in hybrid_results}
            messages = {mid: msg_lookup[mid] for mid, _ in qualified if mid in msg_lookup}
            if not messages:
                return ""

            return self._format_recall_pointers(messages, qualified, max_chars=max_chars)

        except (sqlite3.Error, KeyError, TypeError) as e:
            logger.exception("recall_past_discussions failed: %s", e)
            return ""

    @staticmethod
    def _format_recall_pointers(
        messages: dict[int, dict[str, Any]],
        qualified: list[tuple[int, float]],
        max_chars: int = RECALL_OUTPUT_MAX_CHARS,
    ) -> str:
        """Format scored messages into compact recall pointer strings.

        Each pointer includes role, date, session ID prefix, score, and a short
        content snippet. Output is hard-capped to prevent context bloat.

        Returns:
            Formatted string with header line and one pointer per result.
        """
        parts = []
        for msg_id, score in qualified:
            m = messages.get(msg_id)
            if not m:
                continue

            role = m["role"].upper()
            session = m["session_id"][:15]
            ts = str(m.get("created_at", ""))[:10] if m.get("created_at") else "?"

            snippet = _SearchEngine._extract_snippet(m.get("content", "") or "", max_len=RECALL_SNIPPET_MAX_LEN)

            parts.append(f"[{role} {ts} session:{session} score:{score:.2f}] {snippet}")

        if not parts:
            return ""

        header = "[Relevant past discussion — use perpetual_search for full context]\n"
        result = header + "\n".join(parts)

        if len(result) > max_chars:
            result = result[: max_chars - 3] + "..."

        return result

    @staticmethod
    def _extract_snippet(text: str, max_len: int = RECALL_SNIPPET_MAX_LEN) -> str:
        """Extract a short meaningful snippet from message content.

        Takes the first sentence or phrase up to max_len characters.
        Strips tool call noise and truncates cleanly at word boundaries.
        """
        if not text:
            return ""

        for prefix in ("tool_call:", "function:", "[", "{"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break

        snippet = text.strip()
        for delim in (". ", "\n", "! ", "? "):
            idx = text.find(delim)
            if 0 < idx <= max_len:
                snippet = text[: idx + 1].strip()
                break

        if len(snippet) > max_len:
            snippet = snippet[:max_len]
            last_space = snippet.rfind(" ")
            if last_space > max_len * 0.5:
                snippet = snippet[:last_space]

        return (snippet + "...") if len(snippet) >= max_len else snippet

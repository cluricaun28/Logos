"""Search operations for Perpetual Context.

Handles FTS5 keyword search, semantic vector search, hybrid fusion,
and cross-session recall.
"""
from __future__ import annotations

import json
import logging
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


class _SearchEngine:
    """Mixin: FTS5, semantic, and hybrid search with recall formatting.

    Expects the host class to provide:
      - self._conn, self._lock, self._initialized, self.time_column
    """

    def fts_search(self, query: str, session_id: str = None, top_k: int = 10) -> list[dict[str, Any]]:
        """Full-text search using SQLite FTS5.

        Handles both old schema (timestamp, no metadata in FTS) and new schema.

        Args:
            query: Search query string
            session_id: Optional session filter
            top_k: Maximum results to return

        Returns:
            List of matching message dicts with _rank field
        """
        if not self._initialized or not self._conn:
            return []

        try:
            time_col = self.time_column
            
            # FTS5 uses special syntax for matching — need to escape operators
            search_query = query.strip()
            
            # Always use LIKE-based search to avoid FTS5 expression injection.
            # FTS5 MATCH cannot be safely parameterized in SQLite (the query
            # expression is part of the SQL, not a bound parameter), so we
            # use a parameterized LIKE search with a simple relevance score.
            like_pattern = f"%{search_query}%"

            # Tokenize the query for relevance scoring (simple word count)
            query_words = set(search_query.lower().split())

            # Escape single quotes in each word to prevent SQL injection
            escaped_words = [w.replace("'", "''") for w in list(query_words)[:5]]

            base_sql = f"""
                SELECT m.id, m.session_id, m.role, m.content, 
                       COALESCE(m.metadata, '{{{{}}}}') as metadata,
                       m.{time_col},
                       -- Simple relevance: count of query words found in content (higher = more relevant).
                       -- Negated so ORDER BY rank works the same way as FTS5 (lower = better).
                       -({'+ '.join([f"CASE WHEN LOWER(m.content) LIKE '%{w}%' THEN 1 ELSE 0 END" for w in escaped_words])}) as rank
                FROM messages m
                WHERE m.content LIKE ?
            """
            params: list = [like_pattern]

            # FTS5 join uses m.created_at — time_column property handles fallback at init

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
                    "_rank": float(row[6]),  # FTS5 BM25 rank (lower = better)
                }
                results.append(msg)

            return results

        except Exception as e:
            logger.error("FTS search failed: %s", e)
            return []

    def semantic_search(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 10,
        exclude_session_id: str = None,
    ) -> list[dict[str, Any]]:
        """Semantic search using cosine similarity against stored embeddings.

        Embeds the query vector, loads all non-NULL message embeddings from DB,
        computes cosine similarity, and returns top-k results ranked by similarity score.

        Graceful degradation: if embedding model is unavailable or no messages have
        embeddings yet, returns empty list without error.

        Args:
            query: Search query text to embed.
            session_id: Optional session filter.
            top_k: Maximum results to return.
            exclude_session_id: Optional session ID to skip (for cross-session recall).

        Returns:
            List of message dicts with '_similarity' score field (0-1, higher = more similar).
        """
        if not self._initialized or not self._conn:
            return []

        try:
            # Embed the query — returns None if model unavailable
            from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415
            engine = EmbeddingEngine.get()
            query_vector = engine.embed(query)
            if query_vector is None:
                logger.debug("Semantic search skipped — embedding model unavailable")
                return []

            # Build WHERE clause for session filtering
            where_parts = ["embedding IS NOT NULL AND LENGTH(embedding) >= ?"]
            params: list = [EMBED_DIM * 4]  # Minimum valid blob size

            if session_id:
                where_parts.append("session_id = ?")
                params.append(session_id)
            if exclude_session_id:
                where_parts.append("session_id != ?")
                params.append(exclude_session_id)

            time_col = self.time_column
            cursor = self._conn.execute(
                f"SELECT id, session_id, role, content, metadata, {time_col}, embedding "
                f"FROM messages WHERE {' AND '.join(where_parts)}",
                params,
            )
            rows = cursor.fetchall()

            if not rows:
                logger.debug("Semantic search returned no embedded messages")
                return []

            # Compute cosine similarity for each message with stored embedding
            scored: list[tuple[int, float, dict[str, Any]]] = []
            for row in rows:
                blob = row[6]
                vector = engine.deserialize(blob)
                if vector is None:
                    continue  # Skip corrupted or empty embeddings

                sim = EmbeddingEngine.cosine_similarity(query_vector, vector)
                if sim < COSINE_SIMILARITY_THRESHOLD:
                    continue  # Filter out low-similarity results

                msg = {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {},
                    "created_at": row[5],
                }
                scored.append((row[0], sim, msg))

            # Sort by similarity descending, take top_k
            scored.sort(key=lambda x: -x[1])
            results = []
            for msg_id, sim, msg in scored[:top_k]:
                result = dict(msg)
                result["_similarity"] = round(sim, 4)
                results.append(result)

            return results

        except Exception as e:
            logger.error("Semantic search failed: %s", e)
            return []

    def hybrid_search(
        self,
        query: str,
        session_id: str = None,
        top_k: int = 5,
        exclude_session_id: str = None,
        semantic_weight: float = SEMANTIC_WEIGHT,
        fts5_weight: float = FTS5_WEIGHT,
    ) -> list[dict[str, Any]]:
        """Hybrid search combining FTS5 keyword matching + semantic embeddings.

        Uses weighted scoring to combine BM25 rank from FTS5 with cosine similarity
        from embedding vectors. Messages appearing in both result sets get boosted scores.

        Args:
            query: Search query string.
            session_id: Optional session filter.
            top_k: Maximum results to return.
            exclude_session_id: Optional session ID to skip (for cross-session recall).
            semantic_weight: Weight for cosine similarity (default: SEMANTIC_WEIGHT).
            fts5_weight: Weight for BM25 rank (default: FTS5_WEIGHT).

        Returns:
            List of matching message dicts with '_score' field (weighted hybrid score).
        """
        # Get FTS5 keyword results — fetch extra for fusion
        fts_results = self.fts_search(
            query, session_id=session_id, top_k=top_k * 2
        )

        # Get semantic embedding results — fetch extra for fusion
        semantic_results = self.semantic_search(
            query,
            session_id=session_id,
            top_k=top_k * 2,
            exclude_session_id=exclude_session_id,
        )

        # Combine using weighted scoring: FTS5 rank + cosine similarity.
        # Normalize both scores to [0, 1] range before weighting.
        scored: dict[int, float] = {}
        msg_cache: dict[int, dict[str, Any]] = {}

        # Score FTS5 results — convert BM25 rank (lower=better) to normalized score
        if fts_results:
            min_rank = min(m["_rank"] for m in fts_results)
            max_rank = max(m["_rank"] for m in fts_results)
            rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

            for i, msg in enumerate(fts_results):
                rid = msg["id"]
                # Normalize BM25: invert so higher=better, scale to [0, 1]
                normalized = 1.0 - ((msg["_rank"] - min_rank) / rank_range)
                score = FTS5_WEIGHT * normalized + (1.0 / (i + 1)) * FTS5_WEIGHT * 0.1
                scored[rid] = scored.get(rid, 0) + score
                msg_cache[rid] = msg

        # Score semantic results — cosine similarity already in [0, 1] range
        for i, msg in enumerate(semantic_results):
            rid = msg["id"]
            sim_score = msg["_similarity"]
            score = SEMANTIC_WEIGHT * sim_score + (1.0 / (i + 1)) * SEMANTIC_WEIGHT * 0.1
            scored[rid] = scored.get(rid, 0) + score
            # Prefer semantic result dict if FTS didn't have it (has similarity field)
            if rid not in msg_cache:
                msg_cache[rid] = msg

        # Sort by combined score descending, return top_k
        sorted_msgs = sorted(scored.items(), key=lambda x: -x[1])[:top_k]

        if not sorted_msgs:
            return []

        # Build final results from cache (already have content) or fetch from DB
        msg_ids = [mid for mid, _ in sorted_msgs]
        cached_ids = {rid for rid in msg_cache}
        missing_ids = [rid for rid in msg_ids if rid not in cached_ids]

        db_lookup: dict[int, dict[str, Any]] = {}
        if missing_ids:
            placeholders = ",".join("?" for _ in missing_ids)
            cursor = self._conn.execute(
                f"SELECT id, session_id, role, content, metadata, {self.time_column} "
                f"FROM messages WHERE id IN ({placeholders})",
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
                result = {k: v for k, v in msg_cache[msg_id].items() if not k.startswith("_")}
            elif msg_id in db_lookup:
                result = dict(db_lookup[msg_id])
            else:
                continue  # Shouldn't happen, but skip safely

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
            # Step 1: Hybrid search across ALL sessions (semantic + keyword fusion)
            hybrid_results = self.hybrid_search(
                query, session_id=None, top_k=top_k * RECALL_TOP_K_MULTIPLIER,
                exclude_session_id=exclude_session_id,
            )
            if not hybrid_results:
                return ""

            # Step 2: Filter by score threshold and take top_k.
            # hybrid_search already returns dicts with '_score' (weighted hybrid score).
            qualified = [
                (msg["id"], msg["_score"])
                for msg in hybrid_results[:top_k]
                if msg.get("_score", 0) >= min_score
            ]
            if not qualified:
                return ""

            # Step 3: Build message dict from hybrid_results (already have full content)
            msg_lookup = {msg["id"]: msg for msg in hybrid_results}
            messages = {mid: msg_lookup[mid] for mid, _ in qualified if mid in msg_lookup}
            if not messages:
                return ""

            # Step 4: Format as compact pointers with hard cap
            return self._format_recall_pointers(
                messages, qualified, max_chars=max_chars
            )

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
            session = m["session_id"][:15]  # Truncate long IDs
            ts = str(m.get("created_at", ""))[:10] if m.get("created_at") else "?"

            snippet = PerpetualContextDB._extract_snippet(
                m.get("content", "") or "", max_len=RECALL_SNIPPET_MAX_LEN
            )

            parts.append(
                f"[{role} {ts} session:{session} score:{score:.2f}] "
                f"{snippet}"
            )

        if not parts:
            return ""

        header = "[Relevant past discussion — use perpetual_search for full context]\n"
        result = header + "\n".join(parts)

        # Hard cap on total characters to prevent context bloat
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."

        return result

    @staticmethod
    def _extract_snippet(text: str, max_len: int = RECALL_SNIPPET_MAX_LEN) -> str:
        """Extract a short meaningful snippet from message content.

        Takes the first sentence or phrase up to max_len characters.
        Strips tool call noise and truncates cleanly at word boundaries.
        """
        if not text:
            return ""

        # Strip common prefixes that add noise
        for prefix in ("tool_call:", "function:", "[", "{"):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        # Try to end at a sentence boundary
        snippet = text.strip()
        for delim in (". ", "\n", "! ", "? "):
            idx = text.find(delim)
            if 0 < idx <= max_len:
                snippet = text[:idx + 1].strip()
                break

        # Truncate at word boundary if too long
        if len(snippet) > max_len:
            snippet = snippet[:max_len]
            last_space = snippet.rfind(" ")
            if last_space > max_len * 0.5:
                snippet = snippet[:last_space]

        return (snippet + "...") if len(snippet) >= max_len else snippet

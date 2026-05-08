"""Message storage and retrieval for Perpetual Context.

Handles CRUD operations for conversation messages, embedding generation,
and backfill operations.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Local copies of constants from perpetual_context_db
EMBED_DIM = 384
EMBED_MIN_CONTENT_LEN = 10
EMBED_MAX_CONTENT_LEN = 5000
ACTIVE_WINDOW_SIZE = 5


class _MessageManager:
    """Mixin: message CRUD, embedding storage, and retrieval.

    Expects the host class to provide:
      - self._conn, self._lock, self._initialized, self.time_column
      - self._store_embedding, self._conn, etc.
    """

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] = None,
        timestamp: float = None,
    ) -> int | None:
        """Store a message in the database.

        Handles both old schema (timestamp required) and new schema
        (created_at with default).

        Args:
            session_id: The conversation session ID
            role: Message role (user, assistant, system, tool)
            content: Message text content
            metadata: Optional JSON-serializable metadata dict
            timestamp: Optional explicit timestamp (float). If None, uses current time.

        Returns:
            The message ID if stored successfully, None otherwise
        """
        if not self._initialized or not self._conn:
            return None

        try:
            meta_json = json.dumps(metadata or {})
            
            # Use provided timestamp or generate one with millisecond precision
            ts = timestamp if timestamp is not None else time.time()

            with self._lock:
                # Check which columns exist — inside lock to prevent race condition
                cursor = self._conn.execute("PRAGMA table_info(messages)")
                columns = {row[1] for row in cursor.fetchall()}

                if 'timestamp' in columns and 'created_at' not in columns:
                    # Old schema only (no created_at): use timestamp column
                    self._conn.execute(
                        """INSERT INTO messages (session_id, role, content, metadata, timestamp) 
                           VALUES (?, ?, ?, ?, ?)""",
                        (session_id, role, content, meta_json, ts),
                    )
                else:
                    # New schema or mixed: use created_at column with explicit timestamp
                    # Also set timestamp column if it exists (NOT NULL, no default)
                    if 'timestamp' in columns:
                        self._conn.execute(
                            """INSERT INTO messages (session_id, role, content, metadata, timestamp, created_at) 
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (session_id, role, content, meta_json, ts, ts),
                        )
                    else:
                        self._conn.execute(
                            """INSERT INTO messages (session_id, role, content, metadata, created_at) 
                               VALUES (?, ?, ?, ?, ?)""",
                            (session_id, role, content, meta_json, ts),
                        )

                message_id = self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]

                # Update session metadata
                self._conn.execute(
                    """INSERT INTO session_metadata (session_id, message_count, last_updated)
                       VALUES (?, 1, ?)
                       ON CONFLICT(session_id) DO UPDATE SET 
                           message_count = message_count + 1,
                           last_updated = ?""",
                    (session_id, time.time(), time.time()),
                )

                self._conn.commit()

            # --- Embedding: generate and store vector OUTSIDE the lock ---
            # Model loading is expensive (~80MB download on first use), so we do this
            # after releasing the DB lock. Graceful degradation: if embedding fails,
            # the message is still stored — embedding is an enhancement only.
            self._store_embedding(message_id, role, content)

            return message_id

        except Exception as e:
            logger.error("Failed to add message: %s", e)
            return None

    def _store_embedding(self, message_id: int, role: str, content: str) -> None:
        """Generate and persist embedding for a newly inserted message.

        Runs outside the DB lock since model inference is slow. Skips system/tool
        messages and very short content to avoid wasting compute on noise.

        Args:
            message_id: The ID of the just-inserted message.
            role: Message role (user, assistant, system, tool).
            content: Raw message text.
        """
        # Skip roles that don't carry meaningful semantic content
        if role in ("system", "tool"):
            return
        # Skip very short or empty content — not worth embedding
        if not content or len(content) < EMBED_MIN_CONTENT_LEN:
            return

        from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415
        engine = EmbeddingEngine.get()
        vector = engine.embed(content)
        if vector is None:
            return  # Model unavailable or embed failed — degrade gracefully

        blob = engine.serialize(vector)
        if not blob:
            return

        try:
            with self._lock:
                self._conn.execute(
                    "UPDATE messages SET embedding = ? WHERE id = ?", (blob, message_id)
                )
                self._conn.commit()
        except Exception as e:
            logger.debug("Failed to store embedding for message %d: %s", message_id, e)

    # -----------------------------------------------------------------------
    # Backfill — embed historical messages that missed it on insert
    # -----------------------------------------------------------------------

    def backfill_embeddings(
        self,
        batch_size: int = 100,
        max_messages: int = None,
    ) -> dict[str, int]:
        """Generate embeddings for all historical messages missing them.

        Processes messages in batches to avoid holding the DB lock for too long.
        Skips system/tool roles and very short content (same filters as _store_embedding).

        Args:
            batch_size: Number of messages to process per batch.
            max_messages: Optional cap on total messages to backfill. Use None for all.

        Returns:
            Dict with keys 'total', 'embedded', 'skipped', 'failed' counts.
        """
        if not self._initialized or not self._conn:
            logger.warning("Cannot backfill — database not initialized")
            return {"total": 0, "embedded": 0, "skipped": 0, "failed": 0}

        stats = {"total": 0, "embedded": 0, "skipped": 0, "failed": 0}

        # Build WHERE clause — only messages without embeddings that have meaningful content
        where_parts = [
            "embedding IS NULL OR LENGTH(embedding) < ?",
            "role NOT IN ('system', 'tool')",
            "LENGTH(content) >= ?",
        ]
        params: list = [EMBED_DIM * 4, EMBED_MIN_CONTENT_LEN]

        # Fetch IDs in batches
        limit_clause = f"LIMIT {max_messages}" if max_messages else ""
        cursor = self._conn.execute(
            f"SELECT id, role, content FROM messages "
            f"WHERE {' AND '.join(where_parts)} ORDER BY id ASC {limit_clause}",
            params,
        )
        rows = cursor.fetchall()
        stats["total"] = len(rows)

        if not rows:
            logger.info("No messages need backfilling")
            return stats

        logger.info(
            "Starting embedding backfill for %d messages (batch_size=%d)...",
            stats["total"], batch_size,
        )

        from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415
        engine = EmbeddingEngine.get()

        # Process in batches — embed outside lock, write inside lock
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start: batch_start + batch_size]

            for row in batch:
                msg_id, role, content = row
                try:
                    # Truncate long content before embedding
                    truncated = content[:EMBED_MAX_CONTENT_LEN]
                    vector = engine.embed(truncated)
                    if vector is None:
                        stats["skipped"] += 1
                        continue

                    blob = engine.serialize(vector)
                    if not blob:
                        stats["skipped"] += 1
                        continue

                    with self._lock:
                        self._conn.execute(
                            "UPDATE messages SET embedding = ? WHERE id = ?",
                            (blob, msg_id),
                        )
                        self._conn.commit()
                    stats["embedded"] += 1

                except Exception as e:
                    logger.debug("Backfill failed for message %d: %s", msg_id, e)
                    stats["failed"] += 1

            # Progress log every batch
            processed = min(batch_start + batch_size, len(rows))
            logger.info(
                "Backfill progress: %d/%d messages (%d embedded, %d skipped, %d failed)",
                processed, stats["total"], stats["embedded"], stats["skipped"], stats["failed"],
            )

        logger.info(
            "Embedding backfill complete: %d/%d messages embedded",
            stats["embedded"], stats["total"],
        )
        return stats

    def search_messages_by_pattern(
        self,
        pattern: str,
        session_id: str = None,
        role: str = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Search messages using SQL LIKE-style pattern matching on content.

        This is for exact string/pattern searches (e.g., 'ghp_%' for tokens),
        not semantic search. Use this when you know what you're looking for
        and need the full raw content.

        Args:
            pattern: SQL LIKE pattern (use % as wildcard, _ as single char)
            session_id: Optional session filter
            role: Optional role filter (user, assistant, system, tool)
            limit: Maximum results to return

        Returns:
            List of message dicts with full content
        """
        if not self._initialized or not self._conn:
            return []

        try:
            time_col = self.time_column
            query = f"SELECT id, session_id, role, content, {time_col} FROM messages WHERE 1=1"
            params = []

            # Apply LIKE pattern to content
            query += " AND content LIKE ? ESCAPE '\\'"
            params.append(pattern)

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            if role:
                query += " AND role = ?"
                params.append(role)

            query += f" ORDER BY {time_col} DESC LIMIT ?"
            params.append(limit)

            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],  # Full content, no truncation
                    "created_at": row[4],  # Always use 'created_at' key for API consistency
                })

            return results

        except Exception as e:
            logger.error("Pattern search failed: %s", e)
            return []

    def get_recent_messages(
        self,
        n: int = 10,
        session_id: str = None,
        role: str = None,
    ) -> list[dict[str, Any]]:
        """Get the N most recent messages from the database.

        Returns raw message content in chronological order (oldest first).
        No summarization — just direct DB retrieval.

        Args:
            n: Number of recent messages to retrieve
            session_id: Optional session filter (None for all sessions)
            role: Optional role filter

        Returns:
            List of message dicts ordered oldest-first
        """
        if not self._initialized or not self._conn:
            return []

        try:
            time_col = self.time_column
            query = f"SELECT id, session_id, role, content, {time_col} FROM messages WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            if role:
                query += " AND role = ?"
                params.append(role)

            # Get most recent first, then reverse for chronological order
            query += f" ORDER BY {time_col} DESC LIMIT ?"
            params.append(n)

            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],  # Full content, no truncation
                    "created_at": row[4],  # Always use 'created_at' key for API consistency
                })

            # Reverse to get chronological order (oldest first)
            results.reverse()
            return results

        except Exception as e:
            logger.error("Recent messages query failed: %s", e)
            return []

    def get_message_session(self, message_id: int) -> str | None:
        """Return the session_id for a given message ID.

        Args:
            message_id: The primary key of the message.

        Returns:
            The session_id string, or None if the message doesn't exist.
        """
        if not self._initialized or not self._conn:
            return None
        try:
            with self._lock:
                row = self._conn.execute(
                    "SELECT session_id FROM messages WHERE id = ?", (message_id,)
                ).fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.debug("get_message_session(%d) failed: %s", message_id, e)
            return None

    def get_messages(
        self,
        session_id: str = None,
        role: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve messages with optional filtering.

        Handles both old schema (timestamp, no metadata) and new schema
        (created_at, metadata column).

        Args:
            session_id: Filter by session (None for all sessions)
            role: Filter by role (user, assistant, system, tool)
            limit: Maximum number of results
            offset: Pagination offset

        Returns:
            List of message dicts
        """
        if not self._initialized or not self._conn:
            return []

        try:
            # Detect available columns for backward compatibility
            cursor = self._conn.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}

            query = "SELECT id, session_id, role, content"

            if 'metadata' in columns:
                query += ", metadata"
            else:
                query += ", '' as metadata"  # Default empty JSON

            if 'created_at' in columns:
                query += ", created_at"
            elif 'timestamp' in columns:
                query += ", timestamp as created_at"
            else:
                query += ", 0 as created_at"

            query += " FROM messages WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            if role:
                query += " AND role = ?"
                params.append(role)

            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = self._conn.execute(query, params)
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
                }
                results.append(msg)

            return results

        except Exception as e:
            logger.error("Failed to get messages: %s", e)
            return []

    def get_active_window(self, session_id: str, n: int = None) -> list[dict[str, Any]]:
        """Get the most recent N messages for a session (short-term memory).

        Args:
            session_id: The conversation session ID
            n: Number of recent messages (defaults to ACTIVE_WINDOW_SIZE)

        Returns:
            List of recent message dicts
        """
        if n is None:
            n = ACTIVE_WINDOW_SIZE
        return self.get_messages(session_id=session_id, limit=n)

"""Metadata management for Perpetual Context.

Handles topics, relationships, sessions, knowledge gaps, queries,
and database statistics.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# Local copy of constant from perpetual_context_db
CONTEXT_DRIFT_THRESHOLD = 0.3


class _MetadataManager:
    """Mixin: topics, relationships, sessions, knowledge gaps, query, stats.

    Expects the host class to provide:
      - self._conn, self._lock, self._initialized, self.time_column
    """

    def add_topic(self, session_id: str, topic_name: str, confidence: float = 0.5) -> int | None:
        """Register a new topic for a session.

        Args:
            session_id: The conversation session ID
            topic_name: Name of the topic
            confidence: Confidence score (0.0-1.0)

        Returns:
            Topic ID if created, None if duplicate or error
        """
        if not self._initialized or not self._conn:
            return None

        try:
            with self._lock:
                # Check for existing topic
                cursor = self._conn.execute(
                    "SELECT id FROM topics WHERE session_id = ? AND topic_name = ?",
                    (session_id, topic_name),
                )
                if cursor.fetchone():
                    return None  # Duplicate

                cursor = self._conn.execute(
                    """INSERT INTO topics (session_id, topic_name, confidence) 
                       VALUES (?, ?, ?)""",
                    (session_id, topic_name, confidence),
                )
                topic_id = cursor.lastrowid

                # Update session metadata
                self._conn.execute(
                    """INSERT INTO session_metadata (session_id, topic_count, last_updated)
                       VALUES (?, 1, ?)
                       ON CONFLICT(session_id) DO UPDATE SET 
                           topic_count = topic_count + 1,
                           last_updated = ?""",
                    (session_id, time.time(), time.time()),
                )

                self._conn.commit()
                return topic_id

        except Exception as e:
            logger.error("Failed to add topic: %s", e)
            return None

    def link_topic_to_message(self, topic_id: int, message_id: int, similarity: float = 0.5) -> bool:
        """Link a topic to a message.

        Args:
            topic_id: The topic ID
            message_id: The message ID
            similarity: Similarity score between topic and message

        Returns:
            True if linked successfully
        """
        if not self._initialized or not self._conn:
            return False

        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO topic_messages (topic_id, message_id, similarity) VALUES (?, ?, ?)",
                    (topic_id, message_id, similarity),
                )
                self._conn.commit()
                return True

        except Exception as e:
            logger.error("Failed to link topic to message: %s", e)
            return False

    def get_topic_flow(self, session_id: str) -> list[dict[str, Any]]:
        """Get the topic flow history for a session.

        Args:
            session_id: The conversation session ID

        Returns:
            List of topic dicts with message counts and timestamps
        """
        if not self._initialized or not self._conn:
            return []

        try:
            cursor = self._conn.execute("""
                SELECT t.id, t.topic_name, t.confidence, t.created_at, t.updated_at,
                       COUNT(tm.message_id) as message_count
                FROM topics t
                LEFT JOIN topic_messages tm ON t.id = tm.topic_id
                WHERE t.session_id = ?
                GROUP BY t.id
                ORDER BY t.created_at ASC
            """, (session_id,))

            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "topic_name": row[1],
                    "confidence": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "message_count": row[5] or 0,
                })

            return results

        except Exception as e:
            logger.error("Failed to get topic flow: %s", e)
            return []

    def detect_topic_drift(self, session_id: str, threshold: float = None) -> bool:
        """Detect if the conversation has drifted from its main topics.

        Compares recent messages against established topics and returns True
        if drift is detected (similarity below threshold).

        Args:
            session_id: The conversation session ID
            threshold: Drift detection threshold (defaults to CONTEXT_DRIFT_THRESHOLD)

        Returns:
            True if drift detected, False otherwise
        """
        if threshold is None:
            threshold = CONTEXT_DRIFT_THRESHOLD

        topics = self.get_topic_flow(session_id)
        if not topics:
            return False  # No topics to compare against

        recent_messages = self.get_active_window(session_id, n=3)
        if not recent_messages:
            return False

        # Drift detection: compare recent messages against established topics
        total_match = 0
        for msg in recent_messages:
            content_lower = msg["content"].lower()
            topic_names = [t["topic_name"].lower() for t in topics]
            
            matched = False
            for topic_name in topic_names:
                # Filter out common English stopwords to reduce false positives
                stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                              'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                              'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                              'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                              'and', 'or', 'but', 'not', 'no', 'if', 'then', 'else'}
                topic_words = set(topic_name.split()) - stop_words
                content_words = set(content_lower.split()) - stop_words
                
                if not topic_words:  # Skip topics that are all stopwords
                    continue
                    
                words = topic_words & content_words
                # Require at least 2 shared non-stopwords OR 50%+ of topic words match
                if len(words) >= 2 or (topic_words and len(words) / len(topic_words) >= 0.5):
                    matched = True
                    break
            
            if matched:
                total_match += 1

        drift_ratio = 1.0 - (total_match / len(recent_messages))
        return drift_ratio > threshold

    # -- Relationships -----------------------------------------------------

    def add_relationship(
        self,
        session_id: str,
        source_entity: str,
        target_entity: str,
        relationship_type: str = "related",
        strength: float = 0.5,
    ) -> int | None:
        """Record a relationship between two entities.

        Args:
            session_id: The conversation session ID
            source_entity: Source entity name
            target_entity: Target entity name
            relationship_type: Type of relationship (e.g., 'related', 'uses', 'causes')
            strength: Relationship strength (0.0-1.0)

        Returns:
            Relationship ID if created, None if duplicate or error
        """
        if not self._initialized or not self._conn:
            return None

        try:
            with self._lock:
                cursor = self._conn.execute(
                    """INSERT INTO relationships (session_id, source_entity, target_entity, 
                       relationship_type, strength) VALUES (?, ?, ?, ?, ?)""",
                    (session_id, source_entity, target_entity, relationship_type, strength),
                )
                rel_id = cursor.lastrowid
                self._conn.commit()
                return rel_id

        except Exception as e:
            logger.error("Failed to add relationship: %s", e)
            return None

    def increment_relationship(
        self,
        session_id: str,
        source_entity: str,
        target_entity: str,
        delta: float = 0.1,
    ) -> int | None:
        """Increment relationship strength or create new one (UPSERT).

        If a relationship already exists for this source/target pair, increment its
        strength by `delta` (capped at 1.0). Otherwise insert a new row with initial
        strength equal to `delta`.

        Args:
            session_id: The conversation session ID
            source_entity: Source entity/topic name
            target_entity: Target entity/topic name
            delta: Strength increment per co-occurrence (default 0.1)

        Returns:
            Relationship ID if created/updated, None on error
        """
        if not self._initialized or not self._conn:
            return None

        # Normalize ordering so A↔B and B↔A are the same edge
        s, t = (source_entity.lower(), target_entity.lower())
        if s > t:
            s, t = t, s

        try:
            with self._lock:
                # Check if relationship already exists (either direction)
                cursor = self._conn.execute(
                    """SELECT id, strength FROM relationships
                       WHERE (source_entity = ? AND target_entity = ?)
                          OR (source_entity = ? AND target_entity = ?)""",
                    (s, t, t, s),
                )
                existing = cursor.fetchone()

                if existing:
                    rel_id, current_strength = existing
                    new_strength = min(1.0, current_strength + delta)
                    self._conn.execute(
                        "UPDATE relationships SET strength = ? WHERE id = ?",
                        (new_strength, rel_id),
                    )
                    self._conn.commit()
                    return rel_id
                else:
                    cursor = self._conn.execute(
                        """INSERT INTO relationships (session_id, source_entity, target_entity,
                           relationship_type, strength) VALUES (?, ?, ?, 'related', ?)""",
                        (session_id, s, t, delta),
                    )
                    rel_id = cursor.lastrowid
                    self._conn.commit()
                    return rel_id

        except Exception as e:
            logger.error("Failed to increment relationship (%s ↔ %s): %s", source_entity, target_entity, e)
            return None

    def get_relationships(
        self, 
        session_id: str = None,
        entity: str = None,
        relationship_type: str = None,
    ) -> list[dict[str, Any]]:
        """Query relationships with optional filters.

        Args:
            session_id: Filter by session
            entity: Filter by either source or target entity
            relationship_type: Filter by relationship type

        Returns:
            List of relationship dicts
        """
        if not self._initialized or not self._conn:
            return []

        try:
            query = "SELECT id, session_id, source_entity, target_entity, relationship_type, strength, created_at FROM relationships WHERE 1=1"
            params = []

            if session_id:
                query += " AND session_id = ?"
                params.append(session_id)
            if entity:
                query += " AND (source_entity = ? OR target_entity = ?)"
                params.extend([entity, entity])
            if relationship_type:
                query += " AND relationship_type = ?"
                params.append(relationship_type)

            query += " ORDER BY strength DESC"

            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "session_id": row[1],
                    "source_entity": row[2],
                    "target_entity": row[3],
                    "relationship_type": row[4],
                    "strength": row[5],
                    "created_at": row[6],
                })

            return results

        except Exception as e:
            logger.error("Failed to get relationships: %s", e)
            return []

    # -- Session management ------------------------------------------------

    def get_session_info(self, session_id: str) -> dict[str, Any | None]:
        """Get metadata about a session.

        Args:
            session_id: The conversation session ID

        Returns:
            Dict with message_count, topic_count, last_updated or None if not found
        """
        if not self._initialized or not self._conn:
            return None

        try:
            cursor = self._conn.execute(
                "SELECT session_id, message_count, topic_count, last_updated FROM session_metadata WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "session_id": row[0],
                    "message_count": row[1],
                    "topic_count": row[2],
                    "last_updated": row[3],
                }
            return None

        except Exception as e:
            logger.error("Failed to get session info: %s", e)
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete all data for a session.

        Args:
            session_id: The conversation session ID to delete

        Returns:
            True if deletion succeeded
        """
        if not self._initialized or not self._conn:
            return False

        try:
            with self._lock:
                # Explicit transaction — if any operation fails, roll back to avoid partial deletions
                self._conn.execute("BEGIN")
                
                # Delete in correct order (respect foreign keys)
                # FTS5 triggers handle automatic cleanup when messages are deleted
                try:
                    self._conn.execute("DELETE FROM topic_messages WHERE topic_id IN (SELECT id FROM topics WHERE session_id = ?)", (session_id,))
                    self._conn.execute("DELETE FROM relationships WHERE session_id = ?", (session_id,))
                    self._conn.execute("DELETE FROM topics WHERE session_id = ?", (session_id,))
                    self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                    self._conn.execute("DELETE FROM session_metadata WHERE session_id = ?", (session_id,))
                    # FTS5 cleanup is handled automatically by the messages_ad trigger
                    
                    self._conn.commit()
                    return True
                except Exception:
                    self._conn.rollback()
                    raise  # Re-raise so outer handler logs it

        except Exception as e:
            logger.error("Failed to delete session: %s", e)
            return False



    # -- Utility -----------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with counts and metadata about the database state
        """
        if not self._initialized or not self._conn:
            return {"error": "Database not initialized"}

        try:
            cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
            message_count = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(DISTINCT session_id) FROM messages")
            session_count = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM topics")
            topic_count = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM relationships")
            relationship_count = cursor.fetchone()[0]

            stats = {
                "message_count": message_count,
                "session_count": session_count,
                "topic_count": topic_count,
                "relationship_count": relationship_count,
            }

            return stats

        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {"error": str(e)}

    def query_messages(
        self,
        pattern: str = None,
        session_id: str = None,
        role: str = None,
        ids: list[int] = None,
        time_start: float = None,
        time_end: float = None,
        min_tokens: int = None,
        max_tokens: int = None,
        metadata_key: str = None,
        metadata_value: Any = None,
        query: str = None,
        stats: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Unified message query with comprehensive filtering.

        This is the master query tool — handles time ranges, token counts,
        direct ID lookup, metadata filters, semantic search, and statistics.

        Args:
            pattern: SQL LIKE pattern for content matching (e.g., 'ghp_%')
            session_id: Filter by session ID
            role: Filter by role (user, assistant, system, tool)
            ids: List of specific message IDs to retrieve
            time_start: Unix timestamp filter (messages >= this time)
            time_end: Unix timestamp filter (messages <= this time)
            min_tokens: Minimum token count filter
            max_tokens: Maximum token count filter
            metadata_key: Filter by metadata key name
            metadata_value: Value to match for the metadata key
            query: Search query text (reserved for future use)
            stats: If True, return statistics instead of messages
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            Dict with 'results' list and/or 'stats' dict depending on mode
        """
        if not self._initialized or not self._conn:
            return {"error": "Database not initialized"}

        try:
            time_col = self.time_column

            # ---- STATISTICS MODE ----
            if stats:
                return self._get_query_stats(
                    session_id=session_id, role=role, pattern=pattern,
                    min_tokens=min_tokens, max_tokens=max_tokens,
                    time_start=time_start, time_end=time_end,
                )

            # ---- BUILD SQL QUERY ----
            query_sql = "SELECT id, session_id, role, content, token_count"
            # Check if metadata column exists (backward compatibility)
            try:
                cursor = self._conn.execute("PRAGMA table_info(messages)")
                columns = {row[1] for row in cursor.fetchall()}
                if 'metadata' in columns:
                    query_sql += ", metadata"
                else:
                    query_sql += ", '' as metadata"
            except Exception as e:
                logger.debug("Metadata column check failed: %s", e)
                query_sql += ", '' as metadata"
            query_sql += f", {time_col}"
            query_sql += " FROM messages WHERE 1=1"

            params = []

            # Apply filters
            if pattern is not None:
                query_sql += " AND content LIKE ? ESCAPE '\\'"
                params.append(pattern)

            if session_id:
                query_sql += " AND session_id = ?"
                params.append(session_id)

            if role:
                query_sql += " AND role = ?"
                params.append(role)

            if ids is not None and len(ids) > 0:
                placeholders = ','.join(['?' for _ in ids])
                query_sql += f" AND id IN ({placeholders})"
                params.extend(ids)

            if time_start is not None:
                query_sql += f" AND {time_col} >= ?"
                params.append(time_start)

            if time_end is not None:
                query_sql += f" AND {time_col} <= ?"
                params.append(time_end)

            # Token count filters (approximate — token_count may be 0 for older messages)
            if min_tokens is not None:
                query_sql += " AND token_count >= ?"
                params.append(min_tokens)

            if max_tokens is not None:
                query_sql += " AND token_count <= ?"
                params.append(max_tokens)

            # Metadata filter — use json_extract for proper JSON key lookup.
            # Pass numeric values as-is (not str()) so SQLite type matching works correctly.
            if metadata_key is not None and metadata_value is not None:
                if 'metadata' in columns:
                    if isinstance(metadata_value, (dict, list)):
                        meta_val = json.dumps(metadata_value)
                    elif isinstance(metadata_value, bool):
                        # JSON booleans must be lowercase for SQLite json_extract comparison
                        meta_val = str(metadata_value).lower()
                    else:
                        # Pass numbers as-is; strings as quoted JSON strings
                        meta_val = metadata_value
                    query_sql += " AND json_extract(metadata, ?) = ?"
                    params.extend([f'$.{metadata_key}', meta_val])

            # Order by time descending, apply pagination
            query_sql += f" ORDER BY {time_col} DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = self._conn.execute(query_sql, params)
            rows = cursor.fetchall()

            results = []
            for row in rows:
                msg = {
                    "id": row[0],
                    "session_id": row[1],
                    "role": row[2],
                    "content": row[3],  # Full content, no truncation
                    "token_count": row[4] or 0,
                    "created_at": row[-1],  # Always use 'created_at' key for API consistency
                }
                if 'metadata' in columns and len(row) > 5:
                    try:
                        msg["metadata"] = json.loads(row[5]) if row[5] else {}
                    except (json.JSONDecodeError, TypeError):
                        msg["metadata"] = {}

                results.append(msg)

            return {
                "results": results,
                "total_found": len(results),
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            logger.error("Query messages failed: %s", e)
            return {"error": str(e)}

    def _get_query_stats(
        self,
        session_id: str = None,
        role: str = None,
        pattern: str = None,
        min_tokens: int = None,
        max_tokens: int = None,
        time_start: float = None,
        time_end: float = None,
    ) -> dict[str, Any]:
        """Return statistics about messages matching the given filters."""
        try:
            time_col = self.time_column
            base_sql = "SELECT COUNT(*) FROM messages WHERE 1=1"
            params = []

            if session_id:
                base_sql += " AND session_id = ?"
                params.append(session_id)
            if role:
                base_sql += " AND role = ?"
                params.append(role)
            if pattern is not None:
                base_sql += " AND content LIKE ? ESCAPE '\\'"
                params.append(pattern)
            if min_tokens is not None:
                base_sql += " AND token_count >= ?"
                params.append(min_tokens)
            if max_tokens is not None:
                base_sql += " AND token_count <= ?"
                params.append(max_tokens)
            if time_start is not None:
                base_sql += f" AND {time_col} >= ?"
                params.append(time_start)
            if time_end is not None:
                base_sql += f" AND {time_col} <= ?"
                params.append(time_end)

            cursor = self._conn.execute(base_sql, params)
            total_count = cursor.fetchone()[0]

            # Breakdown by role
            role_query = base_sql.replace("COUNT(*)", "COUNT(DISTINCT role)")
            cursor = self._conn.execute(role_query, params)
            distinct_roles = cursor.fetchone()[0]

            return {
                "total_messages": total_count,
                "distinct_roles": distinct_roles,
                "filters_applied": {
                    "session_id": session_id,
                    "role": role,
                    "pattern": pattern,
                    "min_tokens": min_tokens,
                    "max_tokens": max_tokens,
                    "time_start": time_start,
                    "time_end": time_end,
                },
            }

        except Exception as e:
            logger.error("Query stats failed: %s", e)
            return {"error": str(e)}


    def optimize(self) -> bool:
        """Run database optimization (VACUUM, FTS5 optimization).

        Returns:
            True if optimization succeeded
        """
        if not self._initialized or not self._conn:
            return False

        try:
            with self._lock:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('optimize')")
                self._conn.commit()
                return True

        except Exception as e:
            logger.error("Optimization failed: %s", e)
            return False

    # -- Knowledge gap storage/retrieval -----------------------------------

    def add_knowledge_gap(
        self,
        topic: str,
        query: str,
        confidence: float = 0.5,
        session_id: str = "",
        needs_reference_library_update: bool | None = None,
        first_principles: list[str] | None = None,
        # Legacy alias — kept for backward compatibility
        needs_wiki_update: bool | None = None,    # deprecated: use needs_reference_library_update
    ) -> int | None:
        """Store a knowledge gap in the database.

        Focus: worldview-aligned reference library entries, not ephemeral search results.
        Gaps with needs_reference_library_update=True will be prioritized for permanent
        reference material building from first principles and truth.

        Args:
            topic: Human-readable description (e.g., "First principles of nuclear ethics")
            query: Search query to resolve the gap
            confidence: 0.0-1.0, lower = more uncertain/higher priority for reference library building
            session_id: Session where gap was detected
            needs_reference_library_update: Should this become a permanent RL entry? (default True)
            first_principles: Foundational truths to anchor the entry on
            needs_wiki_update: Deprecated — use needs_reference_library_update instead

        Returns:
            The gap ID if stored successfully, None otherwise
        """
        # Handle legacy parameter alias
        if needs_wiki_update is not None and needs_reference_library_update is None:
            needs_reference_library_update = needs_wiki_update
        if needs_reference_library_update is None:
            needs_reference_library_update = True
        if not self._initialized or not self._conn:
            return None

        try:
            with self._lock:
                cursor = self._conn.execute(
                    """INSERT INTO knowledge_gaps 
                       (topic, query, confidence, session_id, needs_wiki_update, first_principles)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        topic,
                        query,
                        max(0.0, min(1.0, confidence)),
                        session_id,
                        1 if needs_reference_library_update else 0,
                        json.dumps(first_principles or []),
                    ),
                )
                gap_id = cursor.lastrowid
                self._conn.commit()
                return gap_id

        except Exception as e:
            logger.error("Failed to add knowledge gap: %s", e)
            return None

    def get_unresolved_gaps(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get unresolved knowledge gaps from previous sessions.

        Returns gaps that need worldview-aligned reference library entries built from first
        principles and truth — not ephemeral search results.

        Args:
            limit: Maximum number of gaps to return

        Returns:
            List of gap dicts with topic, query, confidence, needs_wiki_update (legacy), etc.
        """
        if not self._initialized or not self._conn:
            return []

        try:
            cursor = self._conn.execute(
                """SELECT id, topic, query, confidence, session_id, created_at, 
                          needs_wiki_update, first_principles
                   FROM knowledge_gaps
                   WHERE resolved = 0
                   ORDER BY confidence ASC, created_at DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()

            gaps = []
            for row in rows:
                first_principles_raw = row[7] or "[]"
                try:
                    first_principles = json.loads(first_principles_raw) if isinstance(first_principles_raw, str) else first_principles_raw
                except (json.JSONDecodeError, TypeError):
                    first_principles = []

                gaps.append({
                    "id": row[0],
                    "topic": row[1],
                    "query": row[2],
                    "confidence": row[3] or 0.5,
                    "session_id": row[4] or "",
                    "created_at": row[5] or 0,
                    "needs_wiki_update": bool(row[6]) if row[6] is not None else True,
                    "first_principles": first_principles,
                })

            return gaps

        except Exception as e:
            logger.error("Failed to get unresolved gaps: %s", e)
            return []

    def resolve_knowledge_gap(
        self,
        gap_id: int,
        resolution_text: str = "",
        session_id: str = "",
        reference_library_entry_written: bool | None = None,
        # Legacy alias — kept for backward compatibility
        wiki_entry_written: bool | None = None,    # deprecated: use reference_library_entry_written
    ) -> bool:
        """Mark a knowledge gap as resolved and store the resolution.

        When reference_library_entry_written=True, this means a permanent worldview-aligned
        reference library entry has been built from first principles and truth — not just
        an ephemeral search result.

        Args:
            gap_id: The gap ID to resolve
            resolution_text: Resolved knowledge content (reference library entry if applicable)
            session_id: Session where resolution occurred
            reference_library_entry_written: Was a permanent RL entry built? (default False)
            wiki_entry_written: Deprecated — use reference_library_entry_written instead

        Returns:
            True if resolved successfully, False otherwise
        """
        # Handle legacy parameter alias
        if wiki_entry_written is not None and reference_library_entry_written is None:
            reference_library_entry_written = wiki_entry_written
        if reference_library_entry_written is None:
            reference_library_entry_written = False
        if not self._initialized or not self._conn:
            return False

        try:
            with self._lock:
                # Truncate only if genuinely excessive (SQLite TEXT max is ~2GB, but 64KB is a practical limit)
                truncation_warning = ""
                text_to_store = resolution_text or ""
                max_len = 65536  # 64KB — generous but prevents abuse
                if len(text_to_store) > max_len:
                    logger.warning("Resolution text truncated from %d to %d chars (gap_id=%d)", len(text_to_store), max_len, gap_id)
                    text_to_store = text_to_store[:max_len]
                
                self._conn.execute(
                    """UPDATE knowledge_gaps
                       SET resolved = 1, resolution_text = ?, resolution_timestamp = ?
                       WHERE id = ?""",
                    (text_to_store, time.time(), gap_id),
                )
                self._conn.commit()
                return True

        except Exception as e:
            logger.error("Failed to resolve knowledge gap %d: %s", gap_id, e)
            return False

    def get_gap_stats(self) -> dict[str, Any]:
        """Get statistics about knowledge gaps.

        Returns:
            Dict with total_gaps, resolved_gaps, unresolved_gaps counts
        """
        if not self._initialized or not self._conn:
            return {"total": 0, "resolved": 0, "unresolved": 0}

        try:
            cursor = self._conn.execute(
                """SELECT 
                       COUNT(*) as total,
                       COALESCE(SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END), 0) as resolved,
                       COALESCE(SUM(CASE WHEN resolved = 0 THEN 1 ELSE 0 END), 0) as unresolved
                   FROM knowledge_gaps"""
            )
            row = cursor.fetchone()
            return {
                "total": row[0],
                "resolved": row[1],
                "unresolved": row[2],
            }

        except Exception as e:
            logger.error("Failed to get gap stats: %s", e)
            return {"total": 0, "resolved": 0, "unresolved": 0}

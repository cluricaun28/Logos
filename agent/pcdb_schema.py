"""Database schema management for Perpetual Context.

Handles table creation, column migration, and backward compatibility.
Extracted from perpetual_context_db.py for single-responsibility compliance.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


class _SchemaManager:
    """Mixin: table creation, migration, and schema maintenance.

    Expects the host class to provide:
      - self._conn: sqlite3.Connection | None
      - self._lock: threading.RLock
      - self._schema_version: int
    """

    def _create_tables(self) -> None:
        """Create all database tables and indexes.

        Handles backward compatibility with existing schemas by adding
        missing columns via ALTER TABLE if they don't exist.

        Uses schema version caching to skip redundant migrations on subsequent calls.
        """
        conn = self._conn
        if not conn:
            raise RuntimeError("Database not initialized")

        # Read schema version from database header — no table dependency.
        # PRAGMA user_version is a SQLite integer stored in the DB header itself,
        # available even before any tables exist. Zero on new databases.
        cursor = conn.execute("PRAGMA user_version")
        self._schema_version = cursor.fetchone()[0]

        if self._schema_version >= 2:
            logger.debug("Schema version %d is current, skipping migrations", self._schema_version)

        # Messages table — stores all conversation turns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT 0
            )
        """)

        # v3 migration (2026-08-19): pre-v3 tables carry
        # CHECK(role IN ('user','assistant','system')) which silently rejects
        # role='tool' rows (PM tool-result persistence) with an IntegrityError
        # swallowed by the sync wrapper. SQLite cannot ALTER a CHECK
        # constraint, so rebuild the table preserving ids and data. Fresh
        # databases already have the 4-role CHECK and skip this entirely.
        _msg_sql_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if _msg_sql_row and _msg_sql_row[0] and "'tool'" not in _msg_sql_row[0]:
            self._migrate_messages_check_v3()

        # Backward compatibility: add missing columns from old schema (only if schema < v2).
        # NOTE: SQLite does NOT support function calls in ALTER TABLE ADD COLUMN defaults.
        # Use constant defaults here; triggers handle auto-timestamping where needed.
        if self._schema_version < 2:
            self._ensure_column("messages", "timestamp", "REAL DEFAULT 0")
            self._ensure_column("messages", "token_count", "INTEGER DEFAULT 0")
            self._ensure_column("messages", "embedding", "BLOB")

        # FTS5 virtual table for full-text search
        # Check if existing FTS table has the right columns, recreate if needed
        fts_has_metadata = False
        try:
            cursor = conn.execute("PRAGMA table_info(messages_fts)")
            fts_columns = [row[1] for row in cursor.fetchall()]
            fts_has_metadata = "metadata" in fts_columns
        except sqlite3.Error:
            logger.debug("FTS metadata column check failed")

        if not fts_has_metadata:
            # Drop old FTS tables and recreate with correct schema
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.execute("DROP TABLE IF EXISTS messages_fts_config")
            conn.execute("DROP TABLE IF EXISTS messages_fts_data")
            conn.execute("DROP TABLE IF EXISTS messages_fts_docsize")
            conn.execute("DROP TABLE IF EXISTS messages_fts_idx")

        # ALWAYS drop old triggers that reference the old FTS schema (2-column insert)
        for trigger in ("messages_ai", "messages_ad", "messages_au"):
            try:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.Error:
                logger.debug("Drop trigger failed")

        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
            USING fts5(content, metadata, session_id, content_bm25)
        """)

        # Triggers to keep FTS5 in sync with messages (always created fresh after DROP)
        conn.execute("""
            CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, metadata, session_id)
                VALUES (new.id, new.content, COALESCE(new.metadata, '{}'), new.session_id);
            END
        """)
        conn.execute("""
            CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
            END
        """)
        conn.execute("""
            CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.id;
                INSERT INTO messages_fts(rowid, content, metadata, session_id)
                VALUES (new.id, new.content, COALESCE(new.metadata, '{}'), new.session_id);
            END
        """)

        # Reindex existing messages into FTS if the table was just recreated
        # (schema migration path drops and recreates, losing all indexed data)
        if not fts_has_metadata:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO messages_fts(rowid, content, metadata, session_id)
                    SELECT id, content, COALESCE(metadata, '{}'), session_id FROM messages
                """)
                _count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
                logger.info("Reindexed %d existing messages into FTS5 after schema migration", _count)
            except sqlite3.Error as e:
                logger.warning("FTS reindex failed: %s", e)

        # Topics table — topic clusters per session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic_name TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                created_at REAL DEFAULT 0,
                updated_at REAL DEFAULT 0
            )
        """)

        # Backward compatibility: ensure topics has required columns (only if schema < v2).
        if self._schema_version < 2:
            self._ensure_column("topics", "created_at", "REAL DEFAULT 0")
            self._ensure_column("topics", "updated_at", "REAL DEFAULT 0")

        # Topic messages — links between topics and messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS topic_messages (
                topic_id INTEGER REFERENCES topics(id),
                message_id INTEGER REFERENCES messages(id),
                similarity REAL,
                PRIMARY KEY (topic_id, message_id)
            )
        """)

        # Relationships table — entity relationships with strength
        conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                source_entity TEXT NOT NULL,
                target_entity TEXT NOT NULL,
                relationship_type TEXT DEFAULT 'related',
                strength REAL DEFAULT 0.5,
                created_at REAL DEFAULT 0
            )
        """)

        # Backward compatibility: ensure relationships has required columns (only if schema < v2).
        if self._schema_version < 2:
            self._ensure_column("relationships", "relationship_type", "TEXT DEFAULT 'related'")
            self._ensure_column("relationships", "strength", "REAL DEFAULT 0.5")
            self._ensure_column("relationships", "created_at", "REAL DEFAULT 0")

        # Session metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_metadata (
                session_id TEXT PRIMARY KEY,
                message_count INTEGER DEFAULT 0,
                topic_count INTEGER DEFAULT 0,
                last_updated REAL DEFAULT 0
            )
        """)

        # Backward compatibility: ensure session_metadata has required columns (only if schema < v2).
        if self._schema_version < 2:
            self._ensure_column("session_metadata", "topic_count", "INTEGER DEFAULT 0")
            self._ensure_column("session_metadata", "last_updated", "REAL DEFAULT 0")

        # Knowledge gaps table — stores unresolved questions from previous sessions
        # Focus: worldview-aligned reference library entries, not ephemeral search results
        conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                query TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                session_id TEXT NOT NULL,
                resolved INTEGER DEFAULT 0,
                resolution_text TEXT DEFAULT '',
                resolution_timestamp REAL DEFAULT 0,
                created_at REAL DEFAULT 0,
                needs_wiki_update INTEGER DEFAULT 1,    -- legacy column name (see: needs_reference_library_update)
                first_principles TEXT DEFAULT ''
            )
        """)

        # Backward compatibility: ensure knowledge_gaps has required columns (only if schema < v2).
        if self._schema_version < 2:
            self._ensure_column("knowledge_gaps", "resolved", "INTEGER DEFAULT 0")
            self._ensure_column("knowledge_gaps", "resolution_text", "TEXT DEFAULT ''")
            self._ensure_column("knowledge_gaps", "resolution_timestamp", "REAL DEFAULT 0")
            # Legacy column: needs_wiki_update (renamed to needs_reference_library_update in docs)
            self._ensure_column("knowledge_gaps", "needs_wiki_update", "INTEGER DEFAULT 1")
            self._ensure_column("knowledge_gaps", "first_principles", "TEXT DEFAULT ''")

        # Indexes for common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_session ON topics(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_entities ON relationships(source_entity, target_entity)")

        # Index for knowledge gap queries (unresolved gaps ordered by confidence)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_resolved ON knowledge_gaps(resolved, confidence ASC, created_at DESC)")

        # Backward compatibility: add timestamp index if using old 'timestamp' column
        try:
            cursor = conn.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}
            if "timestamp" in columns and "created_at" not in columns:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
        except sqlite3.Error:
            logger.debug("Timestamp index creation failed")

        # Auto-timestamp triggers — SQLite doesn't support function-call defaults in ALTER TABLE,
        # so we use constant defaults (0) above and these triggers to set the actual timestamp.
        # Only fires if the column is still at its default value of 0.
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS topics_ai_timestamp AFTER INSERT ON topics
            WHEN new.created_at = 0 OR new.updated_at = 0
            BEGIN
                UPDATE topics SET created_at = strftime('%s', 'now'), updated_at = strftime('%s', 'now')
                WHERE id = new.id;
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS relationships_ai_timestamp AFTER INSERT ON relationships
            WHEN new.created_at = 0
            BEGIN
                UPDATE relationships SET created_at = strftime('%s', 'now')
                WHERE id = new.id;
            END
        """)

        # Write schema version to database header if we ran migrations
        if self._schema_version < 2:
            conn.execute("PRAGMA user_version = 2")
            self._schema_version = 2
            logger.info("Schema migrated to v2 (user_version)")

        conn.commit()

    def _ensure_column(self, table: str, column: str, col_def: str) -> bool:
        """Add a column to a table if it doesn't already exist.

        Args:
            table: Table name
            column: Column name
            col_def: Full column definition (e.g., "TEXT DEFAULT '{}'")

        Returns:
            True if the column was added, False if it already existed or failed
        """
        try:
            cursor = self._conn.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            if column not in columns:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
                logger.debug("Added missing column %s.%s", table, column)
                return True
            return False
        except sqlite3.Error as e:
            # Column might already exist or ALTER not supported (e.g., FTS5 tables).
            # Log and degrade gracefully — don't abort initialization for non-critical columns.
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                return False  # Expected — column exists
            logger.warning("Failed to add optional column %s.%s: %s", table, column, e)
            return False

    def _migrate_messages_check_v3(self) -> None:
        """Rebuild the messages table so its CHECK constraint admits role='tool'.

        Preserves ids (FTS rowids, child-table message_id refs stay valid).
        SQLite cannot ALTER a CHECK constraint, and with foreign_keys=ON it
        refuses to DROP a parent table while child rows exist — so this uses
        the standard 12-step pattern: FKs off (outside the transaction),
        children copied to temp + dropped, parent renamed/recreated/repopulated,
        children recreated pointing at the new parent, then FKs back on and
        validated. Fresh databases already have the 4-role CHECK and never
        reach this method.
        """
        conn = self._conn
        _fk_was_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        if _fk_was_on:
            # No-op inside a transaction — must happen before BEGIN.
            conn.execute("PRAGMA foreign_keys=OFF")
        _children = (
            ("entity_mentions",
             "CREATE TABLE entity_mentions (\n"
             "        message_id INTEGER,\n"
             "        entity_id INTEGER,\n"
             "        FOREIGN KEY(message_id) REFERENCES messages(id),\n"
             "        FOREIGN KEY(entity_id) REFERENCES entities(id)\n"
             "    )"),
            ("topic_messages",
             "CREATE TABLE topic_messages (\n"
             "                topic_id INTEGER REFERENCES topics(id),\n"
             "                message_id INTEGER REFERENCES messages(id),\n"
             "                similarity REAL,\n"
             "                PRIMARY KEY (topic_id, message_id)\n"
             "            )"),
            ("turn_signals",
             "CREATE TABLE turn_signals (\n"
             "                id INTEGER PRIMARY KEY AUTOINCREMENT,\n"
             "                turn_id INTEGER NOT NULL REFERENCES messages(id),\n"
             "                cluster_id INTEGER NOT NULL REFERENCES signal_clusters(cluster_id),\n"
             "                heat_score REAL NOT NULL DEFAULT 0.5 CHECK(heat_score >= 0.0 AND heat_score <= 1.0),\n"
             "                recency_score REAL NOT NULL DEFAULT 0.5 CHECK(recency_score >= 0.0 AND recency_score <= 1.0),\n"
             "                frequency_count INTEGER NOT NULL DEFAULT 1,\n"
             "                depth_score REAL NOT NULL DEFAULT 0.5 CHECK(depth_score >= 0.0 AND depth_score <= 1.0),\n"
             "                centrality_score REAL NOT NULL DEFAULT 0.5 CHECK(centrality_score >= 0.0 AND centrality_score <= 1.0),\n"
             "                composite_score REAL NOT NULL DEFAULT 0.5,\n"
             "                last_updated REAL NOT NULL,\n"
             "                UNIQUE(turn_id, cluster_id)\n"
             "            )"),
        )
        try:
            conn.execute("BEGIN IMMEDIATE")

            # 1-2. Temp-copy each child that exists, then drop it so its FK
            # reference to messages no longer blocks the parent rebuild.
            _present = []
            for _name, _ddl in _children:
                _exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_name,)
                ).fetchone()
                if not _exists:
                    continue
                conn.execute(f"CREATE TEMP TABLE _mig_{_name} AS SELECT * FROM {_name}")
                conn.execute(f"DROP TABLE {_name}")
                _present.append((_name, _ddl))

            # 3-4. Rename old parent, create new parent with 4-role CHECK.
            conn.execute("ALTER TABLE messages RENAME TO messages_v2_backup")
            conn.execute("""
                CREATE TABLE messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system', 'tool')),
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL DEFAULT 0
                )
            """)

            # 5. Preserve legacy/extended columns from the old table.
            _old_cols = {
                row[1] for row in conn.execute("PRAGMA table_info(messages_v2_backup)").fetchall()
            }
            for _col, _def in (
                ("timestamp", "REAL DEFAULT 0"),
                ("token_count", "INTEGER DEFAULT 0"),
                ("embedding", "BLOB"),
                ("chunk_index", "INTEGER DEFAULT 0"),
                ("topic_tags", "TEXT"),
            ):
                if _col in _old_cols:
                    conn.execute(f"ALTER TABLE messages ADD COLUMN {_col} {_def}")

            # 6. Copy rows, preserving ids.
            _new_cols = [row[1] for row in conn.execute("PRAGMA table_info(messages)").fetchall()]
            _shared = [c for c in _new_cols if c in _old_cols]
            conn.execute(
                f"INSERT INTO messages ({', '.join(_shared)}) "
                f"SELECT {', '.join(_shared)} FROM messages_v2_backup"
            )

            # 7-8. Recreate children against the new parent, restore rows.
            for _name, _ddl in _present:
                conn.execute(_ddl)
                if _name == "turn_signals":
                    conn.execute(
                        "CREATE INDEX idx_turn_signals_turn ON turn_signals(turn_id)"
                    )
                    conn.execute(
                        "CREATE INDEX idx_turn_signals_cluster ON turn_signals(cluster_id)"
                    )
                conn.execute(f"INSERT INTO {_name} SELECT * FROM _mig_{_name}")
                conn.execute(f"DROP TABLE _mig_{_name}")

            # 9. Backup parent is now unreferenced — safe to drop.
            conn.execute("DROP TABLE messages_v2_backup")
            conn.execute("COMMIT")
            _count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            logger.info("messages table rebuilt to v3 CHECK (role 'tool' allowed); %d rows preserved", _count)
        except sqlite3.Error as e:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                logger.debug("v3 migration rollback failed (nothing to roll back?)")
            logger.warning("messages v3 CHECK migration failed (non-fatal): %s", e)
        finally:
            if _fk_was_on:
                conn.execute("PRAGMA foreign_keys=ON")
                _bad = conn.execute("PRAGMA foreign_key_check").fetchall()
                if _bad:
                    logger.warning("v3 migration left %d FK inconsistencies: %s", len(_bad), _bad[:5])

    def _migrate_timestamps_to_created_at(self) -> None:
        """Migrate existing timestamp data to created_at for backward compatibility.

        When both 'timestamp' and 'created_at' columns exist (mixed schema),
        copy non-NULL timestamp values to created_at where it's NULL or 0.
        This ensures time-based queries work correctly regardless of which column
        was used during insertion.
        """
        try:
            cursor = self._conn.execute("PRAGMA table_info(messages)")
            columns = {row[1] for row in cursor.fetchall()}

            if "timestamp" not in columns or "created_at" not in columns:
                return  # No migration needed

            # Count rows that need migration (created_at is NULL or 0)
            needs_migration = self._conn.execute("SELECT COUNT(*) FROM messages WHERE created_at IS NULL OR created_at = 0").fetchone()[0]

            if needs_migration == 0:
                logger.debug("No timestamp migration needed — all rows have valid created_at")
                return

            # Copy timestamp values to created_at where created_at is missing/zero
            cursor = self._conn.execute("""
                UPDATE messages
                SET created_at = timestamp
                WHERE (created_at IS NULL OR created_at = 0) AND timestamp IS NOT NULL
            """)
            migrated = cursor.rowcount
            logger.info("Migrated %d/%d rows: copied timestamp → created_at", migrated, needs_migration)

        except sqlite3.Error as e:
            logger.warning("Timestamp migration failed (non-fatal): %s", e)

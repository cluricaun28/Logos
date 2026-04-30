"""Perpetual Context Database — SQLite + FTS5.

Local-first memory system providing infinite recall across sessions.
Combines structured SQL storage with full-text indexing (FTS5) for retrieval.

Architecture:
  - SQLite: Structured storage for messages, topics, relationships
  - FTS5: Full-text search over message content
  - Topic Flow: Automatic topic clustering with drift detection

Usage:
    from agent.perpetual_context_db import PerpetualContextDB
    
    db = PerpetualContextDB(db_path="~/.hermes/perpetual_context.db")
    db.initialize()
    
    # Store a message
    db.add_message(
        session_id="20260421_023037",
        role="user",
        content="Hello, how do I configure Hermes?",
        metadata={"turn": 1}
    )
    
    # Search via FTS5 + SQL
    results = db.hybrid_search("Hermes configuration", top_k=5)
    
    # Track topic flow
    topics = db.get_topic_flow(session_id="20260421_023037")
"""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import sqlite3
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__version__ = "0.11.0"


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

CONTEXT_DRIFT_THRESHOLD = 0.3       # Lower = more sensitive to topic shifts
ACTIVE_WINDOW_SIZE = 5              # Recent turns kept in short-term memory
TOP_K_CHUNKS = 5                    # Archival chunks retrieved per search
AUTO_TAG_ENABLED = True             # Auto-extract topics from messages

# Recall hook configuration — pre-response cross-session awareness
RECALL_TOP_K_MULTIPLIER = 3         # Fetch N× results before scoring to account for session filtering
RECALL_MIN_SCORE = 0.15             # Minimum RRF score threshold; lower = more inclusive, higher = stricter
RECALL_SNIPPET_MAX_LEN = 80         # Max chars per message snippet in recall pointers
RECALL_OUTPUT_MAX_CHARS = 200       # Hard cap on total recall output to prevent context bloat

# Whitelist of valid time column names for SQL interpolation safety
VALID_TIME_COLUMNS = frozenset({"created_at", "timestamp"})


# ---------------------------------------------------------------------------
# Embedding engine — local semantic search via all-MiniLM-L6-v2 (ONNX)
# ---------------------------------------------------------------------------
# Why ONNX/SentenceTransformers: in-process, no HTTP calls to LM Studio,
# ~80MB model download on first use, 384-dim vectors (~1.5KB per message).
# Graceful degradation: if model unavailable, embedding is skipped silently.

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # SentenceTransformers model identifier
EMBED_DIM = 384                          # Output dimension of all-MiniLM-L6-v2
EMBED_MIN_CONTENT_LEN = 10               # Skip embedding for content shorter than this (noise filter)
EMBED_MAX_CONTENT_LEN = 5000             # Truncate very long messages to avoid excessive compute

# Hybrid search weighting — how much semantic vs FTS5 contributes to final score.
# Higher SEMANTIC_WEIGHT means more emphasis on meaning over keyword matching.
SEMANTIC_WEIGHT = 0.4                    # Weight for cosine similarity in hybrid scoring
FTS5_WEIGHT = 0.6                       # Weight for BM25 rank in hybrid scoring

# Cosine similarity threshold — below this, semantic results are discarded as irrelevant
COSINE_SIMILARITY_THRESHOLD = 0.1


class EmbeddingEngine:
    """Lazy-loaded singleton for local embedding generation.

    Uses SentenceTransformers with ONNX runtime to produce 384-dim vectors
    from text. No external HTTP calls — everything runs in-process.

    Design rationale (SRP):
      - Keeps model loading, vector serialization, and similarity math isolated
        from the database layer.
      - Graceful degradation: if model fails to load or embed, operations return
        None/empty rather than crashing. Embedding is an enhancement, not a blocker.
    """

    _instance: Optional["EmbeddingEngine"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None  # Lazy-loaded SentenceTransformer model

    @classmethod
    def get(cls) -> "EmbeddingEngine":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Load the SentenceTransformer model on first use.

        Downloads ~80MB on first call — this takes a minute. Logged clearly.
        """
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                "Loading embedding model '%s' (first use, ~80MB download)...", EMBED_MODEL_NAME
            )
            self._model = SentenceTransformer(EMBED_MODEL_NAME)
            logger.info("Embedding model loaded successfully (%d-dim vectors)", EMBED_DIM)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed — semantic search disabled. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as e:
            logger.error("Failed to load embedding model '%s': %s", EMBED_MODEL_NAME, e)
            self._model = None
        return self._model

    def embed(self, text: str) -> Optional[List[float]]:
        """Generate a 384-dim embedding vector for the given text.

        Args:
            text: Input text to embed (truncated if too long).

        Returns:
            List of floats (length EMBED_DIM), or None on failure.
        """
        model = self._load_model()
        if model is None:
            return None
        try:
            # Truncate very long content to avoid excessive compute
            truncated = text[:EMBED_MAX_CONTENT_LEN]
            vector = model.encode(truncated, convert_to_numpy=True)
            # Convert numpy array to plain Python list for serialization
            return vector.tolist()
        except Exception as e:
            logger.debug("Embedding failed for text (%d chars): %s", len(text), e)
            return None

    @staticmethod
    def serialize(vector: List[float]) -> bytes:
        """Pack a float32 embedding vector into compact binary (BLOB).

        Uses struct.pack for ~1.5KB per message (384 × 4 bytes).
        More efficient than pickle and version-stable across Python releases.

        Args:
            vector: List of floats (must be exactly EMBED_DIM length).

        Returns:
            Binary blob suitable for SQLite BLOB storage.
        """
        if len(vector) != EMBED_DIM:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d", EMBED_DIM, len(vector)
            )
            return b""
        fmt = f"{EMBED_DIM}f"  # e.g., "384f" for 384 float32 values
        return struct.pack(fmt, *vector)

    @staticmethod
    def deserialize(blob: bytes) -> Optional[List[float]]:
        """Unpack a BLOB back into a float list.

        Args:
            blob: Binary data from SQLite BLOB column.

        Returns:
            List of floats, or None if deserialization fails.
        """
        if not blob or len(blob) < EMBED_DIM * 4:
            return None
        try:
            fmt = f"{EMBED_DIM}f"
            return list(struct.unpack(fmt, blob[: EMBED_DIM * 4]))
        except struct.error as e:
            logger.debug("Failed to deserialize embedding BLOB (%d bytes): %s", len(blob), e)
            return None

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors.

        Pure Python implementation — no numpy dependency for the search path.

        Args:
            a: First vector (list of floats).
            b: Second vector (list of floats).

        Returns:
            Similarity score in range [-1, 1]. Higher = more similar.
        """
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class PerpetualContextDB:
    """SQLite + FTS5 memory database.

    Provides infinite recall across sessions with two retrieval modes:
    1. Structured SQL queries (topics, relationships, metadata)
    2. Full-text search via SQLite FTS5 (keyword matching)
    """

    def __init__(self, db_path: str = None):
        """Initialize the database handler.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ~/.hermes/perpetual_context.db if not specified.
        """
        self._db_path = db_path or os.path.expanduser("~/.hermes/perpetual_context.db")
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()  # Changed to RLock for reentrant calls
        self._initialized = False
        self._time_column: Optional[str] = None  # Cached: 'created_at' or 'timestamp'
        self._schema_version: int = 0  # Cached schema version to skip redundant migrations

    @property
    def time_column(self) -> str:
        """Return the active timestamp column name, cached after first detection."""
        if not self._conn or self._initialized is False:
            return "created_at"
        if self._time_column is None:
            try:
                cursor = self._conn.execute("PRAGMA table_info(messages)")
                columns = {row[1] for row in cursor.fetchall()}
                self._time_column = "created_at" if "created_at" in columns else "timestamp"
            except Exception:
                self._time_column = "created_at"
        # Safety check — never interpolate unvalidated column names into SQL
        if self._time_column not in VALID_TIME_COLUMNS:
            logger.warning(
                "Invalid time_column '%s', falling back to 'created_at'",
                self._time_column,
            )
            self._time_column = "created_at"
        return self._time_column

    # -- Lifecycle ---------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize the database.

        Creates tables if they don't exist and verifies FTS5 virtual table is available.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        try:
            with self._lock:
                # Connect to SQLite (create if not exists)
                self._conn = sqlite3.connect(
                    self._db_path, timeout=30, check_same_thread=False
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._create_tables()

                # Migrate existing timestamp data to created_at if needed
                self._migrate_timestamps_to_created_at()

                self._initialized = True
                
                return True

        except Exception as e:
            logger.error("Failed to initialize PerpetualContextDB: %s", e)
            self._conn = None
            return False

    def shutdown(self) -> None:
        """Clean shutdown — close database connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.commit()
                except Exception:
                    pass
                self._conn.close()
                self._conn = None
            self._initialized = False

    # -- Table creation ----------------------------------------------------

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

        # Backward compatibility: add missing columns from old schema (only if schema < v2).
        # NOTE: SQLite does NOT support function calls in ALTER TABLE ADD COLUMN defaults.
        # Use constant defaults here; triggers handle auto-timestamping where needed.

        # FTS5 virtual table for full-text search
        # Check if existing FTS table has the right columns, recreate if needed
        fts_has_metadata = False
        try:
            cursor = conn.execute("PRAGMA table_info(messages_fts)")
            fts_columns = [row[1] for row in cursor.fetchall()]
            fts_has_metadata = 'metadata' in fts_columns
        except Exception:
            pass

        if not fts_has_metadata:
            # Drop old FTS tables and recreate with correct schema
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.execute("DROP TABLE IF EXISTS messages_fts_config")
            conn.execute("DROP TABLE IF EXISTS messages_fts_data")
            conn.execute("DROP TABLE IF EXISTS messages_fts_docsize")
            conn.execute("DROP TABLE IF EXISTS messages_fts_idx")

        # ALWAYS drop old triggers that reference the old FTS schema (2-column insert)
        for trigger in ('messages_ai', 'messages_ad', 'messages_au'):
            try:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except Exception:
                pass

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
            except Exception as e:
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
            self._ensure_column('topics', 'created_at', "REAL DEFAULT 0")
            self._ensure_column('topics', 'updated_at', "REAL DEFAULT 0")

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
            self._ensure_column('relationships', 'relationship_type', "TEXT DEFAULT 'related'")
            self._ensure_column('relationships', 'strength', "REAL DEFAULT 0.5")
            self._ensure_column('relationships', 'created_at', "REAL DEFAULT 0")

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
            self._ensure_column('session_metadata', 'topic_count', "INTEGER DEFAULT 0")
            self._ensure_column('session_metadata', 'last_updated', "REAL DEFAULT 0")

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
            self._ensure_column('knowledge_gaps', 'resolved', "INTEGER DEFAULT 0")
            self._ensure_column('knowledge_gaps', 'resolution_text', "TEXT DEFAULT ''")
            self._ensure_column('knowledge_gaps', 'resolution_timestamp', "REAL DEFAULT 0")
            # Legacy column: needs_wiki_update (renamed to needs_reference_library_update in docs)
            self._ensure_column('knowledge_gaps', 'needs_wiki_update', "INTEGER DEFAULT 1")
            self._ensure_column('knowledge_gaps', 'first_principles', "TEXT DEFAULT ''")

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
        except Exception:
            pass

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
        except Exception as e:
            # Column might already exist or ALTER not supported (e.g., FTS5 tables).
            # Log and degrade gracefully — don't abort initialization for non-critical columns.
            if "duplicate column" in str(e).lower() or "already exists" in str(e).lower():
                return False  # Expected — column exists
            logger.warning("Failed to add optional column %s.%s: %s", table, column, e)
            return False

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
            
            if 'timestamp' not in columns or 'created_at' not in columns:
                return  # No migration needed
            
            # Count rows that need migration (created_at is NULL or 0)
            needs_migration = self._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE created_at IS NULL OR created_at = 0"
            ).fetchone()[0]
            
            if needs_migration == 0:
                logger.debug("No timestamp migration needed — all rows have valid created_at")
                return
            
            # Copy timestamp values to created_at where created_at is missing/zero
            self._conn.execute("""
                UPDATE messages 
                SET created_at = timestamp 
                WHERE (created_at IS NULL OR created_at = 0) AND timestamp IS NOT NULL
            """)
            
            migrated = self._conn.total_changes
            logger.info("Migrated %d/%d rows: copied timestamp → created_at", migrated, needs_migration)
            
        except Exception as e:
            logger.warning("Timestamp migration failed (non-fatal): %s", e)

    # -- Message storage ---------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None,
        timestamp: float = None,
    ) -> Optional[int]:
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
    ) -> Dict[str, int]:
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
    ) -> List[Dict[str, Any]]:
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
    ) -> List[Dict[str, Any]]:
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

    def get_messages(
        self,
        session_id: str = None,
        role: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
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

    def get_active_window(self, session_id: str, n: int = None) -> List[Dict[str, Any]]:
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

    # -- Search ------------------------------------------------------------

    def fts_search(self, query: str, session_id: str = None, top_k: int = 10) -> List[Dict[str, Any]]:
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
            
            # Check if query contains FTS5 operators that can't be safely escaped
            # If so, fall back to LIKE-based search which is more forgiving
            fts_operators = ['[', ']', '+', '-', '(', ')', '.', ',', '?']
            has_fts_ops = any(op in search_query for op in fts_operators)
            
            if has_fts_ops:
                # Fallback to LIKE search for queries with special operators.
                # Add a simple BM25-like relevance score based on word overlap.
                like_pattern = f"%{search_query}%"
                
                # Tokenize the query for relevance scoring (simple word count)
                query_words = set(search_query.lower().split())
                
                base_sql = f"""
                    SELECT m.id, m.session_id, m.role, m.content, 
                           COALESCE(m.metadata, '{{{{}}}}') as metadata,
                           m.{time_col},
                           -- Simple relevance: count of query words found in content (higher = more relevant).
                           -- Negated so ORDER BY rank works the same way as FTS5 (lower = better).
                           -({'+ '.join([f"CASE WHEN LOWER(m.content) LIKE '%{w}%' THEN 1 ELSE 0 END" for w in list(query_words)[:5]])}) as rank
                    FROM messages m
                    WHERE m.content LIKE ?
                """
                params: list = [like_pattern]
            else:
                # Normal FTS5 phrase matching for clean queries
                escaped_query = search_query.replace("'", "''")
                fts_query = f'"{escaped_query}"'
                
                base_sql = f"""
                    SELECT m.id, m.session_id, m.role, m.content, 
                           COALESCE(m.metadata, '{{}}') as metadata,
                           m.{time_col}, rank
                    FROM messages_fts f
                    JOIN messages m ON m.id = f.rowid
                    WHERE messages_fts MATCH {fts_query}
                """
                params: list = []

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
    ) -> List[Dict[str, Any]]:
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
            scored: List[Tuple[int, float, Dict[str, Any]]] = []
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
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining FTS5 keyword matching + semantic embeddings.

        Uses weighted scoring to combine BM25 rank from FTS5 with cosine similarity
        from embedding vectors. Messages appearing in both result sets get boosted scores.

        Args:
            query: Search query string.
            session_id: Optional session filter.
            top_k: Maximum results to return.
            exclude_session_id: Optional session ID to skip (for cross-session recall).

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
        scored: Dict[int, float] = {}
        msg_cache: Dict[int, Dict[str, Any]] = {}

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

        db_lookup: Dict[int, Dict[str, Any]] = {}
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
        messages: Dict[int, Dict[str, Any]],
        qualified: List[Tuple[int, float]],
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


    # -- Topic flow tracking -----------------------------------------------

    def add_topic(self, session_id: str, topic_name: str, confidence: float = 0.5) -> Optional[int]:
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

    def get_topic_flow(self, session_id: str) -> List[Dict[str, Any]]:
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
    ) -> Optional[int]:
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
    ) -> Optional[int]:
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
    ) -> List[Dict[str, Any]]:
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

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
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

    def get_stats(self) -> Dict[str, Any]:
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
        ids: List[int] = None,
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
    ) -> Dict[str, Any]:
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
            except Exception:
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
    ) -> Dict[str, Any]:
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
        first_principles: List[str] | None = None,
        # Legacy alias — kept for backward compatibility
        needs_wiki_update: bool | None = None,    # deprecated: use needs_reference_library_update
    ) -> Optional[int]:
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

    def get_unresolved_gaps(self, limit: int = 10) -> List[Dict[str, Any]]:
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

    def get_gap_stats(self) -> Dict[str, Any]:
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

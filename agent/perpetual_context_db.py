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
        content="Hello, how do I configure Logos?",
        metadata={"turn": 1}
    )

    # Search via FTS5 + SQL
    results = db.hybrid_search("Logos configuration", top_k=5)

    # Track topic flow
    topics = db.get_topic_flow(session_id="20260421_023037")
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import struct
import threading
from typing import Any

from agent.pcdb_messages import _MessageManager
from agent.pcdb_metadata import _MetadataManager
from agent.pcdb_schema import _SchemaManager
from agent.pcdb_search import _SearchEngine

# Lazy-import torch — only loaded when embeddings are actually needed.
# Avoids ~400MB memory / ~10s delay on every import of this module.
torch = None

# Suppress tqdm "Batches" bars from SentenceTransformer.encode().
# ST 5.x resolves show_progress_bar=None from its logger level and passes an
# explicit disable= to trange(); that explicit arg OVERRIDES a class-level
# tqdm.disable=True, so the load-bearing fix is show_progress_bar=False at the
# call site (see EmbeddingEngine.embed below). This module-level disable is
# belt-and-suspenders and matches rl_index.py / semantic_vector/__init__.py.
import tqdm as _tqdm
from logos_constants import logos_env
_tqdm.disable = True

logger = logging.getLogger(__name__)

__version__ = "0.11.0"


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

CONTEXT_DRIFT_THRESHOLD = 0.3  # Lower = more sensitive to topic shifts
ACTIVE_WINDOW_SIZE = 5  # Recent turns kept in short-term memory
TOP_K_CHUNKS = 5  # Archival chunks retrieved per search
AUTO_TAG_ENABLED = True  # Auto-extract topics from messages

# Recall hook configuration — pre-response cross-session awareness
RECALL_TOP_K_MULTIPLIER = 3  # Fetch N× results before scoring to account for session filtering
RECALL_MIN_SCORE = 0.15  # Minimum RRF score threshold; lower = more inclusive, higher = stricter
RECALL_SNIPPET_MAX_LEN = 80  # Max chars per message snippet in recall pointers
RECALL_OUTPUT_MAX_CHARS = 200  # Hard cap on total recall output to prevent context bloat

# Whitelist of valid time column names for SQL interpolation safety
VALID_TIME_COLUMNS = frozenset({"created_at", "timestamp"})


# ---------------------------------------------------------------------------
# Embedding engine — local semantic search via all-MiniLM-L6-v2 (ONNX)
# ---------------------------------------------------------------------------
# Why ONNX/SentenceTransformers: in-process, no HTTP calls to LM Studio,
# ~80MB model download on first use, 384-dim vectors (~1.5KB per message).
# Graceful degradation: if model unavailable, embedding is skipped silently.

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # SentenceTransformers model identifier
EMBED_DIM = 384  # Output dimension of all-MiniLM-L6-v2
EMBED_MIN_CONTENT_LEN = 10  # Skip embedding for content shorter than this (noise filter)
EMBED_MAX_CONTENT_LEN = 5000  # Truncate very long messages to avoid excessive compute

# Hybrid search weighting — how much semantic vs FTS5 contributes to final score.
# Higher SEMANTIC_WEIGHT means more emphasis on meaning over keyword matching.
SEMANTIC_WEIGHT = 0.4  # Weight for cosine similarity in hybrid scoring
FTS5_WEIGHT = 0.6  # Weight for BM25 rank in hybrid scoring

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

    _instance: EmbeddingEngine | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None  # Lazy-loaded SentenceTransformer model
        self._device: str | None = None  # Device the model was loaded on

    @classmethod
    def get(cls) -> EmbeddingEngine:
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @staticmethod
    def _select_device_candidates(torch: Any) -> list[str]:
        """Ordered device candidates for model loading.

        1. ``HERMES_EMBED_DEVICE`` env override (e.g. ``cuda:3``, ``cpu``)
           — tried first and *only*.
        2. Otherwise every CUDA device ranked by free memory (most-free
           first). Devices with < 2 GB free (e.g. cuda:0 under vLLM) are
           tried last — a load there often OOMs, but it can work.
        3. ``cpu`` as the final fallback.

        ``mem_get_info`` itself can raise on a fully-occupied context;
        those devices are ranked as packed (free=0) and still tried.
        """
        forced = logos_env("EMBED_DEVICE", "").strip()
        if forced:
            return [forced]
        ordered: list[tuple[int, int]] = []
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        free, _total = torch.cuda.mem_get_info(i)
                    except Exception:
                        free = 0
                    ordered.append((free, i))
                ordered.sort(key=lambda t: t[0], reverse=True)
        except Exception:
            ordered = []
        min_free = 2 * 1024 ** 3  # MiniLM ONNX needs far less; 2 GB is headroom
        free_gpus = [f"cuda:{i}" for free, i in ordered if free > min_free]
        packed_gpus = [f"cuda:{i}" for free, i in ordered if free <= min_free]
        return free_gpus + packed_gpus + ["cpu"]

    def _load_model(self):
        """Load the SentenceTransformer model on first use.

        Loads from sovereign local path to ensure zero network latency and total sovereignty.

        Device selection walks the candidate list from
        :meth:`_select_device_candidates`; each failure is logged and the
        next candidate tried, so a GPU packed by vLLM no longer silently
        disables semantic search while free GPUs sit idle.
        """
        if self._model is not None:
            return self._model
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except (ImportError, ModuleNotFoundError):
            logger.warning("sentence-transformers not installed — semantic search disabled. Install with: pip install sentence-transformers")
            self._model = None
            return self._model

        local_path = os.path.expanduser("~/.hermes/models/embeddings/all-MiniLM-L6-v2")
        for device in self._select_device_candidates(torch):
            try:
                logger.info("Loading embedding model from '%s' on %s...", local_path, device)
                self._model = SentenceTransformer(local_path, device=device, local_files_only=True)
                self._device = device
                logger.info("Embedding model loaded on %s (%d-dim vectors)", device, EMBED_DIM)
                return self._model
            except Exception as e:
                logger.warning(
                    "Embedding model load failed on %s: %s — trying next device",
                    device, e,
                )
        logger.error(
            "Failed to load embedding model on any device — semantic search "
            "disabled. Set HERMES_EMBED_DEVICE or free a GPU."
        )
        self._model = None
        return self._model

    def embed(self, text: str) -> list[float] | None:
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
            # show_progress_bar=False is required: ST passes an explicit
            # disable= to trange() which overrides the class-level
            # tqdm.disable=True set above, so the bar reappears otherwise
            # (this was the un-suppressed per-message save path — pcdb_messages
            # embeds every stored user/assistant turn, so it fired in the TUI).
            vector = model.encode(truncated, convert_to_numpy=True, show_progress_bar=False)
            # Convert numpy array to plain Python list for serialization
            return vector.tolist()
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError) as e:
            logger.debug("Embedding failed for text (%d chars): %s", len(text), e)
            return None

    @staticmethod
    def serialize(vector: list[float]) -> bytes:
        """Pack a float32 embedding vector into compact binary (BLOB).

        Uses struct.pack for ~1.5KB per message (384 × 4 bytes).
        More efficient than pickle and version-stable across Python releases.

        Args:
            vector: List of floats (must be exactly EMBED_DIM length).

        Returns:
            Binary blob suitable for SQLite BLOB storage.
        """
        if len(vector) != EMBED_DIM:
            logger.warning("Embedding dimension mismatch: expected %d, got %d", EMBED_DIM, len(vector))
            return b""
        fmt = f"{EMBED_DIM}f"  # e.g., "384f" for 384 float32 values
        return struct.pack(fmt, *vector)

    @staticmethod
    def deserialize(blob: bytes) -> list[float] | None:
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
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Pure Python implementation — no numpy dependency for the search path.

        Args:
            a: First vector (list of floats).
            b: Second vector (list of floats).

        Returns:
            Similarity score in range [-1, 1]. Higher = more similar.
        """
        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)


class PerpetualContextDB(
    _SchemaManager,
    _MessageManager,
    _SearchEngine,
    _MetadataManager,
):
    """SQLite + FTS5 memory database.

    Provides infinite recall across sessions with two retrieval modes:
    1. Structured SQL queries (topics, relationships, metadata)
    2. Full-text search via SQLite FTS5 (keyword matching)

    Delegates to mixin classes:
      - _SchemaManager: table creation, migration, schema maintenance
      - _MessageManager: message CRUD, embedding storage, backfill
      - _SearchEngine: FTS5, semantic, hybrid search, recall
      - _MetadataManager: topics, relationships, sessions, gaps, query, stats
    """

    def __init__(self, db_path: str = None):
        """Initialize the database handler.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                ~/.hermes/perpetual_context.db if not specified.
        """
        self._db_path = db_path or os.path.join(logos_env("HOME") or os.path.expanduser("~/.hermes"), "perpetual_context.db")
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()  # Changed to RLock for reentrant calls
        self._initialized = False
        self._time_column: str | None = None  # Cached: 'created_at' or 'timestamp'
        self._schema_version: int = 0  # Cached schema version to skip redundant migrations

    @property
    def time_column(self) -> str:
        """Return the active timestamp column name, cached after first detection.

        Thread-safe: acquires lock before reading self._conn and self._time_column.
        The cached value (self._time_column) is a plain string — safe to read
        outside the lock once assigned, since strings are immutable in Python.
        """
        with self._lock:
            if not self._conn or self._initialized is False:
                return "created_at"
            if self._time_column is None:
                try:
                    cursor = self._conn.execute("PRAGMA table_info(messages)")
                    columns = {row[1] for row in cursor.fetchall()}
                    self._time_column = "created_at" if "created_at" in columns else "timestamp"
                except sqlite3.Error:
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
                # Pre-flight (2026-08-20): if this first boot must rebuild a v1
                # messages table, do it on a throwaway connection so the main
                # connection opens on the final schema. A rebuild on the main
                # connection left its cached schema stale and rejected
                # role='tool' rows until the next restart.
                self._preconnect_v3_check()
                # Connect to SQLite (create if not exists)
                self._conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA busy_timeout=30000")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._create_tables()

                # Migrate existing timestamp data to created_at if needed
                self._migrate_timestamps_to_created_at()

                self._initialized = True

                return True

        except sqlite3.Error as e:
            logger.error("Failed to initialize PerpetualContextDB: %s", e)
            self._conn = None
            return False

    def shutdown(self) -> None:
        """Clean shutdown — close database connection."""
        with self._lock:
            if self._conn:
                try:
                    self._conn.commit()
                except (AttributeError, KeyError, OSError, RuntimeError, TypeError):
                    logger.debug("Commit failed during shutdown")
                # Drain the WAL while we still own the only snapshot —
                # a blocked checkpoint balloons the -wal file and widens
                # the 'database is locked' window for other processes.
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    logger.debug("WAL checkpoint on shutdown skipped")
                self._conn.close()
                self._conn = None
            self._initialized = False

    def reindex_embeddings(self) -> dict[str, Any]:
        """Embed any unembedded messages.

        Designed for periodic maintenance (nightly cron). Skips entirely
        if the message count is below 8,000 — not worth the overhead for
        small databases.

        Returns:
            dict with keys: action_taken, messages_embedded, total_messages
        """
        if self._conn is None:
            logger.debug("Cannot reindex — database not initialized")
            return {"action_taken": "skipped", "reason": "database not initialized"}

        try:
            with self._lock:
                row = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()
            total_count = row[0] if row else 0
            if total_count < 8_000:
                logger.info("Skipping reindex — %d messages below threshold", total_count)
                return {
                    "action_taken": "skipped",
                    "total_messages": total_count,
                    "reason": "below threshold",
                }

            # Embed missing messages using existing backfill method
            backfill_result = self.backfill_embeddings(batch_size=50)
            embedded_count = backfill_result.get("embedded", 0)

            if embedded_count == 0:
                logger.info("No unembedded messages — reindex not needed")
                return {
                    "action_taken": "skipped",
                    "total_messages": total_count,
                    "messages_embedded": 0,
                    "reason": "all embedded",
                }

            return {
                "action_taken": "completed",
                "total_messages": total_count,
                "messages_embedded": embedded_count,
                "backfill_stats": backfill_result,
            }
        except Exception as e:
            logger.error("Reindex failed: %s", e)
            return {"action_taken": "failed", "reason": str(e)}

"""Perpetual Context Memory Provider — Infinite recall across sessions.

A local-first memory system combining SQLite, FTS5 full-text indexing, and
topic flow tracking for keyword retrieval. Provides topic clustering,
relationship management, and graded context depth control.

Architecture:
  - SQLite: Structured storage (messages, topics, relationships)
  - FTS5: Full-text keyword search
  - Topic Flow: Automatic topic clustering with drift detection

Tools exposed to the agent:
  • perpetual_search — Keyword search across all past sessions
  • topic_flow       — View and manage topic clusters per session
  • context_depth    — Control how much historical context is surfaced

Config in ~/.hermes/config.yaml:
  memory:
    provider: perpetual_context
    perpetual_context:
      enabled: true
      db_path: ~/.hermes/perpetual_context.db
"""

from __future__ import annotations

__version__ = "0.11.0"

import json
import logging
import os
import re as _re  # renamed to avoid conflict with topic extraction regex
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add scripts directory for query classifier
_scripts_dir = str(Path.home() / ".hermes" / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

try:
    from query_classifier import QueryClassifier as _QueryClassifierClass
except ImportError:
    _QueryClassifierClass = None  # Graceful fallback if classifier unavailable

from agent.memory_provider import MemoryProvider
from agent.perpetual_context_db import RECALL_OUTPUT_MAX_CHARS
from tools.registry import tool_error

# Import split modules (SRP compliance)
from .extraction_engine import ExtractionEngine, _STOPWORDS
from .tool_handler import ToolHandler
from .context_bridge_builder import ContextBridgeBuilder

logger = logging.getLogger(__name__)

# Configurable truncation limits — adjust for your use case
PREFETCH_TRUNCATION_CHARS = 1500    # Max chars per message in prefetch results (increased from 500)
PRE_COMPRESS_TRUNCATION_CHARS = 8000  # Max chars when archiving before compression (increased from 2000)

# Periodic context injection config — fires every N turns between compressions.
# Keeps the system aware of historical context without waiting for token budget exhaustion.
PERIODIC_INJECTION_INTERVAL = 10     # Turns between injections (configurable via memory.perpetual_context.injection_interval)
PERIODIC_INJECTION_MAX_CHARS = 300   # Hard cap on injected context — pointer, not full content

# Deep Research Engine config — Phase 1: Local Recall auto-hook
DEEP_RESEARCH_ENABLED = True         # Master toggle for deep research loop
RL_SEARCH_TOP_K = 5                  # Results from reference library search
PM_SEARCH_TOP_K = 5                  # Results from perpetual memory hybrid search
GAP_DETECTION_MIN_SCORE = 0.3        # Below this, results considered low-confidence
GAP_DETECTION_MIN_RESULTS = 2        # Fewer than this triggers gap flag for Phase 2 web fallback

# Phase 2: Auto Web Search + Unified Relevance Scoring
WEB_SEARCH_ENABLED = True            # Master toggle for automatic web search triggering
SEARXNG_URL = "http://localhost:8080"  # SearXNG instance URL
WEB_SEARCH_TIMEOUT = 10              # Seconds before timeout
WEB_PRIORITY_THRESHOLD = 0.5         # Trigger web search when priority exceeds this
WEB_SEARCH_TOP_K = 5                 # Max results from SearXNG
UNIFIED_SCORE_WEIGHTS = {            # Source reliability weights for cross-source merge
    "pm": 0.35,   # Perpetual Memory — personal context, episodic memory
    "rl": 0.40,   # Reference Library — curated, worldview-aligned knowledge (highest)
    "web": 0.25,  # Web search — needs verification, time-sensitive (lowest)
}

# Worldview filter: domains to block or flag on web results
WORLDVIEW_BLOCKED_DOMAINS = {
    "reddit.com",   # Low-signal echo chamber
    "quora.com",    # Unverified opinions presented as facts
    "medium.com",   # Paywalled opinion pieces
}

WORLDVIEW_FLAGGED_DOMAINS = {
    "bbc.com": "State-aligned framing",
    "cnn.com": "Progressive institutional capture",
    "nytimes.com": "Elite establishment narrative",
    "reuters.com": "Corporate wire service bias",
    "apnews.com": "Associated Press — establishment framing",
}

# Phase 3: PM → RL Distillation Pipeline
DISTILLATION_ENABLED = True          # Master toggle for auto-distillation triggers
DISTILLATION_SCORE_THRESHOLD = 0.5   # Minimum cluster score to consider for distillation
DISTILLATION_QUEUE_PATH = str(Path.home() / ".hermes" / "staging" / "distillation_queue.json")

# Auto-routing constants moved to retrieval_engine.py — imported below.
# Kept here for backward compatibility with existing code that references them directly.
from .retrieval_engine import classify_query_intent as _classify_query_intent_fn  # noqa: F401
AUTO_ROUTING_KEYWORDS = {
    "decision_trace": {"why", "decision", "chose", "instead of", "reason", "rationale"},
    "file_history": {"file", "edit", "changed"},
    "recent": {"recently", "continue", "pick up"},
}  # Deprecated — use retrieval_engine.AUTO_ROUTING_KEYWORDS instead

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PERPETUAL_SEARCH_SCHEMA = {
    "name": "perpetual_search",
    "description": (
        "Search across all perpetual memory storage backends using hybrid semantic + keyword search. "
        "Combines FTS5 full-text keyword matching with cosine similarity against stored embedding vectors "
        "(weighted fusion: 60% keyword, 40% semantic). Returns the most relevant historical messages.\n\n"
        "PARAMETERS:\\n"
        "• query: Search text (required)\\n"
        "• session_id: Optional session filter\\n"
        "• top_k: Number of results (default 5, max 20)\\n"
        "\\n"
        "EXAMPLES:\\n"
        "• 'Hermes configuration' — Find messages about setup\\n"
        "• 'GPU training' — Find messages mentioning GPU training\\n"
        "• session_id='20260421_023037' — Limit to one session"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}

TOPIC_FLOW_SCHEMA = {
    "name": "topic_flow",
    "description": (
        "View and manage topic clusters for a session. Shows the conversation's "
        "topic evolution, message counts per topic, and allows adding new topics.\n\n"
        "PARAMETERS:\n"
        "• action: 'list', 'add', or 'drift_check' (default 'list')\n"
        "• session_id: Session to analyze (defaults to current)\n"
        "• topic_name: Topic name for 'add' action\n"
        "• confidence: Confidence score 0.0-1.0 for new topics\n"
        "\n"
        "EXAMPLES:\n"
        "• action='list' — Show all topics for current session\n"
        "• action='drift_check' — Detect if conversation has drifted\n"
        "• action='add', topic_name='GPU optimization' — Register new topic"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: list, add, or drift_check", "default": "list"},
            "session_id": {"type": "string", "description": "Session ID (defaults to current)"},
            "topic_name": {"type": "string", "description": "Topic name for 'add' action"},
            "confidence": {"type": "number", "description": "Confidence score 0.0-1.0", "default": 0.5},
        },
    },
}

CONTEXT_DEPTH_SCHEMA = {
    "name": "context_depth",
    "description": (
        "Control how much historical context is surfaced from perpetual memory. "
        "Adjusts the depth of recall based on conversation needs.\n\n"
        "DEPTH LEVELS:\n"
        "• broad_overview: Only main topics and high-level summaries\n"
        "• moderate: Topics + key messages (default)\n"
        "• deep: All topics with detailed message content\n"
        "• expert: Full history with relationships and metadata\n\n"
        "PARAMETERS:\n"
        "• action: 'get', 'set', or 'status'\n"
        "• level: Depth level (for 'set' action)\n"
        "\n"
        "EXAMPLES:\n"
        "• action='get' — Show current depth setting\n"
        "• action='set', level='deep' — Increase recall depth\n"
        "• action='status' — Full memory system status report"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: get, set, or status", "default": "get"},
            "level": {"type": "string", "description": "Depth level: broad_overview, moderate, deep, expert"},
        },
    },
}

GET_MESSAGES_SCHEMA = {
    "name": "get_messages",
    "description": (
        "Search messages using SQL LIKE-style pattern matching on content. "
        "Returns full raw message content — no summarization, no truncation.\n\n"
        "USE THIS WHEN:\n"
        "• You know what you're looking for and need exact matches\n"
        "• Searching for tokens, keys, or specific strings (e.g., 'ghp_%')\n"
        "• You need the complete content of a message without search indexing abstraction\n\n"
        "PARAMETERS:\n"
        "• pattern: SQL LIKE pattern (use % as wildcard, _ as single char)\n"
        "• session_id: Optional session filter\n"
        "• role: Filter by role (user, assistant, system, tool)\n"
        "• limit: Maximum results to return (default 50)\n\n"
        "EXAMPLES:\n"
        "• pattern='ghp_%' — Find all GitHub tokens\n"
        "• pattern='%github token%' — Find messages mentioning github tokens\n"
        "• pattern='%', role='user', limit=10 — Last 10 user messages"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "SQL LIKE pattern (use % as wildcard)"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
            "limit": {"type": "integer", "description": "Maximum results (default 50)", "default": 50},
        },
        "required": ["pattern"],
    },
}

RECENT_MESSAGES_SCHEMA = {
    "name": "recent_messages",
    "description": (
        "Get the N most recent messages from the database. Returns raw content "
        "in chronological order — no summarization, no search indexing.\n\n"
        "USE THIS WHEN:\n"
        "• You need to see what was discussed recently without searching\n"
        "• Reviewing the last few turns of a conversation\n"
        "• Getting raw message content for verification (e.g., checking token length)\n\n"
        "PARAMETERS:\n"
        "• n: Number of recent messages to retrieve (default 10, max 50)\n"
        "• session_id: Optional session filter (None for all sessions)\n"
        "• role: Optional role filter\n\n"
        "EXAMPLES:\n"
        "• n=5 — Last 5 messages across all sessions\n"
        "• n=10, session_id='20260421_124052' — Last 10 in specific session\n"
        "• n=3, role='user' — Last 3 user messages"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "n": {"type": "integer", "description": "Number of recent messages (default 10)", "default": 10},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
        },
    },
}

QUERY_MESSAGES_SCHEMA = {
    "name": "query_messages",
    "description": (
        "Master query tool — comprehensive message filtering with time ranges,\n"
        "token counts, direct ID lookup, metadata filters, and statistics.\n\n"
        "USE THIS WHEN:\n"
        "• You need precise control over what messages to retrieve\n"
        "• Filtering by time range (e.g., 'messages from April 21st')\n"
        "• Looking up specific message IDs directly\n"
        "• Getting statistics about your conversation history\n\n"
        "PARAMETERS:\n"
        "• pattern: SQL LIKE pattern for content (use % as wildcard)\n"
        "• session_id: Filter by session ID\n"
        "• role: Filter by role (user, assistant, system, tool)\n"
        "• ids: List of specific message IDs to retrieve\n"
        "• time_start: Unix timestamp filter (messages >= this time)\n"
        "• time_end: Unix timestamp filter (messages <= this time)\n"
        "• min_tokens: Minimum token count filter\n"
        "• max_tokens: Maximum token count filter\n"
        "• metadata_key: Filter by metadata key name\n"
        "• metadata_value: Value to match for the metadata key\n"
        "• stats: True to return statistics instead of messages\n"
        "• limit: Maximum results (default 100, max 500)\n"
        "• offset: Pagination offset (default 0)\n\n"
        "EXAMPLES:\n"
        "• ids=[542] — Get message #542 directly\n"
        "• pattern='ghp_%', limit=10 — Find all GitHub tokens\n"
        "• time_start=1776780000, time_end=1776790000 — Messages in time range\n"
        "• role='user', min_tokens=500 — Long user messages\n"
        "• stats=True — Get conversation statistics"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "SQL LIKE pattern (use % as wildcard)"},
            "session_id": {"type": "string", "description": "Optional session ID filter"},
            "role": {"type": "string", "description": "Filter by role: user, assistant, system, tool"},
            "ids": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of specific message IDs to retrieve",
            },
            "time_start": {"type": "number", "description": "Unix timestamp filter (messages >= this time)"},
            "time_end": {"type": "number", "description": "Unix timestamp filter (messages <= this time)"},
            "min_tokens": {"type": "integer", "description": "Minimum token count filter"},
            "max_tokens": {"type": "integer", "description": "Maximum token count filter"},
            "metadata_key": {"type": "string", "description": "Filter by metadata key name"},
            "metadata_value": {
                "type": ["string", "boolean", "number"],
                "description": "Value to match for the metadata key",
            },
            "stats": {"type": "boolean", "description": "True to return statistics instead of messages"},
            "limit": {"type": "integer", "description": "Maximum results (default 100)", "default": 100},
            "offset": {"type": "integer", "description": "Pagination offset (default 0)", "default": 0},
        },
    },
}

SMART_RETRIEVE_SCHEMA = {
    "name": "smart_retrieve",
    "description": (
        "Adaptive retrieval engine for Perpetual Memory. Uses different strategies\n"
        "based on the type of information needed, optimized for local hardware.\n\n"
        "RETRIEVAL TYPES:\n"
        "• auto — Let the system classify intent via keyword heuristics (recommended default)\n"
        "• recent — Context from last 20 turns (fastest, O(1) turn ID lookup)\n"
        "• topic — Topic-specific FTS5 search across all sessions\n"
        "• decision_trace — Find where a decision was made and surrounding context\n"
        "• file_history — All edits to a specific file with turn references\n\n"
        "USE THIS WHEN:\n"
        "• You're unsure which strategy to use (use 'auto' — system classifies for you)\n"
        "• You need recent conversation context (use 'recent')\n"
        "• Searching for topic-specific information across sessions (use 'topic')\n"
        "• Tracing why a decision was made earlier (use 'decision_trace')\n"
        "• Finding all edits to a specific file path (use 'file_history')\n\n"
        "PARAMETERS:\n"
        "• query_type: One of 'auto', 'recent', 'topic', 'decision_trace', 'file_history'\n"
        "• query_text: The search query or context identifier\n\n"
        "EXAMPLES:\n"
        "• query_type='auto', query_text='why did we choose SQLite' — System auto-routes to decision_trace\n"
        "• query_type='recent' — Get last 20 turns for immediate context\n"
        "• query_type='topic', query_text='context bridge design' — Find topic discussion\n"
        "• query_type='file_history', query_text='/path/to/file.py' — Get file edit history"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_type": {"type": "string", "description": "Retrieval type: auto, recent, topic, decision_trace, or file_history"},
            "query_text": {"type": "string", "description": "Search query or context identifier"},
        },
        "required": ["query_type", "query_text"],
    },
}

REFERENCE_LIBRARY_SEARCH_SCHEMA = {
    "name": "reference_library_search",
    "description": (
        "MANDATORY FIRST STEP for all factual, historical, political, economic,\\n"
        "media, and worldview questions. Contains curated, worldview-aligned reference\\n"
        "material built from first principles.\\n\\n"
        "USE THIS BEFORE ANY OTHER SEARCH TOOL when answering:\\n"
        "• Questions about history, politics, economics, media bias, or worldview\\n"
        "• Any factual claim that needs verification against curated knowledge\\n"
        "• Research involving people, organizations, or institutions\\n\\n"
        "This is NOT optional. Always check reference_library_search before generating answers\\n"
        "from training data or session memory alone.\\n\\n"
        "PARAMETERS:\\n"
        "• query: Search text (required)\\n"
        "• top_k: Number of results (default 5, max 20)\\n\\n"
        "EXAMPLES:\\n"
        "• 'Elon Musk political influence' — Find entity page with reaction tracking\\n"
        "• 'media bias patterns' — Curated analysis of source credibility\\n"
        "• 'American economic policy post-1964' — Historical reference material"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}

CLASSIFIER_CORRECTION_SCHEMA = {
    "name": "classifier_correction",
    "description": (
        "Log a correction when the query classifier routes to the wrong source.\\n"
        "This trains the system to route similar queries correctly in the future.\\n\\n"
        "USE THIS WHEN:\\n"
        "• The prefetch injected web results for a personal question ('what dogs do I have')\\n"
        "• A contextual query ('any gaps?', 'continue') was treated as general knowledge\\n"
        "• You notice the classifier chose the wrong primary source (PM/RL/Web)\\n\\n"
        "PARAMETERS:\\n"
        "• query: The original misclassified query text\\n"
        "• correct_source: The correct source ('pm', 'rl', or 'web')\\n\\n"
        "EXAMPLES:\\n"
        "• query='What kind of dogs do I have?', correct_source='pm'\\n"
        "• query='Any gaps?', correct_source='pm'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The misclassified query text"},
            "correct_source": {
                "type": "string",
                "enum": ["pm", "rl", "web"],
                "description": "The correct source: pm=Perpetual Memory, rl=Reference Library, web=Web Search"
            },
        },
        "required": ["query", "correct_source"],
    },
}

RETRIEVAL_QUALITY_SCHEMA = {
    "name": "retrieval_quality",
    "description": (
        "Evaluate prefetch retrieval quality. Shows trend analysis, per-metric averages, "
        "and recent low-quality events.\n\n"
        "USE THIS WHEN:\n"
        "• Checking whether prefetch is returning useful context or noise\n"
        "• Diagnosing why certain queries aren't getting relevant results\n"
        "• Reviewing retrieval quality trends over time\n\n"
        "PARAMETERS:\n"
        "• action: 'trend' (default) for overall analysis, 'failures' for recent misses\n"
        "• window: Number of events to analyze (default 20)\n"
        "• threshold: Quality score threshold for failures (default 0.3)\n\n"
        "EXAMPLES:\n"
        "• action='trend' — Show overall retrieval quality trend\n"
        "• action='failures', threshold=0.4 — Show recent low-quality prefetches"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "Action: 'trend' or 'failures'", "default": "trend"},
            "window": {"type": "integer", "description": "Number of events to analyze (default 20)", "default": 20},
            "threshold": {"type": "number", "description": "Quality score threshold for failures (default 0.3)", "default": 0.3},
        },
    },
}

SESSION_SEARCH_SCHEMA = {
    "name": "session_search",
    "description": (
        "Use ONLY for recent conversation context — what the user said/did in the last\n"
        "few turns. NEVER use this tool for facts, history, or analysis.\n\n"
        "STRICT BOUNDARY:\n"
        "• session_search = recent conversation memory only\n"
        "• reference_library_search = factual/historical/worldview reference (use FIRST)\n"
        "• perpetual_search = deep historical recall across all sessions\n\n"
        "If the question requires factual knowledge, use reference_library_search first.\n"
        "session_search is for remembering what was discussed recently, not for\n"
        "answering questions about the world."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {"type": "integer", "description": "Number of results (default 5)", "default": 5},
        },
        "required": ["query"],
    },
}


class PerpetualContextProvider(MemoryProvider):
    """Memory provider using SQLite + FTS5 full-text indexing.

    Provides infinite recall across sessions with keyword retrieval:
    1. Structured SQL queries (topics, relationships, metadata)
    2. Full-text search via SQLite FTS5 (keyword matching)
    """

    def __init__(self):
        self._db = None
        self._session_id: Optional[str] = None
        self._current_depth: str = "moderate"
        self._prefetch_queue: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        # Prefetch injection toggle — disabled by default to avoid injecting
        # irrelevant noise into every turn. The agent has tools (perpetual_search,
        # reference_library_search) and knows when to use them from the system prompt.
        self._prefetch_enabled: bool = False
        # SRP-compliant sub-components — instantiated in initialize() once DB is ready
        self._extraction: Optional[ExtractionEngine] = None
        self._tools: Optional[ToolHandler] = None
        self._bridge_builder: Optional[ContextBridgeBuilder] = None
        # Negative feedback loop components
        self._scorer: Optional[Any] = None  # BridgeQualityScorer (lazy import)
        self._feedback: Optional[Any] = None  # FeedbackState (lazy init)
        # Retrieval quality evaluation
        self._retrieval_scorer: Optional[Any] = None  # RetrievalQualityScorer (lazy init)
        # RL Memory Provider components — hybrid semantic + keyword search over Reference Library
        self._rl_embeddings_loaded: bool = False
        self._rl_model = None
        self._rl_conn = None
        self._classifier = _QueryClassifierClass() if _QueryClassifierClass else None

    def _ensure_subcomponents(self) -> None:
        """Lazy-init sub-components after DB is available."""
        if self._extraction is None and self._db is not None:
            self._extraction = ExtractionEngine()
            # Bridge builder depends on extraction engine + feedback components
            self._bridge_builder = ContextBridgeBuilder(
                extraction_engine=self._extraction,
                scorer=self._scorer,
                feedback_state=self._feedback,
            )
        if self._tools is None and self._db is not None:
            self._tools = ToolHandler(
                db=self._db,
                session_id=self._session_id or "",
                current_depth=self._current_depth,
                prefetch_queue=self._prefetch_queue,
            )

    def _ensure_feedback_loop(self) -> None:
        """Lazy-init negative feedback loop components."""
        if self._scorer is None:
            from .quality_scorer import BridgeQualityScorer
            self._scorer = BridgeQualityScorer()
        if self._feedback is None:
            from .feedback_state import FeedbackState
            self._feedback = FeedbackState()

    def _ensure_retrieval_scorer(self) -> None:
        """Lazy-init retrieval quality scorer."""
        if self._retrieval_scorer is None:
            try:
                from .retrieval_quality import RetrievalQualityScorer
                self._retrieval_scorer = RetrievalQualityScorer()
            except ImportError as e:
                logger.debug("RetrievalQualityScorer unavailable: %s", e)

    def _score_retrieval_quality(
        self,
        query: str,
        priorities: Dict[str, float],
        scored_results: List[Dict[str, Any]],
        formatted_text: str = "",
    ) -> None:
        """Score a prefetch retrieval event for quality tracking.

        Non-blocking — logs results to sliding window for trend analysis.
        Called after each successful prefetch with results.
        """
        self._ensure_retrieval_scorer()
        if self._retrieval_scorer is None:
            return

        try:
            self._retrieval_scorer.score(
                query=query,
                priorities=priorities,
                scored_results=scored_results,
                formatted_text=formatted_text,
                top_k_requested=self._get_depth_limit(),
            )
        except Exception as e:
            logger.debug("Retrieval quality scoring failed: %s", e)

    def _ensure_rl_embeddings(self) -> bool:
        """Lazy-load RL embedding index on first use. Returns True if successful."""
        if self._rl_embeddings_loaded:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            model_path = str(Path.home() / ".hermes" / "models" / "embeddings" / "all-MiniLM-L6-v2")

            # Use GPU if available (Crenshaw server), otherwise CPU (current system)
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

            self._rl_model = SentenceTransformer(model_path, device=device)
            self._rl_conn = sqlite3.connect(str(Path.home() / ".hermes" / "perpetual_context.db"))
            self._rl_embeddings_loaded = True
            logger.info(f"RLMemoryProvider: Embedding model loaded on {device}")
            return True
        except Exception as e:
            logger.warning(f"RLMemoryProvider: Failed to load embedder: {e}. Keyword-only mode.")
            return False

    def _hybrid_rl_search(self, query: str, top_k: int = 3) -> list:
        """Hybrid semantic + keyword search over RL corpus. Returns snippets with file pointers."""
        if not self._rl_model or not self._rl_conn:
            return []

        try:
            import numpy as np

            # Semantic search — encode query and compare against all RL embeddings
            query_emb = self._rl_model.encode([query])[0]
            rows = self._rl_conn.execute(
                "SELECT id, file_path, embedding FROM rl_embeddings WHERE word_count > 50"
            ).fetchall()

            semantic_scores = {}
            for row_id, file_path, emb_bytes in rows:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                sim = float(np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8))
                semantic_scores[file_path] = sim

            # Keyword search (FTS5) — fast exact matching
            fts_rows = self._rl_conn.execute(
                "SELECT file_path, rank FROM rl_fts WHERE rl_fts MATCH ? ORDER BY rank LIMIT 20",
                (query,)
            ).fetchall()
            keyword_scores = {row[0]: row[1] for row in fts_rows}

            # Hybrid scoring: 60% semantic + 40% keyword (normalized)
            max_semantic = max(semantic_scores.values()) if semantic_scores else 1.0
            max_keyword = max(keyword_scores.values()) if keyword_scores else 1.0

            all_files = set(list(semantic_scores.keys()) + list(keyword_scores.keys()))
            final_scores = []

            for file_path in all_files:
                sem = semantic_scores.get(file_path, 0) / max_semantic if max_semantic > 0 else 0
                kw = keyword_scores.get(file_path, 0) / max_keyword if max_keyword > 0 else 0
                hybrid = (sem * 0.6) + (kw * 0.4)
                final_scores.append((hybrid, file_path))

            final_scores.sort(key=lambda x: x[0], reverse=True)

            # Extract snippets for top results — skip YAML frontmatter, find query terms in body text
            rl_dir = Path.home() / ".hermes" / "reference-library"
            results = []
            query_terms = set(query.lower().split())

            for score, file_path in final_scores[:top_k]:
                if score < 0.15:  # Threshold to avoid noise
                    continue

                full_path = rl_dir / file_path
                content = full_path.read_text(encoding='utf-8')
                lines = content.split('\n')

                # Skip YAML frontmatter if present
                content_start = 0
                if lines and lines[0].strip() == '---':
                    for i, line in enumerate(lines[1:], 1):
                        if line.strip() == '---':
                            content_start = i + 1
                            break

                body_lines = lines[content_start:]
                best_score = 0
                best_start = 0

                for i, line in enumerate(body_lines):
                    if line.startswith('#') or not line.strip():
                        continue
                    term_matches = len(query_terms & set(line.lower().split()))
                    if term_matches > best_score:
                        best_score = term_matches
                        best_start = max(0, i - 1)

                snippet_lines = body_lines[best_start:min(best_start + 4, len(body_lines))]
                snippet = ' '.join(l.strip() for l in snippet_lines if l.strip())
                snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet

                results.append({
                    "score": round(score, 3),
                    "file_path": file_path,
                    "snippet": snippet
                })

            return results

        except Exception as e:
            logger.debug(f"RL hybrid search failed: {e}")
            return []

    # -- Phase 2: Auto Web Search + Unified Relevance Scoring ---------------

    def _auto_web_search(self, query: str, top_k: int = WEB_SEARCH_TOP_K) -> list:
        """Call SearXNG for web search results with worldview filtering.

        Returns list of result dicts with url, title, content, score fields.
        Gracefully degrades if SearXNG is unavailable or times out.
        Applies SovereignSieve worldview filtering on all results before return.
        """
        if not WEB_SEARCH_ENABLED:
            return []

        try:
            import requests as _requests

            start = time.perf_counter()
            response = _requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "engines": "google,brave,wikipedia"},
                timeout=WEB_SEARCH_TIMEOUT,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            if response.status_code != 200:
                logger.warning(f"SearXNG returned status {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])[:top_k]

            # Normalize and apply worldview filtering
            filtered = []
            for r in results:
                domain = self._extract_domain(r.get("url", ""))

                # Block known low-signal sources
                if domain in WORLDVIEW_BLOCKED_DOMAINS:
                    logger.debug(f"Blocked web result from {domain}")
                    continue

                entry = {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "content": (r.get("content") or "")[:300],
                    "score": r.get("score", 0),
                    "engine": r.get("engine", ""),
                    "_latency_ms": round(latency_ms, 1),
                }

                # Flag sources with known institutional framing
                if domain in WORLDVIEW_FLAGGED_DOMAINS:
                    entry["worldview_flag"] = "review_recommended"
                    entry["flag_reason"] = f"Source {domain}: {WORLDVIEW_FLAGGED_DOMAINS[domain]}"

                filtered.append(entry)

            logger.info(f"Web search: {len(filtered)} results in {latency_ms:.0f}ms for \"{query[:40]}...\"")
            return filtered

        except _requests.exceptions.Timeout:
            logger.warning("SearXNG request timed out — web search skipped")
            return []
        except Exception as e:
            logger.debug(f"Web search failed: {e}")
            return []

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract domain from URL for worldview filtering."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.lower().replace("www.", "")
        except Exception:
            return url.split("//")[0].split("/")[0].lower() if "//" in url else ""

    def _distill_check(self, query: str, pm_results: list = None) -> Optional[str]:
        """Phase 3: Check for undistilled high-signal clusters matching current context.

        Non-blocking check that identifies PM clusters ready for RL distillation.
        Returns a pointer string if distillation is recommended, or None otherwise.

        Args:
            query: Current user query text.
            pm_results: Results from Perpetual Memory search (optional).

        Returns:
            Pointer string with distillation recommendation, or None.
        """
        if not DISTILLATION_ENABLED:
            return None

        try:
            # Load distillation queue
            queue_path = Path(DISTILLATION_QUEUE_PATH)
            if not queue_path.exists():
                return None

            import json as _json
            with open(queue_path, 'r') as f:
                queue = _json.load(f)

            if not isinstance(queue, list):
                return None

            # Check for undistilled signals matching current query topics
            query_lower = query.lower()
            matched_signals = []

            for signal in queue:
                topic = signal.get("topic", "").lower()
                score = signal.get("score", 0)
                distilled = signal.get("distilled", False)

                # Skip already-distilled signals
                if distilled:
                    continue

                # Check if query matches this topic (simple substring/keyword match)
                if topic.replace("_", " ") in query_lower or topic in query_lower:
                    if score >= DISTILLATION_SCORE_THRESHOLD:
                        matched_signals.append(signal)

            if not matched_signals:
                return None

            # Build recommendation pointer
            top_signal = max(matched_signals, key=lambda s: s.get("score", 0))
            topic_name = top_signal["topic"].replace("_", " ")
            turn_count = top_signal.get("size", len(top_signal.get("turn_ids", [])))

            return (
                f"\n\n[💡 Distillation recommended: '{topic_name}' cluster has {turn_count} turns "
                f"(score: {top_signal['score']:.3f}). Consider running the Logos Engine pipeline "
                f"to distill this into Reference Library. Queue at {queue_path}]"
            )

        except Exception as e:
            logger.debug(f"Distillation check failed: {e}")
            return None

    def _unified_score_results(
        self,
        pm_results: list = None,
        rl_results: list = None,
        web_results: list = None,
    ) -> list:
        """Merge results from all sources with unified relevance scoring.

        Each result gets a normalized score (0-1) weighted by source reliability:
        - RL: 0.40 weight (curated, worldview-aligned knowledge — highest)
        - PM: 0.35 weight (personal context, episodic memory)
        - Web: 0.25 weight (needs verification, time-sensitive — lowest)

        Returns merged list sorted by unified_score descending.
        """
        pm = pm_results or []
        rl = rl_results or []
        web = web_results or []

        scored = []

        # Score RL results
        max_rl = max((r.get("score", 1) for r in rl), default=1)
        for r in rl:
            normalized = min(r.get("score", 0) / max_rl, 1.0) if max_rl > 0 else 0
            unified = normalized * UNIFIED_SCORE_WEIGHTS["rl"]
            scored.append({
                "source": "RL",
                "unified_score": round(unified, 3),
                "raw_score": r.get("score", 0),
                **r,
            })

        # Score PM results
        max_pm = max((r.get("_score", r.get("score", 1)) for r in pm), default=1)
        for r in pm:
            raw = r.get("_score", r.get("score", 0))
            normalized = min(raw / max_pm, 1.0) if max_pm > 0 else 0
            unified = normalized * UNIFIED_SCORE_WEIGHTS["pm"]
            scored.append({
                "source": "PM",
                "unified_score": round(unified, 3),
                "raw_score": raw,
                **r,
            })

        # Score Web results (already filtered through worldview filter)
        max_web = max((r.get("score", 1) for r in web), default=1)
        for r in web:
            normalized = min(r.get("score", 0) / max_web, 1.0) if max_web > 0 else 0
            unified = normalized * UNIFIED_SCORE_WEIGHTS["web"]
            scored.append({
                "source": "Web",
                "unified_score": round(unified, 3),
                "raw_score": r.get("score", 0),
                **r,
            })

        # Sort by unified score descending
        scored.sort(key=lambda x: x["unified_score"], reverse=True)
        return scored

    def _format_unified_injection(self, scored_results: list, max_chars: int = 2000) -> str:
        """Format unified results for injection into agent context.

        Groups by source type with clear attribution and file pointers.
        """
        if not scored_results:
            return ""

        parts = []
        current_source = None
        char_count = 0

        for r in scored_results:
            source = r.get("source", "Unknown")
            score = r.get("unified_score", 0)

            # Source header when switching sources
            if source != current_source:
                if current_source is not None:
                    parts.append("")  # Blank line between sources
                source_labels = {
                    "RL": "[From Reference Library]",
                    "PM": "[From Perpetual Memory]",
                    "Web": "[From Web Search]",
                }
                parts.append(f"\n---\n{source_labels.get(source, f'[From {source}]')}")
                current_source = source

            # Format based on source type
            if source == "RL":
                file_path = r.get("file_path", r.get("name", "Unknown"))
                snippet = (r.get("snippet") or "")[:200]
                line = f"[{file_path} (score: {score:.3f})]\n{snippet}"

            elif source == "PM":
                role = r.get("role", "assistant").upper()
                session = r.get("session_id", "")[:12]
                content = (r.get("content") or "")[:200]
                line = f"[{role} | Session {session} (score: {score:.3f})]\n{content}"

            elif source == "Web":
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                content = (r.get("content") or "")[:200]
                flag = ""
                if r.get("worldview_flag"):
                    flag = f" [FLAG: {r.get('flag_reason', 'review')}] "
                line = f"[{title} ({url}) (score: {score:.3f})]{flag}\n{content}"

            else:
                line = str(r)[:200]

            # Check character budget
            if char_count + len(line) > max_chars:
                parts.append(f"\n... [truncated — more results available]")
                break

            parts.append(line)
            char_count += len(line)

        return "\n\n".join(parts)

    @property
    def name(self) -> str:
        return "perpetual_context"

    # -- Core lifecycle ----------------------------------------------------

    def is_available(self) -> bool:
        """Check if perpetual context provider is available.

        Returns True if sqlite3 (always available in Python stdlib) can be imported.
        """
        try:
            import sqlite3  # Always available in Python stdlib
            return True
        except Exception as e:
            logger.error("PerpetualContext availability check failed: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize the perpetual context database.

        Args:
            session_id: The current session ID
            kwargs: Additional initialization parameters including:
                - hermes_home: Path to ~/.hermes directory
                - platform: Platform name (cli, telegram, etc.)
                - agent_context: Context type (primary, subagent, cron)
        """
        self._session_id = session_id

        # Get config — try kwargs first, fall back to reading config.yaml directly
        pc_config = {}
        _cfg_from_kwargs = kwargs.get("config", {})
        if isinstance(_cfg_from_kwargs, dict):
            pc_config = _cfg_from_kwargs.get("perpetual_context", {}) or {}

        # Fallback: read config.yaml directly if not passed via kwargs
        if not pc_config and "hermes_home" in kwargs:
            try:
                import yaml as _yaml
                cfg_path = os.path.join(kwargs["hermes_home"], "config.yaml")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as _f:
                        _full_cfg = _yaml.safe_load(_f) or {}
                    pc_config = (_full_cfg.get("memory", {}) or {}).get("perpetual_context", {}) or {}
            except Exception:
                pass

        # Prefetch injection toggle — disable to stop auto-injecting context into every turn
        self._prefetch_enabled = pc_config.get("prefetch_enabled", True)

        # Determine DB path — always resolve through ~/.hermes/ first,
        # falling back to hermes_home from kwargs (which may be the project dir).
        db_path = pc_config.get("db_path")
        if not db_path:
            # Default: use ~/.hermes/perpetual_context.db regardless of cwd
            home_dir = os.path.expanduser("~/.hermes")
            db_path = os.path.join(home_dir, "perpetual_context.db")
        else:
            # Expand tildes in explicit paths too
            db_path = os.path.expanduser(db_path)

        # Initialize database
        from agent.perpetual_context_db import PerpetualContextDB
        self._db = PerpetualContextDB(db_path=db_path)

        if not self._db.initialize():
            logger.warning("PerpetualContextDB failed to initialize — provider will be read-only")
            return

        # Log session info
        stats = self._db.get_stats()
        logger.info(
            "PerpetualContext initialized: %d messages, %d sessions, %d topics",
            stats.get("message_count", 0),
            stats.get("session_count", 0),
            stats.get("topic_count", 0),
        )

        # Initialize SRP sub-components now that DB is ready
        self._ensure_subcomponents()

    def system_prompt_block(self) -> str:
        """Return text to include in the system prompt."""
        if not self._db or not self._db._initialized:
            return ""

        from datetime import datetime
        stats = self._db.get_stats()
        msg_count = stats.get('message_count', 0)
        session_count = stats.get('session_count', 0)
        current_time = datetime.now().astimezone().strftime('%A, %B %d, %Y %-I:%M %p (%Z)')

        return (
            f"[Current Time: {current_time}]\n"
            f"[Perpetual Context Memory: {msg_count} messages across "
            f"{session_count} sessions, depth={self._current_depth}]\n"
            f"Infinite recall via Perpetual Memory — every turn stored in local SQLite with FTS5. "
            f"Use `perpetual_search` for past conversations, `reference_library_search` for curated knowledge. "
            f"Reference library at `~/.hermes/reference-library/` — read with `read_file`. "
            f"Check RL before answering factual questions; use web search only if RL has no entry."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant context with intelligent source routing + unified scoring.

        Phase 1+2 of Deep Research & Continuity Engine:
        - Classifies query intent (PM vs RL vs Web priority)
        - Conditionally searches each source based on classification
        - Auto-triggers web search when time-sensitive or gaps detected
        - Merges all results with unified relevance scoring
        - Applies worldview filtering on web results

        Returns formatted text to inject as context, grouped by source type.
        """
        # Phase 3: Distillation check runs independently of PM/RL search.
        # It's a file-based queue check that doesn't need DB initialization.
        distill_ptr = self._distill_check(query) if DISTILLATION_ENABLED else None

        if not self._prefetch_enabled:
            return distill_ptr or ""
        if not self._db or not self._db._initialized:
            return distill_ptr or ""

        try:
            effective_session = session_id or self._session_id or ""

            # --- Step 1: Query Classification (intelligent source routing) ---
            priorities = {"pm_priority": 0.5, "rl_priority": 0.5, "web_priority": 0.0}
            if self._classifier:
                try:
                    priorities = self._classifier.classify(query)
                except Exception as e:
                    logger.debug(f"Query classification failed: {e}")

            # --- Step 2: Collect results from each source (defer formatting) ---
            pm_results = []
            rl_results = []
            web_results = []

            # Conditional PM Search (if personal context likely)
            if priorities["pm_priority"] > 0.3:
                try:
                    pm_results = self._db.hybrid_search(
                        query=query,
                        session_id=effective_session if effective_session else None,
                        top_k=self._get_depth_limit(),
                    ) or []
                except Exception as e:
                    logger.debug("Perpetual memory search failed in prefetch: %s", e)

            # Conditional RL Search (if established knowledge likely)
            if priorities["rl_priority"] > 0.3:
                try:
                    if self._ensure_rl_embeddings():
                        rl_results = self._hybrid_rl_search(query, top_k=RL_SEARCH_TOP_K) or []
                    elif DEEP_RESEARCH_ENABLED:
                        # Fallback to legacy reference_library_search tool
                        self._ensure_subcomponents()
                        if self._tools:
                            rl_json = self._tools.handle_reference_library_search({
                                "query": query,
                                "top_k": RL_SEARCH_TOP_K,
                            })
                            rl_data = json.loads(rl_json)
                            legacy_rl_results = rl_data.get("results", [])
                            # Normalize legacy format to match hybrid search output
                            rl_results = [
                                {
                                    "file_path": r.get("name", "Unknown"),
                                    "snippet": (r.get("snippet") or "")[:300],
                                    "score": r.get("score", 0),
                                }
                                for r in legacy_rl_results[:RL_SEARCH_TOP_K]
                            ]
                except Exception as e:
                    logger.debug("Reference library search failed in prefetch: %s", e)

            # --- Step 3: Auto Web Search (Phase 2) ---
            # Trigger when web_priority is high OR local recall has gaps
            total_local = len(pm_results) + len(rl_results)
            needs_web = (
                priorities["web_priority"] > WEB_PRIORITY_THRESHOLD or
                (total_local < GAP_DETECTION_MIN_RESULTS and DEEP_RESEARCH_ENABLED)
            )

            if needs_web:
                web_results = self._auto_web_search(query, top_k=WEB_SEARCH_TOP_K)

            # --- Step 4: Unified Relevance Scoring (Phase 2) ---
            all_scored = self._unified_score_results(
                pm_results=pm_results,
                rl_results=rl_results,
                web_results=web_results,
            )

            if not all_scored:
                return ""

            # --- Step 5: Format unified injection ---
            result_text = self._format_unified_injection(all_scored)

            # --- Step 6: Score retrieval quality (evaluation harness) ---
            self._score_retrieval_quality(
                query=query,
                priorities=priorities,
                scored_results=all_scored,
                formatted_text=result_text,
            )

            # Add source summary footer
            footer_parts = [
                f"[Local Recall: {len(rl_results)} RL, {len(pm_results)} PM, "
                f"{len(web_results)} Web | "
                f"Priorities: PM={priorities['pm_priority']:.2f}, "
                f"RL={priorities['rl_priority']:.2f}, Web={priorities['web_priority']:.2f}]"
            ]

            # Gap detection flag
            if total_local < GAP_DETECTION_MIN_RESULTS and not web_results:
                footer_parts.append(
                    "[⚠ Gap detected — local recall insufficient, web search unavailable. "
                    "Consider manual research for current/external data.]"
                )

            result_text += "\n\n" + " ".join(footer_parts)

            # Append distillation pointer if one was found at the top of prefetch()
            if distill_ptr:
                result_text += distill_ptr

            return result_text

        except Exception as e:
            logger.debug("Perpetual prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the NEXT turn."""
        with self._lock:
            self._prefetch_queue.append({
                "query": query,
                "session_id": session_id or self._session_id or "",
            })

    def recall_past_discussions(
        self, query: str, exclude_session_id: str, max_chars: int = RECALL_OUTPUT_MAX_CHARS
    ) -> str:
        """Recall relevant discussions from past sessions (excluding current).

        Returns a compact pointer string with session/date/score and short
        content snippets. Only fires when FTS5 finds results above the score
        threshold outside the active window.

        Args:
            query: Search text (typically the user message).
            exclude_session_id: Current session ID to skip.
            max_chars: Hard cap on output length (defaults from DB module constant).

        Returns:
            Formatted pointer string, or empty string if nothing relevant found.
        """
        if not self._db or not self._db._initialized:
            return ""
        try:
            return self._db.recall_past_discussions(
                query=query,
                exclude_session_id=exclude_session_id,
                max_chars=max_chars,
            )
        except (sqlite3.Error, KeyError, TypeError) as e:
            logger.exception("recall_past_discussions failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Persist a completed turn to the backend."""
        if not self._db or not self._db._initialized:
            return

        effective_session = session_id or self._session_id or ""

        # Skip non-primary contexts (cron system prompts would corrupt data)
        agent_context = getattr(self, "_agent_context", "primary")
        if agent_context in ("cron", "subagent"):
            return

        try:
            # Store user message
            self._db.add_message(
                session_id=effective_session,
                role="user",
                content=user_content,
                metadata={"synced_at": time.time()},
            )

            # Store assistant message
            self._db.add_message(
                session_id=effective_session,
                role="assistant",
                content=assistant_content,
                metadata={"synced_at": time.time()},
            )

        except Exception as e:
            logger.exception("Perpetual sync_turn failed: %s", e)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for perpetual memory tools."""
        return [
            PERPETUAL_SEARCH_SCHEMA,
            TOPIC_FLOW_SCHEMA,
            CONTEXT_DEPTH_SCHEMA,
            GET_MESSAGES_SCHEMA,
            RECENT_MESSAGES_SCHEMA,
            QUERY_MESSAGES_SCHEMA,
            SMART_RETRIEVE_SCHEMA,
            REFERENCE_LIBRARY_SEARCH_SCHEMA,
            CLASSIFIER_CORRECTION_SCHEMA,
            RETRIEVAL_QUALITY_SCHEMA,
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call — delegate directly to ToolHandler."""
        if not self._db or not self._db._initialized:
            return json.dumps({"error": "Perpetual context database not initialized"})

        self._ensure_subcomponents()
        if not self._tools:
            return json.dumps({"error": "ToolHandler not initialized"})

        try:
            handler_map = {
                "perpetual_search": self._tools.handle_search,
                "topic_flow": self._tools.handle_topic_flow,
                "context_depth": self._tools.handle_context_depth,
                "get_messages": self._tools.handle_get_messages,
                "recent_messages": self._tools.handle_recent_messages,
                "query_messages": self._tools.handle_query_messages,
                "reference_library_search": self._tools.handle_reference_library_search,
            }

            handler = handler_map.get(tool_name)
            if handler:
                return handler(args)

            # smart_retrieve needs special handling (passes self.smart_retrieve as callback)
            if tool_name == "smart_retrieve":
                return self._tools.handle_smart_retrieve(
                    args, smart_retrieve_fn=self.smart_retrieve
                )

            # classifier_correction — log feedback to improve routing
            if tool_name == "classifier_correction":
                return self._handle_classifier_correction(args)

            # retrieval_quality — evaluate prefetch quality
            if tool_name == "retrieval_quality":
                return self._handle_retrieval_quality(args)

            raise NotImplementedError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.exception("Perpetual context tool error (%s)", tool_name)
            return json.dumps({"error": str(e)})

    def _handle_retrieval_quality(self, args: Dict[str, Any]) -> str:
        """Handle retrieval_quality tool call — evaluate prefetch quality."""
        action = args.get("action", "trend")
        window = min(args.get("window", 20), 100)
        threshold = args.get("threshold", 0.3)

        self._ensure_retrieval_scorer()
        if self._retrieval_scorer is None:
            return json.dumps({"error": "Retrieval quality scorer not available"})

        try:
            if action == "trend":
                trend = self._retrieval_scorer.get_trend(window=window)
                return json.dumps({
                    "action": "trend",
                    **trend,
                })

            elif action == "failures":
                failures = self._retrieval_scorer.get_recent_failures(threshold=threshold)
                trend = self._retrieval_scorer.get_trend(window=window)
                return json.dumps({
                    "action": "failures",
                    "trend_summary": {
                        "avg_score": trend["avg_score"],
                        "events_analyzed": trend["events_analyzed"],
                        "trend_direction": trend["trend"],
                    },
                    "recent_failures": failures,
                    "total_failures_found": len(failures),
                })

            else:
                return json.dumps({"error": f"Unknown action '{action}'. Use 'trend' or 'failures'"})

        except Exception as e:
            logger.exception("Retrieval quality evaluation failed")
            return json.dumps({"error": str(e)})

    def _handle_classifier_correction(self, args: Dict[str, Any]) -> str:
        """Handle classifier_correction tool call — log feedback to improve routing."""
        query = args.get("query", "")
        correct_source = args.get("correct_source", "")

        if not query or not correct_source:
            return json.dumps({"error": "Both 'query' and 'correct_source' are required"})

        try:
            from classifier_feedback import FeedbackLoop
            fb = FeedbackLoop()
            fb.log_correction(query, correct_source)
            stats = fb.get_stats()
            return json.dumps({
                "status": "logged",
                "message": f"Correction logged: '{query}' → {correct_source}",
                "total_corrections": stats["total_corrections"],
            })
        except ImportError:
            return json.dumps({"error": "Feedback loop not available"})
        except Exception as e:
            logger.exception("Classifier correction failed")
            return json.dumps({"error": str(e)})

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""
        if self._db and self._db._initialized:
            try:
                self._db.optimize()
            except Exception:
                pass
            self._db.shutdown()

    # -- Optional hooks ----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Called at the start of each turn with the user message."""
        if not self._db or not self._db._initialized:
            return

        # Periodic maintenance every 100 turns (VACUUM is expensive — avoid frequent calls)
        if turn_number % 100 == 0:
            try:
                self._db.optimize()
            except Exception:
                pass

        # Cache latest message for periodic injection lookup later.
        # on_turn_start fires early; actual injection happens in run_agent.py's pre-response hook.
        self._last_turn_number = turn_number
        self._last_user_message = message

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called at end of session — extract topics and build co-occurrence edges."""
        if not self._db or not self._db._initialized:
            return

        try:
            # Collect all unique topics across the last 10 messages
            all_session_topics = set()

            for msg in messages[-10:]:  # Last 10 messages
                content = msg.get("content", "").lower()
                # Topic extraction: capture meaningful technical phrases only.
                # Pattern 1: Capitalized phrases (e.g., "Python Programming", "Docker Networking")
                # Pattern 2: Technical terms with file extensions or dots (e.g., "perpetual_context.db", "run_agent.py")
                # Pattern 3: CamelCase identifiers (e.g., "PerpetualContextDB", "SmartRetriever")
                topics = _re.findall(
                    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'  # Capitalized phrases
                    r'|(?:[a-zA-Z_]+\.(?:py|md|yaml|json|txt|sh))\b'  # File references
                    r'|(?:[A-Z][a-zA-Z]{2,}(?:[A-Z][a-z]+)+)\b',  # CamelCase identifiers
                    msg.get("content", ""),
                )
                # Flatten tuple results (from alternation groups) and filter stopwords
                flat_topics = []
                for t in topics:
                    if isinstance(t, tuple):
                        t = next((g for g in t if g), "")
                    flat_topics.append(t)
                filtered_topics = [t for t in flat_topics if len(t) > 3 and t.lower() not in _STOPWORDS]

                # Add topics to DB and collect unique set for co-occurrence
                for topic in filtered_topics[:3]:  # Max 3 topics per message (after filtering)
                    topic_stripped = topic.strip()
                    self._db.add_topic(
                        session_id=self._session_id or "",
                        topic_name=topic_stripped,
                        confidence=0.6,
                    )
                    all_session_topics.add(topic_stripped.lower())

            # Build co-occurrence edges between all unique topic pairs in this session
            unique_topics = sorted(all_session_topics)  # Sort for deterministic ordering
            for i, t1 in enumerate(unique_topics):
                for t2 in unique_topics[i + 1:]:
                    self._db.increment_relationship(
                        session_id=self._session_id or "",
                        source_entity=t1,
                        target_entity=t2,
                        delta=0.1,  # Small increment — grows over repeated co-occurrence
                    )

        except Exception as e:
            logger.debug("on_session_end extraction failed: %s", e)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Generates a rich retrieval index for the context bridge.

        Delegates to ContextBridgeBuilder which orchestrates extraction and formatting.
        Negative feedback loop: scores quality after building, records results,
        applies corrections if degradation detected.

        Graceful degradation — never breaks compression if something goes wrong.
        """
        self._ensure_subcomponents()
        self._ensure_feedback_loop()

        # Update bridge builder with latest feedback components (may have just initialized)
        if self._bridge_builder:
            self._bridge_builder._scorer = self._scorer
            self._bridge_builder._feedback = self._feedback

        if not self._bridge_builder:
            return ""

        try:
            # Get correction params from feedback state
            correction_params = None
            if self._feedback and self._feedback.needs_correction():
                correction_params = self._feedback.get_correction_params()

            return self._bridge_builder.build_bridge(messages, correction_params)
        except Exception as e:
            # Robust error handling: never break archival due to bridge generation failure
            logger.warning("Context Bridge generation failed: %s", e)
            return "## Context Bridge\n- Error generating retrieval index. See logs for details."

    @staticmethod
    def _classify_query_intent(query_text: str) -> str:
        """Classify query intent for SmartRetriever auto-routing.

        Delegates to the shared implementation in retrieval_engine.py so both
        periodic injection and explicit smart_retrieve calls use identical logic.
        """
        return _classify_query_intent_fn(query_text)

    def _periodic_injection(self, turn_number: int, message: str) -> Optional[str]:
        """Periodic context injection between compressions.

        Fires every PERIODIC_INJECTION_INTERVAL turns to keep the system aware of
        historical context without waiting for token budget exhaustion. Uses auto-routing
        to pick the right retrieval strategy based on query intent.

        Args:
            turn_number: Current turn number in session
            message: Latest user message text

        Returns:
            Compact context string (~300 chars max) or None if no injection needed
        """
        # Check interval — skip if not time for injection
        if turn_number % PERIODIC_INJECTION_INTERVAL != 0:
            return None

        try:
            # Classify intent and retrieve with auto-routing
            query_type = self._classify_query_intent(message)
            results = self.smart_retrieve(query_type, message)

            if not results:
                return None

            # Format as compact pointer — not full content
            parts = []
            for r in results[:2]:  # Limit to 2 most relevant results
                role = r.get("role", "assistant").title()
                session_id = r.get("session_id", "")[:12]
                snippet = (r.get("content") or "")[:PERIODIC_INJECTION_MAX_CHARS // 2].strip()
                parts.append(f"[{role} | Session {session_id}] {snippet}")

            injected_text = "\n".join(parts)

            # Hard cap enforcement — truncate if needed
            if len(injected_text) > PERIODIC_INJECTION_MAX_CHARS:
                injected_text = injected_text[:PERIODIC_INJECTION_MAX_CHARS - 3] + "..."

            return f"\n[Periodic Context Injection]\n{injected_text}\n"

        except Exception as e:
            # Graceful degradation — injection is enhancement, not critical path
            logger.debug("Periodic injection failed (turn %d): %s", turn_number, e)
            return None

    def get_periodic_context(self) -> Optional[str]:
        """Get periodic context injection for pre-response hook.

        Called from run_agent.py alongside recall_past_discussions(). Returns compact
        historical context pointers (~300 chars max) when it's time for injection,
        or None otherwise. Uses auto-routing to pick the right retrieval strategy.

        Returns:
            Context string if injection is due and results found, else None
        """
        turn_number = getattr(self, "_last_turn_number", 0)
        message = getattr(self, "_last_user_message", "")

        if not turn_number or not message:
            return None

        return self._periodic_injection(turn_number, message)

    def smart_retrieve(self, query_type: str, query_text: str) -> List[Dict[str, Any]]:
        """
        Implements adaptive retrieval strategies based on Meta-Harness principles.

        Args:
            query_type: One of 'auto', 'recent', 'topic', 'decision_trace', 'file_history'.
                Use 'auto' to let the system classify intent via keyword heuristics.
            query_text: The search query or context identifier.

        Returns:
            A list of relevant messages or metadata from Perpetual Memory.
        """
        if not self._db or not self._db._initialized:
            return []

        try:
            # Initialize retrieval modules on first use
            if not hasattr(self, '_retriever'):
                from .retrieval_engine import SmartRetriever
                self._retriever = SmartRetriever(self._db)

            if query_type == "recent":
                # Context from last 20 turns — use recent_messages for speed
                return self._retriever.retrieve("recent", query_text)
            
            elif query_type == "topic":
                # Context about a specific topic across all sessions
                return self._retriever.retrieve("topic", query_text)
            
            elif query_type == "decision_trace":
                # Delegate to SmartRetriever (single source of truth for decision trace)
                return self._retriever.retrieve("decision_trace", query_text)
            
            elif query_type == "file_history":
                # All edits to a specific file
                return self._retriever.retrieve("file_history", query_text)
            
            else:
                logger.warning(f"Unknown retrieval type: {query_type}")
                return []
                
        except Exception as e:
            logger.exception("Smart retrieve failed for type '%s'", query_type)
            return []
    
    def on_memory_write(self, action: str, target: str, content: str, metadata=None) -> None:
        """Mirror built-in memory writes to perpetual storage."""
        if not self._db or not self._db._initialized:
            return

        try:
            self._db.add_message(
                session_id="memory_mirror",
                role="system",
                content=f"[{action}] {target}: {content[:500]}",
                metadata={"mirror": True, "original_action": action},
            )
        except Exception as e:
            logger.debug("on_memory_write failed: %s", e)

    # -- Internal helpers --------------------------------------------------

    def _get_depth_limit(self) -> int:
        """Get result limit based on current depth level."""
        self._ensure_subcomponents()
        if not self._tools:
            return 5
        try:
            return self._tools.get_depth_limit()
        except Exception:
            return 5

# -- Plugin registration ---------------------------------------------------

def register(collector):
    """Register this provider with the Hermes plugin system."""
    collector.register_memory_provider(PerpetualContextProvider())

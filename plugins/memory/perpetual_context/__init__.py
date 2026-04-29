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
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        "MANDATORY FIRST STEP for all factual, historical, political, economic,\n"
        "media, and worldview questions. Contains curated, worldview-aligned reference\n"
        "material built from first principles.\n\n"
        "USE THIS BEFORE ANY OTHER SEARCH TOOL when answering:\n"
        "• Questions about history, politics, economics, media bias, or worldview\n"
        "• Any factual claim that needs verification against curated knowledge\n"
        "• Research involving people, organizations, or institutions\n\n"
        "This is NOT optional. Always check reference_library_search before generating answers\n"
        "from training data or session memory alone.\n\n"
        "PARAMETERS:\n"
        "• query: Search text (required)\n"
        "• top_k: Number of results (default 5, max 20)\n\n"
        "EXAMPLES:\n"
        "• 'Elon Musk political influence' — Find entity page with reaction tracking\n"
        "• 'media bias patterns' — Find curated analysis of source credibility\n"
        "• 'American economic policy post-1964' — Find historical reference material"
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
        """Recall relevant context for the upcoming turn.

        Phase 1 of Deep Research & Continuity Engine — Local Recall auto-hook.
        Searches both Reference Library (curated knowledge) and Perpetual Memory
        (historical conversation data), then detects gaps that may need web fallback.

        Returns formatted text to inject as context, with source attribution
        and gap flags for Phase 2 when local recall is insufficient.
        """
        # Respect prefetch_enabled config — return empty if disabled
        if not self._prefetch_enabled:
            return ""
        if not self._db or not self._db._initialized:
            return ""

        try:
            effective_session = session_id or self._session_id or ""
            parts = []
            rl_results_count = 0
            pm_results_count = 0
            gaps_detected = False

            # --- Phase 1a: Reference Library Search (curated knowledge) ---
            if DEEP_RESEARCH_ENABLED:
                try:
                    self._ensure_subcomponents()
                    if self._tools:
                        rl_json = self._tools.handle_reference_library_search({
                            "query": query,
                            "top_k": RL_SEARCH_TOP_K,
                        })
                        rl_data = json.loads(rl_json)
                        rl_results = rl_data.get("results", [])

                        if rl_results:
                            rl_parts = []
                            for r in rl_results[:RL_SEARCH_TOP_K]:
                                name = r.get("name", "Unknown")
                                snippet = r.get("snippet", "")[:300]
                                score = r.get("score", 0)
                                rl_parts.append(
                                    f"[RL: {name} (score: {score})]\n{snippet}"
                                )
                            parts.append("\n\n---\n\n".join(rl_parts))
                            rl_results_count = len(rl_results)

                except Exception as e:
                    logger.debug("Reference library search failed in prefetch: %s", e)

            # --- Phase 1b: Perpetual Memory Hybrid Search (historical context) ---
            try:
                pm_results = self._db.hybrid_search(
                    query=query,
                    session_id=effective_session if effective_session else None,
                    top_k=self._get_depth_limit(),
                )

                if pm_results:
                    pm_formatted = []
                    for msg in pm_results[:self._get_depth_limit()]:
                        role_label = msg["role"].upper()
                        content = msg.get("content", "")[:PREFETCH_TRUNCATION_CHARS]
                        score = msg.get("_score", 0)
                        pm_formatted.append(
                            f"[PM: {role_label} (relevance: {score:.2f})]\n{content}"
                        )
                    parts.append("\n\n---\n\n".join(pm_formatted))
                    pm_results_count = len(pm_results)

            except Exception as e:
                logger.debug("Perpetual memory search failed in prefetch: %s", e)

            # --- Phase 1c: Gap Detection (triggers Phase 2 web fallback flag) ---
            if DEEP_RESEARCH_ENABLED and parts:
                total_results = rl_results_count + pm_results_count

                # Check for gaps: low result count or low confidence scores
                if total_results < GAP_DETECTION_MIN_RESULTS:
                    gaps_detected = True
                else:
                    # Check average score across all results
                    try:
                        all_scores = []
                        if rl_results_count > 0:
                            for r in rl_data.get("results", [])[:RL_SEARCH_TOP_K]:
                                all_scores.append(r.get("score", 0))
                        if pm_results_count > 0:
                            for msg in pm_results[:self._get_depth_limit()]:
                                all_scores.append(msg.get("_score", 0))
                        if all_scores and (sum(all_scores) / len(all_scores)) < GAP_DETECTION_MIN_SCORE:
                            gaps_detected = True
                    except Exception:
                        pass

            # --- Combine and format results ---
            if not parts:
                return ""

            result_text = "\n\n---\n\n".join(parts)

            # Add source summary and gap flag
            footer_parts = []
            if DEEP_RESEARCH_ENABLED:
                footer_parts.append(
                    f"[Local Recall: {rl_results_count} RL results, "
                    f"{pm_results_count} PM results]"
                )
                if gaps_detected:
                    footer_parts.append(
                        "[⚠ Gap detected — local recall insufficient. "
                        "Consider web research for current/external data.]"
                    )

            result_text += "\n\n" + " ".join(footer_parts)

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

            raise NotImplementedError(f"Unknown tool: {tool_name}")

        except Exception as e:
            logger.exception("Perpetual context tool error (%s)", tool_name)
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
            raise RuntimeError(f"Smart retrieve unavailable ({query_type}): {e}") from e
    
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

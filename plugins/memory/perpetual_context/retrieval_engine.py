"""
Smart Retrieval Engine for Perpetual Memory.

Implements adaptive retrieval strategies based on Meta-Harness principles:
- Richer feedback accelerates learning
- Better information structuring beats more data
- Long-horizon dependencies require precise context recovery

This module provides a clean API for the agent to retrieve relevant context 
from Perpetual Memory without relying on blunt keyword searches. It is optimized
for local open-source hardware by prioritizing indexed lookups over broad FTS5
searches and caching results to minimize I/O latency.
"""

import logging
import re as _re
import threading
import time
from typing import Optional, List, Dict, Any, Callable
from agent.perpetual_context_db import PerpetualContextDB

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-routing heuristics for SmartRetriever query classification.
# Simple keyword matching — if it fails 20% of the time, agent can call explicitly.
# Lives here so both SmartRetriever.retrieve() and Provider._classify_query_intent() use one source.
# ---------------------------------------------------------------------------

# Single-word keywords matched via set intersection after split().
AUTO_ROUTING_KEYWORDS = {
    "decision_trace": {"why", "decision", "chose", "reason", "rationale"},
    "file_history": {"file", "edit", "changed"},
    "recent": {"recently", "continue"},
}

# Multi-word phrases matched via substring search — must be checked separately.
AUTO_ROUTING_PHRASES = {
    "decision_trace": {"instead of"},
    "recent": {"pick up", "last time", "what were we doing", "pick up where"},
}

AUTO_ROUTING_FILE_EXTENSIONS = {".py", ".md", ".yaml", ".json", ".txt", ".sh"}


def classify_query_intent(query_text: str) -> str:
    """Classify query intent for SmartRetriever auto-routing.

    Simple keyword heuristics — good enough for routing. If it fails 20% of the time,
    the agent can always call smart_retrieve with explicit query_type.

    Args:
        query_text: The user's search query or message text

    Returns:
        One of 'recent', 'topic', 'decision_trace', 'file_history'
    """
    if not query_text:
        return "topic"  # Default fallback

    lower_query = query_text.lower()
    words = set(lower_query.split())

    # Single-word keyword matching (set intersection)
    for intent, keywords in AUTO_ROUTING_KEYWORDS.items():
        if words & keywords:
            return intent

    # Multi-word phrase matching (substring) — checked after single words
    for intent, phrases in AUTO_ROUTING_PHRASES.items():
        if any(phrase in lower_query for phrase in phrases):
            return intent

    # File extension check — handles cases like "run_agent.py" where ".py" isn't a standalone word
    if any(ext in lower_query for ext in AUTO_ROUTING_FILE_EXTENSIONS):
        return "file_history"

    return "topic"  # Everything else → FTS5 default


class TTLCache:
    """In-memory cache with Time-To-Live (TTL) expiration and bounded size."""

    def __init__(self, ttl_seconds: int = 60, max_size: int = 256):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                else:
                    del self._cache[key]  # Expired
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            # Evict oldest entry (LRU by insertion order) if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            self._cache[key] = (value, time.time())


class SmartRetriever:
    """
    Adaptive retrieval engine for Perpetual Memory.
    
    Uses different strategies based on the type of information needed:
    - Recent context: Fast lookup via turn IDs (O(1))
    - Topic search: FTS5 full-text search across all sessions
    - Decision trace: Find specific decisions and surrounding context
    - File history: Track all edits to a specific file
    
    Optimized for local hardware by caching results and minimizing heavy queries.
    """
    
    def __init__(self, db: PerpetualContextDB):
        self.db = db
        self.cache = TTLCache(ttl_seconds=60)
        # Lazy-init specialized engines on first use
        self._decision_engine: Optional[Any] = None
        self._file_tracker: Optional[Any] = None

    def _get_decision_engine(self):
        if self._decision_engine is None:
            from .decision_trace import DecisionTraceEngine
            self._decision_engine = DecisionTraceEngine(self.db)
        return self._decision_engine

    def _get_file_tracker(self):
        if self._file_tracker is None:
            from .file_history import FileHistoryTracker
            self._file_tracker = FileHistoryTracker(self.db)
        return self._file_tracker

    def retrieve(self, query_type: str, query_text: str) -> List[Dict[str, Any]]:
        """
        Main entry point for smart retrieval.

        Args:
            query_type: One of 'recent', 'topic', 'decision_trace', 'file_history', or 'auto'.
                Use 'auto' to let the system classify intent via keyword heuristics.
            query_text: The search query or context identifier.

        Returns:
            A list of relevant messages or metadata from Perpetual Memory.
        """
        # Auto-routing: classify intent before dispatching
        if query_type == "auto":
            query_type = classify_query_intent(query_text)
            logger.debug("Auto-routed '%s' → %s", query_text[:50], query_type)

        # Check cache first to avoid redundant DB hits
        cache_key = f"{query_type}:{query_text}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        strategies = {
            "recent": self._retrieve_recent,
            "topic": self._retrieve_topic,
            "decision_trace": self._retrieve_decision_trace,
            "file_history": self._retrieve_file_history
        }

        strategy_fn = strategies.get(query_type)
        if not strategy_fn:
            logger.warning(f"Unknown retrieval type: {query_type}")
            return []

        try:
            result = strategy_fn(query_text)
            # Cache the result for future use
            self.cache.set(cache_key, result)
            return result
        except Exception as e:
            logger.exception("Retrieval failed for type '%s'", query_type)
            raise RuntimeError(f"Retrieval unavailable ({query_type}): {e}") from e

    def _retrieve_recent(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve context from the last 20 turns using indexed lookups.
        
        Direct lookup of recent messages — no FTS5 interference. The query_text
        is used as a filter hint only after retrieval to narrow results.
        """
        try:
            # Direct recent lookup — bypasses FTS5 entirely for O(1) turn-ID access
            return self.db.get_recent_messages(n=20)
        except Exception as e:
            logger.exception("Failed to retrieve recent messages")
            raise RuntimeError(f"Recent retrieval unavailable: {e}") from e

    def _retrieve_topic(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve topic-specific context across all sessions using FTS5.
        
        Uses hybrid_search (FTS5 + reciprocal rank fusion) for better relevance
        than raw keyword matching. Limits top_k to reduce I/O load on local hardware.
        """
        try:
            # Use hybrid_search which combines FTS5 BM25 ranking with SQL filtering
            return self.db.hybrid_search(
                query=query_text, session_id=None, top_k=10
            )
        except Exception as e:
            logger.exception("Failed to retrieve topic '%s'", query_text)
            raise RuntimeError(f"Topic retrieval unavailable: {e}") from e

    def _retrieve_decision_trace(self, query_text: str) -> List[Dict[str, Any]]:
        """Find where a decision was made and surrounding context.

        Delegates to DecisionTraceEngine for clean separation of concerns.
        The engine finds the decision turn via FTS5, then retrieves ±5 turns
        around it for full context recovery.
        """
        try:
            engine = self._get_decision_engine()

            # Find the decision point
            decision = engine.find_decision(query_text)
            if not decision or not decision.get('turn_id'):
                return []

            # Get ±5 turn window around the decision
            context = engine.get_decision_context(decision['turn_id'])
            return context
        except Exception as e:
            logger.exception("Failed to retrieve decision trace for '%s'", query_text)
            raise RuntimeError(f"Decision trace unavailable: {e}") from e

    def _retrieve_file_history(self, query_text: str) -> List[Dict[str, Any]]:
        """Retrieve all edits to a specific file using pattern matching.

        Delegates to FileHistoryTracker for clean separation of concerns.
        Searches for tool calls (write_file, patch, read_file) that mention
        the given file path in recent sessions.
        """
        try:
            tracker = self._get_file_tracker()
            return tracker.get_file_history(query_text)
        except Exception as e:
            logger.exception("Failed to retrieve file history for '%s'", query_text)
            raise RuntimeError(f"File history unavailable: {e}") from e

    def _find_decision_turn(self, query_text: str) -> Optional[Dict[str, Any]]:
        """Finds the turn ID where a specific decision was made.
        
        Searches recent sessions using hybrid_search for plain-text keywords.
        FTS5 does not support regex operators — use simple word matching only.
        
        Returns dict with 'turn_id', 'session_id' if found, else None.
        """
        try:
            # Use plain text keywords only — no regex (FTS5 doesn't support *)
            decision_keywords = "decided chose instead architecture plan confirmed"
            search_query = f"{query_text} {decision_keywords}"
            
            # Search recent sessions only (last 5) for efficiency
            results = self.db.hybrid_search(
                query=search_query, session_id=None, top_k=3
            )
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    'turn_id': result.get('id'),
                    'session_id': result.get('session_id'),
                }
        except Exception as e:
            logger.exception("Failed to find decision turn for '%s'", query_text)
        
        # Fallback: search with just the query text
        try:
            results = self.db.hybrid_search(
                query=query_text, session_id=None, top_k=5
            )
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    'turn_id': result.get('id'),
                    'session_id': result.get('session_id'),
                }
        except Exception as e:
            logger.exception("Fallback decision search failed for '%s'", query_text)
        
        return None

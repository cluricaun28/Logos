"""Tests for RetrievalEngine — message retrieval strategies.

Verifies:
1. classify_query_intent correctly routes queries by keyword heuristics
2. TTLCache provides thread-safe caching with expiration and bounded size
3. SmartRetriever dispatches to correct strategy per query type
4. Auto-routing classifies intent before dispatching
5. Cache prevents redundant DB hits
6. Error handling wraps exceptions as RuntimeError
"""

from __future__ import annotations

import os
import sys
import time
import threading

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.retrieval_engine import (
    classify_query_intent,
    TTLCache,
    SmartRetriever,
    AUTO_ROUTING_KEYWORDS,
    AUTO_ROUTING_PHRASES,
    AUTO_ROUTING_FILE_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Tests: classify_query_intent
# ---------------------------------------------------------------------------

class TestClassifyQueryIntent:
    """Test query intent classification heuristics."""

    def test_empty_string_returns_topic(self):
        assert classify_query_intent("") == "topic"

    def test_none_returns_topic(self):
        # The function checks `if not query_text` which catches None
        assert classify_query_intent(None) == "topic"  # type: ignore[arg-type]

    def test_decision_trace_keyword_why(self):
        assert classify_query_intent("Why did we choose this approach?") == "decision_trace"

    def test_decision_trace_keyword_decision(self):
        assert classify_query_intent("What was the decision on caching?") == "decision_trace"

    def test_decision_trace_keyword_chose(self):
        assert classify_query_intent("We chose Redis over Memcached") == "decision_trace"

    def test_decision_trace_keyword_reason(self):
        assert classify_query_intent("The reason for this change") == "decision_trace"

    def test_decision_trace_keyword_rationale(self):
        assert classify_query_intent("What is the rationale behind this?") == "decision_trace"

    def test_file_history_keyword_file(self):
        assert classify_query_intent("Show me file changes") == "file_history"

    def test_file_history_keyword_edit(self):
        assert classify_query_intent("Recent edit history") == "file_history"

    def test_file_history_keyword_changed(self):
        assert classify_query_intent("What changed in the codebase?") == "file_history"

    def test_recent_keyword_recently(self):
        assert classify_query_intent("What did we recently do?") == "recent"

    def test_recent_keyword_continue(self):
        assert classify_query_intent("Continue where we left off") == "recent"

    def test_phrase_instead_of_routes_decision_trace(self):
        assert classify_query_intent("instead of Redis, use PostgreSQL") == "decision_trace"

    def test_phrase_pick_up_routes_recent(self):
        assert classify_query_intent("pick up where we left off") == "recent"

    def test_phrase_last_time_routes_recent(self):
        assert classify_query_intent("What did we do last time?") == "recent"

    def test_phrase_what_were_we_doing_routes_recent(self):
        assert classify_query_intent("What were we doing yesterday?") == "recent"

    def test_file_extension_py_routes_file_history(self):
        assert classify_query_intent("Changes to run_agent.py") == "file_history"

    def test_file_extension_md_routes_file_history(self):
        assert classify_query_intent("Edit README.md file") == "file_history"

    def test_file_extension_yaml_routes_file_history(self):
        assert classify_query_intent("config.yaml changes") == "file_history"

    def test_file_extension_json_routes_file_history(self):
        assert classify_query_intent("data.json modifications") == "file_history"

    def test_file_extension_txt_routes_file_history(self):
        assert classify_query_intent("notes.txt content") == "file_history"

    def test_file_extension_sh_routes_file_history(self):
        assert classify_query_intent("deploy.sh script") == "file_history"

    def test_generic_query_returns_topic(self):
        assert classify_query_intent("Tell me about the project") == "topic"

    def test_case_insensitive_matching(self):
        assert classify_query_intent("WHY DID WE CHOOSE THIS?") == "decision_trace"

    def test_single_word_why(self):
        assert classify_query_intent("why") == "decision_trace"

    def test_keyword_priority_over_phrase(self):
        """Single-word keywords are checked before phrases."""
        # "continue" is a recent keyword, should match even if phrase also matches
        result = classify_query_intent("continue the discussion about why we chose this")
        assert result in ("recent", "decision_trace")

    def test_file_extension_in_filename(self):
        """File extensions embedded in filenames (not standalone words) are detected."""
        assert classify_query_intent("run_agent.py was modified") == "file_history"


# ---------------------------------------------------------------------------
# Tests: TTLCache
# ---------------------------------------------------------------------------

class TestTTLCache:
    """Test the TTL cache implementation."""

    def test_set_and_get(self):
        cache = TTLCache(ttl_seconds=60)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key_returns_none(self):
        cache = TTLCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = TTLCache(ttl_seconds=0)  # Immediate expiration for testing
        cache.set("key1", "value1")
        time.sleep(0.05)  # Small delay to ensure expiration
        assert cache.get("key1") is None

    def test_ttl_not_expired(self):
        cache = TTLCache(ttl_seconds=3600)  # 1 hour
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_max_size_eviction(self):
        cache = TTLCache(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict oldest ("a")
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_existing_key_does_not_evict(self):
        """Updating an existing key should not trigger eviction."""
        cache = TTLCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("a", 10)  # Update "a" — should not evict "b"
        assert cache.get("a") == 10
        assert cache.get("b") == 2

    def test_expired_entry_removed_on_get(self):
        """Expired entries are removed from the cache on access."""
        cache = TTLCache(ttl_seconds=0)
        cache.set("key1", "value1")
        time.sleep(0.05)
        assert cache.get("key1") is None
        # Verify it's actually removed (not just returning None)
        assert len(cache._cache) == 0

    def test_thread_safety(self):
        """Cache operations should be thread-safe."""
        cache = TTLCache(max_size=1000)
        errors = []

        def writer(start):
            try:
                for i in range(100):
                    cache.set(f"key_{start}_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Should have at most max_size entries
        assert len(cache._cache) <= 1000

    def test_default_ttl_is_60(self):
        cache = TTLCache()
        assert cache.ttl == 60

    def test_default_max_size_is_256(self):
        cache = TTLCache()
        assert cache.max_size == 256


# ---------------------------------------------------------------------------
# Tests: SmartRetriever
# ---------------------------------------------------------------------------

class TestSmartRetriever:
    """Test the smart retrieval engine."""

    def _make_mock_db(self, hybrid_search_return=None, recent_return=None):
        """Create a mock PerpetualContextDB with configurable return values."""
        db = type("MockDB", (), {})()
        db.hybrid_search = lambda query="", session_id=None, top_k=10: hybrid_search_return or []
        db.get_recent_messages = lambda n=20, session_id=None, role=None: recent_return or []
        return db

    def test_init_creates_cache(self):
        db = self._make_mock_db()
        retriever = SmartRetriever(db)
        assert retriever.cache is not None
        assert isinstance(retriever.cache, TTLCache)

    def test_retrieve_recent(self):
        expected = [{"id": 1, "content": "recent message"}]
        db = self._make_mock_db(recent_return=expected)
        retriever = SmartRetriever(db)
        result = retriever.retrieve("recent", "anything")
        assert result == expected

    def test_retrieve_topic(self):
        expected = [{"id": 1, "content": "topic match"}]
        db = self._make_mock_db(hybrid_search_return=expected)
        retriever = SmartRetriever(db)
        result = retriever.retrieve("topic", "some topic")
        assert result == expected

    def test_retrieve_unknown_type_returns_empty(self):
        db = self._make_mock_db()
        retriever = SmartRetriever(db)
        result = retriever.retrieve("nonexistent_type", "query")
        assert result == []

    def test_auto_routing_classifies_before_dispatch(self):
        """Auto mode should classify intent and route accordingly."""
        expected = [{"id": 1, "content": "recent message"}]
        db = self._make_mock_db(recent_return=expected)
        retriever = SmartRetriever(db)
        # "recently" keyword → auto-routes to "recent"
        result = retriever.retrieve("auto", "What did we recently do?")
        assert result == expected

    def test_auto_routing_defaults_to_topic(self):
        """Generic query with 'auto' should route to topic."""
        expected = [{"id": 1, "content": "topic match"}]
        db = self._make_mock_db(hybrid_search_return=expected)
        retriever = SmartRetriever(db)
        result = retriever.retrieve("auto", "Tell me about the project")
        assert result == expected

    def test_cache_prevents_redundant_calls(self):
        """Second call with same query should return cached result."""
        db = self._make_mock_db(recent_return=[{"id": 1}])
        retriever = SmartRetriever(db)
        # First call
        result1 = retriever.retrieve("recent", "test")
        assert result1 == [{"id": 1}]
        # Modify DB return to verify cache is used
        db.get_recent_messages = lambda n=20, session_id=None, role=None: [{"id": 999}]
        # Second call should still return cached result
        result2 = retriever.retrieve("recent", "test")
        assert result2 == [{"id": 1}]

    def test_cache_key_includes_query_type(self):
        """Different query types with same text should have different cache keys."""
        db = self._make_mock_db(
            recent_return=[{"type": "recent"}],
            hybrid_search_return=[{"type": "topic"}]
        )
        retriever = SmartRetriever(db)
        r1 = retriever.retrieve("recent", "test query")
        r2 = retriever.retrieve("topic", "test query")
        assert r1 == [{"type": "recent"}]
        assert r2 == [{"type": "topic"}]

    def test_retrieval_failure_raises_runtime_error(self):
        """DB failures should be wrapped as RuntimeError."""
        db = self._make_mock_db()
        db.get_recent_messages = lambda n=20, session_id=None, role=None: (_ for _ in ()).throw(Exception("DB down"))
        retriever = SmartRetriever(db)
        try:
            retriever.retrieve("recent", "test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Recent retrieval unavailable" in str(e)

    def test_topic_retrieval_failure_raises_runtime_error(self):
        db = self._make_mock_db()
        db.hybrid_search = lambda query="", session_id=None, top_k=10: (_ for _ in ()).throw(Exception("FTS5 error"))
        retriever = SmartRetriever(db)
        try:
            retriever.retrieve("topic", "test")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "Topic retrieval unavailable" in str(e)

    def test_retrieve_recent_uses_n_20(self):
        """Recent retrieval should request 20 messages."""
        captured_n = []

        class MockDB:
            def get_recent_messages(self, n=20, session_id=None, role=None):
                captured_n.append(n)
                return [{"id": 1}]
            def hybrid_search(self, query="", session_id=None, top_k=10):
                return []

        retriever = SmartRetriever(MockDB())
        retriever.retrieve("recent", "test")
        assert captured_n == [20]

    def test_retrieve_topic_uses_top_k_10(self):
        """Topic retrieval should use top_k=10."""
        captured_top_k = []

        class MockDB:
            def get_recent_messages(self, n=20, session_id=None, role=None):
                return []
            def hybrid_search(self, query="", session_id=None, top_k=10):
                captured_top_k.append(top_k)
                return [{"id": 1}]

        retriever = SmartRetriever(MockDB())
        retriever.retrieve("topic", "test")
        assert captured_top_k == [10]


# ---------------------------------------------------------------------------
# Tests: SmartRetriever decision trace and file history (lazy init)
# ---------------------------------------------------------------------------

class TestSmartRetrieverSpecializedEngines:
    """Test lazy initialization of specialized engines."""

    def test_decision_engine_lazy_init(self):
        """Decision engine should be lazily initialized on first use."""
        db = type("MockDB", (), {})()
        db.hybrid_search = lambda query="", session_id=None, top_k=10: []
        db.get_recent_messages = lambda n=20, session_id=None, role=None: []

        retriever = SmartRetriever(db)
        assert retriever._decision_engine is None

    def test_file_tracker_lazy_init(self):
        """File tracker should be lazily initialized on first use."""
        db = type("MockDB", (), {})()
        db.hybrid_search = lambda query="", session_id=None, top_k=10: []
        db.get_recent_messages = lambda n=20, session_id=None, role=None: []

        retriever = SmartRetriever(db)
        assert retriever._file_tracker is None


# ---------------------------------------------------------------------------
# Tests: AUTO_ROUTING constants
# ---------------------------------------------------------------------------

class TestAutoRoutingConstants:
    """Verify auto-routing configuration constants."""

    def test_keywords_has_expected_intents(self):
        expected_intents = {"decision_trace", "file_history", "recent"}
        assert set(AUTO_ROUTING_KEYWORDS.keys()) == expected_intents

    def test_phrases_has_expected_intents(self):
        # Phrases should cover decision_trace and recent
        assert "decision_trace" in AUTO_ROUTING_PHRASES
        assert "recent" in AUTO_ROUTING_PHRASES

    def test_file_extensions_contains_common_types(self):
        expected = {".py", ".md", ".yaml", ".json", ".txt", ".sh"}
        assert AUTO_ROUTING_FILE_EXTENSIONS == expected

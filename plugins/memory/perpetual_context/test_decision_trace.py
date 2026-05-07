#!/usr/bin/env python3
"""Tests for decision_trace.py — Decision Trace Engine.

Verifies:
1. find_decision() returns correct structure when results found
2. find_decision() returns None on empty results
3. find_decision() handles DB exceptions gracefully
4. get_decision_context() returns surrounding turns
5. get_decision_context() handles missing turn_id
6. get_decision_context() clamps to session bounds

Run:
    python -m pytest test_decision_trace.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.decision_trace import DecisionTraceEngine


class TestFindDecision(unittest.TestCase):
    """Test find_decision() method."""

    def setUp(self):
        self.db = MagicMock()
        self.engine = DecisionTraceEngine(db=self.db)

    def test_returns_result_when_found(self):
        self.db.hybrid_search.return_value = [
            {"id": 42, "session_id": "test_session", "content": "We decided to use SQLite"}
        ]
        result = self.engine.find_decision("database choice")
        self.assertIsNotNone(result)
        self.assertEqual(result["turn_id"], 42)
        self.assertEqual(result["session_id"], "test_session")

    def test_context_is_truncated(self):
        long_content = "A" * 500
        self.db.hybrid_search.return_value = [
            {"id": 1, "session_id": "s", "content": long_content}
        ]
        result = self.engine.find_decision("query")
        self.assertLessEqual(len(result["context"]), 200)

    def test_returns_none_when_no_results(self):
        self.db.hybrid_search.return_value = []
        result = self.engine.find_decision("nonexistent decision")
        self.assertIsNone(result)

    def test_returns_none_on_exception(self):
        self.db.hybrid_search.side_effect = Exception("DB error")
        result = self.engine.find_decision("query")
        self.assertIsNone(result)

    def test_calls_hybrid_search_with_correct_params(self):
        self.db.hybrid_search.return_value = []
        self.engine.find_decision("my query")
        self.db.hybrid_search.assert_called_once_with(
            query="my query", session_id=None, top_k=3
        )


class TestGetDecisionContext(unittest.TestCase):
    """Test get_decision_context() method."""

    def setUp(self):
        self.db = MagicMock()
        self.engine = DecisionTraceEngine(db=self.db)

    def test_returns_empty_when_turn_not_found(self):
        self.db._conn.execute.return_value.fetchone.return_value = None
        result = self.engine.get_decision_context(9999)
        self.assertEqual(result, [])

    def test_returns_messages_around_decision(self):
        # Mock the session lookup
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("test_session",)
        self.db._conn.execute.return_value = mock_cursor

        # Create messages with IDs 1-20 in same session
        messages = [
            {"id": i, "session_id": "test_session", "content": f"msg {i}"}
            for i in range(1, 21)
        ]
        self.db.query_messages.return_value = {"results": messages}

        # Get context around turn 10 — should get turns 5-15 (±5)
        result = self.engine.get_decision_context(10)
        self.assertEqual(len(result), 11)  # 5 before + decision + 5 after
        self.assertEqual(result[0]["id"], 5)
        self.assertEqual(result[-1]["id"], 15)

    def test_clamps_to_start_of_session(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("test_session",)
        self.db._conn.execute.return_value = mock_cursor

        messages = [
            {"id": i, "session_id": "test_session", "content": f"msg {i}"}
            for i in range(1, 10)
        ]
        self.db.query_messages.return_value = {"results": messages}

        # Decision at turn 3 — should get turns 1-8 (clamped to start)
        result = self.engine.get_decision_context(3)
        self.assertEqual(result[0]["id"], 1)

    def test_clamps_to_end_of_session(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("test_session",)
        self.db._conn.execute.return_value = mock_cursor

        messages = [
            {"id": i, "session_id": "test_session", "content": f"msg {i}"}
            for i in range(1, 10)
        ]
        self.db.query_messages.return_value = {"results": messages}

        # Decision at turn 8 — should get turns 3-9 (clamped to end)
        result = self.engine.get_decision_context(8)
        self.assertEqual(result[-1]["id"], 9)

    def test_returns_empty_when_turn_not_in_session_messages(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("test_session",)
        self.db._conn.execute.return_value = mock_cursor

        # Messages exist but none match turn_id=999
        messages = [
            {"id": i, "session_id": "test_session", "content": f"msg {i}"}
            for i in range(1, 5)
        ]
        self.db.query_messages.return_value = {"results": messages}

        result = self.engine.get_decision_context(999)
        self.assertEqual(result, [])

    def test_returns_empty_on_exception(self):
        self.db._conn.execute.side_effect = Exception("DB error")
        result = self.engine.get_decision_context(1)
        self.assertEqual(result, [])

    def test_filters_to_correct_session(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("my_session",)
        self.db._conn.execute.return_value = mock_cursor

        # Mix of sessions — only my_session messages should be returned
        messages = [
            {"id": 1, "session_id": "other_session", "content": "wrong"},
            {"id": 2, "session_id": "my_session", "content": "right"},
            {"id": 3, "session_id": "my_session", "content": "also right"},
        ]
        self.db.query_messages.return_value = {"results": messages}

        result = self.engine.get_decision_context(2)
        # Should only contain my_session messages
        for msg in result:
            self.assertEqual(msg["session_id"], "my_session")

    def test_handles_non_dict_query_result(self):
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = ("test_session",)
        self.db._conn.execute.return_value = mock_cursor

        # query_messages returns something that's not a dict
        self.db.query_messages.return_value = "not a dict"
        result = self.engine.get_decision_context(1)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

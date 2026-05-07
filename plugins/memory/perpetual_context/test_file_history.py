#!/usr/bin/env python3
"""Tests for file_history.py — File History Tracker.

Verifies:
1. get_file_history() returns correct structure with turn references
2. get_file_history() handles empty results
3. get_file_history() handles DB exceptions gracefully

NOTE: get_recent_edits() was removed as dead code (never called outside tests,
never used by retrieval_engine.py or any production code).

Run:
    python -m pytest test_file_history.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.file_history import FileHistoryTracker


class TestGetFileHistory(unittest.TestCase):
    """Test get_file_history() method."""

    def setUp(self):
        self.db = MagicMock()
        self.tracker = FileHistoryTracker(db=self.db)

    def test_returns_history_with_turn_references(self):
        self.db.query_messages.return_value = {
            "results": [
                {"id": 1, "session_id": "s1", "content": "wrote file.py"},
                {"id": 2, "session_id": "s1", "content": "patched file.py again"},
            ]
        }
        result = self.tracker.get_file_history("/path/to/file.py")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["turn_id"], 1)
        self.assertEqual(result[0]["session_id"], "s1")

    def test_content_is_truncated(self):
        long_content = "A" * 500
        self.db.query_messages.return_value = {
            "results": [{"id": 1, "session_id": "s", "content": long_content}]
        }
        result = self.tracker.get_file_history("/path.py")
        self.assertLessEqual(len(result[0]["content"]), 200)

    def test_returns_empty_on_no_results(self):
        self.db.query_messages.return_value = {"results": []}
        result = self.tracker.get_file_history("/nonexistent.py")
        self.assertEqual(result, [])

    def test_returns_empty_on_exception(self):
        self.db.query_messages.side_effect = Exception("DB error")
        result = self.tracker.get_file_history("/path.py")
        self.assertEqual(result, [])

    def test_calls_query_with_correct_params(self):
        self.db.query_messages.return_value = {"results": []}
        self.tracker.get_file_history("/my/file.py")
        call_args = self.db.query_messages.call_args
        # Should use pattern with file path and role=tool
        self.assertIn("file.py", call_args[1]["pattern"])
        self.assertEqual(call_args[1]["role"], "tool")
        self.assertEqual(call_args[1]["limit"], 50)

    def test_handles_non_dict_result(self):
        self.db.query_messages.return_value = "not a dict"
        result = self.tracker.get_file_history("/path.py")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

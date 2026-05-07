#!/usr/bin/env python3
"""Tests for extraction_engine.py — Extraction Engine.

Verifies:
1. extract_active_tasks() extracts tasks from user messages with decisions
2. extract_active_tasks() deduplicates by first line
3. extract_active_tasks() limits to 5 most recent
4. extract_file_edits() extracts from structured tool_calls
5. extract_file_edits() extracts from text patterns
6. extract_file_edits() finds related discussions
7. find_related_discussions() scans window around edit turn
8. extract_known_errors() extracts exception types with fix locations
9. extract_known_errors() deduplicates errors
10. extract_knowledge_gaps() detects explicit markers, confidence scores, pending tags
11. All methods return empty list for empty messages

Run:
    python -m pytest test_extraction_engine.py -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.extraction_engine import (
    _STOPWORDS,
    ExtractionEngine,
)


class TestConstants(unittest.TestCase):
    """Verify module-level constants."""

    def test_stopwords_is_frozenset(self):
        self.assertIsInstance(_STOPWORDS, frozenset)

    def test_common_stopwords_present(self):
        for word in ['the', 'and', 'is', 'it', 'of', 'to']:
            self.assertIn(word, _STOPWORDS)


class TestExtractActiveTasks(unittest.TestCase):
    """Test extract_active_tasks() method."""

    def setUp(self):
        self.engine = ExtractionEngine()

    def test_returns_empty_for_no_messages(self):
        result = self.engine.extract_active_tasks([])
        self.assertEqual(result, [])

    def test_extracts_task_from_user_message(self):
        messages = [{"role": "user", "content": "Please fix the login bug"}]
        result = self.engine.extract_active_tasks(messages)
        self.assertEqual(len(result), 1)
        self.assertIn("fix", result[0]["summary"].lower())

    def test_ignores_non_user_messages(self):
        messages = [{"role": "assistant", "content": "I'll fix the bug"}]
        result = self.engine.extract_active_tasks(messages)
        self.assertEqual(result, [])

    def test_deduplicates_by_first_line(self):
        messages = [
            {"role": "user", "content": "Fix login bug\nmore details"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Fix login bug\nfollow up"},
        ]
        result = self.engine.extract_active_tasks(messages)
        self.assertEqual(len(result), 1)
        # Should have both turn IDs
        self.assertIn(0, result[0]["turn_ids"])
        self.assertIn(2, result[0]["turn_ids"])

    def test_limits_to_five_most_recent(self):
        messages = [
            {"role": "user", "content": f"Fix bug number {i}"} for i in range(10)
        ]
        result = self.engine.extract_active_tasks(messages)
        self.assertLessEqual(len(result), 5)

    def test_includes_decisions_from_assistant(self):
        messages = [
            {"role": "user", "content": "Implement database choice"},
            {"role": "assistant", "content": "We decided to use SQLite for simplicity"},
        ]
        result = self.engine.extract_active_tasks(messages)
        self.assertEqual(len(result), 1)
        # Should have extracted the decision
        decisions = result[0].get("decisions", [])
        self.assertGreater(len(decisions), 0)

    def test_description_is_truncated(self):
        long_content = "Implement feature: " + "A" * 500
        messages = [{"role": "user", "content": long_content}]
        result = self.engine.extract_active_tasks(messages)
        self.assertLessEqual(len(result[0]["description"]), 300)

    def test_returns_most_recent_first(self):
        messages = [
            {"role": "user", "content": "Fix bug A"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Fix bug B"},
        ]
        result = self.engine.extract_active_tasks(messages)
        # Most recent should be first
        self.assertIn("B", result[0]["summary"])


class TestExtractFileEdits(unittest.TestCase):
    """Test extract_file_edits() method."""

    def setUp(self):
        self.engine = ExtractionEngine()

    def test_returns_empty_for_no_messages(self):
        result = self.engine.extract_file_edits([])
        self.assertEqual(result, [])

    def test_extracts_from_structured_tool_call(self):
        args = json.dumps({"path": "/home/user/test.py"})
        messages = [{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "write_file", "arguments": args}}],
        }]
        result = self.engine.extract_file_edits(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["path"], "/home/user/test.py")

    def test_extracts_from_text_pattern(self):
        messages = [{"role": "tool", "content": "wrote to /path/to/config.yaml"}]
        result = self.engine.extract_file_edits(messages)
        self.assertGreater(len(result), 0)
        paths = [e["path"] for e in result]
        self.assertTrue(any("config.yaml" in p for p in paths))

    def test_extracts_patch_operations(self):
        args = json.dumps({"path": "/app/main.py"})
        messages = [{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "patch", "arguments": args}}],
        }]
        result = self.engine.extract_file_edits(messages)
        self.assertEqual(len(result), 1)

    def test_limits_to_ten(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"path": f"/file{i}.py"})}}],
            }
            for i in range(15)
        ]
        result = self.engine.extract_file_edits(messages)
        self.assertLessEqual(len(result), 10)

    def test_deduplicates_same_path(self):
        args1 = json.dumps({"path": "/test.py"})
        args2 = json.dumps({"path": "/TEST.PY"})  # Same path, different case
        messages = [
            {"role": "assistant", "tool_calls": [{"function": {"name": "write_file", "arguments": args1}}]},
            {"role": "assistant", "tool_calls": [{"function": {"name": "patch", "arguments": args2}}]},
        ]
        result = self.engine.extract_file_edits(messages)
        # Should be deduplicated (case-insensitive)
        self.assertEqual(len(result), 1)

    def test_includes_related_turns(self):
        messages = [
            {"role": "user", "content": "Edit test.py please"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "write_file", "arguments": json.dumps({"path": "/test.py"})}}],
            },
        ]
        result = self.engine.extract_file_edits(messages)
        self.assertIn("related_turns", result[0])


class TestFindRelatedDiscussions(unittest.TestCase):
    """Test find_related_discussions() helper."""

    def setUp(self):
        self.engine = ExtractionEngine()

    def test_finds_mentions_in_window(self):
        messages = [
            {"role": "user", "content": "Let's edit auth.py"},
            {"role": "assistant", "content": "OK editing auth.py now"},
            {"role": "tool", "content": "wrote to /path/auth.py"},
        ]
        turns, desc = self.engine.find_related_discussions(messages, 2, "/path/auth.py")
        # Should find turn 0 and 1 mentioning auth.py
        self.assertGreater(len(turns), 0)

    def test_skips_edit_turn_itself(self):
        messages = [
            {"role": "tool", "content": "wrote to /test.py"},
        ]
        turns, desc = self.engine.find_related_discussions(messages, 0, "/test.py")
        # Should skip turn 0 (the edit itself)
        self.assertEqual(len(turns), 0)

    def test_respects_window_size(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        # Insert a mention far outside window
        messages[25]["content"] = "edit auth.py here"
        turns, desc = self.engine.find_related_discussions(messages, 0, "/auth.py", window=5)
        # Turn 25 is outside window of 5 from turn 0
        self.assertEqual(len(turns), 0)

    def test_handles_invalid_file_path(self):
        messages = [{"role": "user", "content": "test"}]
        turns, desc = self.engine.find_related_discussions(messages, 0, None)
        self.assertEqual(turns, [])


class TestExtractKnownErrors(unittest.TestCase):
    """Test extract_known_errors() method."""

    def setUp(self):
        self.engine = ExtractionEngine()

    def test_returns_empty_for_no_messages(self):
        result = self.engine.extract_known_errors([])
        self.assertEqual(result, [])

    def test_extracts_type_error(self):
        messages = [{"role": "tool", "content": "TypeError: 'NoneType' object has no attribute 'x'"}]
        result = self.engine.extract_known_errors(messages)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["summary"].startswith("TypeError"))

    def test_extracts_value_error(self):
        messages = [{"role": "tool", "content": "ValueError: invalid literal for int()"}]
        result = self.engine.extract_known_errors(messages)
        self.assertEqual(len(result), 1)

    def test_finds_fix_location(self):
        messages = [{
            "role": "assistant",
            "content": "TypeError: x is None. Fixed in /path/auth.py at line 42"
        }]
        result = self.engine.extract_known_errors(messages)
        self.assertEqual(len(result), 1)
        # Should have found fix location
        self.assertNotEqual(result[0]["fix_location"], "N/A")

    def test_deduplicates_same_error(self):
        messages = [
            {"role": "tool", "content": "TypeError: 'NoneType' object has no attribute"},
            {"role": "tool", "content": "TypeError: 'NoneType' object has no attribute"},
        ]
        result = self.engine.extract_known_errors(messages)
        # Should be deduplicated (same type + similar message)
        self.assertEqual(len(result), 1)

    def test_limits_to_five(self):
        error_types = [
            "TypeError", "ValueError", "KeyError", "RuntimeError",
            "OSError", "IndexError", "AttributeError",
        ]
        messages = [{"role": "tool", "content": f"{et}: unique_msg_{i}"} for i, et in enumerate(error_types)]
        result = self.engine.extract_known_errors(messages)
        self.assertLessEqual(len(result), 5)

    def test_no_errors_in_clean_message(self):
        messages = [{"role": "assistant", "content": "Everything works fine."}]
        result = self.engine.extract_known_errors(messages)
        self.assertEqual(result, [])


class TestExtractKnowledgeGaps(unittest.TestCase):
    """Test extract_knowledge_gaps() method."""

    def setUp(self):
        self.engine = ExtractionEngine()

    def test_returns_empty_for_no_messages(self):
        result = self.engine.extract_knowledge_gaps([])
        self.assertEqual(result, [])

    def test_extracts_explicit_knowledge_gap(self):
        messages = [{"role": "assistant", "content": "knowledge gap: Docker networking"}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(len(result), 1)
        self.assertIn("Docker", result[0]["summary"])

    def test_extracts_rl_entry_needed(self):
        messages = [{"role": "assistant", "content": "RL entry needed: CUDA setup"}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(len(result), 1)

    def test_extracts_gap_marker(self):
        messages = [{"role": "assistant", "content": "[gap] SSL configuration steps"}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(len(result), 1)

    def test_extracts_confidence_score_pattern(self):
        messages = [{"role": "assistant", "content": "confidence: 0.3 — nuclear ethics principles"}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["confidence"], 0.3)

    def test_extracts_pending_tag(self):
        messages = [{"role": "assistant", "content": "[pending] Research on Tailscale DNS"}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(len(result), 1)

    def test_extracts_needs_research_tag(self):
        messages = [{"role": "assistant", "content": "[needs research] GPU memory optimization"}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(len(result), 1)

    def test_limits_to_five(self):
        messages = [
            {"role": "assistant", "content": f"[pending] Gap number {i} unique topic"}
            for i in range(10)
        ]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertLessEqual(len(result), 5)

    def test_ignores_confidence_outside_range(self):
        messages = [{"role": "assistant", "content": "confidence: 1.5 — invalid score"}]
        result = self.engine.extract_knowledge_gaps(messages)
        # Should be ignored (confidence > 1.0)
        self.assertEqual(result, [])

    def test_no_gaps_in_normal_text(self):
        messages = [{"role": "assistant", "content": "The code is working correctly."}]
        result = self.engine.extract_knowledge_gaps(messages)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

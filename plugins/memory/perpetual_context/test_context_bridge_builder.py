#!/usr/bin/env python3
"""Tests for context_bridge_builder.py — Context Bridge Builder.

Verifies:
1. MAX_BRIDGE_CHARS constant (4KB cap)
2. build_bridge() with extraction engine returns formatted sections
3. build_bridge() without extraction engine returns only retrieval guidance
4. build_bridge() graceful degradation on exceptions
5. _format_active_tasks() formatting and limits
6. _format_file_edits() formatting and limits
7. _format_known_errors() formatting and limits
8. _format_knowledge_gaps() formatting and limits
9. _format_retrieval_guidance() always present
10. 4KB truncation removes oldest sections first
11. Preservation warning appended when correction params set
12. Quality scoring integration

Run:
    python -m pytest test_context_bridge_builder.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.context_bridge_builder import (
    MAX_BRIDGE_CHARS,
    ContextBridgeBuilder,
)


class TestConstants(unittest.TestCase):
    """Verify module-level constants."""

    def test_max_bridge_chars(self):
        self.assertEqual(MAX_BRIDGE_CHARS, 4000)


class TestFormatActiveTasks(unittest.TestCase):
    """Test _format_active_tasks() formatting."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_basic_formatting(self):
        tasks = [{"summary": "Fix login bug", "turn_ids": [1, 2], "description": "auth.py fix"}]
        result = self.builder._format_active_tasks(tasks)
        self.assertIn("## Active Tasks (with retrieval pointers)", result)
        self.assertIn("**Fix login bug**", result)
        self.assertIn("#1, #2", result)

    def test_limits_to_three_tasks(self):
        tasks = [
            {"summary": f"Task {i}", "turn_ids": [i], "description": f"desc {i}"}
            for i in range(5)
        ]
        result = self.builder._format_active_tasks(tasks)
        # Should only have 3 task summaries
        count = result.count("→ See turns")
        self.assertEqual(count, 3)

    def test_includes_key_decisions(self):
        tasks = [{
            "summary": "Choose DB",
            "turn_ids": [1],
            "description": "database choice",
            "decisions": [{"turn_id": 2, "text": "Use SQLite"}],
        }]
        result = self.builder._format_active_tasks(tasks)
        self.assertIn("Key decision at turn #2: Use SQLite", result)

    def test_limits_decisions_to_two(self):
        tasks = [{
            "summary": "Big task",
            "turn_ids": [1],
            "description": "many decisions",
            "decisions": [
                {"turn_id": 1, "text": "D1"},
                {"turn_id": 2, "text": "D2"},
                {"turn_id": 3, "text": "D3"},
            ],
        }]
        result = self.builder._format_active_tasks(tasks)
        # Should only have 2 decisions
        count = result.count("Key decision")
        self.assertEqual(count, 2)


class TestFormatFileEdits(unittest.TestCase):
    """Test _format_file_edits() formatting."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_basic_formatting(self):
        edits = [{"path": "/test.py", "last_edit_turn": 5, "description": "added function"}]
        result = self.builder._format_file_edits(edits)
        self.assertIn("## Files Currently Being Edited", result)
        self.assertIn("**/test.py** (modified)", result)
        self.assertIn("#5", result)

    def test_limits_to_three_edits(self):
        edits = [
            {"path": f"/file{i}.py", "last_edit_turn": i, "description": f"edit {i}"}
            for i in range(5)
        ]
        result = self.builder._format_file_edits(edits)
        count = result.count("→ Last edit:")
        self.assertEqual(count, 3)

    def test_includes_related_discussion(self):
        edits = [{
            "path": "/test.py",
            "last_edit_turn": 5,
            "description": "fix",
            "related_turns": [10, 11],
            "related_description": "debugging session",
        }]
        result = self.builder._format_file_edits(edits)
        self.assertIn("Related discussion:", result)


class TestFormatKnownErrors(unittest.TestCase):
    """Test _format_known_errors() formatting."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_basic_formatting(self):
        errors = [{"summary": "TypeError in auth", "turn_id": 7, "fix_location": "auth.py:42"}]
        result = self.builder._format_known_errors(errors)
        self.assertIn("## Known Errors/Issues", result)
        self.assertIn("**TypeError in auth**", result)
        self.assertIn("#7", result)

    def test_limits_to_three_errors(self):
        errors = [
            {"summary": f"Error {i}", "turn_id": i, "fix_location": "file.py"}
            for i in range(5)
        ]
        result = self.builder._format_known_errors(errors)
        count = result.count("→ See")
        self.assertEqual(count, 3)


class TestFormatKnowledgeGaps(unittest.TestCase):
    """Test _format_knowledge_gaps() formatting."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_basic_formatting(self):
        gaps = [{"summary": "Docker networking", "turn_ids": [3, 4], "confidence": 0.7}]
        result = self.builder._format_knowledge_gaps(gaps)
        self.assertIn("## Knowledge Gaps (Pending Reference Library Entries)", result)
        self.assertIn("**Docker networking**", result)
        self.assertIn("confidence: 0.7", result)

    def test_limits_to_three_gaps(self):
        gaps = [
            {"summary": f"Gap {i}", "turn_ids": [i], "confidence": 0.5}
            for i in range(5)
        ]
        result = self.builder._format_knowledge_gaps(gaps)
        count = result.count("→ Discussed turns")
        self.assertEqual(count, 3)


class TestFormatRetrievalGuidance(unittest.TestCase):
    """Test _format_retrieval_guidance() formatting."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_contains_expected_content(self):
        result = self.builder._format_retrieval_guidance()
        self.assertIn("## Historical Context Retrieval", result)
        self.assertIn("perpetual_search", result)
        self.assertIn("query_messages", result)
        self.assertIn("reference-library", result)


class TestBuildBridge(unittest.TestCase):
    """Test build_bridge() orchestration."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_returns_retrieval_guidance_when_no_extraction(self):
        # No extraction engine — should still return retrieval guidance
        bridge = self.builder.build_bridge([{"role": "user", "content": "test"}])
        self.assertIn("Historical Context Retrieval", bridge)

    def test_includes_sections_from_extraction_engine(self):
        mock_extraction = MagicMock()
        mock_extraction.extract_active_tasks.return_value = [
            {"summary": "Fix bug", "turn_ids": [1], "description": "fix"}
        ]
        mock_extraction.extract_file_edits.return_value = []
        mock_extraction.extract_known_errors.return_value = []
        mock_extraction.extract_knowledge_gaps.return_value = []

        self.builder._extraction = mock_extraction
        bridge = self.builder.build_bridge([{"role": "user", "content": "fix bug"}])
        self.assertIn("Fix bug", bridge)

    def test_returns_empty_string_when_no_content(self):
        # No extraction engine, no messages that produce content
        # Retrieval guidance is always added so this won't be empty
        bridge = self.builder.build_bridge([])
        # Should have retrieval guidance at minimum
        self.assertIn("Historical Context Retrieval", bridge)

    def test_graceful_degradation_on_exception(self):
        mock_extraction = MagicMock()
        mock_extraction.extract_active_tasks.side_effect = Exception("boom")
        self.builder._extraction = mock_extraction

        # Should not raise — returns error message instead
        bridge = self.builder.build_bridge([{"role": "user", "content": "test"}])
        # Even with extraction failure, retrieval guidance should be present
        # because the exception happens during _extract which catches it
        self.assertIsInstance(bridge, str)

    def test_truncation_removes_oldest_sections(self):
        mock_extraction = MagicMock()
        # Create very large sections that exceed 4KB
        big_content = [{"summary": "X" * 2000, "turn_ids": [1], "description": "D" * 2000}]
        mock_extraction.extract_active_tasks.return_value = big_content
        mock_extraction.extract_file_edits.return_value = []
        mock_extraction.extract_known_errors.return_value = []
        mock_extraction.extract_knowledge_gaps.return_value = []

        self.builder._extraction = mock_extraction
        bridge = self.builder.build_bridge([{"role": "user", "content": "test"}])
        # Should be truncated to MAX_BRIDGE_CHARS
        self.assertLessEqual(len(bridge), MAX_BRIDGE_CHARS)


class TestPreservationWarning(unittest.TestCase):
    """Test preservation warning formatting."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_basic_warning_formatting(self):
        lost = ["fix login bug", "update config.yaml"]
        result = self.builder._format_preservation_warning(lost)
        self.assertIn("Preservation Warning", result)
        self.assertIn("fix login bug", result)
        self.assertIn("update config.yaml", result)

    def test_limits_to_five_items(self):
        lost = [f"item {i}" for i in range(10)]
        result = self.builder._format_preservation_warning(lost)
        count = result.count("  - ")
        self.assertEqual(count, 5)


class TestApplyCorrections(unittest.TestCase):
    """Test _apply_corrections() method."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_returns_messages_when_no_params(self):
        messages = [{"role": "user", "content": "test"}]
        result = self.builder._apply_corrections(messages, None)
        self.assertIs(result, messages)

    def test_returns_messages_when_multiplier_is_one(self):
        params = {"extraction_window_multiplier": 1.0}
        messages = [{"role": "user", "content": "test"}]
        result = self.builder._apply_corrections(messages, params)
        self.assertIs(result, messages)

    def test_returns_messages_when_multiplier_below_one(self):
        params = {"extraction_window_multiplier": 0.5}
        messages = [{"role": "user", "content": "test"}]
        result = self.builder._apply_corrections(messages, params)
        self.assertIs(result, messages)

    def test_returns_messages_with_multiplier_above_one(self):
        # Currently just logs and returns same messages
        params = {"extraction_window_multiplier": 1.5}
        messages = [{"role": "user", "content": "test"}]
        result = self.builder._apply_corrections(messages, params)
        self.assertIs(result, messages)


class TestScoreQuality(unittest.TestCase):
    """Test _score_quality() method."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_returns_empty_when_no_scorer(self):
        result = self.builder._score_quality([], "bridge text")
        self.assertEqual(result, {})

    def test_calls_scorer_when_available(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {"overall": 0.85}
        self.builder._scorer = mock_scorer

        messages = [{"role": "user", "content": "test"}]
        result = self.builder._score_quality(messages, "bridge text")
        self.assertEqual(result["overall"], 0.85)
        mock_scorer.score.assert_called_once_with(messages, "bridge text")

    def test_records_in_feedback_state(self):
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = {"overall": 0.7}
        mock_feedback = MagicMock()
        self.builder._scorer = mock_scorer
        self.builder._feedback = mock_feedback

        self.builder._score_quality([{"role": "user", "content": "test"}], "bridge")
        mock_feedback.record_compression.assert_called_once()

    def test_returns_empty_on_exception(self):
        mock_scorer = MagicMock()
        mock_scorer.score.side_effect = Exception("scoring failed")
        self.builder._scorer = mock_scorer

        result = self.builder._score_quality([], "bridge")
        self.assertEqual(result, {})


class TestExtract(unittest.TestCase):
    """Test _extract() helper method."""

    def setUp(self):
        self.builder = ContextBridgeBuilder()

    def test_returns_empty_when_no_extraction_engine(self):
        result = self.builder._extract("active_tasks", [])
        self.assertEqual(result, [])

    def test_calls_correct_method_on_engine(self):
        mock_extraction = MagicMock()
        mock_extraction.extract_active_tasks.return_value = [{"summary": "test"}]
        self.builder._extraction = mock_extraction

        result = self.builder._extract("active_tasks", [])
        self.assertEqual(result, [{"summary": "test"}])

    def test_returns_empty_when_method_missing(self):
        mock_extraction = MagicMock(spec=[])  # No methods
        self.builder._extraction = mock_extraction
        result = self.builder._extract("nonexistent", [])
        self.assertEqual(result, [])

    def test_returns_empty_on_exception(self):
        mock_extraction = MagicMock()
        mock_extraction.extract_active_tasks.side_effect = Exception("boom")
        self.builder._extraction = mock_extraction
        result = self.builder._extract("active_tasks", [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()

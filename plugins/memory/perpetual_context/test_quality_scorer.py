#!/usr/bin/env python3
"""Tests for quality_scorer.py — Bridge Quality Scorer.

Verifies:
1. Module-level weight constants sum to 1.0
2. score() returns correct structure with all expected keys
3. Empty/None inputs return zero scores
4. Task extraction from user messages
5. File path extraction from tool calls and text patterns
6. Error type extraction
7. Knowledge gap extraction
8. Preservation scoring (full, partial, none)
9. Section detection in bridge text
10. Lost items identification

Run:
    python -m pytest test_quality_scorer.py -v
"""

from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.quality_scorer import (
    _ERROR_WEIGHT,
    _FILE_WEIGHT,
    _GAP_WEIGHT,
    _TASK_WEIGHT,
    BridgeQualityScorer,
)


class TestConstants(unittest.TestCase):
    """Verify weight constants."""

    def test_task_weight(self):
        self.assertAlmostEqual(_TASK_WEIGHT, 0.40)

    def test_file_weight(self):
        self.assertAlmostEqual(_FILE_WEIGHT, 0.30)

    def test_error_weight(self):
        self.assertAlmostEqual(_ERROR_WEIGHT, 0.20)

    def test_gap_weight(self):
        self.assertAlmostEqual(_GAP_WEIGHT, 0.10)

    def test_weights_sum_to_one(self):
        total = _TASK_WEIGHT + _FILE_WEIGHT + _ERROR_WEIGHT + _GAP_WEIGHT
        self.assertAlmostEqual(total, 1.0)


class TestEmptyInputs(unittest.TestCase):
    """Test behavior with empty or None inputs."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_none_messages_returns_zero_score(self):
        result = self.scorer.score(None, "some bridge text")
        self.assertEqual(result["overall"], 0.0)

    def test_empty_list_messages_returns_zero_score(self):
        result = self.scorer.score([], "some bridge text")
        self.assertEqual(result["overall"], 0.0)

    def test_none_bridge_text_returns_zero_score(self):
        messages = [{"role": "user", "content": "fix the bug"}]
        result = self.scorer.score(messages, None)
        self.assertEqual(result["overall"], 0.0)

    def test_empty_bridge_text_returns_zero_score(self):
        messages = [{"role": "user", "content": "fix the bug"}]
        result = self.scorer.score(messages, "")
        self.assertEqual(result["overall"], 0.0)

    def test_both_empty_returns_zero_with_no_sections(self):
        result = self.scorer.score([], "")
        self.assertEqual(result["overall"], 0.0)
        self.assertEqual(result["bridge_char_count"], 0)
        self.assertEqual(result["sections_present"], [])
        self.assertEqual(result["lost_items"], [])

    def test_empty_score_has_all_keys(self):
        result = self.scorer.score([], "")
        expected_keys = {
            "overall", "active_tasks_preserved", "file_paths_preserved",
            "errors_preserved", "gaps_preserved", "bridge_char_count",
            "sections_present", "lost_items",
        }
        self.assertEqual(set(result.keys()), expected_keys)


class TestTaskExtraction(unittest.TestCase):
    """Test task summary extraction from messages."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_extracts_task_from_user_message(self):
        messages = [{"role": "user", "content": "Please fix the login bug"}]
        tasks = self.scorer._extract_task_summaries(messages)
        self.assertEqual(len(tasks), 1)
        self.assertIn("fix", tasks[0].lower())

    def test_ignores_non_user_messages(self):
        messages = [{"role": "assistant", "content": "I'll fix the bug"}]
        tasks = self.scorer._extract_task_summaries(messages)
        self.assertEqual(len(tasks), 0)

    def test_ignores_empty_content(self):
        messages = [{"role": "user", "content": ""}]
        tasks = self.scorer._extract_task_summaries(messages)
        self.assertEqual(len(tasks), 0)

    def test_limits_to_most_recent_five(self):
        messages = [
            {"role": "user", "content": f"fix bug number {i}"} for i in range(10)
        ]
        tasks = self.scorer._extract_task_summaries(messages)
        self.assertEqual(len(tasks), 5)

    def test_ignores_short_lines(self):
        messages = [{"role": "user", "content": "fix it"}]
        tasks = self.scorer._extract_task_summaries(messages)
        # "fix it" is only 6 chars, below the >10 threshold
        self.assertEqual(len(tasks), 0)

    def test_truncates_long_first_line(self):
        long_content = "implement a very long feature description that goes on and on " * 5
        messages = [{"role": "user", "content": long_content}]
        tasks = self.scorer._extract_task_summaries(messages)
        self.assertEqual(len(tasks), 1)
        self.assertLessEqual(len(tasks[0]), 120)


class TestFilePathExtraction(unittest.TestCase):
    """Test file path extraction from messages."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_extracts_path_from_tool_call(self):
        args = json.dumps({"path": "/home/user/test.py"})
        messages = [{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "write_file", "arguments": args}}],
        }]
        paths = self.scorer._extract_file_paths(messages)
        self.assertIn("/home/user/test.py", paths)

    def test_extracts_path_from_text_pattern(self):
        messages = [{
            "role": "assistant",
            "content": "Edit the file at /path/to/config.yaml please",
        }]
        paths = self.scorer._extract_file_paths(messages)
        self.assertTrue(any("config.yaml" in p for p in paths))

    def test_ignores_non_file_tool_calls(self):
        args = json.dumps({"url": "https://example.com"})
        messages = [{
            "role": "assistant",
            "tool_calls": [{"function": {"name": "browser_navigate", "arguments": args}}],
        }]
        paths = self.scorer._extract_file_paths(messages)
        self.assertEqual(len(paths), 0)

    def test_deduplicates_paths(self):
        messages = [
            {"role": "assistant", "content": "/path/to/file.py"},
            {"role": "assistant", "content": "/path/to/file.py again"},
        ]
        paths = self.scorer._extract_file_paths(messages)
        # Should appear only once due to set dedup
        counts = [p for p in paths if "file.py" in p]
        self.assertEqual(len(counts), 1)

    def test_limits_to_ten(self):
        messages = [
            {"role": "assistant", "content": f"/path/to/file{i}.py"}
            for i in range(20)
        ]
        paths = self.scorer._extract_file_paths(messages)
        self.assertLessEqual(len(paths), 10)

    def test_ignores_short_paths(self):
        messages = [{"role": "assistant", "content": "/a.py"}]
        paths = self.scorer._extract_file_paths(messages)
        # /a.py is only 5 chars, not > 5
        self.assertEqual(len(paths), 0)


class TestErrorExtraction(unittest.TestCase):
    """Test error type extraction from messages."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_extracts_type_error(self):
        messages = [{"role": "tool", "content": "TypeError: 'NoneType' object has no attribute"}]
        errors = self.scorer._extract_error_summaries(messages)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("TypeError"))

    def test_extracts_value_error(self):
        messages = [{"role": "tool", "content": "ValueError: invalid literal for int()"}]
        errors = self.scorer._extract_error_summaries(messages)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("ValueError"))

    def test_extracts_file_not_found(self):
        messages = [{"role": "tool", "content": "FileNotFoundError: [Errno 2] No such file"}]
        errors = self.scorer._extract_error_summaries(messages)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("FileNotFoundError"))

    def test_no_errors_in_clean_message(self):
        messages = [{"role": "assistant", "content": "Everything works fine"}]
        errors = self.scorer._extract_error_summaries(messages)
        self.assertEqual(len(errors), 0)

    def test_deduplicates_errors(self):
        messages = [
            {"role": "tool", "content": "TypeError: x"},
            {"role": "tool", "content": "TypeError: x"},
        ]
        errors = self.scorer._extract_error_summaries(messages)
        # Dedup via dict.fromkeys
        self.assertEqual(len(errors), 1)

    def test_limits_to_five(self):
        error_types = [
            "TypeError", "ValueError", "KeyError", "RuntimeError",
            "OSError", "IndexError", "AttributeError",
        ]
        messages = [{"role": "tool", "content": f"{et}: msg"} for et in error_types]
        errors = self.scorer._extract_error_summaries(messages)
        self.assertLessEqual(len(errors), 5)


class TestGapExtraction(unittest.TestCase):
    """Test knowledge gap extraction from messages."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_extracts_knowledge_gap(self):
        messages = [{"role": "assistant", "content": "knowledge gap: how to configure SSL"}]
        gaps = self.scorer._extract_gap_summaries(messages)
        self.assertEqual(len(gaps), 1)

    def test_extracts_rl_entry(self):
        messages = [{"role": "assistant", "content": "RL entry needed for Docker setup"}]
        gaps = self.scorer._extract_gap_summaries(messages)
        self.assertEqual(len(gaps), 1)

    def test_no_gaps_in_normal_text(self):
        messages = [{"role": "assistant", "content": "The code is working correctly."}]
        gaps = self.scorer._extract_gap_summaries(messages)
        self.assertEqual(len(gaps), 0)


class TestPreservationScoring(unittest.TestCase):
    """Test preservation scoring logic."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_empty_items_returns_perfect_score(self):
        score = self.scorer._score_preservation([], "any text")
        self.assertEqual(score, 1.0)

    def test_all_items_preserved_returns_one(self):
        items = ["fix login bug", "update config"]
        bridge = "We need to fix login bug and update config"
        score = self.scorer._score_preservation(items, bridge)
        self.assertEqual(score, 1.0)

    def test_no_items_preserved_returns_zero(self):
        items = ["fix login bug", "update config"]
        bridge = "The weather is nice today"
        score = self.scorer._score_preservation(items, bridge)
        self.assertAlmostEqual(score, 0.0, places=2)

    def test_partial_credit_for_key_terms(self):
        items = ["implement authentication system for users"]
        bridge = "authentication and system work is ongoing"
        score = self.scorer._score_preservation(items, bridge)
        # Should get partial credit (key terms match)
        self.assertGreater(score, 0.0)

    def test_case_insensitive_matching(self):
        items = ["Fix Login Bug"]
        bridge = "fix login bug was completed"
        score = self.scorer._score_preservation(items, bridge)
        self.assertEqual(score, 1.0)


class TestSectionDetection(unittest.TestCase):
    """Test section detection in bridge text."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_detects_active_tasks_section(self):
        bridge = "## Active Tasks\n- Fix bug"
        sections = self.scorer._detect_sections(bridge)
        self.assertIn("active_tasks", sections)

    def test_detects_file_edits_section(self):
        bridge = "## Files Currently Being Edited\ntest.py"
        sections = self.scorer._detect_sections(bridge)
        self.assertIn("file_edits", sections)

    def test_detects_known_errors_section(self):
        bridge = "## Known Errors\nNone"
        sections = self.scorer._detect_sections(bridge)
        self.assertIn("known_errors", sections)

    def test_detects_knowledge_gaps_section(self):
        bridge = "## Knowledge Gaps\ntbd"
        sections = self.scorer._detect_sections(bridge)
        self.assertIn("knowledge_gaps", sections)

    def test_detects_retrieval_guidance_section(self):
        bridge = "## Historical Context Retrieval\nuse PM"
        sections = self.scorer._detect_sections(bridge)
        self.assertIn("retrieval_guidance", sections)

    def test_no_sections_in_empty_text(self):
        sections = self.scorer._detect_sections("")
        self.assertEqual(len(sections), 0)

    def test_case_insensitive_section_detection(self):
        bridge = "## ACTIVE TASKS\n- Fix bug"
        sections = self.scorer._detect_sections(bridge)
        self.assertIn("active_tasks", sections)


class TestFindLost(unittest.TestCase):
    """Test lost items identification."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_finds_lost_items(self):
        items = ["fix login bug", "update config.yaml"]
        bridge = "The weather is nice"
        lost = self.scorer._find_lost(items, bridge)
        self.assertEqual(len(lost), 2)

    def test_no_lost_when_all_present(self):
        items = ["fix login bug"]
        bridge = "We fixed the fix login bug issue"
        lost = self.scorer._find_lost(items, bridge)
        self.assertEqual(len(lost), 0)


class TestFullScore(unittest.TestCase):
    """Integration test: full score() call with realistic data."""

    def setUp(self):
        self.scorer = BridgeQualityScorer()

    def test_score_with_matching_bridge(self):
        messages = [
            {"role": "user", "content": "Please fix the login bug in auth.py"},
            {"role": "tool", "content": "/home/user/auth.py was edited"},
        ]
        bridge = """## Active Tasks
- Fix the login bug

## Files Currently Being Edited
- /home/user/auth.py"""
        result = self.scorer.score(messages, bridge)
        # Should have high score since items are preserved
        self.assertGreater(result["overall"], 0.5)
        self.assertIn("active_tasks", result["sections_present"])
        self.assertIn("file_edits", result["sections_present"])

    def test_score_with_poor_bridge(self):
        messages = [
            {"role": "user", "content": "Please fix the login bug in auth.py"},
        ]
        bridge = "## Active Tasks\n- Nothing to do"
        result = self.scorer.score(messages, bridge)
        # Should have low task preservation score
        self.assertLess(result["active_tasks_preserved"], 0.5)

    def test_score_result_has_all_expected_keys(self):
        messages = [{"role": "user", "content": "fix something"}]
        bridge = "## Active Tasks\n- fix something"
        result = self.scorer.score(messages, bridge)
        expected_keys = {
            "overall", "active_tasks_preserved", "file_paths_preserved",
            "errors_preserved", "gaps_preserved", "bridge_char_count",
            "sections_present", "lost_items",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_overall_score_is_weighted_average(self):
        messages = [{"role": "user", "content": "fix the bug"}]
        bridge = "## Active Tasks\n- fix the bug"
        result = self.scorer.score(messages, bridge)
        # Overall should be between 0 and 1
        self.assertGreaterEqual(result["overall"], 0.0)
        self.assertLessEqual(result["overall"], 1.0)

    def test_lost_items_capped_at_ten(self):
        messages = [
            {"role": "user", "content": f"fix bug {i}"} for i in range(20)
        ]
        bridge = ""
        result = self.scorer.score(messages, bridge)
        self.assertLessEqual(len(result["lost_items"]), 10)


if __name__ == "__main__":
    unittest.main()

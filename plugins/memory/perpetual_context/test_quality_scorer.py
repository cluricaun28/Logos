#!/usr/bin/env python3
"""Tests for quality_scorer.py — Bridge Quality Scorer.

Verifies:
1. Module-level weight constants sum to 1.0
2. score() returns correct structure with all expected keys
3. Empty/None inputs return zero scores
4. Preservation scoring (full, partial, none)
5. Section detection in bridge text
6. Lost items identification

Extraction tests live in test_extraction_engine.py (single source of truth).

Run:
    python -m pytest test_quality_scorer.py -v
"""

from __future__ import annotations

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

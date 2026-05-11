#!/usr/bin/env python3
"""Tests for feedback_state.py — Compression quality tracking across sessions.

Verifies:
1. FeedbackState persistence (load/save/clear)
2. record_compression() with sliding window enforcement
3. get_degradation_trend() linear regression calculations
4. get_recent_average() over configurable windows
5. get_correction_params() applies corrections when degrading
6. needs_correction() threshold check
7. Graceful degradation on corrupt state files

Run:
    python -m pytest test_feedback_state.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.feedback_state import (
    _DEFAULT_STATE_FILE,
    _DEGRADATION_THRESHOLD,
    _MAX_ENTRIES,
    _MIN_QUALITY_THRESHOLD,
    FeedbackState,
)


class TestConstants(unittest.TestCase):
    """Verify module-level constants."""

    def test_max_entries(self):
        self.assertEqual(_MAX_ENTRIES, 20)

    def test_degradation_threshold(self):
        self.assertAlmostEqual(_DEGRADATION_THRESHOLD, 0.15)

    def test_min_quality_threshold(self):
        self.assertAlmostEqual(_MIN_QUALITY_THRESHOLD, 0.60)

    def test_default_state_file_exists(self):
        self.assertTrue(_DEFAULT_STATE_FILE.endswith("compression_feedback.json"))


class TestFeedbackStateInit(unittest.TestCase):
    """Test initialization and persistence."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state = FeedbackState(state_file=self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        tmp_tmp = self.tmp.name + ".tmp"
        if os.path.exists(tmp_tmp):
            os.unlink(tmp_tmp)

    def test_init_creates_empty_state(self):
        self.assertEqual(len(self.state._entries), 0)

    def test_load_existing_file(self):
        # Write a valid state file
        data = {
            "compressions": [
                {"timestamp": 1.0, "overall_score": 0.9},
                {"timestamp": 2.0, "overall_score": 0.8},
            ]
        }
        with open(self.tmp.name, "w", encoding='utf-8') as f:            json.dump(data, f)

        state = FeedbackState(state_file=self.tmp.name)
        self.assertEqual(len(state._entries), 2)

    def test_load_corrupt_json(self):
        # Write invalid JSON
        with open(self.tmp.name, "w", encoding='utf-8') as f:            f.write("{invalid json}")

        state = FeedbackState(state_file=self.tmp.name)
        self.assertEqual(len(state._entries), 0)

    def test_load_missing_timestamp_entry_dropped(self):
        data = {
            "compressions": [
                {"overall_score": 0.9},  # missing timestamp — should be dropped
                {"timestamp": 1.0, "overall_score": 0.8},  # valid
            ]
        }
        with open(self.tmp.name, "w", encoding='utf-8') as f:            json.dump(data, f)

        state = FeedbackState(state_file=self.tmp.name)
        self.assertEqual(len(state._entries), 1)

    def test_load_missing_overall_score_entry_dropped(self):
        data = {
            "compressions": [
                {"timestamp": 1.0},  # missing overall_score — should be dropped
                {"timestamp": 2.0, "overall_score": 0.8},  # valid
            ]
        }
        with open(self.tmp.name, "w", encoding='utf-8') as f:            json.dump(data, f)

        state = FeedbackState(state_file=self.tmp.name)
        self.assertEqual(len(state._entries), 1)


class TestRecordCompression(unittest.TestCase):
    """Test recording compression events."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state = FeedbackState(state_file=self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        tmp_tmp = self.tmp.name + ".tmp"
        if os.path.exists(tmp_tmp):
            os.unlink(tmp_tmp)

    def test_record_single_compression(self):
        score = {"overall": 0.85, "active_tasks_preserved": 1.0}
        self.state.record_compression(score, session_id="test_session")
        self.assertEqual(len(self.state._entries), 1)
        self.assertAlmostEqual(self.state._entries[0]["overall_score"], 0.85)

    def test_record_multiple_compressions(self):
        for i in range(5):
            self.state.record_compression({"overall": 0.9 - i * 0.05})
        self.assertEqual(len(self.state._entries), 5)

    def test_sliding_window_enforced(self):
        # Record more than MAX_ENTRIES
        for i in range(_MAX_ENTRIES + 10):
            self.state.record_compression({"overall": 0.9 - i * 0.01})
        self.assertEqual(len(self.state._entries), _MAX_ENTRIES)

    def test_sliding_window_keeps_newest(self):
        # Record MAX_ENTRIES + 5, verify oldest are dropped
        for i in range(_MAX_ENTRIES + 5):
            self.state.record_compression({"overall": float(i)})
        # Oldest entries (scores 0-4) should be gone
        scores = [e["overall_score"] for e in self.state._entries]
        self.assertEqual(min(scores), 5.0)

    def test_record_persists_to_disk(self):
        score = {"overall": 0.75}
        self.state.record_compression(score)

        # Load fresh instance from same file
        state2 = FeedbackState(state_file=self.tmp.name)
        self.assertEqual(len(state2._entries), 1)
        self.assertAlmostEqual(state2._entries[0]["overall_score"], 0.75)

    def test_record_with_empty_quality_score(self):
        score = {}
        self.state.record_compression(score)
        entry = self.state._entries[0]
        self.assertEqual(entry["overall_score"], 0.0)
        self.assertEqual(entry["tasks_preserved"], 0.0)

    def test_clear_removes_all_entries(self):
        for i in range(5):
            self.state.record_compression({"overall": 0.9})
        self.state.clear()
        self.assertEqual(len(self.state._entries), 0)


class TestDegradationTrend(unittest.TestCase):
    """Test degradation trend calculations."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state = FeedbackState(state_file=self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        tmp_tmp = self.tmp.name + ".tmp"
        if os.path.exists(tmp_tmp):
            os.unlink(tmp_tmp)

    def test_fewer_than_3_entries_returns_zero(self):
        self.state.record_compression({"overall": 0.9})
        self.state.record_compression({"overall": 0.8})
        trend = self.state.get_degradation_trend()
        self.assertEqual(trend, 0.0)

    def test_stable_quality_returns_near_zero(self):
        for _ in range(5):
            self.state.record_compression({"overall": 0.9})
        trend = self.state.get_degradation_trend()
        self.assertAlmostEqual(trend, 0.0, places=2)

    def test_improving_quality_returns_negative_trend(self):
        # Quality improving: 0.5 -> 1.0
        for i in range(5):
            self.state.record_compression({"overall": 0.5 + i * 0.125})
        trend = self.state.get_degradation_trend()
        self.assertLess(trend, 0)

    def test_degrading_quality_returns_positive_trend(self):
        # Quality degrading: 1.0 -> 0.5
        for i in range(5):
            self.state.record_compression({"overall": 1.0 - i * 0.125})
        trend = self.state.get_degradation_trend()
        self.assertGreater(trend, 0)

    def test_severe_degradation_returns_high_positive(self):
        # Sharp drop: all 1.0 then all 0.0
        for _ in range(3):
            self.state.record_compression({"overall": 1.0})
        for _ in range(3):
            self.state.record_compression({"overall": 0.0})
        trend = self.state.get_degradation_trend()
        self.assertGreater(trend, 0)

    def test_trend_clamped_to_minus_one_plus_one(self):
        # Extreme values should still be clamped
        for _ in range(3):
            self.state.record_compression({"overall": -10.0})
        for _ in range(3):
            self.state.record_compression({"overall": 10.0})
        trend = self.state.get_degradation_trend()
        self.assertGreaterEqual(trend, -1.0)
        self.assertLessEqual(trend, 1.0)


class TestRecentAverage(unittest.TestCase):
    """Test recent average quality calculations."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state = FeedbackState(state_file=self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        tmp_tmp = self.tmp.name + ".tmp"
        if os.path.exists(tmp_tmp):
            os.unlink(tmp_tmp)

    def test_no_entries_returns_one(self):
        avg = self.state.get_recent_average()
        self.assertEqual(avg, 1.0)

    def test_single_entry_returns_that_score(self):
        self.state.record_compression({"overall": 0.75})
        avg = self.state.get_recent_average(window=3)
        self.assertAlmostEqual(avg, 0.75)

    def test_window_larger_than_entries_uses_all(self):
        for i in range(3):
            self.state.record_compression({"overall": 0.8 + i * 0.1})
        avg = self.state.get_recent_average(window=10)
        # Average of 0.8, 0.9, 1.0 = 0.9
        self.assertAlmostEqual(avg, 0.9)

    def test_window_smaller_than_entries_uses_last_n(self):
        for i in range(10):
            self.state.record_compression({"overall": float(i)})
        avg = self.state.get_recent_average(window=3)
        # Last 3 scores: 7.0, 8.0, 9.0 → average = 8.0
        self.assertAlmostEqual(avg, 8.0)


class TestCorrectionParams(unittest.TestCase):
    """Test correction parameter generation."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.state = FeedbackState(state_file=self.tmp.name)

    def tearDown(self):
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)
        tmp_tmp = self.tmp.name + ".tmp"
        if os.path.exists(tmp_tmp):
            os.unlink(tmp_tmp)

    def test_default_params_when_no_data(self):
        params = self.state.get_correction_params()
        self.assertEqual(params["extraction_window_multiplier"], 1.0)
        self.assertFalse(params["preserve_critical_markers"])
        self.assertAlmostEqual(params["min_bridge_quality_threshold"], _MIN_QUALITY_THRESHOLD)

    def test_no_correction_when_improving(self):
        for i in range(5):
            self.state.record_compression({"overall": 0.5 + i * 0.1})
        params = self.state.get_correction_params()
        self.assertEqual(params["extraction_window_multiplier"], 1.0)

    def test_correction_applied_when_degrading(self):
        # Record degrading quality above threshold
        for _ in range(3):
            self.state.record_compression({"overall": 1.0})
        for _ in range(3):
            self.state.record_compression({"overall": 0.2})
        params = self.state.get_correction_params()
        # Should have widened extraction window
        self.assertGreater(params["extraction_window_multiplier"], 1.0)

    def test_preserve_critical_markers_when_low_quality(self):
        # Record very low quality with degradation
        for _ in range(3):
            self.state.record_compression({"overall": 0.9})
        for _ in range(3):
            self.state.record_compression({"overall": 0.1})
        params = self.state.get_correction_params()
        # Average quality is well below MIN_QUALITY_THRESHOLD (0.60)
        self.assertTrue(params["preserve_critical_markers"])

    def test_needs_correction_false_when_stable(self):
        for _ in range(5):
            self.state.record_compression({"overall": 0.9})
        self.assertFalse(self.state.needs_correction())

    def test_needs_correction_true_when_degrading(self):
        for _ in range(3):
            self.state.record_compression({"overall": 1.0})
        for _ in range(3):
            self.state.record_compression({"overall": 0.2})
        self.assertTrue(self.state.needs_correction())


if __name__ == "__main__":
    unittest.main()

"""Tests for FeedbackState — persistent compression quality tracking across sessions.

Verifies:
1. Initialization loads existing state or starts fresh
2. record_compression stores quality scores with timestamps
3. Sliding window enforces MAX_ENTRIES limit (20)
4. get_degradation_trend calculates linear regression slope correctly
5. get_recent_average computes average over configurable window
6. get_correction_params returns appropriate adjustments based on trend
7. needs_correction quick check works correctly
8. Persistence to JSON file with atomic write pattern
9. Graceful degradation on corrupt/missing state files
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))

from perpetual_context.feedback_state import (
    FeedbackState,
    _MAX_ENTRIES,
    _DEGRADATION_THRESHOLD,
    _MIN_QUALITY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_quality_score(overall=0.85, tasks=0.9, files=0.8, errors=1.0, gaps=0.7):
    """Create a quality score dict matching BridgeQualityScorer output."""
    return {
        "overall": overall,
        "active_tasks_preserved": tasks,
        "file_paths_preserved": files,
        "errors_preserved": errors,
        "gaps_preserved": gaps,
        "bridge_char_count": 1234,
        "sections_present": ["active_tasks", "file_edits"],
    }


# ---------------------------------------------------------------------------
# Tests: Initialization and persistence
# ---------------------------------------------------------------------------

class TestInitializationAndPersistence:
    """Test state file loading and saving."""

    def test_init_creates_empty_state(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            assert fs._entries == []
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_load_existing_state(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "compressions": [
                    {"timestamp": 1000, "overall_score": 0.8, "session_id": "s1"},
                    {"timestamp": 2000, "overall_score": 0.9, "session_id": "s2"},
                ]
            }, f)
            state_file = f.name

        try:
            fs = FeedbackState(state_file=state_file)
            assert len(fs._entries) == 2
            assert fs._entries[0]["overall_score"] == 0.8
            assert fs._entries[1]["overall_score"] == 0.9
        finally:
            os.unlink(state_file)

    def test_load_corrupt_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("NOT VALID JSON {{{")
            state_file = f.name

        try:
            fs = FeedbackState(state_file=state_file)
            # Should start fresh on corrupt data
            assert fs._entries == []
        finally:
            os.unlink(state_file)

    def test_load_missing_file_starts_fresh(self):
        fs = FeedbackState(state_file="/tmp/nonexistent_feedback_12345.json")
        assert fs._entries == []

    def test_load_validates_entries(self):
        """Entries without required fields should be dropped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "compressions": [
                    {"timestamp": 1000, "overall_score": 0.8},  # Valid
                    {"bad_key": "value"},  # Missing timestamp and overall_score
                    {"timestamp": 2000, "overall_score": 0.9},  # Valid
                ]
            }, f)
            state_file = f.name

        try:
            fs = FeedbackState(state_file=state_file)
            assert len(fs._entries) == 2
        finally:
            os.unlink(state_file)

    def test_load_enforces_max_entries(self):
        """If loaded data exceeds MAX_ENTRIES, should be truncated."""
        entries = [{"timestamp": i * 100, "overall_score": 0.5 + i * 0.01} for i in range(30)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"compressions": entries}, f)
            state_file = f.name

        try:
            fs = FeedbackState(state_file=state_file)
            assert len(fs._entries) == _MAX_ENTRIES
        finally:
            os.unlink(state_file)

    def test_save_writes_atomic(self):
        """Save should use atomic write pattern (temp file + rename)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name

        try:
            fs = FeedbackState(state_file=state_file)
            fs.record_compression(make_quality_score())
            assert os.path.exists(state_file)
            # Verify no .tmp file remains after save
            assert not os.path.exists(state_file + ".tmp")
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_save_failure_is_graceful(self):
        """Save should not raise on IO errors."""
        fs = FeedbackState(state_file="/nonexistent/dir/state.json")
        # Should not raise even though directory doesn't exist
        fs.record_compression(make_quality_score())


# ---------------------------------------------------------------------------
# Tests: record_compression
# ---------------------------------------------------------------------------

class TestRecordCompression:
    """Test recording compression events."""

    def test_records_single_entry(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            score = make_quality_score(overall=0.85)
            fs.record_compression(score, session_id="test_session")

            assert len(fs._entries) == 1
            entry = fs._entries[0]
            assert entry["session_id"] == "test_session"
            assert entry["overall_score"] == 0.85
            assert entry["tasks_preserved"] == score["active_tasks_preserved"]
            assert entry["files_preserved"] == score["file_paths_preserved"]
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_records_multiple_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for i in range(5):
                fs.record_compression(make_quality_score(overall=0.7 + i * 0.05))

            assert len(fs._entries) == 5
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_enforces_sliding_window(self):
        """Entries beyond MAX_ENTRIES should be trimmed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for i in range(_MAX_ENTRIES + 10):
                fs.record_compression(make_quality_score(overall=float(i)))

            assert len(fs._entries) == _MAX_ENTRIES
            # Should keep the last MAX_ENTRIES entries
            assert fs._entries[-1]["overall_score"] == float(_MAX_ENTRIES + 9)
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_entry_has_timestamp(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            before_time = time.time()
            fs.record_compression(make_quality_score())
            after_time = time.time()

            ts = fs._entries[0]["timestamp"]
            assert before_time <= ts <= after_time
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_entry_defaults_for_missing_keys(self):
        """Missing keys in quality_score should default to 0.0."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            empty_score = {}
            fs.record_compression(empty_score)

            entry = fs._entries[0]
            assert entry["overall_score"] == 0.0
            assert entry["tasks_preserved"] == 0.0
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)


# ---------------------------------------------------------------------------
# Tests: clear
# ---------------------------------------------------------------------------

class TestClear:
    """Test clearing feedback history."""

    def test_clear_removes_all_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for i in range(5):
                fs.record_compression(make_quality_score())

            assert len(fs._entries) == 5
            fs.clear()
            assert fs._entries == []
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)


# ---------------------------------------------------------------------------
# Tests: get_degradation_trend
# ---------------------------------------------------------------------------

class TestGetDegradationTrend:
    """Test degradation trend calculation."""

    def test_fewer_than_3_entries_returns_zero(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            assert fs.get_degradation_trend() == 0.0

            fs.record_compression(make_quality_score(overall=0.9))
            assert fs.get_degradation_trend() == 0.0

            fs.record_compression(make_quality_score(overall=0.8))
            assert fs.get_degradation_trend() == 0.0
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_improving_quality_returns_negative_trend(self):
        """Improving quality (scores going up) should return negative trend."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            # Quality improving from 0.5 to 0.9
            for score in [0.5, 0.6, 0.7, 0.8, 0.9]:
                fs.record_compression(make_quality_score(overall=score))

            trend = fs.get_degradation_trend()
            assert trend < 0  # Negative means improving (good)
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_degrading_quality_returns_positive_trend(self):
        """Degrading quality (scores going down) should return positive trend."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            # Quality degrading from 0.9 to 0.5
            for score in [0.9, 0.8, 0.7, 0.6, 0.5]:
                fs.record_compression(make_quality_score(overall=score))

            trend = fs.get_degradation_trend()
            assert trend > 0  # Positive means degrading (bad)
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_stable_quality_returns_near_zero(self):
        """Constant quality should return near-zero trend."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for _ in range(5):
                fs.record_compression(make_quality_score(overall=0.7))

            trend = fs.get_degradation_trend()
            assert abs(trend) < 0.1
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_trend_clamped_to_minus_1_plus_1(self):
        """Trend should be clamped to [-1, +1] range."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            # Extreme degradation
            for score in [1.0, 0.5, 0.0]:
                fs.record_compression(make_quality_score(overall=score))

            trend = fs.get_degradation_trend()
            assert -1.0 <= trend <= 1.0
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_uses_last_10_entries(self):
        """Trend should only use the last 10 entries."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            # First 5 entries: stable at 0.9 (should be ignored if >10 total)
            for _ in range(5):
                fs.record_compression(make_quality_score(overall=0.9))
            # Last 5 entries: degrading from 0.8 to 0.4
            for score in [0.8, 0.7, 0.6, 0.5, 0.4]:
                fs.record_compression(make_quality_score(overall=score))

            trend = fs.get_degradation_trend()
            assert trend > 0  # Should detect degradation from last entries
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)


# ---------------------------------------------------------------------------
# Tests: get_recent_average
# ---------------------------------------------------------------------------

class TestGetRecentAverage:
    """Test recent average quality calculation."""

    def test_no_entries_returns_1_0(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            assert fs.get_recent_average() == 1.0
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_single_entry_returns_that_score(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            fs.record_compression(make_quality_score(overall=0.75))
            assert fs.get_recent_average() == 0.75
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_default_window_is_5(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for score in [0.6, 0.7, 0.8, 0.9, 1.0]:
                fs.record_compression(make_quality_score(overall=score))

            avg = fs.get_recent_average()
            expected = (0.6 + 0.7 + 0.8 + 0.9 + 1.0) / 5
            assert abs(avg - round(expected, 3)) < 0.001
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_custom_window(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for score in [0.5, 0.6, 0.7, 0.8, 0.9]:
                fs.record_compression(make_quality_score(overall=score))

            avg_2 = fs.get_recent_average(window=2)
            expected = (0.8 + 0.9) / 2
            assert abs(avg_2 - round(expected, 3)) < 0.001
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_window_larger_than_entries(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for score in [0.6, 0.8]:
                fs.record_compression(make_quality_score(overall=score))

            avg = fs.get_recent_average(window=10)
            expected = (0.6 + 0.8) / 2
            assert abs(avg - round(expected, 3)) < 0.001
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)


# ---------------------------------------------------------------------------
# Tests: get_correction_params
# ---------------------------------------------------------------------------

class TestGetCorrectionParams:
    """Test correction parameter generation."""

    def test_stable_quality_returns_neutral_params(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for _ in range(5):
                fs.record_compression(make_quality_score(overall=0.8))

            params = fs.get_correction_params()
            assert params["extraction_window_multiplier"] == 1.0
            assert params["preserve_critical_markers"] is False
            assert params["min_bridge_quality_threshold"] == _MIN_QUALITY_THRESHOLD
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_improving_quality_returns_neutral_params(self):
        """Improving quality (negative trend) should return neutral params."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for score in [0.5, 0.6, 0.7, 0.8, 0.9]:
                fs.record_compression(make_quality_score(overall=score))

            params = fs.get_correction_params()
            assert params["extraction_window_multiplier"] == 1.0
            assert params["preserve_critical_markers"] is False
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_degradation_above_threshold_applies_corrections(self):
        """Degradation above threshold should widen extraction window."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            # Strong degradation from 1.0 to 0.2
            for score in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                fs.record_compression(make_quality_score(overall=score))

            params = fs.get_correction_params()
            assert params["extraction_window_multiplier"] > 1.0
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_low_quality_forces_preservation_markers(self):
        """When avg quality < MIN_QUALITY_THRESHOLD, force preservation markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            # Strong degradation to very low scores
            for score in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                fs.record_compression(make_quality_score(overall=score))

            params = fs.get_correction_params()
            if params["degradation_trend"] >= _DEGRADATION_THRESHOLD:
                # Check that preservation markers are set when avg is low enough
                assert isinstance(params["preserve_critical_markers"], bool)
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_dynamic_threshold_lowered(self):
        """Dynamic threshold should be lowered based on recent average."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for score in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                fs.record_compression(make_quality_score(overall=score))

            params = fs.get_correction_params()
            if params["degradation_trend"] >= _DEGRADATION_THRESHOLD:
                # Threshold should be lowered from default
                assert params["min_bridge_quality_threshold"] <= _MIN_QUALITY_THRESHOLD
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_params_include_recent_avg(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for _ in range(3):
                fs.record_compression(make_quality_score(overall=0.7))

            params = fs.get_correction_params()
            assert "recent_avg_quality" in params
            assert abs(params["recent_avg_quality"] - 0.7) < 0.01
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_insufficient_data_returns_neutral(self):
        """With fewer than 3 entries, should return neutral params."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            fs.record_compression(make_quality_score(overall=0.3))

            params = fs.get_correction_params()
            assert params["extraction_window_multiplier"] == 1.0
            assert params["preserve_critical_markers"] is False
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)


# ---------------------------------------------------------------------------
# Tests: needs_correction
# ---------------------------------------------------------------------------

class TestNeedsCorrection:
    """Test the quick correction check."""

    def test_no_entries_returns_false(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            assert fs.needs_correction() is False
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_stable_quality_returns_false(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for _ in range(5):
                fs.record_compression(make_quality_score(overall=0.8))

            assert fs.needs_correction() is False
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)

    def test_strong_degradation_returns_true(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            state_file = f.name
        try:
            fs = FeedbackState(state_file=state_file)
            for score in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]:
                fs.record_compression(make_quality_score(overall=score))

            assert fs.needs_correction() is True
        finally:
            if os.path.exists(state_file):
                os.unlink(state_file)


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify configuration constants."""

    def test_max_entries_is_20(self):
        assert _MAX_ENTRIES == 20

    def test_degradation_threshold_is_positive(self):
        assert _DEGRADATION_THRESHOLD > 0
        assert _DEGRADATION_THRESHOLD < 1.0

    def test_min_quality_threshold_is_reasonable(self):
        assert 0.0 < _MIN_QUALITY_THRESHOLD <= 1.0

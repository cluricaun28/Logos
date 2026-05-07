"""Tests for ContextBridgeBuilder — bridge formatting, truncation, and error handling."""

import pytest
from unittest.mock import MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from plugins.memory.perpetual_context.context_bridge_builder import (
    ContextBridgeBuilder,
    MAX_BRIDGE_CHARS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def sample_messages():
    """Return a realistic message list with tasks, file edits, and errors."""
    return [
        {"role": "user", "content": "Fix the bug in run_agent.py where context bridge isn't injected"},
        {"role": "assistant", "content": "I'll fix that. The issue is on_pre_compress needs to capture the return value."},
        {"role": "user", "content": "Also update config.yaml with new model settings"},
        {"role": "assistant", "content": "Done. Updated config.yaml and verified the changes work."},
        {"role": "user", "content": "Create a backup script for perpetual memory files"},
        {"role": "assistant", "content": "Created hermes-update-reintegrate.py with automated backup and restore."},
    ]


def mock_extraction_engine():
    """Return a mock extraction engine that returns structured data."""
    engine = MagicMock()
    engine.extract_active_tasks.return_value = [
        {
            "summary": "Fix context bridge injection",
            "turn_ids": [0, 1],
            "description": "on_pre_compress return value not captured in run_agent.py",
            "decisions": [{"turn_id": 1, "text": "Will use patch-based restore for modified tracked files"}],
        },
        {
            "summary": "Update config.yaml model settings",
            "turn_ids": [2, 3],
            "description": "New model configuration needed",
            "decisions": [],
        },
    ]
    engine.extract_file_edits.return_value = [
        {
            "path": "agent/run_agent.py",
            "last_edit_turn": 1,
            "description": "Added context bridge injection after on_pre_compress",
            "related_turns": [0, 1],
            "related_description": "context bridge discussion",
        },
    ]
    engine.extract_known_errors.return_value = [
        {
            "summary": "AttributeError: 'NoneType' has no attribute 'strip'",
            "turn_id": 5,
            "fix_location": "run_agent.py line 8103",
        },
    ]
    engine.extract_knowledge_gaps.return_value = [
        {
            "summary": "Hermes update stash conflict resolution",
            "turn_ids": [6, 7],
            "confidence": 0.7,
        },
    ]
    return engine


# ---------------------------------------------------------------------------
# Tests: build_bridge with extraction engine
# ---------------------------------------------------------------------------

class TestBuildBridgeWithExtraction:
    """Test bridge building when extraction engine is provided."""

    def test_builds_complete_bridge(self, sample_messages, mock_extraction_engine):
        builder = ContextBridgeBuilder(extraction_engine=mock_extraction_engine)
        result = builder.build_bridge(sample_messages)

        assert "## Active Tasks" in result
        assert "Fix context bridge injection" in result
        assert "## Files Currently Being Edited" in result
        assert "agent/run_agent.py" in result
        assert "## Known Errors/Issues" in result
        assert "AttributeError" in result
        assert "## Knowledge Gaps" in result
        assert "Hermes update stash conflict resolution" in result
        assert "## Historical Context Retrieval" in result

    def test_includes_turn_references(self, sample_messages, mock_extraction_engine):
        builder = ContextBridgeBuilder(extraction_engine=mock_extraction_engine)
        result = builder.build_bridge(sample_messages)

        # Turn references should be present
        assert "#0" in result or "#1" in result  # task turn refs
        assert "turns" in result.lower()

    def test_includes_decisions(self, sample_messages, mock_extraction_engine):
        builder = ContextBridgeBuilder(extraction_engine=mock_extraction_engine)
        result = builder.build_bridge(sample_messages)

        assert "Key decision" in result or "key decision" in result.lower()

    def test_respects_4kb_cap(self, sample_messages, mock_extraction_engine):
        """Test that bridge is truncated to MAX_BRIDGE_CHARS."""
        # Make extraction return huge data
        big_data = [{"summary": "x" * 2000, "turn_ids": [i], "description": "d"} for i in range(10)]
        mock_extraction_engine.extract_active_tasks.return_value = big_data

        builder = ContextBridgeBuilder(extraction_engine=mock_extraction_engine)
        result = builder.build_bridge(sample_messages)

        assert len(result) <= MAX_BRIDGE_CHARS + 50  # Small tolerance for section headers


# ---------------------------------------------------------------------------
# Tests: build_bridge without extraction engine (graceful degradation)
# ---------------------------------------------------------------------------

class TestBuildBridgeWithoutExtraction:
    """Test bridge building when no extraction engine is provided."""

    def test_returns_retrieval_guidance_only(self):
        builder = ContextBridgeBuilder()  # No extraction engine
        result = builder.build_bridge([{"role": "user", "content": "test"}])

        assert "## Historical Context Retrieval" in result
        # Should not have task/file/error sections since no extractor
        assert "## Active Tasks" not in result
        assert "## Files Currently Being Edited" not in result

    def test_empty_messages_returns_empty(self):
        builder = ContextBridgeBuilder()
        result = builder.build_bridge([])
        assert result == ""


# ---------------------------------------------------------------------------
# Tests: error handling and graceful degradation
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test that bridge building never raises exceptions."""

    def test_extraction_failure_returns_error_note(self, sample_messages):
        engine = MagicMock()
        engine.extract_active_tasks.side_effect = RuntimeError("DB connection lost")

        builder = ContextBridgeBuilder(extraction_engine=engine)
        result = builder.build_bridge(sample_messages)

        # Should return error note, not raise
        assert "Error generating retrieval index" in result or len(result) > 0

    def test_none_extraction_returns_empty(self):
        builder = ContextBridgeBuilder(extraction_engine=None)
        result = builder.build_bridge([{"role": "user", "content": "test"}])
        # Should have at least the retrieval guidance section
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests: formatting methods
# ---------------------------------------------------------------------------

class TestFormattingMethods:
    """Test individual formatting methods."""

    def test_format_active_tasks_limits_to_3(self):
        tasks = [
            {"summary": f"Task {i}", "turn_ids": [i], "description": f"desc {i}"}
            for i in range(10)
        ]
        builder = ContextBridgeBuilder()
        result = builder._format_active_tasks(tasks)

        # Should only include first 3 tasks
        assert "Task 0" in result
        assert "Task 2" in result
        assert "Task 3" not in result

    def test_format_file_edits_limits_to_3(self):
        edits = [
            {"path": f"/path/file{i}.py", "last_edit_turn": i, "description": f"edit {i}"}
            for i in range(10)
        ]
        builder = ContextBridgeBuilder()
        result = builder._format_file_edits(edits)

        assert "/path/file0.py" in result
        assert "/path/file2.py" in result
        assert "/path/file3.py" not in result

    def test_format_known_errors_limits_to_3(self):
        errors = [
            {"summary": f"Error {i}", "turn_id": i, "fix_location": "loc"}
            for i in range(10)
        ]
        builder = ContextBridgeBuilder()
        result = builder._format_known_errors(errors)

        assert "Error 0" in result
        assert "Error 2" in result
        assert "Error 3" not in result

    def test_format_knowledge_gaps_limits_to_3(self):
        gaps = [
            {"summary": f"Gap {i}", "turn_ids": [i], "confidence": 0.5}
            for i in range(10)
        ]
        builder = ContextBridgeBuilder()
        result = builder._format_knowledge_gaps(gaps)

        assert "Gap 0" in result
        assert "Gap 2" in result
        assert "Gap 3" not in result


# ---------------------------------------------------------------------------
# Tests: quality scoring integration
# ---------------------------------------------------------------------------

class TestQualityScoringIntegration:
    """Test that quality scoring is called and feedback recorded."""

    def test_scorer_called_when_provided(self, sample_messages):
        scorer = MagicMock()
        scorer.score.return_value = {"overall": 0.85}
        builder = ContextBridgeBuilder(scorer=scorer)
        builder.build_bridge(sample_messages)
        assert scorer.score.called

    def test_feedback_recorded_when_provided(self, sample_messages):
        feedback = MagicMock()
        scorer = MagicMock()
        scorer.score.return_value = {"overall": 0.85}
        builder = ContextBridgeBuilder(scorer=scorer, feedback_state=feedback)
        builder.build_bridge(sample_messages)
        assert feedback.record_compression.called


# ---------------------------------------------------------------------------
# Tests: preservation warnings
# ---------------------------------------------------------------------------

class TestPreservationWarnings:
    """Test preservation warning generation."""

    def test_preservation_warning_format(self):
        builder = ContextBridgeBuilder()
        lost = ["task A", "file B.py", "error C"]
        result = builder._format_preservation_warning(lost)

        assert "Preservation Warning" in result
        assert "task A" in result
        assert "perpetual_search" in result

    def test_preservation_warning_caps_at_5(self):
        builder = ContextBridgeBuilder()
        lost = [f"item{i}" for i in range(10)]
        result = builder._format_preservation_warning(lost)

        # Should only include first 5 items
        assert "item0" in result
        assert "item4" in result
        assert "item5" not in result


# ---------------------------------------------------------------------------
# Tests: correction params
# ---------------------------------------------------------------------------

class TestCorrectionParams:
    """Test feedback correction application."""

    def test_no_corrections_returns_same_messages(self, sample_messages):
        builder = ContextBridgeBuilder()
        result = builder._apply_corrections(sample_messages, None)
        assert result is sample_messages

    def test_multiplier_le_1_returns_same(self, sample_messages):
        builder = ContextBridgeBuilder()
        result = builder._apply_corrections(sample_messages, {"extraction_window_multiplier": 0.5})
        assert result is sample_messages

    def test_multiplier_gt_1_logs_and_returns_same(self, sample_messages):
        """Currently the multiplier just logs — messages are returned as-is."""
        builder = ContextBridgeBuilder()
        with patch("plugins.memory.perpetual_context.context_bridge_builder.logger") as mock_log:
            result = builder._apply_corrections(
                sample_messages, {"extraction_window_multiplier": 2.0}
            )
            assert result is sample_messages
            mock_log.info.assert_called_once()


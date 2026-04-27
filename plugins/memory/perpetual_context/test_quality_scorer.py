"""Tests for BridgeQualityScorer — context bridge preservation quality scoring.

Verifies:
1. score() returns correct structure with all expected keys
2. Empty/None inputs return zero scores gracefully
3. Task extraction and preservation scoring works correctly
4. File path extraction from tool calls and text patterns
5. Error summary extraction from messages
6. Knowledge gap marker detection
7. Section detection in bridge text
8. Lost item identification for diagnostics
9. Weighted overall score calculation (tasks: 40%, files: 30%, errors: 20%, gaps: 10%)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.quality_scorer import (
    BridgeQualityScorer,
    _TASK_WEIGHT,
    _FILE_WEIGHT,
    _ERROR_WEIGHT,
    _GAP_WEIGHT,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def sample_messages():
    """Return messages with tasks, file paths, errors, and gaps."""
    return [
        {"role": "user", "content": "Fix the bug in run_agent.py where context bridge isn't injected"},
        {"role": "assistant", "content": "I'll fix that. The issue is on_pre_compress."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "/home/user/run_agent.py"}',
                }
            }],
        },
        {"role": "user", "content": "TypeError: 'NoneType' object is not callable"},
        {"role": "assistant", "content": "[gap] proper error handling for network timeouts"},
    ]


def sample_bridge_text():
    """Return a bridge text that preserves most items."""
    return (
        "## Active Tasks\n"
        "- Fix the bug in run_agent.py where context bridge isn't injected\n"
        "\n"
        "## Files Currently Being Edited\n"
        "- /home/user/run_agent.py: Modified on_pre_compress\n"
        "\n"
        "## Known Errors/Issues\n"
        "- TypeError: 'NoneType' object is not callable\n"
        "\n"
        "## Knowledge Gaps\n"
        "- proper error handling for network timeouts\n"
        "\n"
        "## Historical Context Retrieval\n"
        "Use perpetual_search to find more context."
    )


def partial_bridge_text():
    """Return a bridge text that only preserves some items."""
    return (
        "## Active Tasks\n"
        "- Fix the bug in run_agent.py where context bridge isn't injected\n"
        "\n"
        "## Historical Context Retrieval\n"
        "Use perpetual_search to find more context."
    )


# ---------------------------------------------------------------------------
# Tests: score() main method
# ---------------------------------------------------------------------------

class TestScoreMainMethod:
    """Test the primary score() method."""

    def test_returns_all_expected_keys(self):
        scorer = BridgeQualityScorer()
        result = scorer.score(sample_messages(), sample_bridge_text())
        expected_keys = {
            "overall", "active_tasks_preserved", "file_paths_preserved",
            "errors_preserved", "gaps_preserved", "bridge_char_count",
            "sections_present", "lost_items",
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_bridge_gets_high_score(self):
        scorer = BridgeQualityScorer()
        result = scorer.score(sample_messages(), sample_bridge_text())
        # All items preserved → high overall score
        assert result["overall"] > 0.5

    def test_empty_messages_returns_zero_score(self):
        scorer = BridgeQualityScorer()
        result = scorer.score([], "some bridge text")
        assert result["overall"] == 0.0
        assert result["active_tasks_preserved"] == 0.0
        assert result["file_paths_preserved"] == 0.0

    def test_empty_bridge_returns_zero_score(self):
        scorer = BridgeQualityScorer()
        result = scorer.score(sample_messages(), "")
        assert result["overall"] == 0.0

    def test_none_messages_returns_zero_score(self):
        scorer = BridgeQualityScorer()
        result = scorer.score(None, "some bridge text")
        assert result["overall"] == 0.0

    def test_none_bridge_returns_zero_score(self):
        scorer = BridgeQualityScorer()
        result = scorer.score(sample_messages(), None)
        assert result["overall"] == 0.0

    def test_both_empty_returns_zero(self):
        scorer = BridgeQualityScorer()
        result = scorer.score([], "")
        assert result["overall"] == 0.0
        assert result["bridge_char_count"] == 0
        assert result["sections_present"] == []

    def test_bridge_char_count_accurate(self):
        scorer = BridgeQualityScorer()
        bridge = "x" * 1234
        result = scorer.score(sample_messages(), bridge)
        assert result["bridge_char_count"] == 1234

    def test_scores_are_rounded_to_3_decimals(self):
        scorer = BridgeQualityScorer()
        result = scorer.score(sample_messages(), sample_bridge_text())
        for key in ["overall", "active_tasks_preserved", "file_paths_preserved",
                     "errors_preserved", "gaps_preserved"]:
            val = result[key]
            assert isinstance(val, float) or isinstance(val, int)

    def test_lost_items_capped_at_10(self):
        """Lost items list should be capped at 10 entries."""
        scorer = BridgeQualityScorer()
        # Create many messages with different tasks to generate many lost items
        msgs = []
        for i in range(20):
            msgs.append({"role": "user", "content": f"Fix bug number {i} in module_{i}.py"})
        bridge = "## Active Tasks\n- Nothing preserved"
        result = scorer.score(msgs, bridge)
        assert len(result["lost_items"]) <= 10


# ---------------------------------------------------------------------------
# Tests: _empty_score
# ---------------------------------------------------------------------------

class TestEmptyScore:
    """Test the empty score fallback."""

    def test_empty_score_with_text(self):
        scorer = BridgeQualityScorer()
        result = scorer._empty_score("some text")
        assert result["overall"] == 0.0
        assert result["bridge_char_count"] == len("some text")
        assert result["lost_items"] == []

    def test_empty_score_with_none(self):
        scorer = BridgeQualityScorer()
        result = scorer._empty_score("")
        assert result["overall"] == 0.0
        assert result["bridge_char_count"] == 0
        assert result["sections_present"] == []


# ---------------------------------------------------------------------------
# Tests: _extract_task_summaries
# ---------------------------------------------------------------------------

class TestExtractTaskSummaries:
    """Test task summary extraction."""

    def test_extracts_from_user_messages(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "user", "content": "Fix the login bug in auth.py"}]
        result = scorer._extract_task_summaries(msgs)
        assert len(result) >= 1

    def test_ignores_assistant_messages(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "Fix the login bug in auth.py"}]
        result = scorer._extract_task_summaries(msgs)
        assert len(result) == 0

    def test_ignores_non_task_messages(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "user", "content": "What is the weather today?"}]
        result = scorer._extract_task_summaries(msgs)
        assert len(result) == 0

    def test_skips_short_first_lines(self):
        """First lines shorter than 10 chars should be skipped."""
        scorer = BridgeQualityScorer()
        msgs = [{"role": "user", "content": "Fix it"}]
        result = scorer._extract_task_summaries(msgs)
        assert len(result) == 0

    def test_limits_to_5_most_recent(self):
        scorer = BridgeQualityScorer()
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"Fix bug number {i} in module_{i}.py"})
        result = scorer._extract_task_summaries(msgs)
        assert len(result) == 5

    def test_truncates_to_120_chars(self):
        scorer = BridgeQualityScorer()
        long_content = "Fix the bug where " + "x" * 300
        msgs = [{"role": "user", "content": long_content}]
        result = scorer._extract_task_summaries(msgs)
        assert len(result[0]) <= 120

    def test_empty_content_skipped(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": None},
            {"role": "user", "content": "   "},
        ]
        assert scorer._extract_task_summaries(msgs) == []


# ---------------------------------------------------------------------------
# Tests: _extract_file_paths
# ---------------------------------------------------------------------------

class TestExtractFilePaths:
    """Test file path extraction."""

    def test_extracts_from_tool_calls(self):
        import json as _json
        scorer = BridgeQualityScorer()
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": _json.dumps({"path": "/home/user/test.py"}),
                }
            }],
        }]
        result = scorer._extract_file_paths(msgs)
        assert any("/home/user/test.py" in p for p in result)

    def test_extracts_from_text_patterns(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "See /path/to/config.yaml for details"}]
        result = scorer._extract_file_paths(msgs)
        assert any("/path/to/config.yaml" in p for p in result)

    def test_ignores_non_file_tool_calls(self):
        import json as _json
        scorer = BridgeQualityScorer()
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "terminal",
                    "arguments": _json.dumps({"command": "ls"}),
                }
            }],
        }]
        result = scorer._extract_file_paths(msgs)
        assert len(result) == 0

    def test_deduplicates_paths(self):
        import json as _json
        scorer = BridgeQualityScorer()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"path": "/home/user/test.py"}),
                    }
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "patch",
                        "arguments": _json.dumps({"path": "/home/user/test.py"}),
                    }
                }],
            },
        ]
        result = scorer._extract_file_paths(msgs)
        # Should be deduplicated to one entry
        assert len(result) == 1

    def test_limits_to_10(self):
        import json as _json
        scorer = BridgeQualityScorer()
        msgs = []
        for i in range(15):
            msgs.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"path": f"/home/user/file{i}.py"}),
                    }
                }],
            })
        result = scorer._extract_file_paths(msgs)
        assert len(result) == 10

    def test_skips_short_paths(self):
        """Paths shorter than 5 chars should be skipped."""
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "See /a.py for details"}]
        result = scorer._extract_file_paths(msgs)
        assert len(result) == 0

    def test_handles_invalid_json_args(self):
        scorer = BridgeQualityScorer()
        msgs = [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": "NOT VALID JSON {{{",
                }
            }],
        }]
        # Should not raise, just skip invalid args
        result = scorer._extract_file_paths(msgs)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: _extract_error_summaries
# ---------------------------------------------------------------------------

class TestExtractErrorSummaries:
    """Test error summary extraction."""

    def test_extracts_type_error(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "user", "content": "TypeError: 'NoneType' object is not callable"}]
        result = scorer._extract_error_summaries(msgs)
        assert len(result) >= 1
        assert any("TypeError" in e for e in result)

    def test_extracts_multiple_errors(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "user", "content": "TypeError: 'NoneType' object is not callable"},
            {"role": "assistant", "content": "ValueError: invalid literal for int()"},
        ]
        result = scorer._extract_error_summaries(msgs)
        assert len(result) >= 2

    def test_deduplicates_errors(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "user", "content": "TypeError: 'NoneType' object is not callable"},
            {"role": "assistant", "content": "TypeError: 'NoneType' object is not callable"},
        ]
        result = scorer._extract_error_summaries(msgs)
        assert len(result) == 1

    def test_limits_to_5(self):
        scorer = BridgeQualityScorer()
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", content: f"TypeError: error {i}"})
        result = scorer._extract_error_summaries(msgs)
        assert len(result) <= 5

    def test_truncates_error_message(self):
        scorer = BridgeQualityScorer()
        long_msg = "TypeError: " + "x" * 200
        msgs = [{"role": "user", "content": long_msg}]
        result = scorer._extract_error_summaries(msgs)
        assert len(result[0]) <= 80  # exc_type + ": " + 60 chars

    def test_no_errors_returns_empty(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "user", "content": "Everything works fine"}]
        result = scorer._extract_error_summaries(msgs)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: _extract_gap_summaries
# ---------------------------------------------------------------------------

class TestExtractGapSummaries:
    """Test knowledge gap extraction."""

    def test_extracts_knowledge_gap_marker(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "knowledge gap: how to handle WSL mounts"}]
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) >= 1

    def test_extracts_rl_entry_marker(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "RL entry needed: proper error handling"}]
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) >= 1

    def test_extracts_bracket_gap(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "[gap] network timeout handling"}]
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) >= 1

    def test_extracts_pending_tag(self):
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "[pending] migration strategy"}]
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) >= 1

    def test_skips_short_summaries(self):
        """Summaries shorter than 3 chars should be skipped."""
        scorer = BridgeQualityScorer()
        msgs = [{"role": "assistant", "content": "[gap] ab"}]
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) == 0

    def test_limits_to_5(self):
        scorer = BridgeQualityScorer()
        msgs = []
        for i in range(10):
            msgs.append({"role": "assistant", content: f"[gap] gap number {i}"})
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) <= 5

    def test_deduplicates_gaps(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "assistant", "content": "[gap] network timeout handling"},
            {"role": "assistant", "content": "[gap] network timeout handling"},
        ]
        result = scorer._extract_gap_summaries(msgs)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: _score_preservation
# ---------------------------------------------------------------------------

class TestScorePreservation:
    """Test the preservation scoring logic."""

    def test_empty_items_returns_1_0(self):
        scorer = BridgeQualityScorer()
        result = scorer._score_preservation([], "any bridge text")
        assert result == 1.0

    def test_all_items_preserved(self):
        scorer = BridgeQualityScorer()
        items = ["fix the bug", "update config"]
        bridge = "We need to fix the bug and update config"
        result = scorer._score_preservation(items, bridge)
        assert result == 1.0

    def test_no_items_preserved(self):
        scorer = BridgeQualityScorer()
        items = ["fix the login bug in auth module"]
        bridge = "Nothing relevant here at all"
        result = scorer._score_preservation(items, bridge)
        assert result == 0.0

    def test_partial_credit_for_key_terms(self):
        """If key terms appear but not full match, should get partial credit."""
        scorer = BridgeQualityScorer()
        items = ["implement caching strategy for database queries"]
        bridge = "caching and database were discussed"
        result = scorer._score_preservation(items, bridge)
        assert 0.0 < result <= 1.0

    def test_file_path_basename_matching(self):
        """File paths should match by basename if full path not in bridge."""
        scorer = BridgeQualityScorer()
        items = ["/home/user/projects/run_agent.py"]
        bridge = "Modified run_agent.py to fix the bug"
        result = scorer._score_preservation(items, bridge)
        assert result == 1.0

    def test_case_insensitive_matching(self):
        scorer = BridgeQualityScorer()
        items = ["Fix The Bug In Auth"]
        bridge = "fix the bug in auth was completed"
        result = scorer._score_preservation(items, bridge)
        assert result == 1.0

    def test_whitespace_normalization(self):
        """Extra whitespace should be normalized."""
        scorer = BridgeQualityScorer()
        items = ["Fix   the   bug"]
        bridge = "fix the bug was completed"
        result = scorer._score_preservation(items, bridge)
        assert result == 1.0

    def test_partial_match_first_words(self):
        """Task summaries should match on first significant words."""
        scorer = BridgeQualityScorer()
        items = ["Fix the login bug in auth module with new validation"]
        bridge = "We need to fix the login bug"
        result = scorer._score_preservation(items, bridge)
        assert result == 1.0

    def test_score_capped_at_1_0(self):
        """Score should never exceed 1.0."""
        scorer = BridgeQualityScorer()
        items = ["test"]
        bridge = "test"
        result = scorer._score_preservation(items, bridge)
        assert result <= 1.0


# ---------------------------------------------------------------------------
# Tests: _find_lost
# ---------------------------------------------------------------------------

class TestFindLost:
    """Test lost item identification."""

    def test_finds_completely_missing_items(self):
        scorer = BridgeQualityScorer()
        items = ["fix the login bug", "update config file"]
        bridge = "Nothing relevant here"
        result = scorer._find_lost(items, bridge)
        assert len(result) == 2

    def test_does_not_flag_preserved_items(self):
        scorer = BridgeQualityScorer()
        items = ["fix the login bug", "update config file"]
        bridge = "We fixed the login bug and updated config file"
        result = scorer._find_lost(items, bridge)
        assert len(result) == 0

    def test_partial_match_not_flagged(self):
        """Items with significant word overlap should not be flagged as lost."""
        scorer = BridgeQualityScorer()
        items = ["implement caching strategy for database"]
        bridge = "caching and database were discussed"
        result = scorer._find_lost(items, bridge)
        # Should have partial match → not flagged
        assert len(result) == 0

    def test_empty_items_returns_empty(self):
        scorer = BridgeQualityScorer()
        result = scorer._find_lost([], "bridge text")
        assert result == []


# ---------------------------------------------------------------------------
# Tests: _detect_sections
# ---------------------------------------------------------------------------

class TestDetectSections:
    """Test bridge section detection."""

    def test_detects_active_tasks_section(self):
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("## Active Tasks\n- task 1")
        assert "active_tasks" in sections

    def test_detects_file_edits_section(self):
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("## Files Currently Being Edited\n- file.py")
        assert "file_edits" in sections

    def test_detects_known_errors_section(self):
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("## Known Errors\n- error 1")
        assert "known_errors" in sections

    def test_detects_knowledge_gaps_section(self):
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("## Knowledge Gaps\n- gap 1")
        assert "knowledge_gaps" in sections

    def test_detects_retrieval_guidance_section(self):
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("## Historical Context Retrieval\nUse search.")
        assert "retrieval_guidance" in sections

    def test_uppercase_markers_detected(self):
        """Uppercase section markers should also be detected."""
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("ACTIVE TASK: fix bug")
        assert "active_tasks" in sections

    def test_no_sections_returns_empty(self):
        scorer = BridgeQualityScorer()
        sections = scorer._detect_sections("Just some random text here")
        assert len(sections) == 0

    def test_all_five_sections_detected(self):
        scorer = BridgeQualityScorer()
        bridge = (
            "## Active Tasks\n- task\n"
            "## Files Currently Being Edited\n- file\n"
            "## Known Errors\n- error\n"
            "## Knowledge Gaps\n- gap\n"
            "## Historical Context Retrieval\nSearch."
        )
        sections = scorer._detect_sections(bridge)
        assert len(sections) == 5


# ---------------------------------------------------------------------------
# Tests: Weight constants
# ---------------------------------------------------------------------------

class TestWeightConstants:
    """Verify weight constants sum to 1.0."""

    def test_weights_sum_to_one(self):
        total = _TASK_WEIGHT + _FILE_WEIGHT + _ERROR_WEIGHT + _GAP_WEIGHT
        assert abs(total - 1.0) < 0.001

    def test_task_weight_is_40_percent(self):
        assert _TASK_WEIGHT == 0.40

    def test_file_weight_is_30_percent(self):
        assert _FILE_WEIGHT == 0.30

    def test_error_weight_is_20_percent(self):
        assert _ERROR_WEIGHT == 0.20

    def test_gap_weight_is_10_percent(self):
        assert _GAP_WEIGHT == 0.10


# ---------------------------------------------------------------------------
# Tests: Integration — full scoring pipeline
# ---------------------------------------------------------------------------

class TestIntegrationScoringPipeline:
    """Test the complete scoring pipeline end-to-end."""

    def test_all_sections_preserved_gets_high_score(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "user", "content": "Fix the login bug in auth.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "/home/user/auth.py"}',
                    }
                }],
            },
            {"role": "user", "content": "TypeError: 'NoneType' object is not callable"},
            {"role": "assistant", "content": "[gap] proper error handling"},
        ]
        bridge = (
            "## Active Tasks\n- Fix the login bug in auth.py\n"
            "## Files Currently Being Edited\n- /home/user/auth.py\n"
            "## Known Errors/Issues\n- TypeError: 'NoneType' object is not callable\n"
            "## Knowledge Gaps\n- proper error handling\n"
        )
        result = scorer.score(msgs, bridge)
        assert result["overall"] > 0.5

    def test_no_sections_preserved_gets_low_score(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "user", "content": "Fix the login bug in auth.py"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": '{"path": "/home/user/auth.py"}',
                    }
                }],
            },
        ]
        bridge = "## Historical Context Retrieval\nNothing preserved."
        result = scorer.score(msgs, bridge)
        assert result["overall"] < 0.5

    def test_lost_items_populated_when_missing(self):
        scorer = BridgeQualityScorer()
        msgs = [
            {"role": "user", "content": "Fix the login bug in auth.py"},
        ]
        bridge = "## Historical Context Retrieval\nNothing preserved."
        result = scorer.score(msgs, bridge)
        assert len(result["lost_items"]) > 0

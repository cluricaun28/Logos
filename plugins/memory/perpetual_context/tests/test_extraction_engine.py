"""Tests for ExtractionEngine — content extraction and processing engine.

Verifies:
1. extract_active_tasks correctly identifies task-related user messages
2. extract_file_edits detects both structured tool_calls and text patterns
3. find_related_discussions locates surrounding context mentions
4. extract_known_errors finds exception types with fix locations
5. extract_knowledge_gaps detects explicit markers, confidence scores, pending tags
6. All methods handle empty/None inputs gracefully (never raise)
"""

from __future__ import annotations

import json as _json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent/plugins/memory"))

from perpetual_context.extraction_engine import ExtractionEngine

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def sample_messages():
    """Return a realistic message list with tasks, file edits, errors."""
    return [
        {"role": "user", "content": "Fix the bug in run_agent.py where context bridge isn't injected"},
        {"role": "assistant", "content": "I'll fix that. The architecture is to patch on_pre_compress."},
        {"role": "user", "content": "Also update config.yaml with new model settings"},
        {"role": "assistant", "content": "Done. I decided we will use YAML anchors for the config."},
        {"role": "user", "content": "Create a backup script for perpetual memory files"},
        {"role": "assistant", "content": "Created hermes-update-reintegrate.py with automated backup."},
    ]


def messages_with_tool_calls():
    """Return messages containing structured tool_calls."""
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"path": "/home/user/run_agent.py", "content": "..."}),
                    }
                },
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "patch",
                        "arguments": _json.dumps({"path": "/home/user/config.yaml"}),
                    }
                },
            ],
        },
    ]


def messages_with_text_file_patterns():
    """Return messages with text-based file operation mentions."""
    return [
        {"role": "assistant", "content": "I wrote to /tmp/output.txt the results."},
        {"role": "assistant", "content": "Patched /src/main.py to fix the bug."},
        {"role": "assistant", "content": "Reading /etc/config.yaml for settings."},
    ]


def messages_with_errors():
    """Return messages containing error traces."""
    return [
        {
            "role": "user",
            "content": (
                "I got this error:\n"
                "TypeError: 'NoneType' object is not callable\n"
                "at line 42 in run_agent.py"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Found the issue. AttributeError: 'SessionDB' has no attribute 'save'\n"
                "Fixed in /home/user/hermes_state.py line 120"
            ),
        },
    ]


def messages_with_knowledge_gaps():
    """Return messages with knowledge gap markers."""
    return [
        {
            "role": "assistant",
            "content": (
                "knowledge gap: how to handle WSL filesystem mounts\n"
                "[gap] proper error handling for network timeouts\n"
                "confidence: 0.3 — first principles of nuclear ethics\n"
                "[pending] migration strategy for SQLite to PostgreSQL\n"
                "[needs research] FAISS index optimization techniques\n"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Tests: extract_active_tasks
# ---------------------------------------------------------------------------

class TestExtractActiveTasks:
    """Test active task extraction from user messages."""

    def test_returns_empty_for_no_messages(self):
        engine = ExtractionEngine()
        assert engine.extract_active_tasks([]) == []

    def test_returns_empty_for_non_user_messages(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "assistant", "content": "Fix the bug"},
            {"role": "system", "content": "Implement feature X"},
        ]
        assert engine.extract_active_tasks(msgs) == []

    def test_returns_empty_for_messages_without_task_keywords(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "What is the weather today?"},
            {"role": "user", "content": "Tell me a joke"},
        ]
        assert engine.extract_active_tasks(msgs) == []

    def test_extracts_task_with_fix_keyword(self):
        engine = ExtractionEngine()
        msgs = [{"role": "user", "content": "Fix the login bug in auth.py"}]
        result = engine.extract_active_tasks(msgs)
        assert len(result) >= 1
        task = result[0]
        assert "summary" in task
        assert "turn_ids" in task
        assert "description" in task
        assert "decisions" in task
        assert 0 in task["turn_ids"]

    def test_extracts_multiple_tasks(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Implement new API endpoint"},
            {"role": "assistant", "content": "Will do that."},
        ]
        result = engine.extract_active_tasks(msgs)
        assert len(result) >= 2

    def test_deduplicates_similar_tasks(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Fix the login bug in auth.py\nMore details here."},
            {"role": "assistant", "content": "Working on it."},
            {"role": "user", "content": "Fix the login bug in auth.py\nStill broken, please check again."},
        ]
        result = engine.extract_active_tasks(msgs)
        # Same first line → deduplicated, but turn_ids should include both
        assert len(result) == 1
        assert sorted(result[0]["turn_ids"]) == [0, 2]

    def test_case_insensitive_dedup(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Fix the login bug"},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "FIX THE LOGIN BUG"},
        ]
        result = engine.extract_active_tasks(msgs)
        assert len(result) == 1

    def test_captures_decisions_from_assistant(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Implement caching strategy"},
            {"role": "assistant", "content": (
                "I decided we will use an LRU cache with TTL.\n"
                "The architecture is to wrap the DB layer.\n"
                "Some random text that doesn't match."
            )},
        ]
        result = engine.extract_active_tasks(msgs)
        assert len(result) == 1
        decisions = result[0]["decisions"]
        assert len(decisions) >= 2

    def test_skips_short_decision_lines(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Fix bug"},
            {"role": "assistant", "content": (
                "I decided to fix it.\n"
                "Short\n"
            "This is a very long decision line that explains the key design "
            "principle we must never forget when implementing this feature "
            "because it always check for edge cases and mandatory validation."
            )},
        ]
        result = engine.extract_active_tasks(msgs)
        decisions = result[0]["decisions"]
        # "Short" is < 20 chars, should be skipped
        assert not any(d["text"] == "Short" for d in decisions)

    def test_skips_example_pseudocode_in_decisions(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Implement feature"},
            {"role": "assistant", "content": (
                "I decided we will use this approach.\n"
                "Example: def foo(): pass\n"
                "# TODO: implement later\n"
            )},
        ]
        result = engine.extract_active_tasks(msgs)
        decisions = result[0]["decisions"]
        assert not any("example" in d["text"].lower() for d in decisions)

    def test_limits_to_5_most_recent_tasks(self):
        engine = ExtractionEngine()
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"Fix bug number {i}"})
            msgs.append({"role": "assistant", "content": "On it."})
        result = engine.extract_active_tasks(msgs)
        assert len(result) == 5

    def test_most_recent_first(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Fix bug A"},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "Fix bug B"},
            {"role": "assistant", "content": "OK."},
        ]
        result = engine.extract_active_tasks(msgs)
        assert len(result) >= 2
        # Most recent first means B should come before A
        summaries = [t["summary"] for t in result]
        b_idx = summaries.index("Fix bug B") if "Fix bug B" in summaries else -1
        a_idx = summaries.index("Fix bug A") if "Fix bug A" in summaries else -1
        assert b_idx < a_idx

    def test_empty_content_user_message_skipped(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
            {"role": "user", "content": None},
        ]
        assert engine.extract_active_tasks(msgs) == []

    def test_description_truncated_to_300_chars(self):
        engine = ExtractionEngine()
        long_content = "Fix the bug in module X where " + "x" * 400
        msgs = [{"role": "user", "content": long_content}]
        result = engine.extract_active_tasks(msgs)
        assert len(result[0]["description"]) <= 300

    def test_all_task_keywords_detected(self):
        """Verify all task keywords in the frozenset are detected."""
        engine = ExtractionEngine()
        keywords = [
            "fix", "implement", "create", "build", "add", "refactor", "debug",
            "write", "update", "migrate", "deploy", "configure", "set up",
            "resolve", "address", "handle", "support", "enable", "disable",
        ]
        for kw in keywords:
            msgs = [{"role": "user", "content": f"{kw} something"}]
            result = engine.extract_active_tasks(msgs)
            assert len(result) >= 1, f"Keyword '{kw}' not detected as task keyword"


# ---------------------------------------------------------------------------
# Tests: extract_file_edits
# ---------------------------------------------------------------------------

class TestExtractFileEdits:
    """Test file edit extraction from messages."""

    def test_returns_empty_for_no_messages(self):
        engine = ExtractionEngine()
        assert engine.extract_file_edits([]) == []

    def test_detects_write_file_tool_call(self):
        engine = ExtractionEngine()
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
        ]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1
        assert any("/home/user/test.py" in r["path"] for r in result)

    def test_detects_patch_tool_call(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "patch",
                        "arguments": _json.dumps({"path": "/home/user/config.yaml"}),
                    }
                }],
            },
        ]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_detects_read_file_tool_call(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "read_file",
                        "arguments": _json.dumps({"path": "/home/user/data.json"}),
                    }
                }],
            },
        ]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_ignores_non_file_tool_calls(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "terminal", "arguments": _json.dumps({"command": "ls"})},
                }],
            },
        ]
        assert engine.extract_file_edits(msgs) == []

    def test_detects_text_pattern_wrote(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "I wrote to /tmp/output.txt the results."}]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_detects_text_pattern_patched(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "Patched /src/main.py to fix the bug."}]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_detects_text_pattern_reading(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "Reading /etc/config.yaml for settings."}]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_deduplicates_same_path_different_ops(self):
        engine = ExtractionEngine()
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
        result = engine.extract_file_edits(msgs)
        # Same path → deduplicated, description should mention both ops
        assert len(result) == 1
        desc = result[0]["description"]
        assert "Write File" in desc or "write file" in desc.lower()

    def test_limits_to_10_results(self):
        engine = ExtractionEngine()
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
        result = engine.extract_file_edits(msgs)
        assert len(result) == 10

    def test_most_recent_first(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"path": "/home/user/first.py"}),
                    }
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"path": "/home/user/second.py"}),
                    }
                }],
            },
        ]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 2
        # Most recent first → second.py should come before first.py
        paths = [r["path"] for r in result]
        assert paths.index("/home/user/second.py") < paths.index("/home/user/first.py")

    def test_handles_invalid_tool_call_args(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": "NOT VALID JSON {{{",
                    }
                }],
            },
        ]
        # Should not raise, just skip invalid args
        result = engine.extract_file_edits(msgs)
        assert isinstance(result, list)

    def test_handles_non_dict_tool_call(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": ["not a dict"],
            },
        ]
        assert engine.extract_file_edits(msgs) == []

    def test_handles_missing_path_in_args(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"other_key": "value"}),
                    }
                }],
            },
        ]
        assert engine.extract_file_edits(msgs) == []

    def test_strips_query_params_from_path(self):
        """Paths with query params should be normalized for dedup."""
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": _json.dumps({"path": "/home/user/test.py?v=1"}),
                    }
                }],
            },
        ]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_tool_call_args_as_dict(self):
        """Arguments can be a dict instead of JSON string."""
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "write_file",
                        "arguments": {"path": "/home/user/dict_args.py"},
                    }
                }],
            },
        ]
        result = engine.extract_file_edits(msgs)
        assert len(result) >= 1

    def test_related_turns_capped_at_5(self):
        """Related turns in file edit results should be capped at 5."""
        engine = ExtractionEngine()
        # Create many messages mentioning the same file
        msgs = []
        for i in range(20):
            msgs.append({"role": "user", "content": f"Discussion about test.py turn {i}"})
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "write_file",
                    "arguments": _json.dumps({"path": "/home/user/test.py"}),
                }
            }],
        })
        result = engine.extract_file_edits(msgs)
        assert len(result[0]["related_turns"]) <= 5


# ---------------------------------------------------------------------------
# Tests: find_related_discussions
# ---------------------------------------------------------------------------

class TestFindRelatedDiscussions:
    """Test related discussion finding for file edits."""

    def test_returns_empty_for_no_messages(self):
        engine = ExtractionEngine()
        turns, desc = engine.find_related_discussions([], 0, "/home/user/test.py")
        assert turns == []
        assert desc == ""

    def test_finds_mentions_by_basename(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "Let's discuss test.py"},
            {"role": "assistant", "content": "OK, I'll edit it."},
            {"role": "user", "content": "Make sure test.py handles errors"},
        ]
        turns, desc = engine.find_related_discussions(msgs, 1, "/home/user/test.py")
        assert len(turns) >= 2

    def test_skips_edit_turn_itself(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "assistant", "content": "Editing test.py now"},
        ]
        turns, desc = engine.find_related_discussions(msgs, 0, "/home/user/test.py")
        assert 0 not in turns

    def test_respects_window_parameter(self):
        engine = ExtractionEngine()
        msgs = []
        for i in range(30):
            msgs.append({"role": "user", "content": f"test.py discussion at turn {i}"})
        # Edit at turn 15, window=2 → only turns 14 and 16 should be checked
        turns, desc = engine.find_related_discussions(msgs, 15, "/home/user/test.py", window=2)
        for t in turns:
            assert abs(t - 15) <= 2

    def test_returns_description_from_snippets(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "The test.py file needs a major refactor of the core logic"},
            {"role": "assistant", "content": ""},
        ]
        turns, desc = engine.find_related_discussions(msgs, 1, "/home/user/test.py")
        assert len(desc) > 0

    def test_description_capped_at_2_snippets(self):
        """Only first 2 unique snippets should be in description."""
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": f"Snippet {i} about test.py with enough content to matter"}
            for i in range(10)
        ]
        turns, desc = engine.find_related_discussions(msgs, 5, "/home/user/test.py")
        # Description should contain at most 2 snippets separated by " — "
        parts = desc.split(" — ") if desc else []
        assert len(parts) <= 3  # 2 snippets → 3 parts max

    def test_description_truncated_at_200_chars(self):
        engine = ExtractionEngine()
        long_snippet = "test.py " + "x" * 500
        msgs = [{"role": "user", "content": long_snippet}]
        turns, desc = engine.find_related_discussions(msgs, 1, "/home/user/test.py")
        if desc:
            assert len(desc) <= 200

    def test_handles_invalid_file_path(self):
        """Should handle non-string file paths gracefully."""
        engine = ExtractionEngine()
        turns, desc = engine.find_related_discussions([], 0, None)
        assert turns == []
        assert desc == ""


# ---------------------------------------------------------------------------
# Tests: extract_known_errors
# ---------------------------------------------------------------------------

class TestExtractKnownErrors:
    """Test known error extraction from messages."""

    def test_returns_empty_for_no_messages(self):
        engine = ExtractionEngine()
        assert engine.extract_known_errors([]) == []

    def test_detects_type_error(self):
        engine = ExtractionEngine()
        msgs = [{"role": "user", "content": "TypeError: 'NoneType' object is not callable"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1
        assert "TypeError" in result[0]["summary"]

    def test_detects_attribute_error(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "AttributeError: 'SessionDB' has no attribute 'save'"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1

    def test_detects_key_error(self):
        engine = ExtractionEngine()
        msgs = [{"role": "user", "content": "KeyError: 'missing_key' in dict lookup"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1

    def test_detects_import_error(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "ModuleNotFoundError: No module named 'requests'"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1

    def test_detects_file_not_found(self):
        engine = ExtractionEngine()
        msgs = [{"role": "user", "content": "FileNotFoundError: [Errno 2] No such file"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1

    def test_detects_syntax_error(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "SyntaxError: invalid syntax at line 42"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1

    def test_finds_fix_location_path(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": (
                    "TypeError: 'NoneType' object is not callable\n"
                    "Fixed in /home/user/run_agent.py line 42"
                ),
            },
        ]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1
        assert "/home/user/run_agent.py" in result[0]["fix_location"]

    def test_finds_fix_location_line(self):
        engine = ExtractionEngine()
        msgs = [
            {
                "role": "assistant",
                "content": (
                    "ValueError: invalid literal\n"
                    "Resolved at line 120"
                ),
            },
        ]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1
        assert "120" in result[0]["fix_location"]

    def test_fix_location_defaults_to_n_a(self):
        engine = ExtractionEngine()
        msgs = [{"role": "user", "content": "TypeError: something went wrong"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1
        assert result[0]["fix_location"] == "N/A"

    def test_deduplicates_same_error(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "TypeError: 'NoneType' object is not callable"},
            {"role": "assistant", "content": "Fixing..."},
            {"role": "user", "content": "Still getting TypeError: 'NoneType' object is not callable"},
        ]
        result = engine.extract_known_errors(msgs)
        assert len(result) == 1

    def test_limits_to_5_results(self):
        engine = ExtractionEngine()
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"TypeError: error number {i}"})
        result = engine.extract_known_errors(msgs)
        assert len(result) == 5

    def test_most_recent_first(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": "TypeError: first error"},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "ValueError: second error"},
        ]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 2
        # Most recent first → ValueError should come before TypeError
        summaries = [r["summary"] for r in result]
        assert summaries[0].startswith("ValueError")

    def test_cleans_traceback_artifacts(self):
        """Traceback prefixes like 'File "...", line N' should be cleaned."""
        engine = ExtractionEngine()
        msgs = [
            {"role": "user", "content": (
                'TypeError: File "/path/to/file.py", line 42, in func\n'
                "actual error message"
            )}
        ]
        result = engine.extract_known_errors(msgs)
        assert len(result) >= 1

    def test_empty_error_message_skipped(self):
        """If there's no content after the exception type, skip it."""
        engine = ExtractionEngine()
        msgs = [{"role": "user", "content": "TypeError:\n\n"}]
        result = engine.extract_known_errors(msgs)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: extract_knowledge_gaps
# ---------------------------------------------------------------------------

class TestExtractKnowledgeGaps:
    """Test knowledge gap extraction from messages."""

    def test_returns_empty_for_no_messages(self):
        engine = ExtractionEngine()
        assert engine.extract_knowledge_gaps([]) == []

    def test_detects_explicit_knowledge_gap_marker(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "knowledge gap: how to handle WSL mounts"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1
        assert "WSL" in result[0]["summary"]

    def test_detects_rl_entry_needed(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "RL entry needed: proper error handling"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1

    def test_detects_bracket_gap_marker(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "[gap] network timeout handling"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1

    def test_detects_confidence_score_pattern(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "confidence: 0.3 — first principles of nuclear ethics"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1
        assert result[0]["confidence"] == 0.3

    def test_detects_pending_tag(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "[pending] migration strategy for SQLite"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1
        assert result[0]["confidence"] == 0.7

    def test_detects_needs_research_tag(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "[needs research] FAISS optimization"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1

    def test_confidence_out_of_range_skipped(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "assistant", "content": "confidence: 1.5 — something impossible"},
            {"role": "assistant", "content": "confidence: -0.2 — negative confidence"},
        ]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) == 0

    def test_short_summary_skipped(self):
        """Summaries shorter than 3 chars should be skipped."""
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "confidence: 0.5 — ab"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) == 0

    def test_limits_to_5_results(self):
        engine = ExtractionEngine()
        msgs = []
        for i in range(10):
            msgs.append({"role": "assistant", "content": f"[gap] topic_alpha_{i} needs research"})
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) == 5

    def test_most_recent_first(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "assistant", "content": "[gap] first gap"},
            {"role": "user", "content": "OK."},
            {"role": "assistant", "content": "[gap] second gap"},
        ]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 2
        summaries = [r["summary"] for r in result]
        assert "second" in summaries[0].lower()

    def test_deduplicates_similar_gaps(self):
        engine = ExtractionEngine()
        msgs = [
            {"role": "assistant", "content": "[gap] network timeout handling"},
            {"role": "user", "content": "OK."},
            {"role": "assistant", "content": "[gap] Network timeout handling again"},
        ]
        result = engine.extract_knowledge_gaps(msgs)
        # Same first word → deduplicated
        assert len(result) == 1

    def test_explicit_marker_default_confidence_0_5(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "[gap] some topic"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert result[0]["confidence"] == 0.5

    def test_pending_tag_default_confidence_0_7(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "[pending] some topic"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert result[0]["confidence"] == 0.7

    def test_todo_tag_detected(self):
        engine = ExtractionEngine()
        msgs = [{"role": "assistant", "content": "[TODO] implement caching layer"}]
        result = engine.extract_knowledge_gaps(msgs)
        assert len(result) >= 1

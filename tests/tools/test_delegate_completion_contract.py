#!/usr/bin/env python3
"""
Tests for the P1 delegation completion contract, fan-in reconciliation,
and resumable children (work order 20260826-review-followup, items A + F).

Specification-as-tests:
  - child prompts must require a fenced ``json completion_report`` block
  - orchestrator machine-checks completed == total -> contract_mismatch
    warning block (never a hard fail)
  - fan_in() catches the 144-vs-129 expected-set-vs-artifacts class
  - timed-out/partial children persist {done_items, output_dir,
    task_context}; resume_from injects the done-set into the next child
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

from tools.delegate_tool import (
    _build_child_system_prompt,
    _derive_done_items,
    _run_single_child,
    build_resume_note,
    delegate_task,
    fan_in,
    load_delegation_state,
    parse_completion_report,
    persist_delegation_state,
)


def _make_mock_parent(depth=0):
    """Mock parent agent with the fields delegate_task expects."""
    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    return parent


# ---------------------------------------------------------------------------
# parse_completion_report
# ---------------------------------------------------------------------------

def test_parse_completion_report_ok():
    text = (
        "Summary of work done.\n"
        "```json completion_report\n"
        '{"completed": 12, "total": 12, "output_paths": ["/tmp/a.md"], '
        '"failures": []}\n'
        "```"
    )
    report = parse_completion_report(text)
    assert report == {
        "completed": 12,
        "total": 12,
        "output_paths": ["/tmp/a.md"],
        "failures": [],
    }


def test_parse_completion_report_last_block_wins():
    text = (
        "Example of the shape:\n```json completion_report\n"
        '{"completed": 0, "total": 99}\n```\n'
        "Real report:\n```json completion_report\n"
        '{"completed": 3, "total": 3}\n```\n'
    )
    assert parse_completion_report(text)["completed"] == 3


def test_parse_completion_report_missing_or_malformed():
    assert parse_completion_report("plain summary, no block") is None
    assert parse_completion_report("```json completion_report\n{nope\n```") is None
    assert parse_completion_report(None) is None
    assert parse_completion_report("") is None


def test_parse_completion_report_coerces_string_ints_and_rejects_bools():
    text = (
        "```json completion_report\n"
        '{"completed": "7", "total": "8", "failures": "one bad item"}\n'
        "```"
    )
    report = parse_completion_report(text)
    assert report["completed"] == 7
    assert report["total"] == 8
    # non-list failures degrades to [] (never a crash)
    assert report["failures"] == []
    # bool is not a count
    text2 = (
        "```json completion_report\n"
        '{"completed": true, "total": 8}\n'
        "```"
    )
    assert parse_completion_report(text2)["completed"] is None


# ---------------------------------------------------------------------------
# Child system prompt requires the block
# ---------------------------------------------------------------------------

def test_child_prompt_requires_completion_report_block():
    prompt = _build_child_system_prompt(goal="do the thing")
    assert "completion_report" in prompt
    assert "```json completion_report" in prompt
    # Contract fields are named so the child knows the expected shape.
    for field in ("completed", "total", "output_paths", "failures"):
        assert field in prompt


# ---------------------------------------------------------------------------
# Orchestrator-side mismatch warning (no hard fail)
# ---------------------------------------------------------------------------

def _mock_child(final_response, api_calls=2):
    child = MagicMock()
    child.model = "test-model"
    child.session_prompt_tokens = 0
    child.session_completion_tokens = 0
    child._subagent_id = "child-contract-test"
    child.run_conversation.return_value = {
        "final_response": final_response,
        "completed": True,
        "interrupted": False,
        "api_calls": api_calls,
        "messages": [{"role": "assistant", "content": final_response}],
    }
    return child


def _run_child_direct(child, **kwargs):
    """Run _run_single_child directly against a mock child (no AIAgent)."""
    return _run_single_child(0, "goal", child, _make_mock_parent(), **kwargs)


def test_contract_mismatch_surfaces_warning_without_failing():
    summary = (
        "Wrote reports.\n```json completion_report\n"
        '{"completed": 129, "total": 144, "output_paths": [], '
        '"failures": ["15 items never produced"]}\n```'
    )
    entry = _run_child_direct(_mock_child(summary))
    assert entry["status"] == "completed"  # NOT hard-failed
    assert entry["completion_contract"]["completed"] == 129
    mismatch = entry["contract_mismatch"]
    assert mismatch["completed"] == 129
    assert mismatch["total"] == 144
    assert "15 items never produced" in mismatch["failures"]
    assert "UNVERIFIED" in mismatch["warning"]


def test_contract_match_no_warning():
    summary = (
        "Done.\n```json completion_report\n"
        '{"completed": 12, "total": 12, "output_paths": ["/tmp/x.md"], '
        '"failures": []}\n```'
    )
    entry = _run_child_direct(_mock_child(summary))
    assert entry["completion_contract"]["completed"] == 12
    assert "contract_mismatch" not in entry


def test_missing_block_is_tolerated():
    entry = _run_child_direct(_mock_child("All good, no machine block here."))
    assert entry["status"] == "completed"
    assert entry["completion_contract"] is None
    assert "contract_mismatch" not in entry


# ---------------------------------------------------------------------------
# fan_in — the 144-vs-129 class
# ---------------------------------------------------------------------------

def test_fan_in_catches_144_vs_129_class(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    expected = [f"report_{i:03d}" for i in range(144)]
    # 129 of 144 artifacts arrived; the last 15 were silently dropped.
    for item in expected[:129]:
        (out / f"{item}.md").write_text("ok")
    result = fan_in(expected, str(out))
    assert len(result["merged"]) == 129
    assert result["missing"] == expected[129:]
    assert len(result["missing"]) == 15
    assert result["dupes"] == {}


def test_fan_in_dupes_and_nested_dirs(tmp_path):
    out = tmp_path / "out"
    (out / "nested").mkdir(parents=True)
    (out / "a.md").write_text("1")
    (out / "a.json").write_text("2")  # dupe for 'a'
    (out / "nested" / "b.md").write_text("3")  # nested match
    result = fan_in(["a", "b", "c"], str(out))
    assert result["merged"] == ["b"]
    assert result["missing"] == ["c"]
    assert sorted(result["dupes"]["a"]) == ["a.json", "a.md"]


def test_fan_in_missing_output_dir_reports_all_missing(tmp_path):
    result = fan_in(["x", "y"], str(tmp_path / "never-created"))
    assert result["merged"] == []
    assert result["missing"] == ["x", "y"]


# ---------------------------------------------------------------------------
# Resume payload round-trip
# ---------------------------------------------------------------------------

def test_resume_payload_round_trip():
    payload = {
        "done_items": ["b.md", "a.md", "a.md"],
        "output_dir": "/tmp/out",
        "task_context": {"goal": "G", "context": "C"},
    }
    path = persist_delegation_state(
        "child-42",
        done_items=payload["done_items"],
        output_dir=payload["output_dir"],
        task_context=payload["task_context"],
    )
    assert path and path.endswith("child-42.json")
    loaded = load_delegation_state("child-42")
    assert loaded["schema"] == "delegation_resume_v1"
    assert loaded["child_id"] == "child-42"
    assert loaded["done_items"] == ["a.md", "b.md"]  # deduped + sorted
    assert loaded["output_dir"] == "/tmp/out"
    assert loaded["task_context"] == payload["task_context"]


def test_resume_state_rejects_unsafe_child_id():
    assert persist_delegation_state(
        "../evil", done_items=[], output_dir=None, task_context=None
    ) is None
    assert load_delegation_state("../evil") is None
    assert load_delegation_state("no-such-child") is None


def test_build_resume_note_lists_done_items():
    note = build_resume_note({"done_items": ["item_000.md", "item_001.md"]})
    assert "already complete" in note
    assert "verify and continue with the remainder" in note
    assert "- item_000.md" in note and "- item_001.md" in note


# ---------------------------------------------------------------------------
# Timeout persistence + resume_from wiring through delegate_task
# ---------------------------------------------------------------------------

def test_timeout_persists_resume_state(tmp_path):
    out = tmp_path / "artifacts"
    out.mkdir()
    (out / "done_0.md").write_text("x")

    child = MagicMock()
    child._subagent_id = "child-timeout-test"
    child.model = "test-model"
    child.get_activity_summary.return_value = {"api_call_count": 3}

    def _slow_run(*args, **kwargs):
        time.sleep(5)

    child.run_conversation.side_effect = _slow_run
    child.messages = []

    entry = _run_single_child(
        0,
        "fan out items",
        child,
        _make_mock_parent(),
        child_timeout=0.5,
        output_dir=str(out),
        task_context={"goal": "fan out items"},
    )
    assert entry["status"] == "timeout"
    assert entry["child_id"] == "child-timeout-test"
    assert entry["resume_state_path"]
    state = load_delegation_state("child-timeout-test")
    assert state is not None
    # Artifact on disk is the done-set ground truth.
    assert state["done_items"] == ["done_0.md"]
    assert state["output_dir"] == str(out)
    assert state["task_context"]["goal"] == "fan out items"
    assert "resume_from='child-timeout-test'" in entry["resume_hint"]


def test_resume_from_injects_done_set(tmp_path):
    persist_delegation_state(
        "child-src",
        done_items=["item_0.md", "item_1.md"],
        output_dir=str(tmp_path / "out"),
        task_context={"goal": "G"},
    )
    with patch("tools.delegate_tool._build_child_agent") as mock_build, patch(
        "tools.delegate_tool._run_single_child"
    ) as mock_run:
        mock_build.return_value = MagicMock()
        mock_run.return_value = {
            "task_index": 0,
            "status": "completed",
            "summary": "resumed ok",
            "api_calls": 1,
            "duration_seconds": 1.0,
        }
        out = delegate_task(
            goal="continue the fan-out",
            resume_from="child-src",
            parent_agent=_make_mock_parent(),
        )
    result = json.loads(out)
    assert result["results"][0]["status"] == "completed"
    kwargs = mock_run.call_args.kwargs
    assert kwargs["output_dir"] == str(tmp_path / "out")
    ctx = kwargs["task_context"]["context"]
    assert "already complete" in ctx
    assert "item_0.md" in ctx and "item_1.md" in ctx
    # The child's context also carries the note for the system prompt path.
    build_ctx = mock_build.call_args.kwargs["context"]
    assert "already complete" in build_ctx


def test_resume_from_unknown_id_fails_loudly():
    out = delegate_task(
        goal="fresh", resume_from="ghost-child", parent_agent=_make_mock_parent()
    )
    data = json.loads(out)
    assert "error" in data
    assert "ghost-child" in data["error"]
    assert "resume_from" in data["error"]


def test_derive_done_items_merges_artifacts_and_claims(tmp_path):
    out = tmp_path / "out"
    (out / "nested").mkdir(parents=True)
    (out / "art.md").write_text("x")
    (out / "nested" / "deep.md").write_text("y")
    msgs = [
        {
            "role": "assistant",
            "content": (
                "```json completion_report\n"
                '{"completed": 1, "total": 2, "output_paths": '
                '["/tmp/claimed.md"], "failures": ["b"]}\n```'
            ),
        }
    ]
    done = _derive_done_items(str(out), msgs)
    assert done == ["/tmp/claimed.md", "art.md", "nested/deep.md"]

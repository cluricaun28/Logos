"""Tests for ``run_agent.append_usage_event`` — the fleet-wide per-call usage
stamp that feeds the harness-performance rollup (``<home>/logs/usage.jsonl``).

The function must be dead-simple and dead-safe: it appends one JSON line per
LLM call and must NEVER raise (it sits in the hot agent loop).
"""
import json

from run_agent import append_usage_event


def test_writes_jsonl_to_logs_dir(tmp_path):
    append_usage_event(tmp_path, {"session": "S1", "in": 10, "out": 5, "total": 15})
    p = tmp_path / "logs" / "usage.jsonl"
    assert p.exists(), "usage.jsonl must be created under <home>/logs/"
    rec = json.loads(p.read_text().strip())
    assert rec["session"] == "S1"
    assert rec["total"] == 15


def test_appends_one_line_per_call(tmp_path):
    append_usage_event(tmp_path, {"session": "S1", "total": 1})
    append_usage_event(tmp_path, {"session": "S2", "total": 2})
    lines = (tmp_path / "logs" / "usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["session"] == "S1"
    assert json.loads(lines[1])["session"] == "S2"


def test_accepts_str_home(tmp_path):
    append_usage_event(str(tmp_path), {"session": "S1"})
    assert (tmp_path / "logs" / "usage.jsonl").exists()


def test_never_raises_on_bad_home():
    # Best-effort: bad home values must be swallowed, never propagate.
    append_usage_event(None, {"session": "X"})
    append_usage_event(object(), {"session": "Y"})


def test_non_serializable_value_falls_back_to_str(tmp_path):
    append_usage_event(tmp_path, {"session": "S1", "weird": object()})
    rec = json.loads((tmp_path / "logs" / "usage.jsonl").read_text().strip())
    assert rec["session"] == "S1"


def test_empty_record_still_writes_line(tmp_path):
    append_usage_event(tmp_path, {})
    lines = (tmp_path / "logs" / "usage.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {}

"""Regression tests: every context-engine.jsonl event carries a session id.

Work order P2 (2026-08-26): calibration and task_aware_prune events were
emitted WITHOUT a session field, so engine telemetry could not be
attributed to a session. Only archive events (emitted by run_agent) had
one. These tests pin the fix — spec: every event the engines emit contains
a non-empty "session", equal to the id passed to on_session_start, or
"none" when no session start was seen.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.context_engine.semantic_vector import SemanticVectorContextEngine
from plugins.context_engine.rolling_window import RollingWindowContextEngine

SV = importlib.import_module("plugins.context_engine.semantic_vector")
RW = importlib.import_module("plugins.context_engine.rolling_window")


class MockEmbedder:
    def embed(self, text):
        return [1.0, 0.0]


def make_sv(**kw):
    e = SemanticVectorContextEngine(**kw)
    e.context_length = kw.pop("context_length", 600_000)
    e.threshold_tokens = int(e.context_length * e.threshold_percent)
    e._embedding_engine = MockEmbedder()
    return e


@pytest.fixture
def capture(monkeypatch):
    events = []
    monkeypatch.setattr(SV, "context_engine_log", events.append)
    monkeypatch.setattr(RW, "context_engine_log", events.append)
    return events


class TestSessionTagging:
    def test_sv_calibration_carries_session(self, capture):
        e = make_sv()
        e.on_session_start("sess-abc")
        e._last_archive_post_est = 1000
        e._last_archive_path = "semantic"
        e.update_from_response({"prompt_tokens": 5000})
        cal = [ev for ev in capture if ev.get("type") == "calibration"]
        assert cal, "calibration event not emitted"
        assert cal[-1]["session"] == "sess-abc"

    def test_sv_defaults_to_none_without_session_start(self, capture):
        e = make_sv()
        e._last_archive_post_est = 100
        e._last_archive_path = "semantic"
        e.update_from_response({"prompt_tokens": 400})
        cal = [ev for ev in capture if ev.get("type") == "calibration"]
        assert cal and cal[-1]["session"] == "none"

    def test_sv_session_rotates_on_new_session(self, capture):
        e = make_sv()
        e.on_session_start("first")
        e.on_session_start("second")
        e._last_archive_post_est = 100
        e.update_from_response({"prompt_tokens": 400})
        cal = [ev for ev in capture if ev.get("type") == "calibration"]
        assert cal[-1]["session"] == "second"

    def test_rw_calibration_carries_session(self, capture):
        e = RollingWindowContextEngine()
        e.on_session_start("rw-sess")
        e._last_archive_post_est = 100
        e._last_archive_path = "rw_fallback"
        e.update_from_response({"prompt_tokens": 900})
        cal = [ev for ev in capture if ev.get("type") == "calibration"]
        assert cal, "rw calibration event not emitted"
        assert cal[-1]["session"] == "rw-sess"

    def test_task_aware_prune_site_emits_session(self):
        # The prune event fires deep in the rolling fallback; guard the
        # emit site directly (repo precedent: source-line guard tests).
        src = Path(SV.__file__).read_text(encoding="utf-8")
        idx = src.find('"type": "task_aware_prune"')
        assert idx != -1, "task_aware_prune emit site vanished"
        window = src[idx : idx + 260]
        assert '"session":' in window, (
            "task_aware_prune event no longer emits a session field"
        )

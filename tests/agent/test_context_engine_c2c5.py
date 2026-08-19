"""Tests for c2 (archive instrumentation) + c3 (rw kwarg wiring) + c5
(per-topic rolling tail), 2026-08-19.

Pure logic — mocked embedder, no GPU, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.context_engine import context_engine_log, estimate_content_tokens
from plugins.context_engine.semantic_vector import SemanticVectorContextEngine
from plugins.context_engine.rolling_window import RollingWindowContextEngine


class MockEmbedder:
    """Deterministic 2-d embeddings by keyword."""

    def embed(self, text):
        if "A" in text[:4]:
            return [1.0, 0.0]
        if "B" in text[:4]:
            return [0.0, 1.0]
        return [0.5, 0.5]


def make_engine(**kw):
    e = SemanticVectorContextEngine(**kw)
    e.context_length = kw.pop("context_length", 600_000)
    e.threshold_tokens = int(e.context_length * e.threshold_percent)
    e._embedding_engine = MockEmbedder()
    return e


def make_topic_msgs(n, tokens_each):
    """n non-system messages alternating topicA/topicB, plus one system msg."""
    chars = tokens_each * 4
    out = [{"role": "system", "content": "sys"}]
    for i in range(n):
        topic = "A" if i % 2 == 0 else "B"
        out.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"{topic}-turn{i} " + "x" * (chars - 12),
            }
        )
    return out


def contents(messages):
    return [m.get("content", "") for m in messages]


class TestEstimateContentTokens:
    def test_counts_content_only(self):
        msgs = [
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": None, "tool_calls": [{"a": "bbb"}]},
            {"role": "tool", "content": "y" * 800},
        ]
        assert estimate_content_tokens(msgs) == 300

    def test_empty_and_none(self):
        assert estimate_content_tokens([]) == 0
        assert estimate_content_tokens(None) == 0


class TestContextEngineLog:
    def test_writes_jsonl(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        context_engine_log({"type": "test", "value": 42})
        f = tmp_path / "logs" / "context-engine.jsonl"
        assert f.exists()
        line = json.loads(f.read_text().strip().splitlines()[-1])
        assert line["type"] == "test"
        assert line["value"] == 42
        assert "ts" in line

    def test_appends(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        context_engine_log({"type": "a"})
        context_engine_log({"type": "b"})
        lines = (tmp_path / "logs" / "context-engine.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_never_raises_on_bad_values(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        context_engine_log({"type": "bad", "value": object()})  # default=str
        context_engine_log({"type": "bad2", "value": {"nested": [1, 2]}})


class TestActiveTail:
    """c5: per-topic rolling tail."""

    def test_tail_bounded_keeps_last_k_per_topic(self):
        """active_tail_turns=4: each active topic keeps only its last 4 turns.
        context_length=600K → threshold 450K. Input 40 msgs × 20K = 800K
        (over threshold). Kept = 9 protected + 4 A + 4 B = 17 × 20K = 340K
        (< 450K, so no rolling-window fallback)."""
        e = make_engine(active_tail_turns=4, protect_first_n=3, protect_last_n=6)
        msgs = make_topic_msgs(40, 20_000)
        result = e.archive(msgs, current_tokens=460_000)
        got = contents(result)
        # A turns: 0,2,...,38. Tail keeps 32,34,36,38.
        assert any(c.startswith("A-turn32") for c in got)
        assert any(c.startswith("A-turn34") for c in got)
        # A turns 0,2,...,30 are gone. (A-turn0 and B-turn1 are inside the
        # protected first-3 band: msgs 0,1,2 = system + A-turn0 + B-turn1.)
        old_a = {f"A-turn{i} " for i in range(2, 32, 2)}
        assert not any(any(c.startswith(p) for p in old_a) for c in got)
        # B turns: 1,3,...,39. Tail keeps 33,35,37,39.
        old_b = {f"B-turn{i} " for i in range(3, 33, 2)}
        assert not any(any(c.startswith(p) for p in old_b) for c in got)
        assert any(c.startswith("B-turn33") for c in got)
        assert e.archive_count == 1
        assert e._last_archive_path == "semantic"

    def test_legacy_keeps_all_active_turns(self):
        """active_tail_turns=0 (default): below-threshold → nothing pruned;
        nothing active is ever dropped by the tail logic."""
        e = make_engine(active_tail_turns=0, context_length=2_000_000)
        msgs = make_topic_msgs(20, 20_000)  # 400K < 1.5M threshold
        result = e.archive(msgs, current_tokens=460_000)
        assert len(result) == len(msgs)
        assert e._last_archive_path == "semantic_below_threshold"

    def test_dormant_topic_fully_pruned_even_with_tail(self):
        """A topic untouched for dormancy_decay turns is Dormant → all its
        turns pruned, regardless of the tail."""
        e = make_engine(
            active_tail_turns=10, dormancy_decay=3, protect_first_n=1, protect_last_n=2
        )
        # 12 turns: A,B,A,B,A,B,A,B,A,B,A,B — A last seen turn 10, B turn 11.
        # gap for A = 11-10 = 1 < 3 → still Active. Make it longer:
        msgs = make_topic_msgs(14, 20_000)
        # Force topic A dormant by hand after assignment is not possible
        # pre-archive; instead set dormancy_decay=2: A gap=1... still active.
        # Use an explicit sequence: A-only then 6 B-only turns.
        chars = 20_000 * 4
        custom = [{"role": "system", "content": "sys"}]
        for i in range(3):
            custom.append({"role": "user", "content": f"A-turn{i} " + "x" * (chars - 12)})
        for i in range(6):
            custom.append({"role": "assistant", "content": f"B-turn{i} " + "x" * (chars - 12)})
        e.dormancy_decay = 3
        result = e.archive(custom, current_tokens=460_000)
        got = contents(result)
        assert not any(c.startswith("A-turn") for c in got)
        assert any(c.startswith("B-turn") for c in got)


class TestCalibration:
    def test_semantic_engine_pairs_actual_vs_estimate(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = SemanticVectorContextEngine()
        e._last_archive_post_est = 50_000
        e._last_archive_path = "rw_fallback"
        e.update_from_response({"prompt_tokens": 52_500, "completion_tokens": 100, "total_tokens": 52_600})
        f = tmp_path / "logs" / "context-engine.jsonl"
        lines = [json.loads(l) for l in f.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["type"] == "calibration"
        assert lines[0]["estimated"] == 50_000
        assert lines[0]["actual"] == 52_500
        assert lines[0]["ratio"] == 1.05
        assert lines[0]["path"] == "rw_fallback"
        # state cleared — a second response writes nothing
        e.last_prompt_tokens = 0
        e.update_from_response({"prompt_tokens": 60_000})
        lines = [json.loads(l) for l in f.read_text().strip().splitlines()]
        assert len(lines) == 1

    def test_no_calibration_without_prior_archive(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = SemanticVectorContextEngine()
        e.update_from_response({"prompt_tokens": 60_000})
        f = tmp_path / "logs" / "context-engine.jsonl"
        assert not f.exists()

    def test_rw_fallback_sets_est_and_path(self):
        e = SemanticVectorContextEngine(window_size=0, archive_target=0.5)
        e.context_length = 262_144
        e.threshold_tokens = int(262_144 * 0.75)
        msgs = make_topic_msgs(40, 20_000)
        e._rolling_window_fallback(msgs)
        assert e._last_archive_path == "rw_fallback"
        assert e._last_archive_post_est > 0
        assert e.archive_count == 1


class TestRollingWindowWiring:
    """C9-B: the rolling_window engine must honor its full config section."""

    def test_threshold_percent_wired(self):
        e = RollingWindowContextEngine(threshold_percent=0.6)
        assert e.threshold_percent == 0.6
        e.update_model(model="m", context_length=100_000)
        assert e.threshold_tokens == 60_000

    def test_fallback_knobs_wired(self):
        e = RollingWindowContextEngine(
            archive_target=0.25, hard_ceiling_percent=0.8, danger_zone_percent=0.85
        )
        assert e.archive_target == 0.25
        assert e.hard_ceiling_percent == 0.8
        assert e.danger_zone_percent == 0.85

    def test_defaults_preserved_when_unset(self):
        e = RollingWindowContextEngine()
        assert e.threshold_percent == 0.75
        assert e.archive_target == 0.0

    def test_archive_increments_count_and_sets_markers(self):
        e = RollingWindowContextEngine(window_size=10, max_tokens=10_000_000)
        e.context_length = 100_000
        e.threshold_tokens = 75_000
        msgs = [{"role": "user", "content": "x" * 1000 * i} for i in range(1, 40)]
        result = e.archive(msgs, current_tokens=90_000)
        assert e.archive_count == 1
        assert e._last_archive_path == "rolling_window"
        assert e._last_archive_post_est > 0
        assert len(result) <= 10

    def test_rw_calibration_parity(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = RollingWindowContextEngine(window_size=10, max_tokens=10_000_000)
        e._last_archive_post_est = 10_000
        e._last_archive_path = "rolling_window"
        e.update_from_response({"prompt_tokens": 11_000, "completion_tokens": 5, "total_tokens": 11_005})
        f = tmp_path / "logs" / "context-engine.jsonl"
        lines = [json.loads(l) for l in f.read_text().strip().splitlines()]
        assert len(lines) == 1
        assert lines[0]["type"] == "calibration"
        assert lines[0]["engine"] == "rolling_window"
        assert lines[0]["ratio"] == 1.1

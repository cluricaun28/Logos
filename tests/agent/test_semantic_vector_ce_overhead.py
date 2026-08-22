"""Tests for C-E (2026-08-22): static-overhead estimator calibration.

Measured 2026-08-21 (96 c2 pairs): the chars//4 message-content estimate
undercounts the REAL prompt 2.7-3.1x (median 2.73) because it omits the
static payload (system prompt + tool schemas + injections ≈ 50-55K).
C-E calibrates `_overhead_est` from the c2 pairs and adds it back ONLY at
full-context threshold comparisons, so:
  1. the below-threshold short-circuit in _semantic_archive stops no-op'ing
     (the c5 per-topic tail is actually applied),
  2. the rolling-window emergency brake engages on time,
  3. the c2 calibration JSONL gains an honest `overhead` key.

Pure logic — mocked embedder, no GPU, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.context_engine import estimate_content_tokens
from plugins.context_engine.semantic_vector import SemanticVectorContextEngine


class MockEmbedder:
    """Deterministic 2-d embeddings by keyword (same as c2c5 tests)."""

    def embed(self, text):
        if "A" in text[:4]:
            return [1.0, 0.0]
        if "B" in text[:4]:
            return [0.0, 1.0]
        return [0.5, 0.5]


def make_engine(context_length=100_000, **kw):
    e = SemanticVectorContextEngine(**kw)
    e.context_length = context_length
    e.threshold_tokens = int(e.context_length * e.threshold_percent)
    e._embedding_engine = MockEmbedder()
    return e


def topic_msgs(n, tokens_each, topic="A"):
    """n non-system messages of topic, plus one tiny system msg."""
    chars = tokens_each * 4
    out = [{"role": "system", "content": "sys"}]
    for i in range(n):
        out.append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"{topic}-turn{i} " + "x" * (chars - 12),
        })
    return out


def contents(messages):
    return [m.get("content", "") for m in messages]


class TestOverheadCalibration:
    def test_overhead_from_c2_pair(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = SemanticVectorContextEngine()
        e._last_archive_post_est = 20_000
        e._last_archive_path = "semantic"
        e.update_from_response(
            {"prompt_tokens": 120_000, "completion_tokens": 100, "total_tokens": 120_100}
        )
        assert e._overhead_est == 100_000
        f = tmp_path / "logs" / "context-engine.jsonl"
        line = json.loads(f.read_text().strip().splitlines()[-1])
        assert line["type"] == "calibration"
        # additive key — the existing keys must all still be there
        assert line["estimated"] == 20_000
        assert line["actual"] == 120_000
        assert line["ratio"] == 6.0
        assert line["path"] == "semantic"
        assert line["overhead"] == 100_000

    def test_overhead_clamped_at_zero(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = SemanticVectorContextEngine()
        e._overhead_est = 90_000  # previous session measurement
        e._last_archive_post_est = 200_000
        e._last_archive_path = "rw_fallback"
        e.update_from_response({"prompt_tokens": 150_000})
        assert e._overhead_est == 0
        line = json.loads((tmp_path / "logs" / "context-engine.jsonl").read_text().strip().splitlines()[-1])
        assert line["overhead"] == 0

    def test_no_pair_no_overhead_change(self):
        e = SemanticVectorContextEngine()
        e._overhead_est = 55_000
        e.update_from_response({"prompt_tokens": 60_000})  # no prior archive pair
        assert e._overhead_est == 55_000

    def test_session_reset_clears_overhead(self):
        e = make_engine()
        e._overhead_est = 55_000
        e.on_session_reset()
        assert e._overhead_est == 0

    def test_estimate_total_is_additive(self):
        e = make_engine()
        msgs = [{"role": "user", "content": "x" * 400}]
        assert e._estimate_total(msgs) == 100  # overhead 0
        e._overhead_est = 50_000
        assert e._estimate_total(msgs) == 50_100
        # estimate_content_tokens itself is untouched (message-only)
        assert estimate_content_tokens(msgs) == 100


class TestShortCircuitFix:
    """The headline C-E bug: below-threshold short-circuit used to return
    the ORIGINAL list (c5 tail never applied) because the message-only
    estimate undercounts. With calibrated overhead it now prunes."""

    def _engine_and_msgs(self, overhead):
        # 4 topic-A turns × 10K tokens = 40K content tokens < 75K threshold.
        # With 50K static overhead the real prompt is 90K > 75K → archive
        # must fire. active_tail_turns=1 keeps only the last A turn.
        e = make_engine(
            active_tail_turns=1, protect_first_n=1, protect_last_n=1
        )
        e._overhead_est = overhead
        msgs = topic_msgs(4, 10_000)
        return e, msgs

    def test_zero_overhead_reproduces_today_behavior(self):
        """Regression: overhead == 0 → 40K < 75K → short-circuit, original
        list returned, nothing pruned (today's exact behavior)."""
        e, msgs = self._engine_and_msgs(0)
        result = e.archive(msgs)  # current_tokens=None
        assert len(result) == len(msgs)
        assert e._last_archive_path == "semantic_below_threshold"
        assert e.archive_count == 0

    def test_overhead_makes_short_circuit_apply_keep_set(self):
        e, msgs = self._engine_and_msgs(50_000)
        result = e.archive(msgs)  # current_tokens=None
        got = contents(result)
        # NOTE: _inject_state_map prepends a topic state map onto the last
        # assistant message's content, so match on long content runs that
        # the (truncated-name) state map cannot contain.
        assert any(f"A-turn3 {'x' * 100}" in c for c in got)
        assert not any("A-turn1 " in c for c in got)
        assert not any("A-turn2 " in c for c in got)
        assert e._last_archive_path == "semantic"
        assert e.archive_count == 1

    def test_explicit_current_tokens_still_authoritative(self):
        """When run_agent passes the measured prompt_tokens, the estimate
        (overhead or not) is not used for the trigger — behavior unchanged.
        80K measured is over the 75K threshold but under the 90K danger
        zone, so the plain semantic path runs."""
        e, msgs = self._engine_and_msgs(50_000)
        result = e.archive(msgs, current_tokens=80_000)
        assert e._last_archive_path == "semantic"
        assert e.archive_count == 1


class TestFallbackTriggerFix:
    """Second C-E site: post-prune result compared against the threshold.
    The message-only number said 'fine'; content + overhead crosses it, so
    the rolling-window brake must engage (it used to be ~3x too late)."""

    def _engine_and_msgs(self, overhead):
        # 4 turns × 10K = 40K content. tail=2 keeps 2 turns = 20K result.
        # With 60K overhead the real post-prune prompt is 80K > 75K →
        # fallback; without it 20K < 75K → no fallback.
        e = make_engine(
            active_tail_turns=2, protect_first_n=1, protect_last_n=1
        )
        e._overhead_est = overhead
        msgs = topic_msgs(4, 10_000)
        return e, msgs

    def test_zero_overhead_no_fallback(self):
        """overhead 0 → 40K content < 75K threshold → today's short-circuit:
        original list, nothing pruned."""
        e, msgs = self._engine_and_msgs(0)
        result = e.archive(msgs)
        assert e._last_archive_path == "semantic_below_threshold"
        assert len(result) == len(msgs)
        assert e.archive_count == 0

    def test_overhead_engages_rolling_window_brake(self):
        e, msgs = self._engine_and_msgs(60_000)
        result = e.archive(msgs)
        assert e._last_archive_path == "rw_fallback"
        # Brake dropped the oldest non-system turns to get under target
        assert len(result) < len(msgs)
        assert not any(c.startswith("A-turn1") for c in contents(result))


class TestRollingWindowFallbackOverhead:
    """Inside _rolling_window_fallback the drop-oldest counter now includes
    the overhead, so the loop drops to a smaller working set and the
    floor/ceiling stop conditions see the true total."""

    def _run(self, overhead, n=10):
        e = make_engine(archive_target=0.3, protect_first_n=1, protect_last_n=1)
        e._overhead_est = overhead
        msgs = topic_msgs(n, 10_000)
        return e._rolling_window_fallback(msgs), e

    def test_zero_overhead_unchanged(self):
        result, e = self._run(0)
        # target = 30K of 100K; content 100K → drop to ~3 messages
        assert e._last_archive_path == "rw_fallback"
        assert e.archive_count == 1

    def test_overhead_drops_more(self):
        result_no, _ = self._run(0)
        result_yes, _ = self._run(50_000)
        # 50K overhead: loop keeps dropping until content ≤ 30K - 50K,
        # i.e. to the working-window floor — fewer messages survive.
        assert len(result_yes) < len(result_no)

    def test_overhead_never_drops_system(self):
        result, _ = self._run(80_000)
        assert any(m.get("role") == "system" for m in result)

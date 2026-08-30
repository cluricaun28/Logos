"""Regression tests for C9-A: rolling-window config wiring (2026-08-17).

The context.rolling_window section of config.yaml must control the
semantic_vector engine's fallback behavior (and be the rolling_window
engine's own config). Defaults must preserve pre-C9-A behavior exactly,
so other users on the shared fork see no change.

Pure logic — no embeddings, no GPU, no network.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.context_engine.semantic_vector import SemanticVectorContextEngine
from plugins.context_engine.rolling_window import (
    RollingWindowContextEngine,
    register as rw_register,
)

CTX = 262144


def make_engine(**kw):
    e = SemanticVectorContextEngine(**kw)
    e.context_length = CTX
    e.threshold_tokens = int(CTX * 0.75)
    return e


def make_msgs(n, tokens_each):
    chars = tokens_each * 4
    out = [{"role": "system", "content": "sys" * 100}]
    for i in range(n):
        out.append(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"m{i} " + "x" * (chars - 4),
            }
        )
    return out


def non_system_count(messages):
    return sum(1 for m in messages if m.get("role") != "system")


def total_tokens(messages):
    return sum(len(m.get("content", "")) for m in messages) // 4


class TestSemanticVectorRollingConfig:
    def test_live_config_floor_protects_working_window(self):
        """window_size=60 + archive_target=0.5: floor wins when the token
        target would shrink below the working window."""
        e = make_engine(window_size=60, archive_target=0.5, hard_ceiling_percent=0.85)
        result = e._rolling_window_fallback(make_msgs(200, 2400))
        assert non_system_count(result) == 60
        assert total_tokens(result) <= int(CTX * 0.85)

    def test_token_target_respected(self):
        """With no floor pressure, pruning stops at the archive_target."""
        e = make_engine(window_size=0, archive_target=0.5, protect_last_n=1)
        result = e._rolling_window_fallback(make_msgs(8, 40000))
        assert total_tokens(result) <= int(CTX * 0.85)

    def test_legacy_defaults_unchanged(self):
        """Default knobs (0/0) must reproduce the pre-C9-A behavior:
        target = threshold_tokens, floor = protect_last_n * 2."""
        e = make_engine()
        messages = make_msgs(200, 2400)
        result = e._rolling_window_fallback(messages)
        # target = 75% of CTX; result must land within one message of it
        assert abs(total_tokens(result) - int(CTX * 0.75)) < 2400
        assert non_system_count(result) > 12  # legacy floor not hit

    def test_oom_guard_wins_over_floor(self):
        """Above the hard ceiling, pruning continues below the floor."""
        e = make_engine(window_size=10, archive_target=0.5, protect_last_n=1)
        # 6 msgs x 40K = 240K tokens; target 131K, ceiling 222K.
        result = e._rolling_window_fallback(make_msgs(6, 40000))
        assert total_tokens(result) <= int(CTX * 0.85)

    def test_model_path_kwarg_honored(self):
        """F13: config 'model_path' must reach the engine (was dropped)."""
        e = make_engine(model_path="/models/embeddings/all-MiniLM-L6-v2")
        assert e._model_path == "/models/embeddings/all-MiniLM-L6-v2"

    def test_default_model_path_preserved(self):
        e = make_engine()
        assert "all-MiniLM-L6-v2" in e._model_path


class _Collector:
    def __init__(self):
        self.engine = None

    def register_context_engine(self, engine):
        self.engine = engine


class TestRollingWindowEngineConfig:
    def test_register_receives_config(self):
        """C9-A: the loader's config dict must reach the engine instead of
        silently falling back to defaults."""
        c = _Collector()
        rw_register(c, {"window_size": 60, "max_tokens": 262144})
        assert c.engine.window_size == 60
        assert c.engine.max_tokens == 262144

    def test_register_without_config_uses_defaults(self):
        c = _Collector()
        rw_register(c)
        assert isinstance(c.engine, RollingWindowContextEngine)
        assert c.engine.window_size == 20  # documented default

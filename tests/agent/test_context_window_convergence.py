"""Stage 3 (2026-09-23 context-window project): fake-loop convergence.

Simulates a full agent session against a deterministic provider model:
    actual_prompt_tokens = full_payload_estimate + STATIC_OVERHEAD
with STATIC_OVERHEAD = 50,000 (the ~50-55K static payload measured by C-E:
system prompt + tool schemas + injections).

Verifies the dynamic the production incident was missing:
  1. The learned _payload_delta converges to the true overhead in a few
     samples (EMA).
  2. The pre-send calibrated estimate tracks the REAL prompt_tokens, so the
     threshold gate fires BEFORE the request that would blow the window —
     not after a context-length error.
  3. Archive fires once, near the real 85% (222,822 of 262,144), and does
     NOT re-fire every turn (no spam).
  4. The pre-send hard cap holds on every turn: send estimate never
     exceeds context_length - 4096.

Real-data anchor (exx context-engine.jsonl, 9/23): the old content-only
estimate undercounted real prompt_tokens by a stable ~1.73x (median 2.1x,
max 22.7x) — the last pre-incident archive saw est=12,600 vs actual=53,531.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.context_engine import (
    estimate_content_tokens,
    estimate_messages_tokens_full,
    estimate_request_tokens_full,
)
from plugins.context_engine.rolling_window import (
    HARD_CAP_RESERVE,
    RollingWindowContextEngine,
)

CONTEXT_LENGTH = 262_144  # real vLLM max_model_len for Qwen3.8-27B
STATIC_OVERHEAD = 50_000  # measured ~50-55K (C-E comment)
TOOL_CHARS = 20_000  # ~5K tokens per tool result, typical terminal output


def _make_engine():
    e = RollingWindowContextEngine(
        context_length=CONTEXT_LENGTH,
        threshold_percent=0.85,
        window_size=40,
        max_tokens=CONTEXT_LENGTH,
        protect_first_n=3,
        protect_last_n=12,
        task_aware=False,
    )
    e.update_model("Qwen3.8-27B", CONTEXT_LENGTH)
    assert e.threshold_tokens == int(CONTEXT_LENGTH * 0.85)
    return e


def _turn_messages(n_turns: int):
    msgs = [
        {"role": "system", "content": "You are a research agent. " * 200},
        {"role": "user", "content": "Do a big research task."},
    ]
    for i in range(n_turns):
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "terminal",
                             "arguments": json.dumps({"command": f"cmd {i}"})},
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "output line " * (TOOL_CHARS // 12),
        })
    return msgs


class FakeProvider:
    """actual = full-payload chars//4 + STATIC_OVERHEAD (char-based,
    matching the estimator's base rate so the learned delta isolates the
    static payload exactly)."""

    def __init__(self):
        self.calls = 0

    def respond(self, messages):
        self.calls += 1
        base = estimate_messages_tokens_full(messages)
        actual = base + STATIC_OVERHEAD
        return {"prompt_tokens": actual, "completion_tokens": 200,
                "total_tokens": actual + 200}


def test_convergence_and_archive_brake():
    engine = _make_engine()
    provider = FakeProvider()

    n = 4
    messages = _turn_messages(n)
    sends = []            # (turn, pre_send_est, actual, cap_held)
    archive_turns = []    # turns on which archive fired
    first_archive_actual = None

    def _append_turn(msgs, i):
        """Append one assistant(tool_call)+tool pair to a COPY of msgs."""
        out = list(msgs)
        out.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "terminal",
                             "arguments": json.dumps({"command": f"cmd {i}"})},
            }],
        })
        out.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "output line " * (TOOL_CHARS // 12),
        })
        return out

    for turn in range(1, 60):
        # Session grows by one assistant+tool turn each iteration, on the
        # LIVE (possibly already-archived) message list.
        n += 1
        messages = _append_turn(messages, n)
        # ── pre-send: hard cap + record estimate ──
        capped = engine.enforce_hard_cap(messages)
        est = engine.estimate_full_request(capped)
        # Production guard records the DETERMINISTIC base (no learned
        # delta) — mirrors _apply_pre_send_context_guard in run_agent.py.
        engine.record_send_estimate(estimate_request_tokens_full(capped))
        # ── send + provider responds with REAL tokens ──
        usage = provider.respond(capped)
        actual = usage["prompt_tokens"]
        engine.update_from_response(usage)
        sends.append((turn, est, actual, est <= CONTEXT_LENGTH - HARD_CAP_RESERVE))

        # ── archive brake: fires on the REAL count crossing threshold ──
        if engine.should_archive(actual):
            if first_archive_actual is None:
                first_archive_actual = actual
            archive_turns.append(turn)
            messages = engine.archive(messages, current_tokens=actual)

        if turn >= 45 and not engine.should_archive(actual):
            break  # converged below threshold; stop

    # 1. Delta converged to the TRUE static overhead — tight tolerance:
    #    with base-only recording the EMA hits O exactly (fake is exact);
    #    the old base+delta recording plateaued at O/2 and FAILS this.
    assert engine._payload_samples >= 5
    assert abs(engine._payload_delta - STATIC_OVERHEAD) <= 1_000, \
        f"delta {engine._payload_delta} vs true {STATIC_OVERHEAD}"
    # the calibrated pre-send estimate tracks the REAL prompt_tokens
    last_est, last_actual = sends[-1][1], sends[-1][2]
    assert abs(last_est - last_actual) <= 2_000, \
        f"calibrated est {last_est} not tracking real {last_actual}"

    # 2. Archive fired — and fired on the REAL ~85% line, not long after
    assert archive_turns, "archive never fired"
    assert first_archive_actual is not None
    assert first_archive_actual >= int(CONTEXT_LENGTH * 0.85) * 0.97, \
        f"archive fired too early: {first_archive_actual}"
    # and never blew past the hard ceiling before firing
    assert first_archive_actual < CONTEXT_LENGTH - HARD_CAP_RESERVE, \
        f"request blew the window before archive fired: {first_archive_actual}"

    # 3. No spam: after the first archive, the next turn is under threshold
    idx = archive_turns[0] - 1  # 0-based index of first archive turn in sends
    next_est, next_actual = sends[idx + 1][1], sends[idx + 1][2]
    assert next_actual < engine.threshold_tokens, \
        f"archive spam: next turn still {next_actual} >= {engine.threshold_tokens}"
    # and few total archive fires across the whole session
    assert len(archive_turns) <= 3, f"archive fired {len(archive_turns)}x — spam"

    # 4. Hard cap held on every pre-send
    for turn, est, actual, held in sends:
        assert held, f"turn {turn}: pre-send est {est} > cap {CONTEXT_LENGTH - HARD_CAP_RESERVE}"

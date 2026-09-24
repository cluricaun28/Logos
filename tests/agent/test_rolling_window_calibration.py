"""Stage 1 (2026-09-23 context-window project): rolling_window calibration.

Verifies the payload-delta calibration + full-request estimation +
deterministic hard cap that replace the content-only (2-22x undercount)
estimates for threshold/danger-zone/cap decisions.

Rollout rule: rolling_window is the UNUSED engine — this test file must pass
before the same hooks touch semantic_vector (active) or ContextCompressor.
"""
import copy
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


def _make_engine(**overrides):
    base = dict(
        context_length=100_000,
        threshold_percent=0.85,  # 85,000 tokens
        window_size=20,
        max_tokens=100_000,
        protect_first_n=3,
        protect_last_n=6,
        task_aware=False,
    )
    base.update(overrides)
    e = RollingWindowContextEngine(**base)
    # mirror production: run_agent passes context_length via update_model
    e.update_model("test-model", e.context_length)
    return e


def _tool_session(n_tool_msgs=20, chars_per_result=20_000):
    """Tool-heavy session: 20 big terminal/read_file results."""
    msgs = [{"role": "system", "content": "You are an agent. " * 100}]
    msgs.append({"role": "user", "content": "Do a big research task."})
    for i in range(n_tool_msgs):
        msgs.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": "terminal",
                    "arguments": json.dumps({"command": f"cmd {i}"}),
                },
            }],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": "line of output " * (chars_per_result // 15),
        })
    msgs.append({"role": "assistant", "content": "Done, here is the summary."})
    return msgs


# -- estimation functions ------------------------------------------------------

def test_full_counts_tool_calls_that_content_only_misses():
    """The deterministic gap the new estimator closes: tool_calls JSON.

    content-only counts string bodies; full ALSO counts tool_calls
    (function name + arguments). A session of big arguments + small
    results exposes that gap (string-result-heavy sessions are ~equal
    because the result bodies dominate both estimates).
    """
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({
            "role": "assistant", "content": "tiny",
            "tool_calls": [{
                "id": f"c{i}", "type": "function",
                "function": {"name": "run",
                             "arguments": json.dumps({"payload": "x" * 8000})},
            }],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    content = estimate_content_tokens(msgs)
    full = estimate_messages_tokens_full(msgs)
    # 10 x ~8000 chars of args (~20K tokens) that content-only misses
    assert full - content > 10_000
    assert full > 10 * content


def test_string_result_heavy_session_content_close_to_full():
    """Honest relationship: big string tool results are counted by BOTH,
    so content-only ≈ full there. The learned delta (not the recount) is
    what closes the static-payload gap in that regime."""
    msgs = _tool_session(n_tool_msgs=20, chars_per_result=20_000)
    content = estimate_content_tokens(msgs)
    full = estimate_messages_tokens_full(msgs)
    assert full >= content
    assert full < content * 1.25  # within ~25%: result bodies dominate both


def test_full_request_includes_system_and_schemas():
    msgs = _tool_session(n_tool_msgs=4)
    system = "S " * 10_000
    tools = [{"type": "function", "function": {"name": f"t{i}", "parameters": {"p": "x" * 100}}}
             for i in range(50)]
    base = estimate_messages_tokens_full(msgs)
    req = estimate_request_tokens_full(msgs, system_prompt=system, tools=tools)
    assert req > base + 2_000  # system (~2.5K) + schemas
    assert req > base + len(system) // 4 - 50


def test_full_counts_dict_tool_call_arguments():
    # vLLM receives arguments as parsed dict — estimator must handle both
    args_payload = {"path": "/x" * 500}
    m_str = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps(args_payload),
                    },
                }
            ],
        }
    ]
    m_dict = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": dict(args_payload),
                    },
                }
            ],
        }
    ]
    assert estimate_messages_tokens_full(m_str) > 100
    assert abs(estimate_messages_tokens_full(m_str)
               - estimate_messages_tokens_full(m_dict)) <= 10


# -- delta calibration ----------------------------------------------------------

def test_payload_delta_learns_from_real_usage():
    e = _make_engine()
    msgs = _tool_session(n_tool_msgs=8)
    est = e.estimate_full_request(msgs, system_prompt="sys " * 500,
                                  tools=[{"function": {"name": "t"}}] * 30)
    # simulate a real response: provider says the prompt was 4K bigger
    real = est + 4_000
    e.record_send_estimate(est)
    e.update_from_response({"prompt_tokens": real, "completion_tokens": 100,
                            "total_tokens": real + 100})
    # next estimate must carry the learned delta
    assert e._payload_delta == 4_000
    assert e.estimate_full_request(msgs, system_prompt="sys " * 500,
                                   tools=[{"function": {"name": "t"}}] * 30) == real
    assert e.last_prompt_tokens == real


def test_payload_delta_ema_smoothing_and_nonnegative():
    e = _make_engine()
    msgs = _tool_session(n_tool_msgs=4)
    est = e.estimate_full_request(msgs)
    e.record_send_estimate(est)
    e.update_from_response({"prompt_tokens": est + 8_000, "completion_tokens": 1,
                            "total_tokens": est + 8_001})
    assert e._payload_delta == 8_000
    # second sample 2K below -> EMA midpoint 5_000
    est2 = e.estimate_full_request(msgs)  # includes 8_000 delta now
    e.record_send_estimate(est2)
    e.update_from_response({"prompt_tokens": est2 + 2_000, "completion_tokens": 1,
                            "total_tokens": est2 + 2_001})
    assert e._payload_delta == 5_000
    # actual smaller than estimate -> observed clamps to 0 (no negative delta)
    e.record_send_estimate(50_000)
    e.update_from_response({"prompt_tokens": 10_000, "completion_tokens": 1,
                            "total_tokens": 10_001})
    assert e._payload_delta >= 0
    assert e._payload_samples == 3


def test_no_pairing_without_send_estimate():
    e = _make_engine()
    e.update_from_response({"prompt_tokens": 50_000, "completion_tokens": 1,
                            "total_tokens": 50_001})
    assert e._payload_delta == 0
    assert e._payload_samples == 0


# -- hard cap -------------------------------------------------------------------

def test_hard_cap_noop_when_under_cap():
    e = _make_engine(context_length=500_000)
    msgs = _tool_session(n_tool_msgs=5)
    out = e.enforce_hard_cap(msgs, system_prompt="s", tools=None)
    assert out == msgs  # identical content, under cap


def test_hard_cap_trims_largest_tool_result_first():
    e = _make_engine(context_length=100_000)  # cap = 95_904
    msgs = _tool_session(n_tool_msgs=20, chars_per_result=20_000)  # ~100K+
    # make message #5's tool result the unique largest
    big = {"role": "tool", "tool_call_id": "call_5",
           "content": "Z" * 60_000}
    msgs[11] = big
    before = [copy.deepcopy(m) for m in msgs]
    out = e.enforce_hard_cap(msgs)
    # the largest (60K) is trimmed first; original list untouched
    assert before[11]["content"] == "Z" * 60_000
    assert len(out[11]["content"]) < 60_000
    assert "[hard-capped]" in out[11]["content"]
    # estimate now under cap
    cap = e.context_length - HARD_CAP_RESERVE
    assert e.estimate_full_request(out) <= cap


def test_hard_cap_respects_min_floor():
    e = _make_engine(context_length=50_000, hard_cap_min_tool_chars=2_000)
    # 30 x ~7.4K chars (~56K tokens) vs cap 45_904 -> trims fire
    msgs = _tool_session(n_tool_msgs=30, chars_per_result=8_000)
    out = e.enforce_hard_cap(msgs)
    trimmed = any(
        a.get("content") != b.get("content")
        for a, b in zip(msgs, out) if a.get("role") == "tool"
    )
    assert trimmed  # floor was actually exercised
    for m in out:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            assert len(m["content"]) >= 2_000


def test_hard_cap_deterministic_and_bounded():
    e = _make_engine(context_length=60_000)
    msgs = _tool_session(n_tool_msgs=40, chars_per_result=30_000)
    out1 = e.enforce_hard_cap(copy.deepcopy(msgs))
    out2 = e.enforce_hard_cap(copy.deepcopy(msgs))
    assert out1 == out2  # deterministic


def test_hard_cap_no_context_length_is_noop():
    e = _make_engine(context_length=0)
    msgs = _tool_session(n_tool_msgs=5)
    assert e.enforce_hard_cap(msgs) == msgs


# -- should_archive still uses the real token count -----------------------------

def test_should_archive_threshold_85pct():
    e = _make_engine(context_length=100_000, threshold_percent=0.85)
    assert e.threshold_tokens == 85_000
    assert not e.should_archive(84_999)
    assert e.should_archive(85_001)
    # no arg -> uses last real prompt_tokens
    e.last_prompt_tokens = 90_000
    assert e.should_archive()

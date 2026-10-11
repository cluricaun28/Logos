"""Regression test — context compression must send enable_thinking: False.

Qwen3.x thinking mode can exhaust the summarizer's token budget with zero
visible content: ``content=None`` → the compressor stores an EMPTY summary
→ context amnesia at every compaction (silent — no exception, no log).

The same bug class was fixed in audit_service / synthesis_service as "C7"
(2026-08-17) but context compression was missed. 10/11 probes (fresh
process, local vLLM Qwen3.8-27B):

    :8000 mt=8 thinking ON  -> choices=1 content=None  (all tokens spent
                                       in reasoning_content)
    :8000 mt=8 thinking OFF -> choices=1 content='ok'

RL of record: technology/rl-fts-malformed-root-cause-2026-10-11.md (aux
section).

This test pins the fix: the compressor's call_llm kwargs must carry the
thinking-off extra_body, so a future refactor that drops it fails the
gate instead of silently re-introducing empty summaries.
"""
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor

_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def _make_compressor():
    return ContextCompressor(
        model="qwen3.8-27b-dflash",
        base_url="http://127.0.0.1:8001/v1",
        api_key="sk-test-not-real",
        provider="custom",
        api_mode="chat_completions",
        config_context_length=131072,
        quiet_mode=True,
    )


def _mock_response(content="## Active Task\nNone."):
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    resp.choices = [choice]
    return resp


_TURNS = [
    {"role": "user", "content": "Fix the login bug in auth.py"},
    {"role": "assistant", "content": "Fixed it — the session cookie was expired."},
]


def test_compression_call_sends_thinking_off():
    c = _make_compressor()
    with patch(
        "agent.context_compressor.call_llm", return_value=_mock_response()
    ) as mock_llm:
        summary = c._generate_summary(_TURNS)
    # Summary was produced (mock content, prefixed)
    assert summary is not None and summary.strip()
    mock_llm.assert_called_once()
    kwargs = mock_llm.call_args.kwargs
    assert kwargs["task"] == "compression"
    assert kwargs["max_tokens"] > 0
    assert kwargs["extra_body"] == _THINKING_OFF, (
        "compression call lost the enable_thinking:False extra_body — "
        "Qwen3.x thinking mode will burn the token budget and return "
        "content=None (silent empty summary). See C7."
    )


def test_compression_retry_on_main_keeps_thinking_off():
    """The fallback retry re-enters _generate_summary — the extra_body must
    survive it (same call_kwargs, so this guards the retry path too)."""
    c = _make_compressor()
    c.summary_model = "aux-model-x"
    c.model = "qwen3.8-27b-dflash"

    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Exception("model_not_found: aux-model-x")
        return _mock_response()

    with patch("agent.context_compressor.call_llm", side_effect=fake_call_llm):
        summary = c._generate_summary(_TURNS)
    assert summary is not None and summary.strip()
    # First attempt failed, retry on main succeeded — both sent thinking-off.
    assert len(calls) == 2
    for k in calls:
        assert k["extra_body"] == _THINKING_OFF

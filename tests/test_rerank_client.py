"""Tests for the flag-gated rerank client (Phase 1b of the retrieval upgrade).

Contract under test:
  * ``enabled()`` parses RERANK_ENABLED (off by default).
  * ``rerank()`` returns indices in descending score order.
  * ``rerank()`` returns ``None`` (never raises) on any failure: service
    down, timeout, malformed payload, length mismatch, empty input.
  * Docs are truncated to RERANK_DOC_CHARS before the wire.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rerank_client  # noqa: E402


class _FakeResp:
    """Minimal file-like response: json.load() only needs .read()."""

    def __init__(self, payload):
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._data


def _score_resp(scores):
    return _FakeResp({"data": [{"index": i, "score": s} for i, s in enumerate(scores)]})


class TestEnabled:
    @pytest.mark.parametrize(
        ("val", "expected"),
        [
            (None, False),
            ("", False),
            ("0", False),
            ("false", False),
            ("off", False),
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("on", True),
            (" yes ", True),
        ],
    )
    def test_env_parsing(self, monkeypatch, val, expected):
        if val is None:
            monkeypatch.delenv("RERANK_ENABLED", raising=False)
        else:
            monkeypatch.setenv("RERANK_ENABLED", val)
        assert rerank_client.enabled() is expected


class TestRerank:
    def test_orders_by_score_descending(self, monkeypatch):
        monkeypatch.setattr(
            rerank_client.urllib.request, "urlopen",
            lambda req, timeout=None: _score_resp([0.2, 0.9, 0.5]),
        )
        assert rerank_client.rerank("q", ["a", "b", "c"]) == [1, 2, 0]

    def test_ties_keep_original_order(self, monkeypatch):
        # stable sort: equal scores preserve first-stage order
        monkeypatch.setattr(
            rerank_client.urllib.request, "urlopen",
            lambda req, timeout=None: _score_resp([0.5, 0.5, 0.1]),
        )
        assert rerank_client.rerank("q", ["a", "b", "c"]) == [0, 1, 2]

    def test_service_down_returns_none(self, monkeypatch):
        def boom(req, timeout=None):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(rerank_client.urllib.request, "urlopen", boom)
        assert rerank_client.rerank("q", ["a", "b"]) is None

    def test_malformed_payload_returns_none(self, monkeypatch):
        monkeypatch.setattr(
            rerank_client.urllib.request, "urlopen",
            lambda req, timeout=None: _FakeResp({"nope": 1}),
        )
        assert rerank_client.rerank("q", ["a", "b"]) is None

    def test_score_count_mismatch_returns_none(self, monkeypatch):
        # server returned fewer scores than docs sent
        monkeypatch.setattr(
            rerank_client.urllib.request, "urlopen",
            lambda req, timeout=None: _score_resp([0.5]),
        )
        assert rerank_client.rerank("q", ["a", "b"]) is None

    def test_empty_docs_returns_none(self):
        assert rerank_client.rerank("q", []) is None

    def test_empty_query_returns_none(self):
        assert rerank_client.rerank("", ["a"]) is None

    def test_docs_truncated_to_wire_limit(self, monkeypatch):
        captured = {}

        def fake(req, timeout=None):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _score_resp([0.9, 0.1])

        monkeypatch.setattr(rerank_client.urllib.request, "urlopen", fake)
        rerank_client.rerank("q", ["short", "x" * 5000])
        assert len(captured["payload"]["documents"][0]) == 5
        assert len(captured["payload"]["documents"][1]) == rerank_client.DEFAULT_DOC_CHARS
        assert captured["payload"]["queries"] == ["q"]

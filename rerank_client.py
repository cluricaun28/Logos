"""Rerank client — flag-gated second-stage cross-encoder ranking.

Phase 1b of the retrieval upgrade (plan of record:
``reference-library/technology/rl-retrieval-upgrade-plan.md``).

First-stage retrieval (FTS5 / hybrid) is cheap and indexes the whole
corpus, so it stays the candidate generator. The reranker
(Qwen3-Reranker-8B served by vLLM's scoring API) scores (query, document)
pairs jointly — far more accurate per pair, ~50x more expensive — so it
only ever sees the top-K candidates the first stage already surfaced.

Contract:
  * ``rerank(query, docs)`` returns the indices of ``docs`` in descending
    relevance order, or ``None`` on ANY failure (service down, timeout,
    malformed response, length mismatch). ``None`` means "keep the original
    order" — the reranker must never make a search worse than the first
    stage, and must never raise.
  * Zero new dependencies: stdlib ``urllib`` only.

Environment (all optional; the feature is OFF unless RERANK_ENABLED):
  RERANK_ENABLED    "1"/"true"/"yes"/"on" (case-insensitive, trimmed)
  RERANK_URL        default http://127.0.0.1:8101/score
  RERANK_TIMEOUT    seconds, default 4.0
  RERANK_DOC_CHARS  max chars of each doc sent to the model, default 1400
  RERANK_BLEND      CE weight in the score blend, 0..1, default 0.5
                    (1.0 = full cross-encoder reorder; 0.0 = first stage)
"""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8101/score"
DEFAULT_TIMEOUT = 4.0
# Multi-site evidence windows (book_index.rerank_doc / rl_search
# _rerank_windows) run to ~1400 chars; the old 700 default silently
# truncated them back to the first window (10/20 A/B battery, run 2).
DEFAULT_DOC_CHARS = 1400
DEFAULT_BLEND = 0.5
_MODEL = "rerank"

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def enabled() -> bool:
    """Flag-gated: ON only when RERANK_ENABLED is a truthy value."""
    return os.environ.get("RERANK_ENABLED", "").strip().lower() in _TRUE_VALUES


def _url() -> str:
    return os.environ.get("RERANK_URL", "").strip() or DEFAULT_URL


def _timeout() -> float:
    try:
        return float(os.environ.get("RERANK_TIMEOUT", "") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _doc_chars() -> int:
    try:
        return max(64, int(os.environ.get("RERANK_DOC_CHARS", "") or DEFAULT_DOC_CHARS))
    except ValueError:
        return DEFAULT_DOC_CHARS


def blend_weight() -> float:
    """CE weight in the score blend (RERANK_BLEND, 0..1, default 0.5).

    1.0 = full cross-encoder reorder (measured 10/20: loses to the first
    stage on this corpus — one-line mentions outrank exact title matches),
    0.0 = first-stage order untouched. The blend is the plan's "rerank
    score blended into the fused score" — the first-stage rank term keeps
    an exact match anchored so a single-mention page cannot leapfrog it.
    """
    try:
        w = float(os.environ.get("RERANK_BLEND", "") or DEFAULT_BLEND)
    except ValueError:
        return DEFAULT_BLEND
    return min(1.0, max(0.0, w))


def blend(scores: list[float], weight: float | None = None) -> list[int]:
    """Order docs by ``(1-w) * first-stage rank + w * min-max(CE score)``.

    Input order IS the first-stage order (index i = rank i). The rank term
    is 1/(i+1) — rank 1 = 1.0, decaying. Ties keep original order (stable).
    """
    if weight is None:
        weight = blend_weight()
    n = len(scores)
    if n <= 1:
        return list(range(n))
    smin, smax = min(scores), max(scores)
    rng = (smax - smin) if smax != smin else 1.0
    fused = [
        (1.0 - weight) * (1.0 / (i + 1)) + weight * (s - smin) / rng
        for i, s in enumerate(scores)
    ]
    return sorted(range(n), key=lambda i: -fused[i])


def rerank_scored(
    query: str,
    docs: list[str],
    url: str | None = None,
    timeout: float | None = None,
) -> list[float] | None:
    """CE scores for ``docs`` in input order, or ``None`` on any failure.

    Shared transport for ``rerank`` (pure reorder) and ``blend`` (score
    fusion). Never raises.
    """
    if not query or not docs:
        return None
    payload = {
        "model": _MODEL,
        "queries": [query],
        "documents": [d[: _doc_chars()] for d in docs],
    }
    req = urllib.request.Request(
        url or _url(),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout or _timeout()) as resp:
            data = json.load(resp)
    except Exception:  # noqa: BLE001 — any failure = fallback, never raise
        return None
    try:
        scores = [float(e["score"]) for e in data["data"]]
        if len(scores) != len(docs):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return scores


def rerank(
    query: str,
    docs: list[str],
    url: str | None = None,
    timeout: float | None = None,
) -> list[int] | None:
    """Order ``docs`` by relevance to ``query`` via the vLLM /score API.

    Returns a list of indices into ``docs`` (descending relevance; ties keep
    original order — stable sort), or ``None`` on any failure so the caller
    can fall back to its first-stage ordering.
    """
    scores = rerank_scored(query, docs, url=url, timeout=timeout)
    if scores is None:
        return None
    return sorted(range(len(docs)), key=lambda i: -scores[i])

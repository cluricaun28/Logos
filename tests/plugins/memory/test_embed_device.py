"""Tests for embedding device selection and the silent-FTS-only fix (t3).

Bug under test: when vLLM occupies cuda:0, the RL index build loaded the
embedding model on cuda:0, OOM'd, and the build "succeeded" with
files_embedded=0 — semantic search silently degraded to FTS-only.

Fix under test:
  1. EmbeddingEngine._select_device_candidates ranks GPUs by free memory,
     honors HERMES_EMBED_DEVICE, and falls back to CPU.
  2. _load_model walks candidates, logging each failure.
  3. rl_builder.build_index reports embeddings_ok and logs an ERROR when
     0 embeddings were produced for a non-empty index.
  4. rl_builder.reindex_stale warns when files were re-indexed but none
     embedded.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import types
from pathlib import Path

import pytest

from agent.perpetual_context_db import EmbeddingEngine
from plugins.memory.perpetual_context import rl_builder, rl_schema
from plugins.memory.perpetual_context.rl_builder import RL_EMBED_DIM
from plugins.memory.perpetual_context.rl_index import _extract_file_info

GB = 1024 ** 3


def _fake_torch(free: dict[int, int], raise_on: set[int] | None = None) -> types.SimpleNamespace:
    """Stand-in for the torch surface used by _select_device_candidates."""
    raise_on = raise_on or set()

    def mem_get_info(i: int) -> tuple[int, int]:
        if i in raise_on:
            raise RuntimeError("CUDA error: out of memory")
        return free[i], 100 * GB

    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: bool(free),
            device_count=lambda: len(free),
            mem_get_info=mem_get_info,
        )
    )


# ── _select_device_candidates ──────────────────────────────────────────


def test_env_override_is_only_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_EMBED_DEVICE", "cuda:3")
    candidates = EmbeddingEngine._select_device_candidates(_fake_torch({0: 90 * GB}))
    assert candidates == ["cuda:3"]


def test_cpu_only_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_EMBED_DEVICE", raising=False)
    candidates = EmbeddingEngine._select_device_candidates(_fake_torch({}))
    assert candidates == ["cpu"]


def test_gpus_ranked_by_free_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_EMBED_DEVICE", raising=False)
    torch = _fake_torch({0: 1 * GB, 1: 90 * GB, 2: 3 * GB})
    candidates = EmbeddingEngine._select_device_candidates(torch)
    # most-free first, packed (<2GB) last, CPU final
    assert candidates == ["cuda:1", "cuda:2", "cuda:0", "cpu"]


def test_mem_query_failure_ranked_as_packed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_EMBED_DEVICE", raising=False)
    # cuda:0 has free memory but the query itself OOMs (fully-occupied ctx)
    torch = _fake_torch({0: 90 * GB, 1: 5 * GB}, raise_on={0})
    candidates = EmbeddingEngine._select_device_candidates(torch)
    assert candidates == ["cuda:1", "cuda:0", "cpu"]


def test_packed_gpus_still_tried_before_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HERMES_EMBED_DEVICE", raising=False)
    torch = _fake_torch({0: 0.5 * GB})
    candidates = EmbeddingEngine._select_device_candidates(torch)
    assert candidates == ["cuda:0", "cpu"]


# ── _load_model fallback chain ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_engine_singleton():
    EmbeddingEngine._instance = None
    yield
    EmbeddingEngine._instance = None


def test_load_model_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _FakeST:
        def __init__(self, path, device="cpu", local_files_only=True):
            calls.append(device)
            if device != "cpu":
                raise RuntimeError("CUDA OOM at load")

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", _FakeST, raising=False
    )
    monkeypatch.setattr(
        EmbeddingEngine, "_select_device_candidates", staticmethod(
            lambda torch: ["cuda:0", "cpu"]
        )
    )

    engine = EmbeddingEngine()
    model = engine._load_model()
    assert model is not None
    assert calls == ["cuda:0", "cpu"]
    assert engine._device == "cpu"


def test_load_model_all_candidates_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeST:
        def __init__(self, path, device="cpu", local_files_only=True):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer", _FakeST, raising=False
    )
    monkeypatch.setattr(
        EmbeddingEngine, "_select_device_candidates", staticmethod(
            lambda torch: ["cuda:0"]
        )
    )

    engine = EmbeddingEngine()
    assert engine._load_model() is None
    assert engine._device is None


# ── rl_builder loud failure signals ────────────────────────────────────


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("PRAGMA journal_mode=MEMORY")
    rl_schema.create_tables(conn)
    return conn


def _make_rl_base(base: Path) -> None:
    (base / "ideas").mkdir(parents=True)
    (base / "ideas" / "alpha.md").write_text(
        "# Alpha\n\nAlpha body content for embedding tests.\n", encoding="utf-8"
    )
    (base / "ideas" / "beta.md").write_text(
        "# Beta\n\nBeta body content for embedding tests.\n", encoding="utf-8"
    )


def _good_embed(texts, batch_size=64):
    return [[0.01] * RL_EMBED_DIM for _ in texts]


def _dead_embed(texts, batch_size=64):
    return [None] * len(texts)


def test_build_index_flags_fts_only(tmp_path: Path) -> None:
    _make_rl_base(tmp_path)
    conn = _make_conn()
    lock = threading.RLock()
    stats = rl_builder.build_index(
        conn=conn, lock=lock, rl_base=str(tmp_path),
        file_info_extractor=_extract_file_info, embed_batch_fn=_dead_embed,
    )
    assert stats["files_indexed"] == 2
    assert stats["files_embedded"] == 0
    assert stats["embeddings_ok"] is False
    conn.close()


def test_build_index_ok_when_embedded(tmp_path: Path) -> None:
    _make_rl_base(tmp_path)
    conn = _make_conn()
    lock = threading.RLock()
    stats = rl_builder.build_index(
        conn=conn, lock=lock, rl_base=str(tmp_path),
        file_info_extractor=_extract_file_info, embed_batch_fn=_good_embed,
    )
    assert stats["files_embedded"] == 2
    assert stats["embeddings_ok"] is True
    conn.close()


def test_reindex_stale_warns_on_zero_embedded(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _make_rl_base(tmp_path)
    conn = _make_conn()
    lock = threading.RLock()
    rl_builder.build_index(
        conn=conn, lock=lock, rl_base=str(tmp_path),
        file_info_extractor=_extract_file_info, embed_batch_fn=_good_embed,
    )

    # Make one file stale with an explicit mtime bump (VPS clocks are coarse)
    target = tmp_path / "ideas" / "alpha.md"
    future = time.time() + 30
    os.utime(target, (future, future))

    with caplog.at_level("WARNING", logger="plugins.memory.perpetual_context.rl_builder"):
        count = rl_builder.reindex_stale(
            conn=conn, lock=lock,
            file_info_extractor=_extract_file_info,
            embed_batch_fn=_dead_embed,
            update_file_fn=lambda *a, **k: None,
            rl_base=str(tmp_path),
        )

    assert count == 0
    assert any("0 embedded" in r.message for r in caplog.records)
    conn.close()

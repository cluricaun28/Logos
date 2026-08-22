"""Tests for schema-v2 RL index chunking (full-body search coverage).

Bug under test: pre-v2, ``_extract_file_info`` truncated ``rl_files.body``
to RL_EMBED_MAX_CONTENT_LEN (8000 chars) at extract time. FTS5 keyword
search and the file-level embedding were both blind to content past char
8000. The 2026-08-22 fleet RL-index scan hit this live: a verification
phrase sitting past char 8000 in a 32KB page was unfindable by keyword
search.

Fix under test:
  1. ``_extract_file_info`` stores the FULL cleaned body.
  2. ``_chunk_text`` splits bodies into overlapping, word-bounded chunks
     (RL_CHUNK_SIZE=1000 / RL_CHUNK_OVERLAP=150 — sized for MiniLM's
     256-token window).
  3. build_index / update_file / reindex_stale maintain rl_chunks rows,
     the rl_chunks_fts index, and per-chunk embeddings in
     rl_chunk_embeddings.
  4. ``_fts_search`` prefers chunk FTS (best-rank-per-file) and falls
     back to the legacy file-level FTS when rl_chunks is empty — so a
     pre-v2 DB behaves exactly as before.
  5. ``_semantic_search`` merges file-level + chunk embeddings and
     aggregates max-similarity-per-file.
  6. remove_file cleans all three v2 tables.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.perpetual_context import rl_builder, rl_schema, rl_search
from plugins.memory.perpetual_context.rl_builder import (
    RL_EMBED_DIM,
    _chunk_text,
)
from plugins.memory.perpetual_context.rl_index import _extract_file_info

SENTINEL_HEAD = "headsentinel"
SENTINEL_TAIL = "tailsentinel"
FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit"

VOCAB = [SENTINEL_HEAD, SENTINEL_TAIL, "alpha", "beta", "gamma"]


def fake_vec(text: str) -> list[float]:
    """Deterministic bag-of-words vector over a tiny vocabulary."""
    v = [0.0] * RL_EMBED_DIM
    t = text.lower()
    for i, w in enumerate(VOCAB):
        if w in t:
            v[i] = 1.0
    return v


class FakeEngine:
    def embed(self, text: str) -> list[float]:
        return fake_vec(text)


def fake_embed_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    return [fake_vec(t) for t in texts]


def fake_embed_single(text: str) -> list[float]:
    return fake_vec(text)


def _write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: \"Test {path.stem}\"\ncategory: topics\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _long_body(tail_word: str = SENTINEL_TAIL, head_word: str = SENTINEL_HEAD,
               total: int = 20_000) -> str:
    """~20K char body with distinct sentinels at head and past char 8000."""
    body = f"{head_word} " + (FILLER + " ") * 2000
    body = body[:total - len(tail_word) - 20] + f" filler pad {tail_word} end"
    return body


@pytest.fixture
def rl_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A throwaway RL tree + index DB, isolated from the real home.

    RL_BASE_DIR is pointed at the tmp tree so _extract_file_info
    relativizes file paths exactly as in production (where the passed
    rl_base and RL_BASE_DIR are the same directory).
    """
    monkeypatch.setenv("LOGOS_HOME", str(tmp_path / "home"))
    base = tmp_path / "rl"
    base.mkdir()
    monkeypatch.setattr(
        "plugins.memory.perpetual_context.rl_index.RL_BASE_DIR", str(base)
    )
    return {
        "base": base,
        "db": str(tmp_path / "rl_index.db"),
        "lock": threading.RLock(),
    }


def _build(rl_env: dict) -> dict:
    conn = sqlite3.connect(rl_env["db"], timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        rl_schema.create_tables(conn)
        return rl_builder.build_index(
            conn=conn,
            lock=rl_env["lock"],
            rl_base=str(rl_env["base"]),
            file_info_extractor=_extract_file_info,
            embed_batch_fn=fake_embed_batch,
        )
    finally:
        conn.close()


def _search(rl_env: dict, query: str, **kw) -> list[dict]:
    conn = sqlite3.connect(rl_env["db"], timeout=30)
    try:
        return rl_search.search(
            conn=conn, query=query, embedding_engine=FakeEngine(), **kw
        )
    finally:
        conn.close()


def _fp(rl_env: dict, rel: str) -> str:
    """file_path as _extract_file_info records it — relative to the RL
    base (the fixture monkeypatches RL_BASE_DIR to the tmp tree)."""
    return rel


# ── _chunk_text ────────────────────────────────────────────────────────


def test_chunk_text_short_single_chunk():
    assert _chunk_text("hello world") == ["hello world"]
    assert _chunk_text("") == []
    assert _chunk_text("   \n  ") == []


def test_chunk_text_coverage_overlap_progress():
    for seed in range(20):
        import random

        random.seed(seed)
        n = random.randint(0, 3000)
        base = " ".join(f"u{i:04d}" for i in range(n))
        chunks = _chunk_text(base)
        s = base.strip()
        if not s:
            assert chunks == []
            continue
        assert chunks
        offs = [s.find(c) for c in chunks]
        assert all(o >= 0 for o in offs)
        assert offs[0] == 0
        assert offs[-1] + len(chunks[-1]) == len(s), "tail not covered"
        for i in range(1, len(offs)):
            assert offs[i - 1] < offs[i], "must make progress"
            assert offs[i] < offs[i - 1] + len(chunks[i - 1]), "coverage gap"
        assert all(len(c) <= 1001 for c in chunks)


def test_chunk_text_no_mid_word_splits():
    text = " ".join(f"word{i:02d}" for i in range(400))
    for c in _chunk_text(text):
        assert c == c.strip()


# ── extraction: full body, no truncation ──────────────────────────────


def test_extract_keeps_full_body(tmp_path: Path):
    p = tmp_path / "long.md"
    body = _long_body()
    _write_md(p, body)
    info = _extract_file_info(p)
    assert info is not None
    assert len(info["body"]) > 8000, "body must no longer be truncated at 8000"
    assert SENTINEL_TAIL in info["body"]
    assert info["body"].index(SENTINEL_TAIL) > 8000


# ── build: chunk FTS finds content past char 8000 (the core regression) ─


def test_build_fts_finds_phrase_past_8000(rl_env):
    _write_md(rl_env["base"] / "topics" / "long-page.md", _long_body())
    stats = _build(rl_env)
    assert stats["files_indexed"] == 1
    assert stats["chunks_indexed"] > 8, "20K body must span many chunks"
    assert stats["chunks_embedded"] == stats["chunks_indexed"]

    # FTS-only visibility: query a phrase that exists ONLY past char 8000.
    results = _search(rl_env, SENTINEL_TAIL, semantic_weight=0.0,
                      fts5_weight=1.0)
    assert results, "chunk FTS must find the tail sentinel"
    assert results[0]["file_path"] == _fp(rl_env, "topics/long-page.md")

    # The head sentinel still works (no regression on early content).
    results = _search(rl_env, SENTINEL_HEAD, semantic_weight=0.0,
                      fts5_weight=1.0)
    assert results
    assert results[0]["file_path"] == _fp(rl_env, "topics/long-page.md")


def test_build_chunks_cover_full_body(rl_env):
    body = _long_body()
    _write_md(rl_env["base"] / "topics" / "cov.md", body)
    _build(rl_env)

    fp = _fp(rl_env, "topics/cov.md")
    conn = sqlite3.connect(rl_env["db"])
    chunks = [r[0] for r in conn.execute(
        "SELECT body FROM rl_chunks WHERE file_path = ?"
        " ORDER BY chunk_idx", (fp,)
    )]
    n_rows = conn.execute("SELECT count(*) FROM rl_chunks").fetchone()[0]
    n_fts = conn.execute(
        "SELECT count(*) FROM rl_chunks_fts_docsize").fetchone()[0]
    conn.close()
    # The tail sentinel (past char 8000) must live in the stored chunks.
    assert SENTINEL_TAIL in " ".join(chunks)
    assert SENTINEL_HEAD in chunks[0]
    assert n_fts == n_rows, "chunk FTS drifted from content"


# ── semantic: tail-only match ranks above head-only file ──────────────


def test_semantic_tail_beats_head(rl_env):
    # A: tail sentinel only past char 8000; B: head sentinel + 'gamma'.
    _write_md(rl_env["base"] / "a.md", _long_body())
    _write_md(rl_env["base"] / "b.md",
              _long_body(tail_word="gamma pad", head_word=SENTINEL_HEAD))
    _build(rl_env)

    results = _search(rl_env, SENTINEL_TAIL, fts5_weight=0.0,
                      semantic_weight=1.0)
    paths = [r["file_path"] for r in results]
    assert _fp(rl_env, "a.md") in paths
    if _fp(rl_env, "b.md") in paths:
        assert paths.index(_fp(rl_env, "a.md")) < paths.index(
            _fp(rl_env, "b.md")), "matching tail chunk must outrank B"


# ── update_file / reindex_stale / remove_file chunk paths ─────────────


def _update_file(rl_env, path: Path) -> bool:
    conn = sqlite3.connect(rl_env["db"], timeout=30)
    try:
        return rl_builder.update_file(
            conn=conn, lock=rl_env["lock"], file_path_str=str(path),
            file_info_extractor=_extract_file_info,
            embed_fn=fake_embed_single, embed_dim=RL_EMBED_DIM,
        )
    finally:
        conn.close()


def test_update_file_rechunks(rl_env):
    p = rl_env["base"] / "grow.md"
    _write_md(p, "short body alpha beta")
    _build(rl_env)

    fp = _fp(rl_env, "grow.md")
    conn = sqlite3.connect(rl_env["db"])
    n_before = conn.execute(
        "SELECT count(*) FROM rl_chunks WHERE file_path = ?", (fp,)
    ).fetchone()[0]
    conn.close()
    assert n_before == 1

    _write_md(p, _long_body())
    assert _update_file(rl_env, p)

    conn = sqlite3.connect(rl_env["db"])
    n_after = conn.execute(
        "SELECT count(*) FROM rl_chunks WHERE file_path = ?", (fp,)
    ).fetchone()[0]
    n_emb = conn.execute(
        "SELECT count(*) FROM rl_chunk_embeddings WHERE file_path = ?",
        (fp,),
    ).fetchone()[0]
    conn.close()
    assert n_after > 8
    assert n_emb == n_after

    results = _search(rl_env, SENTINEL_TAIL, semantic_weight=0.0,
                      fts5_weight=1.0)
    assert results and results[0]["file_path"] == fp


def test_reindex_stale_rechunks(rl_env):
    p = rl_env["base"] / "stale.md"
    _write_md(p, "old short body")
    _build(rl_env)

    _write_md(p, _long_body())
    time.sleep(0.02)  # ensure mtime actually changed

    conn = sqlite3.connect(rl_env["db"], timeout=30)
    try:
        updated = rl_builder.reindex_stale(
            conn=conn, lock=rl_env["lock"],
            file_info_extractor=_extract_file_info,
            embed_batch_fn=fake_embed_batch,
            update_file_fn=lambda *_: False,
            rl_base=str(rl_env["base"]), embed_dim=RL_EMBED_DIM,
        )
    finally:
        conn.close()
    assert updated == 1

    fp = _fp(rl_env, "stale.md")
    conn = sqlite3.connect(rl_env["db"])
    n_chunks = conn.execute(
        "SELECT count(*) FROM rl_chunks WHERE file_path = ?", (fp,)
    ).fetchone()[0]
    conn.close()
    assert n_chunks > 8
    results = _search(rl_env, SENTINEL_TAIL, semantic_weight=0.0,
                      fts5_weight=1.0)
    assert results and results[0]["file_path"] == fp


def test_remove_file_cleans_chunks(rl_env):
    p = rl_env["base"] / "gone.md"
    _write_md(p, _long_body())
    _build(rl_env)
    fp = _fp(rl_env, "gone.md")

    conn = sqlite3.connect(rl_env["db"], timeout=30)
    try:
        assert rl_builder.remove_file(
            conn=conn, lock=rl_env["lock"], file_path_str=str(p),
            rl_base=str(rl_env["base"]),
        )
        for table in ("rl_chunks", "rl_chunk_embeddings", "rl_files"):
            n = conn.execute(
                f"SELECT count(*) FROM {table} WHERE file_path = ?", (fp,)
            ).fetchone()[0]
            assert n == 0, f"{table} still has rows"
    finally:
        conn.close()

    # Chunk FTS must no longer match the removed file
    conn = sqlite3.connect(rl_env["db"])
    rows = conn.execute(
        "SELECT count(*) FROM rl_chunks_fts WHERE rl_chunks_fts MATCH ?",
        (SENTINEL_TAIL,),
    ).fetchone()[0]
    conn.close()
    assert rows == 0


# ── backward compatibility: pre-v2 DBs ─────────────────────────────────


def _make_v1_db(rl_env, body: str) -> None:
    """Simulate a pre-v2 index: v1 tables only, 8000-char truncated body."""
    conn = sqlite3.connect(rl_env["db"], timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    rl_schema.create_tables(conn)
    for t in ("rl_chunk_embeddings", "rl_chunks_fts", "rl_chunks"):
        conn.execute(f"DROP TABLE IF EXISTS {t}")
    conn.execute(
        "INSERT INTO rl_files (file_path, title, frontmatter, body,"
        " category, mtime, size, indexed_at)"
        " VALUES ('v1.md', 'V1', '{}', ?, 'topics', ?, ?, ?)",
        (body[:8000], time.time(), len(body), time.time()),
    )
    conn.execute(
        "INSERT INTO rl_embeddings (file_path, embedding, mtime,"
        " created_at) VALUES (?, ?, ?, ?)",
        ("v1.md", b"\x00" * RL_EMBED_DIM * 4, time.time(), time.time()),
    )
    conn.commit()
    conn.close()


def test_v1_db_search_falls_back_no_crash(rl_env):
    _make_v1_db(rl_env, f"{SENTINEL_HEAD} " + (FILLER + " ") * 500)

    results = _search(rl_env, SENTINEL_HEAD)
    # Legacy FTS over the truncated body still finds the head sentinel...
    assert any(r["file_path"] == "v1.md" for r in results), results
    # ...and the tail sentinel (past 8000, never indexed) is a miss —
    # pre-v2 behavior, not a crash.
    results = _search(rl_env, SENTINEL_TAIL, semantic_weight=0.0,
                      fts5_weight=1.0)
    assert all(r["file_path"] != "v1.md" for r in results)


def test_v2_code_on_v1_db_then_rebuild(rl_env):
    """Migrate-by-rebuild: v2 code reads a v1 DB, then a full build fixes it."""
    _make_v1_db(rl_env, _long_body())
    _search(rl_env, SENTINEL_HEAD)  # must not crash on v1-shaped DB

    _write_md(rl_env["base"] / "v1.md", _long_body())
    stats = _build(rl_env)
    assert stats["chunks_indexed"] > 8
    # After rebuild the tail sentinel is findable
    results = _search(rl_env, SENTINEL_TAIL, semantic_weight=0.0,
                      fts5_weight=1.0)
    assert results and results[0]["file_path"] == _fp(rl_env, "v1.md")


def test_missing_chunk_fts_degrades_not_crashes(rl_env):
    """Crash-mid-build shape: chunk rows exist but the FTS table is gone.

    _self_heal_fts repairs this at next init; until then a live search
    must degrade (FTS branch returns []) instead of raising.
    """
    _write_md(rl_env["base"] / "drift.md", _long_body())
    _build(rl_env)
    conn = sqlite3.connect(rl_env["db"])
    conn.execute("DROP TABLE rl_chunks_fts")
    conn.commit()
    conn.close()

    results = _search(rl_env, SENTINEL_TAIL)  # must not raise
    # Semantic branch (chunk embeddings) still surfaces the file.
    assert any(r["file_path"] == _fp(rl_env, "drift.md") for r in results), \
        results

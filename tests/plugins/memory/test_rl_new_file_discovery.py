"""New-file discovery in reindex_stale (2026-10-15).

Bug under test: reindex_stale only scanned rows already tracked in
rl_files, so a file NEWLY added to the RL tree never entered the index —
it stayed invisible to search until a full build_index() ran. Measured
live on the fleet box 2026-10-15: 17 on-disk files missing from rl_files
(the first 10 book-catalog recognition pages + 7 audit/worklog/skill
pages); on 5 topical queries the unindexed pages ranked absent from
top-20, and after ingesting via update_file all 10 ranked #1.

Fix under test: reindex_stale walks the RL tree and adds untracked .md
files to the stale set, so the nightly incremental pass picks up new
pages without a full rebuild.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugins.memory.perpetual_context import rl_builder, rl_schema
from plugins.memory.perpetual_context.rl_builder import RL_EMBED_DIM
from plugins.memory.perpetual_context.rl_index import _extract_file_info

VOCAB = ["zebra", "quince", "alpha"]


def fake_vec(text: str) -> list[float]:
    v = [0.0] * RL_EMBED_DIM
    t = text.lower()
    for i, w in enumerate(VOCAB):
        if w in t:
            v[i] = 1.0
    return v


def _write_md(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\ntitle: \"Test {path.stem}\"\ncategory: topics\n---\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def rl_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Throwaway RL tree + index DB, isolated from the real home."""
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


def _build(rl_env: dict) -> None:
    conn = sqlite3.connect(rl_env["db"], timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        rl_schema.create_tables(conn)
        rl_builder.build_index(
            conn=conn,
            lock=rl_env["lock"],
            rl_base=str(rl_env["base"]),
            file_info_extractor=_extract_file_info,
            embed_batch_fn=lambda texts, batch_size=64: [fake_vec(t) for t in texts],
        )
    finally:
        conn.close()


def _reindex(rl_env: dict) -> int:
    conn = sqlite3.connect(rl_env["db"], timeout=30)
    try:
        return rl_builder.reindex_stale(
            conn=conn,
            lock=rl_env["lock"],
            file_info_extractor=_extract_file_info,
            embed_batch_fn=lambda texts, batch_size=64: [fake_vec(t) for t in texts],
            update_file_fn=lambda *_: False,
            rl_base=str(rl_env["base"]),
            embed_dim=RL_EMBED_DIM,
        )
    finally:
        conn.close()


def test_reindex_stale_discovers_new_files(rl_env):
    a = rl_env["base"] / "existing.md"
    _write_md(a, "existing page alpha")
    _build(rl_env)

    # New file appears on disk AFTER the initial build.
    new = rl_env["base"] / "corpus-catalog" / "brand-new-book.md"
    _write_md(new, "brand new book zebra quince")
    time.sleep(0.02)

    updated = _reindex(rl_env)
    assert updated >= 1

    conn = sqlite3.connect(rl_env["db"])
    rows = conn.execute(
        "SELECT file_path FROM rl_files WHERE file_path LIKE '%brand-new-book%'"
    ).fetchall()
    n_chunks = conn.execute(
        "SELECT count(*) FROM rl_chunks WHERE file_path LIKE '%brand-new-book%'"
    ).fetchone()[0]
    conn.close()
    assert len(rows) == 1
    assert n_chunks >= 1


def test_reindex_stale_no_new_files_is_noop(rl_env):
    a = rl_env["base"] / "only.md"
    _write_md(a, "only page quince")
    _build(rl_env)
    assert _reindex(rl_env) == 0

"""Tests for the flag-gated rerank second stage in rl_search.search().

Phase 1b of the retrieval upgrade. Invariant: with RERANK_ENABLED off (the
default) or the rerank service down (rerank() -> None), the result order is
identical to the pure first-stage (FTS5) hybrid order.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import uuid

import pytest

_PLUGIN_TESTS = os.path.dirname(os.path.abspath(__file__))
_PLUGINS_MEMORY = os.path.dirname(os.path.dirname(_PLUGIN_TESTS))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_PLUGINS_MEMORY)))
for _p in (_REPO_ROOT, _PLUGINS_MEMORY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perpetual_context import rl_search  # noqa: E402


def _make_rl_db(tmp_path, rows):
    """Legacy-schema scratch RL index: rl_files + rl_index_fts (no chunks).

    Unique file name per call — a test may build several indexes in the
    same tmp_path (each _search call is independent).
    """
    db = tmp_path / f"rl_{uuid.uuid4().hex[:10]}.db"
    con = sqlite3.connect(str(db))
    con.execute(
        "CREATE TABLE rl_files("
        " file_path TEXT PRIMARY KEY, title TEXT, category TEXT, body TEXT)"
    )
    con.execute(
        """CREATE VIRTUAL TABLE rl_index_fts USING fts5(
        file_path, title, category, body, tokenize='porter unicode61')"""
    )
    for (fp, title, cat, body) in rows:
        con.execute("INSERT INTO rl_files VALUES (?,?,?,?)", (fp, title, cat, body))
        con.execute(
            "INSERT INTO rl_index_fts(file_path,title,category,body) VALUES (?,?,?,?)",
            (fp, title, cat, body),
        )
    con.commit()
    con.close()
    return db


def _rows():
    return [
        ("/rl/alpha.md", "Alpha", "tech",
         "zfs storage pool configuration and vdev layout details"),
        ("/rl/beta.md", "Beta", "tech",
         "zfs replication and the storage network"),
        ("/rl/gamma.md", "Gamma", "tech",
         "unrelated notes about baking bread"),
    ]


def _search(tmp_path, monkeypatch, env=None, rerank_fn=None):
    if env is None:
        monkeypatch.delenv("RERANK_ENABLED", raising=False)
    else:
        monkeypatch.setenv("RERANK_ENABLED", env)
    if rerank_fn is not None:
        monkeypatch.setattr(rl_search.rerank_client, "rerank", rerank_fn)
    db = _make_rl_db(tmp_path, _rows())
    conn = sqlite3.connect(str(db), check_same_thread=False)
    try:
        rows = rl_search.search(conn=conn, query="zfs storage", top_k=5)
    finally:
        conn.close()
    return rows


class TestRerankFlag:
    def test_flag_off_returns_first_stage_order(self, tmp_path, monkeypatch):
        rows = _search(tmp_path, monkeypatch)
        # sanity: FTS actually matched (search() swallows errors into []);
        # gamma has no query terms and must not appear. Ordering between
        # alpha/beta is bm25's (length-normalized) — not asserted.
        assert len(rows) == 2
        assert {r["file_path"] for r in rows} == {"/rl/alpha.md", "/rl/beta.md"}

    def test_flag_on_reorders_pool(self, tmp_path, monkeypatch):
        baseline = _search(tmp_path, monkeypatch)  # flag off: pure FTS order
        flipped = _search(
            tmp_path,
            monkeypatch,
            env="1",
            rerank_fn=lambda q, docs: list(reversed(range(len(docs)))),
        )
        assert [r["file_path"] for r in flipped] == list(
            reversed([r["file_path"] for r in baseline])
        )

    def test_flag_on_service_down_keeps_first_stage_order(self, tmp_path, monkeypatch):
        rows_down = _search(tmp_path, monkeypatch, env="1", rerank_fn=lambda q, docs: None)
        rows_off = _search(tmp_path, monkeypatch)
        assert [r["file_path"] for r in rows_down] == [
            r["file_path"] for r in rows_off
        ]

    def test_flag_on_ignores_rerank_when_single_candidate(self, tmp_path, monkeypatch):
        # len(all_ranked) == 1 -> no rerank call (nothing to reorder)
        called = {}

        def spy(q, docs):
            called["yes"] = True
            return [0]

        if True:
            monkeypatch.delenv("RERANK_ENABLED", raising=False)
            monkeypatch.setenv("RERANK_ENABLED", "1")
            monkeypatch.setattr(rl_search.rerank_client, "rerank", spy)
            db = _make_rl_db(tmp_path, [("/rl/alpha.md", "Alpha", "tech",
                                         "zfs storage pool")])
            conn = sqlite3.connect(str(db), check_same_thread=False)
            try:
                rows = rl_search.search(conn=conn, query="zfs storage", top_k=5)
            finally:
                conn.close()
        assert rows[0]["file_path"] == "/rl/alpha.md"
        assert "yes" not in called

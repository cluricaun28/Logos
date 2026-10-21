#!/usr/bin/env python3
"""Rerank A/B battery — Phase 1b gate of the retrieval upgrade.

Plan of record: reference-library/technology/rl-retrieval-upgrade-plan.md
Gate: search battery A/B (recall@1 / recall@5 / nDCG@5 on a labeled query
set). Sweeps RERANK_BLEND (the plan's "weights measured on the A/B
battery") — each weight is a full pass over both arms:

  * baseline — first-stage FTS/hybrid order, top-5 (today's exact
    behavior; deterministic, so one pass suffices for all weights)
  * rerank   — first stage + cross-encoder score BLEND
    ((1-w)*rank + w*min-max(CE score)) via the production call sites:
    tools.book_search_tool._rerank_docs + rerank_client for the book arm,
    plugins...rl_search.search (blends internally) for the RL arm.

Relevance is binary (the labeled target in the top-5). Corpus-gap labels
(target not present in the index) are reported as SKIP, never as 0-recall.
Service down during a pass -> the row records fallback=True; the battery
never treats a fallback as a win.

NOT a pytest (manual gate): filename does not match test_*.py. Run:

    RERANK_ENABLED=1 /home/exx/logos/venv/bin/python tests/rerank_ab_battery.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)  # flat import layout: rerank_client at repo root

import rerank_client  # noqa: E402

TOP_N = 5
BLEND_WEIGHTS = [0.35, 0.5, 0.65, 0.85, 1.0]
BOOK_DB = "/data1/archive/book_index.db"
BOOK_INDEX_PY = "/data1/archive/book_index.py"
RL_DB = os.path.expanduser("~/.hermes/rl_index.db")


# ---------------- book (FL) battery ----------------
# (query, label-substring to mark relevance, notes)
BOOK_QUERIES = [
    ("Ahab Pequod whale ship captain", "MOBY-DICK", "semantic+lexical"),
    ("Darcy Elizabeth Pemberley proposal", "PRIDE AND PREJUDICE", "lexical"),
    ("mad scientist reanimates corpse galvanism", "FRANKENSTEIN", "semantic, no title-term overlap"),
    ("Henry David Thoreau Walden Pond experiment", "WALDEN", "lexical"),
    ("Huck Finn Jim raft Mississippi river", "HUCKLEBERRY FINN", "lexical"),
    ("Shakespeare merchant of Venice Shylock bond", "MERCHANT OF VENICE", "hard negative: many Shakespeare docs"),
    ("Britannica 1911 entry origin of the word democracy", "DEMOCRACY", "eb1911, source-distinguishing"),
    ("Talmud Bavah Metzia chakham opening", "METZIA", "sefaria"),
    ("Walden Thoreau vs his brother", "WALDEN", "hard negative: 'brother' pulls other docs"),
    ("Frankenstein monster creator Victor Walton", "FRANKENSTEIN", "hard negative: shared 'creator/Victor' terms"),
]


def _load_book_index():
    spec = importlib.util.spec_from_file_location("book_index_ab", BOOK_INDEX_PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def book_arm(bi, query: str, limit: int = TOP_N):
    """One first-stage fetch + optional rerank stage. Returns (rows, meta).

    The rerank stage runs the production path: wrapper doc builder
    (tools.book_search_tool._rerank_docs = title/author + multi-site
    windows) + rerank_client scored + blend — i.e. the battery measures
    the shipped code, not a re-implementation.
    """
    toks = bi._fts_tokens(query)
    fts_q = " AND ".join(bi._fts_term(t) for t in toks)
    fetch = min(max(limit * 5, 40), getattr(bi, "SEARCH_BUFFER", 100))
    con = sqlite3.connect(bi.DB)
    t0 = time.time()
    rows = con.execute(
        "SELECT rowid, title, author, source, path, bm25(books_fts) AS s "
        "FROM books_fts WHERE books_fts MATCH ? ORDER BY s LIMIT ?",
        (fts_q, fetch)).fetchall()
    fetch_s = time.time() - t0
    baseline = rows[:limit]
    reranked, fb = None, False
    if rerank_client.enabled() and rows:
        t1 = time.time()
        import tools.book_search_tool as bst  # noqa: E402 (repo root on path)
        as_dicts = [{"title": r[1], "author": r[2], "rowid": r[0],
                     "snippet": ""} for r in rows]
        docs = bst._rerank_docs(bi, query, as_dicts)
        scores = rerank_client.rerank_scored(query, docs)
        rr_s = time.time() - t1
        if scores is None:
            fb = True
        else:
            order = rerank_client.blend(scores)
            reranked = [rows[i] for i in order][:limit]
    else:
        rr_s = 0.0
    con.close()
    return baseline, (reranked if not fb else None), fb, fetch_s, rr_s


def book_relevance(row, label: str) -> bool:
    hay = f"{row[1]} {row[2]} {row[4]}".lower()
    return label.lower() in hay


# ---------------- RL hybrid battery ----------------
# (query, exact file_path label, notes)
RL_QUERIES = [
    ("how to park a vllm engine so it does not resurrect", "technology/inference-fleet-topology.md", "ops, specific"),
    ("Stalwart local mail migration plan cutover", "crenshaw/operations/crenshaw-local-mail-migration-plan.md", "ops, specific"),
    ("Qwen3 reranker hf-overrides sequence classification recipe", "technology/rl-retrieval-upgrade-plan.md", "this plan itself"),
    ("which GPU card hosts the comfyui media stack", "crenshaw/it/crenshaw-gpu-demand-management.md", "hard negative: many GPU docs"),
    ("owner's settled views on rabbinic authority above scripture", "people/patrick-daley.md", "theological, specific"),
    ("Noahide law classifies christianity as idolatry", "religion/bible/lxx-christ-text-virgin-birth.md", "theological — SKIP if path wrong"),
]


def _rl_label_exists(conn, fp: str) -> bool:
    try:
        return bool(conn.execute(
            "SELECT 1 FROM rl_files WHERE file_path=? LIMIT 1", (fp,)).fetchone())
    except sqlite3.OperationalError:
        return False


def rl_rr_arm(conn, query: str, all_ranked: list) -> tuple[list, float, bool]:
    """Rerank arm over a CACHED first-stage pool (production code path).

    Mirrors the rerank block in rl_search.search() exactly: top
    RERANK_POOL candidates -> _rerank_docs -> rerank_scored -> blend
    (RERANK_BLEND from env) -> pool + remainder. The first stage (the
    ~12s MiniLM scan) is weight-independent, so the battery runs it once
    per query and re-blends the same pool across weights — a tighter A/B.
    """
    import rl_search  # noqa: E402
    t0 = time.time()
    if not all_ranked or not (rerank_client.enabled() and len(all_ranked) > 1):
        return all_ranked[:TOP_N], time.time() - t0, False
    pool_n = min(rl_search.RERANK_POOL, len(all_ranked))
    pool = all_ranked[:pool_n]
    docs = rl_search._rerank_docs(conn, [fp for fp, _ in pool], query)
    scores = rerank_client.rerank_scored(query, docs)
    if scores is None:
        return all_ranked[:TOP_N], time.time() - t0, True
    order = rerank_client.blend(scores)
    final_pool = [pool[i] for i in order] + all_ranked[pool_n:]
    return final_pool[:TOP_N], time.time() - t0, False


def ndcg_at(rank: int | None, n: int = TOP_N) -> float:
    """Binary-relevance nDCG@n for exactly one relevant doc (IDCG = 1)."""
    if rank is None or rank > n:
        return 0.0
    return 1.0 / (math.log2(rank + 1))


def _book_row(bi, query: str, label: str, note: str):
    os.environ["RERANK_ENABLED"] = "0"
    base, _, _, fb_s, _ = book_arm(bi, query)
    base_found = next((i + 1 for i, r in enumerate(base)
                       if book_relevance(r, label)), None)
    # corpus pre-check: is the label in the index at all?
    in_corpus = False
    try:
        toks = [bi._fts_term(t) for t in bi._fts_tokens(label)]
        con = sqlite3.connect(bi.DB)
        f = " AND ".join(toks)
        in_corpus = bool(con.execute(
            "SELECT 1 FROM books_fts WHERE books_fts MATCH ? LIMIT 1",
            (f,)).fetchone()) if f else False
        con.close()
    except Exception:
        in_corpus = True  # unknown -> do not skip
    os.environ["RERANK_ENABLED"] = "1"
    _, rr, fb, _, rr_s = book_arm(bi, query)
    if rr is None:
        rr_rank = None
        rr_found = False
    else:
        rr_found = next((i + 1 for i, r in enumerate(rr)
                         if book_relevance(r, label)), None)
        rr_rank = rr_found
    rec = {"q": query, "label": label, "note": note,
           "in_corpus": in_corpus,
           "base_rank": base_found, "rr_rank": rr_rank,
           "fallback": fb, "fetch_s": round(fb_s, 2),
           "rr_s": round(rr_s, 3),
           "base_top": [f"{r[1]} [{r[3]}]" for r in base],
           "rr_top": [f"{r[1]} [{r[3]}]" for r in (rr or [])]}
    mark = "SKIP(corpus)" if not in_corpus else \
           f"base={base_found} rr={rr_found}{' [FB]' if fb else ''}"
    print(f"  [{mark}] {query[:60]}")
    return rec


def _rl_row(conn, query: str, label: str, note: str, all_ranked: list):
    if not all_ranked:
        print(f"  [ERR empty-first-stage] {query[:55]}")
        return None
    base_top5 = all_ranked[:TOP_N]
    base_rank = next((i + 1 for i, (fp, _) in enumerate(base_top5)
                      if fp == label), None)
    os.environ["RERANK_ENABLED"] = "1"
    rr, rr_s, fb = rl_rr_arm(conn, query, all_ranked)
    rr_rank = next((i + 1 for i, (fp, _) in enumerate(rr)
                    if fp == label), None)
    mark = f"base={base_rank} rr={rr_rank}{' [FB]' if fb else ''}"
    print(f"  [{mark}] {query[:55]}")
    return {"q": query, "label": label, "note": note,
            "base_rank": base_rank, "rr_rank": rr_rank,
            "rr_s": round(rr_s, 2),
            "base_top": [fp for fp, _ in base_top5],
            "rr_top": [fp for fp, _ in rr]}


def summarize(rows, bkey, rkey):
    tot = len(rows)
    br1 = sum(1 for r in rows if r.get(bkey) == 1)
    br5 = sum(1 for r in rows if r.get(bkey))
    rr1 = sum(1 for r in rows if r.get(rkey) == 1)
    rr5 = sum(1 for r in rows if r.get(rkey))
    b_ndcg = sum(ndcg_at(r.get(bkey)) for r in rows) / tot if tot else 0
    r_ndcg = sum(ndcg_at(r.get(rkey)) for r in rows) / tot if tot else 0
    fixes = sum(1 for r in rows
                if r.get(rkey) == 1 and r.get(bkey) != 1)
    worse = sum(1 for r in rows if r.get(bkey) and not r.get(rkey))
    return {"n": tot, "base_r@1": br1, "base_r@5": br5,
            "rerank_r@1": rr1, "rerank_r@5": rr5,
            "base_ndcg5": round(b_ndcg, 3), "rerank_ndcg5": round(r_ndcg, 3),
            "rerank_fixes_top1": fixes,
            "rerank_breaks_recall5": worse}


def main():
    os.environ["RERANK_ENABLED"] = "1"
    os.environ["RERANK_TIMEOUT"] = "15"
    os.environ["RERANK_DOC_CHARS"] = "1400"  # full multi-window docs

    bi = _load_book_index() if os.path.exists(BOOK_DB) else None
    conn = sqlite3.connect(f"file:{RL_DB}?mode=ro", uri=True)

    # rl_search lives under plugins/memory/perpetual_context
    sys.path.insert(0, os.path.join(REPO, "plugins", "memory", "perpetual_context"))

    # first stage: try the real MiniLM engine (true hybrid), fall back to FTS-only
    engine, engine_note = None, "FTS-only first stage (engine import failed)"
    try:
        from agent.perpetual_context_db import EmbeddingEngine  # type: ignore
        engine = EmbeddingEngine()
        engine_note = "MiniLM in-process (true hybrid first stage)"
    except Exception as e:  # noqa: BLE001
        engine_note = f"FTS-only first stage ({type(e).__name__})"
    print(f"first stage: {engine_note}")

    # RL first stage: ONCE per query (deterministic; the ~12s MiniLM scan is
    # weight-independent). The sweep re-blends the same pool per weight.
    import rl_search  # noqa: E402
    rl_first: dict[str, list] = {}
    for query, label, note in RL_QUERIES:
        if not _rl_label_exists(conn, label):
            print(f"  [SKIP(label-absent)] {query[:55]} -> {label}")
            continue
        t0 = time.time()
        all_ranked, _ = rl_search._first_stage(
            conn=conn, query=query, pool=50, category=None,
            semantic_weight=rl_search.RL_SEMANTIC_WEIGHT,
            fts5_weight=rl_search.RL_FTS5_WEIGHT,
            embedding_engine=engine)
        rl_first[query] = all_ranked
        print(f"  first-stage {time.time()-t0:.1f}s {len(all_ranked)} cands: "
              f"{query[:50]}", flush=True)

    results = {}
    base_rows = None
    for w in BLEND_WEIGHTS:
        os.environ["RERANK_BLEND"] = str(w)
        print(f"\n===== BLEND w={w} =====")
        book_rows, rl_rows = [], []
        if bi:
            for query, label, note in BOOK_QUERIES:
                book_rows.append(_book_row(bi, query, label, note))
        for query, label, note in RL_QUERIES:
            if query in rl_first:
                rec = _rl_row(conn, query, label, note, rl_first[query])
                if rec:
                    rl_rows.append(rec)
        s_book = summarize(book_rows, "base_rank", "rr_rank")
        s_rl = summarize(rl_rows, "base_rank", "rr_rank")
        results[str(w)] = {"book": s_book, "rl": s_rl,
                           "rows": {"book": book_rows, "rl": rl_rows}}
        if base_rows is None:
            base_rows = {"book": book_rows, "rl": rl_rows}
        print(f"  book: r@1 {s_book['base_r@1']}->{s_book['rerank_r@1']} "
              f"r@5 {s_book['base_r@5']}->{s_book['rerank_r@5']} "
              f"ndcg {s_book['base_ndcg5']}->{s_book['rerank_ndcg5']}")
        print(f"  rl:   r@1 {s_rl['base_r@1']}->{s_rl['rerank_r@1']} "
              f"r@5 {s_rl['base_r@5']}->{s_rl['rerank_r@5']} "
              f"ndcg {s_rl['base_ndcg5']}->{s_rl['rerank_ndcg5']}")
    conn.close()

    def gain(kv):
        w, r = kv
        return r["book"]["rerank_ndcg5"] + r["rl"]["rerank_ndcg5"]
    best_w, _ = max(results.items(), key=gain)
    first_key = list(results)[0]
    base_score = (results[first_key]["book"]["base_ndcg5"]
                  + results[first_key]["rl"]["base_ndcg5"])
    print("\n== SUMMARY (baseline is weight-invariant) ==")
    print(json.dumps({
        "baseline": {"book": results[first_key]["book"],
                     "rl": results[first_key]["rl"],
                     "ndcg_total": round(base_score, 3)},
        "by_weight": {w: {"book": r["book"], "rl": r["rl"]}
                      for w, r in results.items()},
        "best_weight": best_w,
        "best_ndcg_total": round(gain((best_w, results[best_w])), 3),
    }, indent=1))
    with open("/tmp/rerank_ab_results.json", "w") as f:
        json.dump({"baseline_rows": base_rows, "by_weight": results,
                   "best_weight": best_w}, f, indent=1)
    print("details -> /tmp/rerank_ab_results.json")


if __name__ == "__main__":
    main()

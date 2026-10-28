#!/usr/bin/env python3
"""book_search tool — full-text search over the local preservation corpus.

The "alternate internet": raw book / Sefaria (Talmud) / CCEL / law / Catholic
Encyclopedia text, one search away, with provenance (title, author, source,
path, snippet) on every hit.

Ladder (knowledge architecture):
    RL index  ->  book catalog  ->  BOOK TEXT (this tool)  ->  RL page on deep
    (curated)      (what we have)     (raw words, deep check)     (written from
                                                                 engagement)

Use this as the deep-check tier when a claim may live in the raw book text but
isn't surfaced in the Reference Library. It sits *under* reference_library_search
(curated) — the raw archive is complete and sovereign; RL stays curated/honest.

The index + build logic live in the user-space operational module
``~/.hermes/scripts/book_index.py`` (single source of truth). This tool is a thin
registered wrapper: it imports that module in-process and returns structured JSON.
It degrades gracefully — when the index isn't built, check_fn returns False and the
tool is hidden (no error, just absent).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3

import rerank_client

# Candidate locations for the corpus-index module (user-space operational script).
_MODULE_CANDIDATES = (
    os.path.expanduser("~/.hermes/scripts/book_index.py"),
    "/data1/archive/book_index.py",
    os.path.expanduser("~/.logos/scripts/book_index.py"),
)

_mod = None
_load_attempted = False
_check_cache = None


def _bi():
    """Import the corpus-index module once (lazy), or None if not found."""
    global _mod, _load_attempted
    if _mod is not None or _load_attempted:
        return _mod
    _load_attempted = True
    for p in _MODULE_CANDIDATES:
        if os.path.exists(p):
            try:
                spec = importlib.util.spec_from_file_location("book_index_mod", p)
                m = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(m)
                _mod = m
                return m
            except Exception:
                continue
    return None


def check_book_index_ready() -> bool:
    """Tool is available only when the index DB exists and has docs.

    Bounded probe + per-process memo. This check runs on EVERY tool-discovery
    pass; a full ``COUNT(*)`` over the 260 GB FTS5 index walked the entire
    index on each pass (measured 10/18: 1.1 TB of reads in a single
    test-suite run, stalling the gateway test batch). A ``LIMIT 1`` rowid
    probe is O(1), and a definitive True is memoized for the process
    lifetime (the nightly build repopulates, never removes, the DB).
    False is NOT memoized — the first-ever build flips it to True on the
    next discovery pass.
    """
    global _check_cache
    if _check_cache is True:
        return True
    m = _bi()
    if m is None or not os.path.exists(getattr(m, "DB", "")):
        return False
    try:
        import sqlite3
        con = sqlite3.connect(m.DB)
        try:
            row = con.execute("SELECT rowid FROM books_fts LIMIT 1").fetchone()
        finally:
            con.close()
        _check_cache = bool(row)
        return _check_cache
    except Exception:
        return False


def _rerank_docs(m, query: str, rows: list[dict]) -> list[str]:
    """Build cross-encoder docs for a candidate pool (rerank stage only).

    Reads the body column by rowid (O(1) lookups per the 10/18 FTS5 cost
    notes) and delegates to ``book_index.rerank_doc``. Any failure degrades
    to the old title/author/snippet form — the rerank stage is an upgrade,
    never a breaker.
    """
    try:
        toks = m._fts_tokens(query)
        rowids = [r["rowid"] for r in rows if r.get("rowid") is not None]
        bodies: dict[int, str] = {}
        if rowids:
            ph = ",".join("?" * len(rowids))
            con = sqlite3.connect(m.DB)
            try:
                cur = con.execute(
                    f"SELECT rowid, body FROM books_fts WHERE rowid IN ({ph})",
                    rowids)
                bodies = {rid: (b or "") for rid, b in cur.fetchall()}
            finally:
                con.close()
        return [
            m.rerank_doc(r["title"], r.get("author", ""), bodies.get(r["rowid"], ""), toks)
            for r in rows
        ]
    except Exception:  # noqa: BLE001 - degrade, never break the search
        return [f"{r.get('title', '')} — {r.get('author', '')}. {r.get('snippet', '')}"
                for r in rows]


def book_search(query: str = "", limit: int = 8, source: str = None) -> str:
    if not query:
        return json.dumps({"error": "query is required."})
    m = _bi()
    if m is None:
        return json.dumps({
            "error": "corpus index module not found "
                     "(~/.hermes/scripts/book_index.py).",
        })
    if not os.path.exists(m.DB):
        return json.dumps({"error": "corpus index not built yet (no book_index.db)."})
    use_rerank = rerank_client.enabled()
    try:
        if use_rerank:
            # The reranker needs a candidate pool bigger than the return size.
            # Buffer stays inside the measured FTS5 fast window (10/18 on the
            # 260GB index: top-100 = 6.4s, top-150 = 39s — non-linear past 100).
            fetch = min(max(limit * 5, 40), getattr(m, "SEARCH_BUFFER", 100))
        else:
            fetch = limit
        rows = m.search_structured(query, limit=fetch, source=source)
    except Exception as e:  # noqa: BLE001 - surface as tool error, not a crash
        return json.dumps({"error": f"search failed: {type(e).__name__}: {e}"})
    if use_rerank and rows:
        # Second stage: cross-encoder reorders the pool. ``None`` = service
        # down / failure -> keep the first-stage order verbatim (output shape
        # and truncation are identical either way).
        # CE docs = title/author + multi-site windows around the query-term
        # hit positions (book_index.rerank_doc). A single head/snippet window
        # let one-line mentions (list pages, chronologies) outscore the target
        # doc whose matched section sits mid-body (10/20 A/B battery).
        docs = _rerank_docs(m, query, rows)
        # Blend (RERANK_BLEND, default 0.5): first-stage rank stays anchored
        # so a one-line-mention page cannot leapfrog an exact match — pure
        # CE reorder measured worse on the 10/20 A/B battery.
        scores = rerank_client.rerank_scored(query, docs)
        if scores is not None:
            order = rerank_client.blend(scores)
            rows = [rows[i] for i in order]
        rows = rows[:limit]
    else:
        rows = rows[:limit]
    return json.dumps({"count": len(rows), "results": rows})


BOOK_SEARCH_SCHEMA = {
    "name": "book_search",
    "description": (
        "Full-text search over the local preservation corpus — the 'alternate "
        "internet'. Books (Project Gutenberg), Sefaria (Talmud/rabbinic, our source "
        "of record), CCEL, US law, Catholic Encyclopedia 1913. Use as the deep-check "
        "tier when a claim may be in the RAW book text but isn't surfaced in the "
        "Reference Library. Every hit carries provenance (title, author, source, "
        "path, snippet). Ladder: RL index -> book catalog -> book text (this) -> RL "
        "page written only on deep engagement. Prefer reference_library_search for "
        "curated/worldview-framed answers; use this to verify against the primary "
        "text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query. Tokens are AND-combined.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 8).",
            },
            "source": {
                "type": "string",
                "description": (
                    "Optional source filter: gutenberg, sefaria, ccel, law, "
                    "cathen, books, maps, stackoverflow, math-stackexchange, "
                    "physics-stackexchange, enwiki, arxiv. Omit to search all."
                ),
            },
        },
        "required": ["query"],
    },
}


# --- Registry ---
from tools.registry import registry  # noqa: E402

registry.register(
    name="book_search",
    toolset="library",
    schema=BOOK_SEARCH_SCHEMA,
    handler=lambda args, **kw: book_search(
        query=args.get("query", ""),
        limit=args.get("limit", 8),
        source=args.get("source"),
    ),
    check_fn=check_book_index_ready,
    emoji="\U0001F4DA",
)

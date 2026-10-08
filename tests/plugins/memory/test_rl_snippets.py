"""Tests for the per-term snippet-match fix (WS3 follow-up, 2026-10-07).

Bug under test: ``_extract_snippets_batch`` located the snippet window with
``body.find(query)`` — the FULL query string as one substring. A real query
is a keyword list ("dmarc aggregate report domain"), which almost never
appears verbatim in the body, so the lookup missed and silently fell back to
``body[0:300]`` (page top). Measured live: 0/534 recent search results were
true match windows (59% were literally the page's first 300 chars).

Fix under test: ``snippet_match_index`` tries the full phrase first, then
individual terms (>2 chars, in query order), and returns -1 when nothing
matches. ``_extract_snippets_batch`` uses it, so a snippet window now centers
on a real term hit.

The same full-string pattern in ``tool_handler`` (topics/entities +
britannica archive fallback paths) was fixed in-place; those paths are
covered by the full gate, not unit tests here (they need a populated RL
tree).
"""
from __future__ import annotations

import sqlite3

from plugins.memory.perpetual_context.rl_search import (
    _extract_snippets_batch,
    snippet_match_index,
)


# ---------------------------------------------------------------------------
# snippet_match_index — pure function
# ---------------------------------------------------------------------------

class TestSnippetMatchIndex:
    def test_full_phrase_match_wins(self):
        body = "prefix text here the quick brown fox suffix"
        assert snippet_match_index(
            body, "quick brown fox", ["quick", "brown", "fox"],
        ) == body.find("quick brown fox")

    def test_falls_back_to_first_term(self):
        # "dmarc aggregate report" never appears verbatim; "dmarc" does.
        body = "x " * 50 + "dmarc settings page y " * 20 + "report elsewhere"
        assert snippet_match_index(
            body, "dmarc aggregate report", ["dmarc", "aggregate", "report"],
        ) == body.find("dmarc")

    def test_short_terms_skipped(self):
        # Only 2-char terms available -> skipped -> no match.
        body = "a b c d e f g h i j k l"
        assert snippet_match_index(body, "of to the", ["of", "to", "the"]) == -1

    def test_no_match_returns_minus_one(self):
        assert snippet_match_index("hello world", "zzz qqq", ["zzz", "qqq"]) == -1

    def test_case_insensitive(self):
        body = "THE QUICK BROWN FOX"
        # "the " prefix = 4 chars; the lowercased phrase match starts at 4.
        assert snippet_match_index(body, "quick brown fox", []) == 4

    def test_phrase_beats_term(self):
        body = "foo quick then the full phrase quick brown fox later"
        idx = snippet_match_index(body, "quick brown fox", ["quick"])
        assert idx == body.find("quick brown fox")


# ---------------------------------------------------------------------------
# _extract_snippets_batch — needs a real sqlite conn with the rl_files schema
# ---------------------------------------------------------------------------

def _make_conn(rows: list[tuple[str, str]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE rl_files (file_path TEXT, body TEXT)")
    conn.executemany("INSERT INTO rl_files VALUES (?, ?)", rows)
    return conn


class TestExtractSnippetsBatch:
    def test_term_hit_centers_window(self):
        filler = "filler " * 80  # 400 chars of noise
        body = filler + "the dmarc aggregate report is here " + filler
        conn = _make_conn([("tech/dmarc.md", body)])
        out = _extract_snippets_batch(
            conn=conn, file_paths=["tech/dmarc.md"],
            query="dmarc aggregate report domain",
        )
        snip = out["tech/dmarc.md"]
        # Window must contain the real hit, not just the page top.
        assert "dmarc aggregate report" in snip

    def test_no_hit_falls_back_to_page_top(self):
        body = "first line of page " * 20 + "nothing relevant here at all"
        conn = _make_conn([("tech/other.md", body)])
        out = _extract_snippets_batch(
            conn=conn, file_paths=["tech/other.md"],
            query="zzz qqq www",
        )
        assert out["tech/other.md"].startswith("first line of page")

    def test_empty_inputs(self):
        assert _extract_snippets_batch(conn=None, file_paths=["a"], query="q") == {}
        assert _extract_snippets_batch(conn=_make_conn([]), file_paths=[], query="q") == {}

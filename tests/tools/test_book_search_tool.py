"""Tests for tools/book_search_tool.py — the 'alternate internet' corpus search.

Covers: registration, schema shape, check_fn gating (built vs empty), handler
hits with provenance, source filtering, and graceful degradation (no module).
"""
import importlib
import json
import os
import sqlite3
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _make_scratch_db(tmp_path, rows):
    """Create a minimal book_index FTS5 db with the given (t,a,s,p,b) rows."""
    db = tmp_path / "book_index.db"
    con = sqlite3.connect(str(db))
    con.execute("""CREATE VIRTUAL TABLE books_fts USING fts5(
        path, title, author, source, body, tokenize='porter unicode61')""")
    con.execute("CREATE TABLE manifest(path TEXT PRIMARY KEY, mtime REAL, size INTEGER, note TEXT)")
    for (t, a, s, p, b) in rows:
        con.execute(
            "INSERT INTO books_fts(path,title,author,source,body) VALUES(?,?,?,?,?)",
            (p, t, a, s, b),
        )
    con.commit()
    con.close()
    return str(db)


def _fresh_tool_module(monkeypatch, db_path):
    """Re-import tools.book_search_tool with BOOK_INDEX_DB pointed at db_path."""
    monkeypatch.setenv("BOOK_INDEX_DB", db_path)
    sys.modules.pop("tools.book_search_tool", None)
    sys.modules.pop("book_index_mod", None)
    import tools.book_search_tool as mod
    mod._mod = None
    mod._load_attempted = False
    return mod


class TestRegistration:
    def test_registered_under_library(self, monkeypatch, tmp_path):
        from tools.registry import registry
        _fresh_tool_module(monkeypatch, str(tmp_path / "x.db"))
        entry = registry.get_entry("book_search")
        assert entry is not None
        assert entry.name == "book_search"
        assert entry.toolset == "library"

    def test_schema_shape(self, monkeypatch, tmp_path):
        from tools.registry import registry
        _fresh_tool_module(monkeypatch, str(tmp_path / "x.db"))
        schema = registry.get_schema("book_search")
        assert schema["name"] == "book_search"
        assert "query" in schema["parameters"]["properties"]
        assert "query" in schema["parameters"]["required"]
        assert "limit" in schema["parameters"]["properties"]
        assert "source" in schema["parameters"]["properties"]


class TestCheckFn:
    def test_false_when_zero_docs(self, monkeypatch, tmp_path):
        db = _make_scratch_db(tmp_path, [])
        mod = _fresh_tool_module(monkeypatch, db)
        assert mod.check_book_index_ready() is False

    def test_true_when_docs_present(self, monkeypatch, tmp_path):
        db = _make_scratch_db(tmp_path, [
            ("Book One", "Author A", "gutenberg", "/x/1.txt",
             "the quick red fox jumps over the lazy dog redemption"),
        ])
        mod = _fresh_tool_module(monkeypatch, db)
        assert mod.check_book_index_ready() is True


class TestHandler:
    def test_empty_query_error(self, monkeypatch, tmp_path):
        db = _make_scratch_db(tmp_path, [("B", "A", "gutenberg", "/x/1.txt", "hello world")])
        mod = _fresh_tool_module(monkeypatch, db)
        out = json.loads(mod.book_search(query=""))
        assert "error" in out

    def test_returns_hits_with_provenance(self, monkeypatch, tmp_path):
        db = _make_scratch_db(tmp_path, [
            ("Religious Affections", "Jonathan Edwards", "gutenberg", "/x/100.txt",
             "the affections of the heart and redemption of the soul"),
            ("Other Book", "Someone", "gutenberg", "/x/200.txt",
             "unrelated content about gardening and tomatoes"),
        ])
        mod = _fresh_tool_module(monkeypatch, db)
        out = json.loads(mod.book_search(query="redemption", limit=5))
        assert out["count"] == 1
        r = out["results"][0]
        assert r["title"] == "Religious Affections"
        assert r["author"] == "Jonathan Edwards"
        assert r["source"] == "gutenberg"
        assert "path" in r and "snippet" in r and "score" in r

    def test_source_filter(self, monkeypatch, tmp_path):
        db = _make_scratch_db(tmp_path, [
            ("Sef Book", "Hebrew", "sefaria", "/s/1.txt", "eruv and sabbath observance"),
            ("Gut Book", "Someone", "gutenberg", "/g/1.txt", "eruv and sabbath observance"),
        ])
        mod = _fresh_tool_module(monkeypatch, db)
        out = json.loads(mod.book_search(query="eruv", source="sefaria"))
        assert out["count"] == 1
        assert out["results"][0]["source"] == "sefaria"


class TestGracefulDegradation:
    def test_missing_module_returns_error(self, monkeypatch, tmp_path):
        db = str(tmp_path / "nope.db")
        mod = _fresh_tool_module(monkeypatch, db)
        monkeypatch.setattr(mod, "_MODULE_CANDIDATES", ("/nonexistent/book_index.py",))
        mod._mod = None
        mod._load_attempted = False
        out = json.loads(mod.book_search(query="anything"))
        assert "error" in out

    def test_no_db_returns_error(self, monkeypatch, tmp_path):
        # module found (real script) but DB path does not exist
        db = str(tmp_path / "does_not_exist.db")
        mod = _fresh_tool_module(monkeypatch, db)
        out = json.loads(mod.book_search(query="anything"))
        assert "error" in out

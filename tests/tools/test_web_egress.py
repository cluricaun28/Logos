"""Tests for web_egress — fleet-wide web egress guards.

Covers:
- URL extraction cache: put/get, TTL expiry, key normalization
  (trailing slash, host case), pruning to max rows, empty-content skip.
- Domain cooldown: strike escalation (15m/1h/6h/24h), expiry, success
  reset, unknown-domain pass-through, state file persistence.
- looks_blocked heuristic on block-page text.
- enabled() kill switch.
- Fail-open: corrupt state must never raise from the public API.

All tests use a tmp_path state dir — no fleet state is touched.
"""

from __future__ import annotations

import importlib
import json
import time

import pytest


@pytest.fixture()
def egress(tmp_path, monkeypatch):
    """Fresh web_egress module bound to a tmp state dir.

    The module reads config at import time, so we set env then reload.
    """
    import web_egress as we

    monkeypatch.setenv("HERMES_WEB_EGRESS_DIR", str(tmp_path))
    importlib.reload(we)
    return we


# ── enabled() ────────────────────────────────────────────────────────────────

def test_enabled_default_on(egress):
    assert egress.enabled() is True


def test_enabled_env_kill_switch(egress, monkeypatch):
    monkeypatch.setenv("HERMES_WEB_EGRESS_ENABLED", "0")
    assert egress.enabled() is False
    monkeypatch.setenv("HERMES_WEB_EGRESS_ENABLED", "no")
    assert egress.enabled() is False


# ── looks_blocked ────────────────────────────────────────────────────────────

def test_looks_blocked_recognizes_block_pages(egress):
    for text in (
        "Just a moment...",
        "Access Denied",
        "are you a robot?",
        "Too Many Requests",
        "captcha",
        "403 Forbidden",
        "Enable JavaScript and reload the page to continue",
        "Checking your browser before accessing the site",
    ):
        assert egress.looks_blocked(text), text


def test_looks_blocked_ignores_normal_content(egress):
    for text in (
        "The Federal Reserve raised rates by 25 basis points.",
        "In 1964, the Civil Rights Act was signed into law.",
        "",
        None,
    ):
        assert not egress.looks_blocked(text), repr(text)


def test_looks_blocked_ignores_long_bodies(egress):
    # >600 chars is content, even if it happens to contain a marker word
    assert not egress.looks_blocked("captcha " * 200)


# ── key normalization ────────────────────────────────────────────────────────

def test_key_normalization(egress):
    base = egress._key("https://Example.COM/article")
    assert base == egress._key("https://example.com/article/")
    assert base == egress._key(" https://example.com/article ")
    # query preserved
    assert egress._key("https://x.com/a?b=1") != egress._key("https://x.com/a")
    # bad url degrades to input, no raise
    egress._key("not a url")  # just must not raise


# ── URL extraction cache ─────────────────────────────────────────────────────

def test_cache_put_get_roundtrip(egress):
    egress.cache_put("https://example.com/article", "content A")
    assert egress.cache_get("https://example.com/article") == "content A"


def test_cache_normalization_hits(egress):
    egress.cache_put("https://Example.COM/article/", "content B")
    assert egress.cache_get("https://example.com/article") == "content B"
    assert egress.cache_get("https://EXAMPLE.com/article") == "content B"


def test_cache_ttl_expiry(egress):
    egress.cache_put("https://example.com/x", "old")
    we = egress
    with we._lock:
        conn = we._conn()
        conn.execute(
            "UPDATE web_extract_cache SET fetched_at = ? WHERE url = ?",
            (time.time() - (we.CACHE_TTL_HOURS + 1) * 3600, we._key("https://example.com/x")),
        )
        conn.commit()
    assert egress.cache_get("https://example.com/x") is None


def test_cache_within_ttl_returned(egress):
    egress.cache_put("https://example.com/y", "fresh")
    assert egress.cache_get("https://example.com/y") == "fresh"


def test_cache_prunes_to_max_rows(egress, monkeypatch, tmp_path):
    import web_egress as we_mod

    monkeypatch.setenv("HERMES_WEB_CACHE_MAX_ROWS", "5")
    importlib.reload(we_mod)
    for i in range(10):
        we_mod.cache_put(f"https://example.com/p{i}", f"c{i}")
    remaining = {
        row[0] for row in we_mod._conn().execute("SELECT url FROM web_extract_cache").fetchall()
    }
    assert len(remaining) == 5
    assert we_mod.cache_get("https://example.com/p0") is None
    assert we_mod.cache_get("https://example.com/p9") == "c9"


def test_cache_empty_content_not_stored(egress):
    egress.cache_put("https://example.com/empty", "")
    assert egress.cache_get("https://example.com/empty") is None


def test_cache_bad_url_is_noop(egress):
    egress.cache_put(None, "x")  # type: ignore[arg-type]
    assert egress.cache_get(None) is None  # type: ignore[arg-type]


# ── Domain cooldown ──────────────────────────────────────────────────────────

def test_unknown_domain_not_locked(egress):
    locked, why = egress.cooldown_locked("https://fresh.example.com/page")
    assert locked is False
    assert why == ""


def test_first_strike_locks_15_minutes(egress):
    egress.record_block("https://blocked.example.com/a")
    locked, why = egress.cooldown_locked("https://blocked.example.com/b")
    assert locked is True
    assert "blocked.example.com" in why
    assert "15" in why
    # other domains unaffected
    assert egress.cooldown_locked("https://other.example.com/c")[0] is False


def test_strikes_escalate(egress):
    egress.record_block("https://esc.example.com/x")   # strike 1 -> 15m
    egress.record_block("https://esc.example.com/x")   # strike 2 -> 1h
    locked, why = egress.cooldown_locked("https://esc.example.com/next")
    assert locked is True
    assert "60" in why


def test_fourth_strike_locks_24h(egress):
    for _ in range(4):
        egress.record_block("https://bad.example.com/1")
    locked, why = egress.cooldown_locked("https://bad.example.com/next")
    assert locked is True
    assert "1440" in why


def test_success_resets_strikes(egress):
    egress.record_block("https://flaky.example.com/a")
    egress.record_success("https://flaky.example.com/a")
    data = json.loads(egress._cooldown_path().read_text())
    assert "flaky.example.com" not in data
    locked, _ = egress.cooldown_locked("https://flaky.example.com/b")
    assert locked is False
    # next block starts back at strike 1
    egress.record_block("https://flaky.example.com/c")
    data = json.loads(egress._cooldown_path().read_text())
    assert data["flaky.example.com"]["strikes"] == 1


def test_cooldown_expires(egress):
    egress.record_block("https://exp.example.com/a")
    data = json.loads(egress._cooldown_path().read_text())
    data["exp.example.com"]["until"] = time.time() - 60
    egress._save_cooldowns(data)
    locked, _ = egress.cooldown_locked("https://exp.example.com/next")
    assert locked is False


def test_cooldown_state_file_persisted(egress):
    egress.record_block("https://disk.example.com/a")
    data = json.loads(egress._cooldown_path().read_text())
    assert data["disk.example.com"]["strikes"] == 1
    assert data["disk.example.com"]["step"] == 15


# ── Fail-open ────────────────────────────────────────────────────────────────

def test_corrupt_cooldown_file_is_fail_open(egress):
    egress._cooldown_path().write_text("{not json!!")
    locked, why = egress.cooldown_locked("https://whatever.example.com/x")
    assert locked is False
    assert why == ""
    # a block still records fine (file rewritten)
    egress.record_block("https://whatever.example.com/x")
    locked, _ = egress.cooldown_locked("https://whatever.example.com/y")
    assert locked is True


def test_corrupt_cache_db_is_fail_open(egress):
    we = egress
    # close the live conn so the next access reopens against the file
    if we._cache_conn is not None:
        we._cache_conn.close()
        we._cache_conn = None
    db = we._STATE_DIR / "web_extract_cache.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_bytes(b"not a sqlite database at all")
    for suffix in ("-wal", "-shm"):
        extra = db.with_suffix(db.suffix + suffix)
        if extra.exists():
            extra.unlink()
    # must not raise
    assert we.cache_get("https://fail.example.com/x") is None
    we.cache_put("https://fail.example.com/x", "ok")  # must not raise
    # recover once the bad file is gone
    db.unlink()
    if we._cache_conn is not None:
        try:
            we._cache_conn.close()
        except Exception:
            pass
        we._cache_conn = None
    we.cache_put("https://fail.example.com/x", "ok")
    assert we.cache_get("https://fail.example.com/x") == "ok"


def test_bad_status_exception_detection(egress):
    class Resp:
        def __init__(self, status: int):
            self.status_code = status

    class StatusError(Exception):
        def __init__(self, status: int):
            super().__init__(f"error {status}")
            self.response = Resp(status)

    assert egress._blocked_by_status(StatusError(403)) is True
    assert egress._blocked_by_status(StatusError(429)) is True
    assert egress._blocked_by_status(StatusError(503)) is True
    assert egress._blocked_by_status(StatusError(404)) is False
    assert egress._blocked_by_status(StatusError(500)) is False
    assert egress._blocked_by_status(Exception("connection reset")) is False
    assert egress._blocked_by_status("no such exception") is False


# ── Integration: web_tools wrapper behavior (unit-level) ────────────────────

def test_web_tools_guard_short_circuits_when_locked(egress, monkeypatch, tmp_path):
    """_local_extract_with_fallback must not call _extract_once when locked."""
    # ensure state dir for the reloaded module matches
    import tools.web_tools as wt

    calls = []
    monkeypatch.setattr(wt, "_extract_once", lambda url: calls.append(url) or {"success": True, "markdown": "x"})
    egress.record_block("https://locked.example.com/a")
    result = wt._local_extract_with_fallback("https://locked.example.com/page")
    assert result["success"] is False
    assert "cooldown" in result["error"]
    assert calls == []


def test_web_tools_guard_serves_cache(egress, monkeypatch):
    import tools.web_tools as wt

    calls = []
    monkeypatch.setattr(wt, "_extract_once", lambda url: calls.append(url) or {"success": True, "markdown": "x"})
    egress.cache_put("https://cached.example.com/p", "cached body")
    result = wt._local_extract_with_fallback("https://cached.example.com/p")
    assert result["success"] is True
    assert result["markdown"] == "cached body"
    assert result.get("cached") is True
    assert calls == []


def test_web_tools_guard_records_block_on_403(egress, monkeypatch):
    import tools.web_tools as wt

    class Resp:
        status_code = 403

    class HTTPError(Exception):
        def __init__(self):
            super().__init__("403")
            self.response = Resp()

    def boom(url):
        raise HTTPError()

    monkeypatch.setattr(wt, "_extract_once", boom)
    result = wt._local_extract_with_fallback("https://blocked.example.com/z")
    assert result["success"] is False
    locked, _ = egress.cooldown_locked("https://blocked.example.com/next")
    assert locked is True


def test_web_tools_guard_caches_success(egress, monkeypatch):
    import tools.web_tools as wt

    monkeypatch.setattr(wt, "_extract_once", lambda url: {"success": True, "markdown": "real body"})
    result = wt._local_extract_with_fallback("https://good.example.com/p")
    assert result["success"] is True
    assert egress.cache_get("https://good.example.com/p") == "real body"

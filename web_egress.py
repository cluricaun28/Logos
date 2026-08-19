"""Web egress guards — fleet-wide protection against IP blocking.

Two guards, both local, stdlib-only, and fail-open (a guard error must
never break web extraction):

1. **Extraction cache** (``web_extract_cache.db``, SQLite WAL)
   A URL we already scraped successfully within the TTL is served from
   disk instead of re-hitting the target. This kills the dominant
   traffic pattern: repeat hits to the same URL across turns, sessions,
   and cron jobs.

2. **Domain block cooldown** (``domain_cooldowns.json``)
   When a scrape fails with a bot-block signal (HTTP 403/429/503 or a
   challenge-page body), the domain enters a backoff:
   1st -> 15 min, 2nd -> 1 h, 3rd -> 6 h, 4th+ -> 24 h.
   While locked, all tiers short-circuit *before* sending traffic.
   A clean success resets the domain.

Egress context: local Firecrawl scrapes egress from OUR IP (no
FIRECRAWL_API_KEY by default); Camofox egress from our IP with a
browser fingerprint. Residential line, but the fleet shares one IP —
search tier (SearXNG) is unaffected because target sites never see it.

Config (env):
  HERMES_WEB_EGRESS_DIR        state dir (default ~/.hermes/state/web-egress)
  HERMES_WEB_CACHE_TTL_HOURS   cache TTL, default 48
  HERMES_WEB_CACHE_MAX_ROWS    prune cap, default 4000
  HERMES_WEB_EGRESS_ENABLED    "0"/"false"/"no" disables both guards
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

_STATE_DIR = Path(
    os.getenv("HERMES_WEB_EGRESS_DIR", str(Path.home() / ".hermes" / "state" / "web-egress"))
)
CACHE_TTL_HOURS = float(os.getenv("HERMES_WEB_CACHE_TTL_HOURS", "48"))
CACHE_MAX_ROWS = int(os.getenv("HERMES_WEB_CACHE_MAX_ROWS", "4000"))
COOLDOWN_STEPS_MIN = (15, 60, 360, 1440)
BLOCK_STATUS = frozenset({403, 429, 503})
# Short bodies matching these are bot walls, not content.
_BLOCK_MARKERS = (
    "just a moment",
    "attention required",
    "cf-browser-verification",
    "are you a robot",
    "captcha",
    "access denied",
    "request blocked",
    "cloudflare",
    "too many requests",
    "403 forbidden",
    "enable javascript",
    "checking your browser",
)


def _key(url: str) -> str:
    """Cache key normalization: lowercase scheme+host, drop trailing slash.

    Scheme is preserved (http and https can differ); query is preserved.
    """
    if not url:
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    return parts._replace(
        scheme=parts.scheme.lower(), netloc=parts.netloc.lower(), path=path
    ).geturl()

_lock = threading.RLock()
_cache_conn: sqlite3.Connection | None = None


def enabled() -> bool:
    return os.getenv("HERMES_WEB_EGRESS_ENABLED", "1").lower() not in ("0", "false", "no")


def domain_of(url: str) -> str:
    """Apex-ish domain key (lower, no port, no userinfo, no leading www.)."""
    try:
        netloc = urlsplit(url).netloc.lower()
    except ValueError:
        return ""
    if not netloc:
        return ""
    netloc = netloc.rsplit("@", 1)[-1].split(":", 1)[0]
    return netloc[4:] if netloc.startswith("www.") else netloc


def looks_blocked(text: str) -> bool:
    """True when a scraped body looks like a challenge/bot wall, not content."""
    if not text or len(text) > 600:
        return False
    low = text.lower()
    return any(m in low for m in _BLOCK_MARKERS)


def _status_of(exc: BaseException) -> int | None:
    resp = getattr(exc, "response", None)
    st = getattr(resp, "status_code", None)
    return st if isinstance(st, int) else None


def _blocked_by_status(exc: BaseException) -> bool:
    st = _status_of(exc)
    return st is not None and st in BLOCK_STATUS


# ── Extraction cache ─────────────────────────────────────────────────────────


def _conn() -> sqlite3.Connection | None:
    global _cache_conn
    if _cache_conn is not None:
        return _cache_conn
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(_STATE_DIR / "web_extract_cache.db"), check_same_thread=False, timeout=5.0
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=3000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS web_extract_cache ("
            " url TEXT PRIMARY KEY,"
            " content TEXT NOT NULL,"
            " fetched_at REAL NOT NULL)"
        )
        conn.commit()
        _cache_conn = conn
        return conn
    except Exception as e:  # noqa: S110 — fail-open
        logger.debug("web_egress cache unavailable: %s", e)
        return None


def cache_get(url: str) -> str | None:
    """Cached markdown for url if within TTL, else None (and pruned)."""
    if not enabled():
        return None
    key = _key(url)
    if not key:
        return None
    try:
        with _lock:
            conn = _conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT content, fetched_at FROM web_extract_cache WHERE url=?", (key,)
            ).fetchone()
            if row is None:
                return None
            content, fetched_at = row
            if time.time() - fetched_at > CACHE_TTL_HOURS * 3600:
                conn.execute("DELETE FROM web_extract_cache WHERE url=?", (key,))
                conn.commit()
                return None
            return content
    except Exception as e:  # noqa: S110 — fail-open
        logger.debug("web_egress cache_get failed for %s: %s", str(url)[:80], e)
        return None


def cache_put(url: str, content: str) -> None:
    if not enabled() or not url or not content:
        return
    key = _key(url)
    if not key:
        return
    try:
        with _lock:
            conn = _conn()
            if conn is None:
                return
            conn.execute(
                "INSERT OR REPLACE INTO web_extract_cache (url, content, fetched_at) VALUES (?,?,?)",
                (key, content, time.time()),
            )
            total = conn.execute("SELECT COUNT(*) FROM web_extract_cache").fetchone()[0]
            over = total - CACHE_MAX_ROWS
            if over > 0:
                conn.execute(
                    "DELETE FROM web_extract_cache WHERE url IN ("
                    " SELECT url FROM web_extract_cache ORDER BY fetched_at ASC LIMIT ?)",
                    (over,),
                )
            conn.commit()
    except Exception as e:  # noqa: S110 — fail-open
        logger.debug("web_egress cache_put failed for %s: %s", url[:80], e)


# ── Domain block cooldown ────────────────────────────────────────────────────


def _cooldown_path() -> Path:
    return _STATE_DIR / "domain_cooldowns.json"


def _load_cooldowns() -> dict:
    try:
        raw = _cooldown_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cooldowns(data: dict) -> None:
    path = _cooldown_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def cooldown_locked(url: str) -> tuple[bool, str]:
    """(locked, reason) for url's domain. Fail-open: (False, "")."""
    if not enabled():
        return False, ""
    domain = domain_of(url)
    if not domain:
        return False, ""
    try:
        with _lock:
            entry = _load_cooldowns().get(domain)
        if not entry:
            return False, ""
        until = float(entry.get("until", 0))
        if time.time() < until:
            mins_left = int((until - time.time()) // 60) + 1
            step = entry.get("step", "?")
            return True, (
                f"{domain} locked ~{mins_left}m (backoff {step}m, "
                f"strike {entry.get('strikes', '?')})"
            )
        return False, ""
    except Exception as e:  # noqa: S110 — fail-open
        logger.debug("web_egress cooldown_locked failed for %s: %s", url[:80], e)
        return False, ""


def record_block(url: str) -> None:
    """Escalate the domain's cooldown on a bot-block signal."""
    if not enabled():
        return
    domain = domain_of(url)
    if not domain:
        return
    try:
        with _lock:
            data = _load_cooldowns()
            entry = data.get(domain) or {}
            strikes = min(int(entry.get("strikes", 0)) + 1, len(COOLDOWN_STEPS_MIN))
            step = COOLDOWN_STEPS_MIN[strikes - 1]
            data[domain] = {
                "strikes": strikes,
                "step": step,
                "until": time.time() + step * 60,
                "last": time.time(),
            }
            _save_cooldowns(data)
        logger.info("web_egress: block signal on %s (strike %d, backoff %dm)", domain, strikes, step)
    except Exception as e:  # noqa: S110 — fail-open
        logger.debug("web_egress record_block failed for %s: %s", url[:80], e)


def record_success(url: str) -> None:
    """Reset a domain's strikes after a clean scrape."""
    if not enabled():
        return
    domain = domain_of(url)
    if not domain:
        return
    try:
        with _lock:
            data = _load_cooldowns()
            entry = data.get(domain)
            if not entry or int(entry.get("strikes", 0)) == 0:
                return
            data.pop(domain, None)
            _save_cooldowns(data)
        logger.info("web_egress: %s recovered, cooldown reset", domain)
    except Exception as e:  # noqa: S110 — fail-open
        logger.debug("web_egress record_success failed for %s: %s", url[:80], e)

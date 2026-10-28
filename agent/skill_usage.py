"""Per-user skill usage store + state.db backfill (SKILL-DEMOTE-1028).

Stamps the last time each skill was loaded (skill_view) or maintained
(skill_manage) into ``<HERMES_HOME>/.skill_usage.json``. The prompt
builder reads the store and ages Core (full-description) skills out of
the injected index when they go unused — see
``SkillInjectionManager.demote_stale_skills`` / ``exclude_dropped``.

The store is the ONLY state this feature needs: nothing is deleted from
disk, demoted/dropped skills stay reachable via skill_view/skills_list,
and one fresh load re-stamps them.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)

STORE_NAME = ".skill_usage.json"


def usage_store_path(store_path: Union[str, Path, None] = None) -> Path:
    """Resolve the usage store location (per-user, in HERMES_HOME)."""
    if store_path is not None:
        return Path(store_path)
    from logos_constants import get_logos_home

    return get_logos_home() / STORE_NAME


def load_usage(store_path: Union[str, Path, None] = None) -> Dict[str, float]:
    """Load the usage store. Missing/corrupt file → {} (feature inert)."""
    p = usage_store_path(store_path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def record_skill_use(
    name: str,
    ts: Optional[float] = None,
    store_path: Union[str, Path, None] = None,
) -> None:
    """Record a skill load/maintenance at ``ts`` (default now).

    Keeps the latest stamp per skill; never raises — usage tracking must
    not perturb the tool call that produced it.
    """
    if not name:
        return
    p = usage_store_path(store_path)
    stamp = float(ts if ts is not None else time.time())
    cur: Dict[str, float] = {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            cur = {str(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, TypeError):
        cur = {}
    if stamp < cur.get(str(name), 0.0):
        return  # older stamp — no-op
    cur[str(name)] = stamp
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        from utils import atomic_json_write

        atomic_json_write(p, cur)
    except Exception as e:  # noqa: BLE001 — must never raise
        logger.debug("skill_usage record failed for %r: %s", name, e)


def backfill_from_state_db(
    db_path: Union[str, Path, None] = None,
    store_path: Union[str, Path, None] = None,
) -> int:
    """Seed the usage store from state.db history.

    Scans assistant rows for ``skill_view`` tool_calls and records the
    latest timestamp per skill name. Never overwrites a fresher existing
    stamp (live usage wins over history). Read-only on state.db.

    Returns the number of distinct skills found in the scan (0 when the
    db is missing or unreadable).
    """
    if db_path is None:
        from logos_constants import get_logos_home

        db_path = get_logos_home() / "state.db"
    db_path = Path(db_path)
    if not db_path.exists():
        return 0
    store_p = usage_store_path(store_path)
    existing = load_usage(store_p)
    found: Dict[str, float] = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        logger.debug("skill backfill: cannot open %s: %s", db_path, e)
        return 0
    try:
        rows = conn.execute(
            "SELECT timestamp, tool_calls FROM messages "
            "WHERE role='assistant' AND tool_calls IS NOT NULL "
            "AND tool_calls LIKE '%skill_view%'"
        )
        for ts_raw, tc_raw in rows:
            if not tc_raw:
                continue
            try:
                arr = json.loads(tc_raw)
            except (ValueError, TypeError):
                continue
            if not isinstance(arr, list):
                continue
            for entry in arr:
                try:
                    fn = entry.get("function") if isinstance(entry, dict) else None
                    if not fn or fn.get("name") != "skill_view":
                        continue
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        args = json.loads(args)
                    if not isinstance(args, dict):
                        continue
                    nm = args.get("name")
                    if not nm:
                        continue
                    nm = str(nm)
                    ts = float(ts_raw or 0.0)
                    if ts > found.get(nm, -1.0):
                        found[nm] = ts
                except (ValueError, TypeError, AttributeError):
                    continue
    except sqlite3.Error as e:
        logger.debug("skill backfill: scan failed on %s: %s", db_path, e)
        return 0
    finally:
        conn.close()

    if not found:
        return 0
    changed = False
    for k, v in found.items():
        if v > existing.get(k, -1.0):
            existing[k] = v
            changed = True
    if changed:
        try:
            store_p.parent.mkdir(parents=True, exist_ok=True)
            from utils import atomic_json_write

            atomic_json_write(store_p, existing)
        except Exception as e:  # noqa: BLE001
            logger.debug("skill backfill: store write failed: %s", e)
    return len(found)

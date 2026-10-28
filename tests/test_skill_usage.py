"""Usage-based skill demotion (SKILL-DEMOTE-1028).

Core (full-description) skills that go unused age out of the injected
index: >30d (config) unused → name-only Available tier; >90d unused →
dropped from the index entirely. Still on disk, still reachable via
skill_view/skills_list — only the per-turn injection tax stops.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.skill_injection import SkillInjectionManager  # noqa: E402
from agent.skill_usage import (  # noqa: E402
    load_usage,
    record_skill_use,
    backfill_from_state_db,
)
import agent.skill_usage as su  # noqa: E402

DAY = 86400.0


# ── Usage store ─────────────────────────────────────────────────────────

def test_record_and_load_roundtrip(tmp_path):
    p = tmp_path / ".skill_usage.json"
    record_skill_use("axolotl", store_path=p)
    loaded = load_usage(store_path=p)
    assert abs(loaded["axolotl"] - time.time()) < 5


def test_record_keeps_latest(tmp_path):
    p = tmp_path / ".skill_usage.json"
    record_skill_use("axolotl", ts=time.time() - 100, store_path=p)
    record_skill_use("axolotl", ts=time.time(), store_path=p)
    assert load_usage(store_path=p)["axolotl"] > time.time() - 5


def test_load_missing_or_corrupt_returns_empty(tmp_path):
    assert load_usage(store_path=tmp_path / "nope.json") == {}
    p = tmp_path / "corrupt.json"
    p.write_text("{not json")
    assert load_usage(store_path=p) == {}


# ── Demotion in the injection manager ───────────────────────────────────

def _configs():
    return {
        "used-core": {"category": "core", "priority": "high", "description": "d", "dependencies": []},
        "stale-core": {"category": "core", "priority": "high", "description": "d", "dependencies": []},
        "never-core": {"category": "core", "priority": "high", "description": "d", "dependencies": []},
        "low-skill": {"category": "core", "priority": "low", "description": "d", "dependencies": []},
    }


def test_demote_recent_use_stays_core():
    m = SkillInjectionManager()
    m.build_graph(_configs())
    m.demote_stale_skills({"used-core": time.time()}, time.time() - 30 * DAY)
    assert m.graph["used-core"].tier == 0


def test_demote_unused_core_moves_to_available_tier():
    m = SkillInjectionManager()
    m.build_graph(_configs())
    m.demote_stale_skills({"stale-core": time.time() - 31 * DAY}, time.time() - 30 * DAY)
    assert m.graph["stale-core"].tier == 1
    # still injected (name-only) but not in Core
    s = m.get_injection_set(max_tier=1)
    assert "stale-core" in s
    block = m.format_injection_block(s)
    core_section = block.split("**Available (load on-demand):**")[0]
    assert "``stale-core``" not in core_section


def test_demote_never_used_core_demotes_after_backfill():
    """A skill absent from the usage store means 'never loaded since
    backfill ran' — it must not pay the Core tax."""
    m = SkillInjectionManager()
    m.build_graph(_configs())
    m.demote_stale_skills({"used-core": time.time()}, time.time() - 30 * DAY)
    assert m.graph["never-core"].tier == 1


def test_demote_zero_days_disables():
    m = SkillInjectionManager()
    m.build_graph(_configs())
    m.demote_stale_skills({}, time.time())  # cutoff == now → nothing stale
    assert m.graph["stale-core"].tier == 0
    assert m.graph["never-core"].tier == 0


def test_demote_low_priority_untouched():
    m = SkillInjectionManager()
    m.build_graph(_configs())
    m.demote_stale_skills({}, time.time() - 999 * DAY)
    assert m.graph["low-skill"].tier == 1  # already tier 1, unchanged
    assert m.graph["low-skill"].priority == "low"


def test_exclude_dropped_removes_from_graph():
    m = SkillInjectionManager()
    m.build_graph(_configs())
    m.exclude_dropped(
        {"stale-core": time.time() - 95 * DAY, "never-core": 0.0,
         "used-core": time.time()},
        time.time() - 90 * DAY,
    )
    assert "stale-core" not in m.graph
    assert "never-core" not in m.graph
    assert "used-core" in m.graph
    assert "low-skill" in m.graph


# ── Backfill from state.db ──────────────────────────────────────────────

def _make_state_db(path, rows):
    """rows: list of (timestamp, tool_calls_json)."""
    c = sqlite3.connect(str(path))
    c.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, tool_call_id TEXT, "
        "tool_calls TEXT, tool_name TEXT, timestamp REAL, token_count INTEGER, "
        "finish_reason TEXT, reasoning TEXT, reasoning_content TEXT, "
        "reasoning_details TEXT, codex_reasoning_items TEXT, codex_message_items TEXT)"
    )
    c.executemany(
        "INSERT INTO messages (session_id, role, content, tool_calls, timestamp) "
        "VALUES ('s1', 'assistant', '', ?, ?)",
        [(tc, ts) for ts, tc in rows],
    )
    c.commit()
    c.close()


def test_backfill_parses_variants(tmp_path):
    now = time.time()
    dbp = tmp_path / "state.db"
    _make_state_db(
        dbp,
        [
            (now - 3600, json.dumps(
                [{"id": "x", "type": "function", "function": {"name": "skill_view",
                  "arguments": json.dumps({"name": "axolotl"})}}])),
            (now - 7200, json.dumps(
                [{"id": "y", "type": "function", "function": {"name": "skill_view",
                  "arguments": {"name": "dpo-training"}}}])),
            (now - 9000, json.dumps(
                [{"id": "z", "type": "function", "function": {"name": "skill_view",
                  "arguments": json.dumps({"name": "axolotl"})}}])),
            (now - 9999, json.dumps(
                [{"id": "w", "type": "function", "function": {"name": "terminal",
                  "arguments": {"command": "echo"}}}])),
        ],
    )
    store = tmp_path / "store.json"
    n = backfill_from_state_db(dbp, store)
    assert n >= 2
    u = json.loads(store.read_text())
    # latest wins
    assert abs(u["axolotl"] - (now - 3600)) < 5
    assert abs(u["dpo-training"] - (now - 7200)) < 5
    assert "terminal" not in u


def test_backfill_missing_db_is_noop(tmp_path):
    store = tmp_path / "store.json"
    assert backfill_from_state_db(tmp_path / "missing.db", store) == 0
    assert not store.exists()


def test_backfill_preserves_fresher_existing_entries(tmp_path):
    now = time.time()
    dbp = tmp_path / "state.db"
    _make_state_db(
        dbp,
        [(now - 5 * DAY, json.dumps(
            [{"id": "x", "type": "function", "function": {"name": "skill_view",
              "arguments": json.dumps({"name": "axolotl"})}}]))],
    )
    store = tmp_path / "store.json"
    record_skill_use("axolotl", ts=now, store_path=store)
    backfill_from_state_db(dbp, store)
    u = json.loads(store.read_text())
    assert abs(u["axolotl"] - now) < 5  # live stamp beats 5-day-old backfill

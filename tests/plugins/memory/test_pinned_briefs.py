"""Tests for pinned project briefs (p1).

The system prompt is protected from every pruning path, so a brief
pinned into it persists for the life of the project. These tests cover:
rendering, priority order, expiry archiving, per-brief and total caps,
corrupt-file isolation, and the mtime-fingerprint cache (the update path
overwrites files in place — dir mtime alone would serve stale blocks).
"""
from __future__ import annotations

import os
import threading
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from plugins.memory.perpetual_context import pinned_briefs
from plugins.memory.perpetual_context.pinned_briefs import (
    _parse_frontmatter,
    get_pinned_block,
    invalidate_cache,
    load_briefs,
    render_briefs,
)

BRIEF = """---
project: {project}
{extra}
---
## Goal
{body}
"""


@pytest.fixture(autouse=True)
def _no_cache():
    invalidate_cache()
    yield
    invalidate_cache()


# ── frontmatter ─────────────────────────────────────────────────────────


def test_frontmatter_parse_full():
    meta, body = _parse_frontmatter(
        "---\nproject: x\nexpires: 2026-12-31\npriority: 2\n---\nhello\n"
    )
    assert meta == {"project": "x", "expires": "2026-12-31", "priority": "2"}
    assert body.strip() == "hello"


def test_frontmatter_absent():
    meta, body = _parse_frontmatter("just body text\n")
    assert meta == {}
    assert body == "just body text\n"


# ── load_briefs ─────────────────────────────────────────────────────────


def test_missing_dir_yields_nothing(tmp_path: Path):
    assert load_briefs(tmp_path / "nope") == []


def test_single_brief_loaded(tmp_path: Path):
    (tmp_path / "proj.md").write_text(
        BRIEF.format(project="alpha", extra="priority: 1", body="Ship it."),
        encoding="utf-8",
    )
    briefs = load_briefs(tmp_path)
    assert len(briefs) == 1
    assert briefs[0]["project"] == "alpha"
    assert briefs[0]["priority"] == 1
    assert "Ship it." in briefs[0]["body"]


def test_expired_brief_archived(tmp_path: Path):
    past = (datetime.now().astimezone() - timedelta(days=1)).date().isoformat()
    (tmp_path / "old.md").write_text(
        BRIEF.format(project="old", extra=f"expires: {past}", body="done"),
        encoding="utf-8",
    )
    assert load_briefs(tmp_path) == []
    assert (tmp_path / "archive" / "old.md").exists()
    assert not (tmp_path / "old.md").exists()


def test_future_brief_stays(tmp_path: Path):
    future = (datetime.now().astimezone() + timedelta(days=30)).date().isoformat()
    (tmp_path / "live.md").write_text(
        BRIEF.format(project="live", extra=f"expires: {future}", body="ongoing"),
        encoding="utf-8",
    )
    briefs = load_briefs(tmp_path)
    assert [b["project"] for b in briefs] == ["live"]


def test_unparseable_expiry_keeps_brief(tmp_path: Path):
    (tmp_path / "x.md").write_text(
        BRIEF.format(project="x", extra="expires: someday", body="keep me"),
        encoding="utf-8",
    )
    assert len(load_briefs(tmp_path)) == 1


def test_corrupt_file_does_not_sink_others(tmp_path: Path):
    (tmp_path / "a.md").mkdir()  # directory matching *.md → read fails
    (tmp_path / "b.md").write_text(
        BRIEF.format(project="b", extra="", body="fine"), encoding="utf-8"
    )
    briefs = load_briefs(tmp_path)
    assert [b["project"] for b in briefs] == ["b"]


# ── render_briefs ───────────────────────────────────────────────────────


def test_render_priority_order(tmp_path: Path):
    (tmp_path / "z.md").write_text(
        BRIEF.format(project="zeta", extra="priority: 1", body="first"),
        encoding="utf-8",
    )
    (tmp_path / "a.md").write_text(
        BRIEF.format(project="alpha", extra="", body="second"), encoding="utf-8"
    )
    out = render_briefs(load_briefs(tmp_path))
    assert out.index("### zeta") < out.index("### alpha")


def test_render_until_label(tmp_path: Path):
    briefs = [
        {"name": "x", "project": "x", "priority": 1, "expires": "2026-12-31",
         "path": "/tmp/x.md", "body": "b", "max_chars": 3000},
        {"name": "y", "project": "y", "priority": 2, "expires": "",
         "path": "/tmp/y.md", "body": "b", "max_chars": 3000},
    ]
    out = render_briefs(briefs)
    assert "until 2026-12-31" in out
    assert "until unpinned" in out


def test_per_brief_truncation(tmp_path: Path):
    (tmp_path / "big.md").write_text(
        BRIEF.format(project="big", extra="max_chars: 20", body="A" * 100),
        encoding="utf-8",
    )
    out = render_briefs(load_briefs(tmp_path))
    assert "…truncated — full brief at" in out
    assert "A" * 21 not in out


def test_total_cap_drops_overflow(tmp_path: Path):
    (tmp_path / "one.md").write_text(
        BRIEF.format(project="one", extra="priority: 1", body="B" * 100),
        encoding="utf-8",
    )
    (tmp_path / "two.md").write_text(
        BRIEF.format(project="two", extra="priority: 2", body="C" * 100),
        encoding="utf-8",
    )
    out = render_briefs(load_briefs(tmp_path), max_total_chars=100)
    assert "### one" in out
    assert "### two" not in out


def test_total_cap_truncates_second_brief(tmp_path: Path):
    (tmp_path / "one.md").write_text(
        BRIEF.format(project="one", extra="priority: 1", body="B" * 100),
        encoding="utf-8",
    )
    (tmp_path / "two.md").write_text(
        BRIEF.format(project="two", extra="priority: 2", body="C" * 100),
        encoding="utf-8",
    )
    out = render_briefs(load_briefs(tmp_path), max_total_chars=150)
    assert "### one" in out
    assert "### two" in out  # rendered within the remaining 50-char budget
    assert "…truncated" in out


# ── get_pinned_block cache ─────────────────────────────────────────────


def test_block_cached_then_invalidated_on_overwrite(tmp_path: Path):
    """Overwriting an existing file must not serve a stale block."""
    (tmp_path / "p.md").write_text(
        BRIEF.format(project="p", extra="", body="v1"), encoding="utf-8"
    )
    assert "v1" in get_pinned_block(tmp_path)

    # Same content length, newer mtime — dir mtime unchanged.
    (tmp_path / "p.md").write_text(
        BRIEF.format(project="p", extra="", body="v2"), encoding="utf-8"
    )
    st = (tmp_path / "p.md").stat()
    os.utime(tmp_path / "p.md", (st.st_atime, st.st_mtime + 5))

    assert "v2" in get_pinned_block(tmp_path)


def test_block_stable_across_calls(tmp_path: Path):
    (tmp_path / "p.md").write_text(
        BRIEF.format(project="p", extra="", body="stable"), encoding="utf-8"
    )
    first = get_pinned_block(tmp_path)
    second = get_pinned_block(tmp_path)
    assert first == second
    assert "## Pinned Project Briefs" in first


def test_empty_dir_returns_empty(tmp_path: Path):
    # tmp_path is an existing empty directory
    assert get_pinned_block(tmp_path) == ""


# ── plugin integration ──────────────────────────────────────────────────


def test_system_prompt_block_appends_pinned(tmp_path: Path):
    from plugins.memory.perpetual_context import PerpetualContextProvider

    (tmp_path / "proj.md").write_text(
        BRIEF.format(project="proj", extra="", body="the goal"),
        encoding="utf-8",
    )
    p = PerpetualContextProvider()
    p._db = types.SimpleNamespace(
        _initialized=True,
        get_stats=lambda: {"message_count": 1, "session_count": 1, "topic_count": 0},
    )
    p._current_depth = "moderate"
    p._lock = threading.RLock()
    p._pinned_enabled = True
    p._pinned_dir = tmp_path
    out = p.system_prompt_block()
    assert "## Pinned Project Briefs" in out
    assert "the goal" in out


def test_pinned_disabled_leaves_block_untouched(tmp_path: Path):
    from plugins.memory.perpetual_context import PerpetualContextProvider

    (tmp_path / "proj.md").write_text(
        BRIEF.format(project="proj", extra="", body="the goal"),
        encoding="utf-8",
    )
    p = PerpetualContextProvider()
    p._db = types.SimpleNamespace(
        _initialized=True,
        get_stats=lambda: {"message_count": 1, "session_count": 1, "topic_count": 0},
    )
    p._current_depth = "moderate"
    p._lock = threading.RLock()
    p._pinned_enabled = False
    p._pinned_dir = tmp_path
    out = p.system_prompt_block()
    assert "Pinned Project Briefs" not in out

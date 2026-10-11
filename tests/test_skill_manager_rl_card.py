"""Regression test — RL lookup card frontmatter must be valid YAML.

10/11 defect (measured): two live RL cards had MALFORMED frontmatter —
the generator wrote ``description: {description[:200]}`` unquoted, and a
200-char truncation of a description that itself starts with a double
quote leaves an UNCLOSED scalar:

    description: "Use when two scripts that must be mutually exclusive
    each hold their OWN flock file ... still holds     <- cut at 200, no closing "

``yaml.safe_load`` fails with "while scanning a quoted scalar". Live
victims 10/11: skills/flock-lockfile-migration.md,
skills/sqlite-fts5-concurrent-writer-drift.md (generated pre-restart by
the old code; fixed at 03:32 in prod working tree, gated here).

The fix wraps the truncated description in ``json.dumps`` — a JSON
string is a valid YAML double-quoted scalar (escape-compatible
subset), so any truncation point yields parseable frontmatter.

RL of record: technology/aux-thinking-mode-empty-content-2026-10-11.md
(sibling finding, same 10/11 dig) — card defect is tracked in
active-tasks.md.
"""
import os

import pytest
import yaml

from tools.skill_manager_tool import _generate_rl_lookup_card


def _card_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


_SKILL_BODY = (
    "---\n"
    "name: test-skill\n"
    "description: %s\n"
    "---\n"
    "\n"
    "# Test Skill\n"
    "\n"
    "## When to Use\n"
    "- when things break\n"
)


def test_card_frontmatter_valid_yaml_colon_and_quote(tmp_path, monkeypatch):
    """The live failure shape: description starts with a quote, contains
    colons, and is long enough that [:200] cuts mid-string."""
    _card_home(tmp_path, monkeypatch)
    # 230 chars, opens with a literal double quote (never closed)
    desc = '"' + "Use when two scripts that must be mutually exclusive " * 6
    assert len(desc) > 200 and desc[0] == '"'
    card = _generate_rl_lookup_card(
        "test-skill", _SKILL_BODY % desc, category="devops"
    )
    assert card is not None
    text = card.read_text(encoding="utf-8")
    end = text.find("---", 3)
    fm = yaml.safe_load(text[3:end])  # raises on the pre-fix shape
    assert fm["name"] == "test-skill"
    assert fm["category"] == "devops"
    # Truncated to 200 but round-trips exactly through YAML
    assert fm["description"] == desc[:200]


def test_card_short_plain_description(tmp_path, monkeypatch):
    _card_home(tmp_path, monkeypatch)
    desc = "Simple description with: a colon."
    card = _generate_rl_lookup_card(
        "test-skill", _SKILL_BODY % desc, category=None
    )
    assert card is not None
    text = card.read_text(encoding="utf-8")
    end = text.find("---", 3)
    fm = yaml.safe_load(text[3:end])
    assert fm["description"] == desc
    assert fm["category"] == "general"


def test_card_no_description_falls_back(tmp_path, monkeypatch):
    _card_home(tmp_path, monkeypatch)
    body = "---\nname: test-skill\n---\n\n# Test Skill\n"
    card = _generate_rl_lookup_card("test-skill", body)
    assert card is not None
    text = card.read_text(encoding="utf-8")
    end = text.find("---", 3)
    fm = yaml.safe_load(text[3:end])
    assert fm["description"] == "Skill for task automation"

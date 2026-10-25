"""Tests for the one-file user profile read (owner steer 10/25,
USER-PROFILE-ONEFILE-1025) — AIAgent._read_user_profile_core.

Component-level coverage of the read path: page resolution under
get_logos_home(), section extraction, and the over-cap hard cap.
The injection branch in _build_system_prompt is config-gated
(user_profile_page set = core-section read; unset = legacy USER.md
block) and rides on the memory_tool extraction/redirect tests plus the
functional smoke on the live RL page.
"""

from __future__ import annotations

import run_agent as run_agent_module
from run_agent import AIAgent


def _bare_agent(page: str, limit: int = 2200) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._user_profile_page = page
    agent._user_profile_char_limit = limit
    return agent


def _write_page(tmp_path, rel: str, text: str):
    path = tmp_path / "reference-library" / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class TestReadUserProfileCore:
    def test_reads_core_section_from_page(self, tmp_path, monkeypatch):
        _write_page(
            tmp_path,
            "people/p.md",
            "# P\n\nintro\n\n"
            "## Agent Operating Core\n\n- rule one\n- rule two\n\n"
            "## Work\n\njob\n",
        )
        monkeypatch.setattr(run_agent_module, "get_logos_home", lambda: tmp_path)
        out = _bare_agent("people/p.md")._read_user_profile_core()
        assert out == "## Agent Operating Core\n\n- rule one\n- rule two"

    def test_missing_page_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(run_agent_module, "get_logos_home", lambda: tmp_path)
        assert _bare_agent("people/nope.md")._read_user_profile_core() is None

    def test_page_without_core_section_returns_none(self, tmp_path, monkeypatch):
        _write_page(tmp_path, "people/p.md", "# P\n\n## Other\n\ntext\n")
        monkeypatch.setattr(run_agent_module, "get_logos_home", lambda: tmp_path)
        assert _bare_agent("people/p.md")._read_user_profile_core() is None

    def test_over_cap_is_hard_capped(self, tmp_path, monkeypatch):
        _write_page(
            tmp_path,
            "people/p.md",
            "## Agent Operating Core\n\n" + "x" * 3000 + "\n\n## Next\n\ny\n",
        )
        monkeypatch.setattr(run_agent_module, "get_logos_home", lambda: tmp_path)
        out = _bare_agent("people/p.md", limit=1000)._read_user_profile_core()
        assert out is not None
        assert len(out) <= 1000

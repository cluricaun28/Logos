"""Tests for idle-tool demotion (selective injection, phase 2).

Demotion drops promoted (deferred-tier) tools back to index-only after N
unused API-call rounds. The curated essential set never auto-demotes.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import model_tools  # noqa: F401 — ensure router deps importable
# Make the repo's tool_router importable the same way model_tools does.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plugins" / "tool_router"))


def _fresh_router():
    """Reset both singleton layers (model_tools wrapper + tool_router)."""
    try:
        import tool_router as tr
        tr.reset_tool_router()
    except Exception:
        pass
    model_tools._tool_router_instance = None
    return model_tools._get_tool_router()


@pytest.fixture(autouse=True)
def _reset_router():
    _fresh_router()
    yield
    _fresh_router()


def _make_stub_agent(n_turns: int = 3):
    """Minimal agent stub exposing only the demotion-relevant state."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)  # skip heavy __init__
    agent._selective_demote_after_turns = n_turns
    agent._promoted_tools = {}
    agent._demoted_tools = set()
    agent.valid_tool_names = {"web_search", "browser_navigate"}
    agent.tools = [
        {"type": "function", "function": {"name": "web_search"}},
        {"type": "function", "function": {"name": "browser_navigate"}},
    ]
    return agent


class TestIdleDemotion:
    def test_demote_after_n_unused_rounds(self):
        from run_agent import AIAgent

        agent = _make_stub_agent(n_turns=3)
        AIAgent._record_promotion(agent, "browser_navigate", 1)

        # Rounds 2 and 3: not yet idle enough.
        AIAgent._maybe_demote_tools(agent, 2)
        AIAgent._maybe_demote_tools(agent, 3)
        assert "browser_navigate" in agent.valid_tool_names
        assert "browser_navigate" not in agent._demoted_tools

        # Round 4: 4 - 1 = 3 >= 3 → demoted.
        AIAgent._maybe_demote_tools(agent, 4)
        assert "browser_navigate" not in agent.valid_tool_names
        assert "browser_navigate" not in agent._promoted_tools
        assert "browser_navigate" in agent._demoted_tools
        names = [t["function"]["name"] for t in agent.tools]
        assert "browser_navigate" not in names
        # Untouched tools stay.
        assert "web_search" in agent.valid_tool_names
        assert "web_search" in names

    def test_refresh_prevents_demotion(self):
        from run_agent import AIAgent

        agent = _make_stub_agent(n_turns=3)
        AIAgent._record_promotion(agent, "browser_navigate", 1)
        # Used again at round 2 → clock resets.
        AIAgent._record_promotion(agent, "browser_navigate", 2)
        AIAgent._maybe_demote_tools(agent, 4)  # 4-2=2 < 3
        assert "browser_navigate" in agent.valid_tool_names

    def test_disabled_when_n_is_zero(self):
        from run_agent import AIAgent

        agent = _make_stub_agent(n_turns=0)
        agent._selective_demote_after_turns = 0
        AIAgent._record_promotion(agent, "browser_navigate", 1)
        AIAgent._maybe_demote_tools(agent, 99)
        assert "browser_navigate" in agent.valid_tool_names

    def test_record_promotion_noop_when_disabled(self):
        from run_agent import AIAgent

        agent = _make_stub_agent(n_turns=0)
        agent._selective_demote_after_turns = 0
        # Must not raise, must not track.
        AIAgent._record_promotion(agent, "browser_navigate", 1)
        assert agent._promoted_tools == {}

    def test_essential_tools_never_tracked(self):
        """The dispatch refresh path only updates already-promoted tools,
        so curated essentials (never in _promoted_tools) can't be demoted."""
        from run_agent import AIAgent

        agent = _make_stub_agent(n_turns=3)
        assert "web_search" not in agent._promoted_tools
        AIAgent._maybe_demote_tools(agent, 99)
        assert "web_search" in agent.valid_tool_names


class TestDemotedIndex:
    def test_recently_demoted_section(self):
        router = _fresh_router()
        assert router is not None
        # Non-static tool → appears under Recently Demoted.
        index = router.get_deferred_index(
            injected_names={"web_search"},
            extra_deferred={"weather_fetch"},
        )
        assert "Recently Demoted" in index
        assert "'''weather_fetch'''" in index
        assert "'''web_search'''" not in index  # injected → not listed as a tool

    def test_static_deferred_tool_not_duplicated(self):
        router = _fresh_router()
        # browser_navigate is in the static deferred set → listed once in
        # Browser Suite, not re-listed under Recently Demoted.
        index = router.get_deferred_index(
            injected_names=set(),
            extra_deferred={"browser_navigate"},
        )
        assert "Recently Demoted" not in index
        assert index.count("'''browser_navigate'''") == 1

    def test_demoted_section_empty_when_nothing_demoted(self):
        router = _fresh_router()
        index = router.get_deferred_index(injected_names={"web_search"})
        assert "Recently Demoted" not in index

    def test_cache_key_includes_demoted_set(self):
        router = _fresh_router()
        a = router.get_deferred_index(injected_names=set())
        b = router.get_deferred_index(
            injected_names=set(), extra_deferred={"weather_fetch"}
        )
        assert "Recently Demoted" not in a
        assert "Recently Demoted" in b
        # Back to no demotion → cached 'a' result, not b's.
        c = router.get_deferred_index(injected_names=set())
        assert "Recently Demoted" not in c

    def test_model_tools_passthrough(self):
        idx = model_tools.get_deferred_tools_index(
            injected_names=set(),
            extra_deferred={"weather_fetch"},
        )
        assert idx and "Recently Demoted" in idx and "weather_fetch" in idx

    def test_promoted_tool_still_promotable_after_demotion(self, monkeypatch):
        """Re-promotion path: after demotion the tool re-enters on next call."""
        from run_agent import AIAgent
        import tools.registry as reg

        fake_schema = {
            "type": "function",
            "function": {"name": "browser_navigate", "description": "x", "parameters": {}},
        }
        monkeypatch.setattr(
            reg.registry, "get_definitions",
            lambda names, quiet=False: [fake_schema] if "browser_navigate" in names else [],
        )

        agent = _make_stub_agent(n_turns=1)
        # Remove it to simulate the demoted state.
        agent.valid_tool_names.discard("browser_navigate")
        agent.tools = [
            t for t in agent.tools
            if t["function"]["name"] != "browser_navigate"
        ]
        agent._demoted_tools.add("browser_navigate")

        promoted = AIAgent._promote_deferred_tool(agent, "browser_navigate", 7)
        assert promoted
        assert "browser_navigate" in agent.valid_tool_names
        # And promotion clears the demoted flag + tracks for next idle period.
        assert "browser_navigate" not in agent._demoted_tools
        assert agent._promoted_tools.get("browser_navigate") == 7

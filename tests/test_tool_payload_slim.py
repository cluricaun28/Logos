"""Tool payload slim (2026-08-21) — selective-injection re-balance.

Covers:
  - Re-audited essential/deferred tier split (process + vision_analyze
    promoted; skills_list/clarify + PM extended suite demoted).
  - Deferred index no longer double-lists force-injected tools.
  - PM provider tiering: core injected, extended deferred but dispatchable.
  - Deferred promotion on first call (registry + memory-provider paths).
  - agent.selective_force_toolsets opt-in (default: router split is truth).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Router tier split
# ---------------------------------------------------------------------------

def test_router_tier_split_2026_08_21():
    from plugins.tool_router.tool_router import (
        PM_CORE_TOOLS,
        PM_EXTENDED_TOOLS,
        ToolRouter,
    )

    r = ToolRouter()
    # Promoted to essential by the 90-day usage audit
    for tool in ("process", "vision_analyze", "terminal", "execute_code",
                 "delegate_task", "cronjob", "read_file", "patch"):
        assert r.is_essential(tool), f"{tool} should be essential"
    # Demoted to deferred
    for tool in ("browser_navigate", "browser_vision", "send_message",
                 "text_to_speech", "clarify", "skills_list", "image_generate"):
        assert r.is_deferred(tool), f"{tool} should be deferred"
    # PM core stays essential-tier; PM extended is deferred
    for tool in PM_CORE_TOOLS:
        assert not r.is_deferred(tool), f"PM core {tool} must not be deferred"
    for tool in PM_EXTENDED_TOOLS:
        assert r.is_deferred(tool), f"PM extended {tool} should be deferred"


def test_essential_definitions_include_pm_core():
    from plugins.tool_router.tool_router import ToolRouter

    r = ToolRouter()
    fake_defs = [
        {"function": {"name": "perpetual_search"}},
        {"function": {"name": "read_file"}},
        {"function": {"name": "browser_navigate"}},
    ]
    names = {d["function"]["name"] for d in r.get_essential_definitions(fake_defs)}
    assert {"perpetual_search", "read_file"} <= names
    assert "browser_navigate" not in names


def test_deferred_index_excludes_injected_tools():
    from plugins.tool_router.tool_router import ToolRouter

    r = ToolRouter()
    full = r.get_deferred_index()
    assert "browser_navigate" in full
    assert "query_messages" in full

    # Force-injected tools must not be double-listed
    partial = r.get_deferred_index(injected_names={"browser_navigate", "query_messages"})
    assert "browser_navigate" not in partial
    assert "query_messages" not in partial
    # Unrelated deferred tools remain
    assert "browser_vision" in partial

    # Everything injected → empty index
    all_names = set()
    for line in full.splitlines():
        if "'''" in line:
            all_names.add(line.split("'''")[1])
    assert r.get_deferred_index(injected_names=all_names) == ""


# ---------------------------------------------------------------------------
# PM provider tiering
# ---------------------------------------------------------------------------

def test_pm_provider_core_only_injection():
    from plugins.memory.perpetual_context import PerpetualContextProvider
    from plugins.memory.perpetual_context import schemas as pm_schemas

    p = PerpetualContextProvider()
    core = {s["name"] for s in p.get_tool_schemas()}
    all_names = {s["name"] for s in p.get_all_tool_schemas()}

    assert core == pm_schemas.CORE_TOOL_NAMES
    assert len(all_names) == 10
    # extended = full minus core, exactly the deferred PM suite
    from plugins.tool_router.tool_router import PM_EXTENDED_TOOLS
    assert all_names - core == PM_EXTENDED_TOOLS


def test_memory_manager_extended_schemas_exclude_core():
    from agent.memory_manager import MemoryManager

    core = [{"name": "perpetual_search"}]
    extended = [{"name": "query_messages"}, {"name": "topic_flow"}]
    fake_provider = SimpleNamespace(
        name="fake-pm",
        get_tool_schemas=lambda: list(core),
        get_all_tool_schemas=lambda: list(core + extended),
    )

    mm = MemoryManager.__new__(MemoryManager)
    mm._providers = [fake_provider]
    assert [s["name"] for s in mm.get_all_tool_schemas()] == ["perpetual_search"]
    assert [s["name"] for s in mm.get_extended_tool_schemas()] == ["query_messages", "topic_flow"]


# ---------------------------------------------------------------------------
# Selective injection end-to-end (model_tools)
# ---------------------------------------------------------------------------

def _selective_defs(**kwargs):
    from model_tools import get_selective_tool_definitions

    with patch("hermes_cli.config.load_config", return_value={}):
        return get_selective_tool_definitions(quiet_mode=True, **kwargs)


def test_selective_injection_defers_browser_suite():
    defs = _selective_defs(enabled_toolsets=None)
    names = {d["function"]["name"] for d in defs}
    # Essentials present
    assert "terminal" in names and "read_file" in names
    # Browser suite is deferred, NOT force-injected (the 2026-08-21 fix:
    # force_essential=enabled_toolsets used to nullify the whole split)
    assert "browser_navigate" not in names
    assert "clarify" not in names
    # PM tools are NOT registry tools — they ride the memory-manager path
    # (covered by the provider/manager tests above), so this layer must
    # not claim them either way.
    assert "perpetual_search" not in names
    assert "query_messages" not in names


def test_deferred_index_no_overlap_with_injected():
    from model_tools import get_deferred_tools_index

    defs = _selective_defs(enabled_toolsets=None)
    names = {d["function"]["name"] for d in defs}
    index = get_deferred_tools_index(injected_names=names)
    # No tool appears both injected and indexed
    for name in names:
        assert f"'''{name}'''" not in index, f"{name} double-listed"


def test_force_toolsets_config_opt_in():
    from run_agent import AIAgent

    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config",
               return_value={"agent": {"selective_force_toolsets": ["clarify"]}}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    names = {t["function"]["name"] for t in agent.tools}
    # clarify (deferred by default) is force-included via config
    assert "clarify" in names
    # ...but browser suite stays deferred (not force-listed)
    assert "browser_navigate" not in names


def test_default_config_has_no_forced_toolsets():
    from run_agent import AIAgent

    with patch("run_agent.OpenAI"), \
         patch("hermes_cli.config.load_config", return_value={}):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent._selective_force_toolsets is None
    names = {t["function"]["name"] for t in agent.tools}
    assert "clarify" not in names  # deferred, not forced


# ---------------------------------------------------------------------------
# Deferred promotion on first call
# ---------------------------------------------------------------------------

def _bare_agent():
    from run_agent import AIAgent

    a = AIAgent.__new__(AIAgent)
    a.tools = [{"type": "function", "function": {"name": "terminal"}}]
    a.valid_tool_names = {"terminal"}
    a._memory_manager = None
    a.selective_injection = True
    return a


def test_promote_registry_tool():
    a = _bare_agent()
    assert a._promote_deferred_tool("clarify") is True
    assert "clarify" in a.valid_tool_names
    assert any(t["function"]["name"] == "clarify" for t in a.tools)
    # Idempotent
    n = len(a.tools)
    assert a._promote_deferred_tool("clarify") is True
    assert len(a.tools) == n


def test_promote_memory_provider_tool():
    a = _bare_agent()
    fake_mm = SimpleNamespace(
        get_extended_tool_schemas=lambda: [
            {"name": "query_messages", "description": "d", "parameters": {}},
        ],
    )
    a._memory_manager = fake_mm
    assert a._promote_deferred_tool("query_messages") is True
    assert "query_messages" in a.valid_tool_names
    promoted = [t for t in a.tools if t["function"]["name"] == "query_messages"]
    assert len(promoted) == 1
    assert promoted[0]["type"] == "function"


def test_promote_unknown_tool_returns_false():
    a = _bare_agent()
    assert a._promote_deferred_tool("definitely_not_a_tool_xyz") is False
    assert "definitely_not_a_tool_xyz" not in a.valid_tool_names

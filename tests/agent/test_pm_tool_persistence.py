"""Tests for PM tool-result persistence (2026-08-19).

Closes the recall gap: tool output was never synced to Perpetual Context
(only user+assistant), so hard-pruned tool results were unrecoverable.
Now this turn's tool results sync as role='tool' (truncated 3000+3000),
zero embedding cost (tool role already skipped), FTS auto-indexed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FakeDB:
    def __init__(self):
        self._initialized = True
        self.stored = []

    def add_message(self, session_id, role, content, metadata=None, timestamp=None):
        self.stored.append((session_id, role, content, metadata))
        return len(self.stored)


class _LegacyProvider:
    """Provider without tool_results support (e.g. Honcho)."""
    name = "legacy"

    def __init__(self):
        self.calls = []

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        self.calls.append((user_content, assistant_content, session_id))


class _ModernProvider:
    name = "modern"

    def __init__(self):
        self.calls = []

    def sync_turn(self, user_content, assistant_content, *, session_id="", tool_results=None):
        self.calls.append((user_content, assistant_content, session_id, tool_results))


def make_provider():
    from plugins.memory.perpetual_context import PerpetualContextProvider

    p = PerpetualContextProvider.__new__(PerpetualContextProvider)
    p._db = FakeDB()
    p._session_id = "sess-1"
    p._agent_context = "primary"
    import threading
    p._lock = threading.RLock()
    return p


class TestProviderSyncTurnTools:
    def test_tool_results_persisted_with_role_and_metadata(self):
        p = make_provider()
        p.sync_turn("hello", "world", tool_results=[("terminal", "some command output")])
        roles = [r[1] for r in p._db.stored]
        assert roles == ["user", "assistant", "tool"]
        _, role, content, meta = p._db.stored[2]
        assert role == "tool"
        assert content == "some command output"
        assert meta["tool"] == "terminal"

    def test_long_tool_result_truncated_head_and_tail(self):
        p = make_provider()
        big = "H" * 3000 + "M" * 2000 + "T" * 3000  # 8000 chars
        p.sync_turn("u", "a", tool_results=[("read_file", big)])
        content = p._db.stored[2][2]
        assert content.startswith("H" * 3000)
        assert content.endswith("T" * 3000)
        assert "...[truncated]..." in content
        assert "M" * 100 not in content  # middle dropped
        assert len(content) < 8000

    def test_short_and_empty_results_skipped(self):
        p = make_provider()
        p.sync_turn("u", "a", tool_results=[("terminal", "ok"), ("read_file", None), ("x", "y" * 9)])
        assert [r[1] for r in p._db.stored] == ["user", "assistant"]

    def test_no_tool_results_is_legacy_behavior(self):
        p = make_provider()
        p.sync_turn("u", "a")
        assert [r[1] for r in p._db.stored] == ["user", "assistant"]

    def test_cron_context_still_skips_entirely(self):
        p = make_provider()
        p._agent_context = "cron"
        p.sync_turn("u", "a", tool_results=[("terminal", "output here")])
        assert p._db.stored == []


class TestMemoryManagerRouting:
    def test_tool_results_routed_by_signature(self):
        from agent.memory_manager import MemoryManager

        legacy = _LegacyProvider()
        modern = _ModernProvider()
        mm = MemoryManager.__new__(MemoryManager)
        mm._providers = [legacy, modern]
        tools = [("terminal", "out1"), ("read_file", "out2")]
        mm.sync_all("user msg", "asst msg", session_id="s1", tool_results=tools)
        # legacy got the old 3-arg call (no crash)
        assert legacy.calls == [("user msg", "asst msg", "s1")]
        # modern got the tool_results
        assert modern.calls[0][3] == tools

    def test_no_tool_results_uses_legacy_call_for_all(self):
        from agent.memory_manager import MemoryManager

        modern = _ModernProvider()
        mm = MemoryManager.__new__(MemoryManager)
        mm._providers = [modern]
        mm.sync_all("u", "a", session_id="s1")
        assert modern.calls[0][3] is None


class TestRunAgentTurnSlicing:
    def test_collects_only_this_turns_tool_results(self):
        from run_agent import AIAgent

        class Stub:
            _session_messages = [
                {"role": "user", "content": "previous question"},
                {"role": "assistant", "content": "previous answer"},
                {"role": "tool", "name": "terminal", "content": "old output"},
                {"role": "user", "content": "current question"},
                {"role": "assistant", "content": "", "tool_calls": [{"x": 1}]},
                {"role": "tool", "name": "terminal", "content": "fresh output"},
                {"role": "tool", "name": "read_file", "content": "file body"},
                {"role": "assistant", "content": "final"},
            ]

        class StubMM:
            def __init__(self):
                self.tool_results = "unset"

            def sync_all(self, u, a, tool_results=None):
                self.tool_results = tool_results

            def queue_prefetch_all(self, q, session_id=""):
                pass

        stub = Stub()
        stub._memory_manager = StubMM()
        AIAgent._sync_external_memory_for_turn(
            stub, original_user_message="current question",
            final_response="final", interrupted=False,
        )
        assert stub._memory_manager.tool_results == [
            ("terminal", "fresh output"),
            ("read_file", "file body"),
        ]

    def test_no_tool_messages_gives_none(self):
        from run_agent import AIAgent

        class Stub:
            _session_messages = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]

        class StubMM:
            def __init__(self):
                self.tool_results = "unset"

            def sync_all(self, u, a, tool_results=None):
                self.tool_results = tool_results

            def queue_prefetch_all(self, q, session_id=""):
                pass

        stub = Stub()
        stub._memory_manager = StubMM()
        AIAgent._sync_external_memory_for_turn(
            stub, original_user_message="q", final_response="a", interrupted=False,
        )
        assert stub._memory_manager.tool_results is None

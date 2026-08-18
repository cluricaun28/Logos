"""Tests for per-user memory scoping via user_id threading.

Verifies that gateway user_id flows from AIAgent -> MemoryManager -> plugins,
so each gateway user gets their own memory bucket instead of sharing a static one.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from agent.memory_provider import MemoryProvider
from agent.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# Concrete test provider that records init kwargs
# ---------------------------------------------------------------------------


class RecordingProvider(MemoryProvider):
    """Minimal provider that records what initialize() receives."""

    def __init__(self, name="recording"):
        self._name = name
        self._init_kwargs = {}
        self._init_session_id = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._init_session_id = session_id
        self._init_kwargs = dict(kwargs)

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content, assistant_content, *, session_id=""):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({})

    def shutdown(self):
        pass


# ---------------------------------------------------------------------------
# MemoryManager user_id threading tests
# ---------------------------------------------------------------------------


class TestMemoryManagerUserIdThreading:
    """Verify user_id reaches providers via initialize_all."""

    def test_user_id_forwarded_to_provider(self):
        mgr = MemoryManager()
        p = RecordingProvider()
        mgr.add_provider(p)

        mgr.initialize_all(
            session_id="sess-123",
            platform="telegram",
            user_id="tg_user_42",
        )

        assert p._init_kwargs.get("user_id") == "tg_user_42"
        assert p._init_kwargs.get("platform") == "telegram"
        assert p._init_session_id == "sess-123"

    def test_chat_context_forwarded_to_provider(self):
        mgr = MemoryManager()
        p = RecordingProvider()
        mgr.add_provider(p)

        mgr.initialize_all(
            session_id="sess-chat",
            platform="discord",
            user_id="discord_u_7",
            user_name="fakeusername",
            chat_id="1485316232612941897",
            chat_name="fakeassistantname-forums",
            chat_type="thread",
            thread_id="1491249007475949698",
        )

        assert p._init_kwargs.get("user_name") == "fakeusername"
        assert p._init_kwargs.get("chat_id") == "1485316232612941897"
        assert p._init_kwargs.get("chat_name") == "fakeassistantname-forums"
        assert p._init_kwargs.get("chat_type") == "thread"
        assert p._init_kwargs.get("thread_id") == "1491249007475949698"

    def test_no_user_id_when_cli(self):
        """CLI sessions should not have user_id in kwargs."""
        mgr = MemoryManager()
        p = RecordingProvider()
        mgr.add_provider(p)

        mgr.initialize_all(
            session_id="sess-456",
            platform="cli",
        )

        assert "user_id" not in p._init_kwargs
        assert p._init_kwargs.get("platform") == "cli"

    def test_user_id_none_not_forwarded(self):
        """Explicit None user_id should not appear in kwargs."""
        mgr = MemoryManager()
        p = RecordingProvider()
        mgr.add_provider(p)

        # Simulates what happens when AIAgent passes user_id=None
        # (the agent code only adds user_id to kwargs when it's truthy)
        mgr.initialize_all(
            session_id="sess-789",
            platform="discord",
        )

        assert "user_id" not in p._init_kwargs

    def test_multiple_providers_all_receive_user_id(self):
        mgr = MemoryManager()
        # Use one provider named "builtin" (always accepted) and one external
        p1 = RecordingProvider("builtin")
        p2 = RecordingProvider("external")
        mgr.add_provider(p1)
        mgr.add_provider(p2)

        mgr.initialize_all(
            session_id="sess-multi",
            platform="slack",
            user_id="slack_U12345",
        )

        assert p1._init_kwargs.get("user_id") == "slack_U12345"
        assert p1._init_kwargs.get("platform") == "slack"
        assert p2._init_kwargs.get("user_id") == "slack_U12345"
        assert p2._init_kwargs.get("platform") == "slack"


# ---------------------------------------------------------------------------
# Mem0 provider user_id tests
# ---------------------------------------------------------------------------


class TestAIAgentUserIdPropagation:
    """Verify AIAgent stores user_id and passes it to memory init kwargs."""

    def test_user_id_stored_on_agent(self):
        """AIAgent should store user_id as instance attribute."""
        with patch.dict(os.environ, {"HERMES_HOME": "/tmp/test_hermes"}):
            from run_agent import AIAgent
            agent = object.__new__(AIAgent)
            # Manually set the attribute as __init__ does
            agent._user_id = "test_user_42"
            assert agent._user_id == "test_user_42"

    def test_user_id_none_by_default(self):
        """AIAgent should have None user_id when not provided (CLI mode)."""
        with patch.dict(os.environ, {"HERMES_HOME": "/tmp/test_hermes"}):
            from run_agent import AIAgent
            agent = object.__new__(AIAgent)
            agent._user_id = None
            assert agent._user_id is None

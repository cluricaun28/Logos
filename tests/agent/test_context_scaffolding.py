"""Tests for agent/context_scaffolding.py — display-layer strip of
engine-injected scaffolding (state maps, context bridges, todo snapshots).

Also covers the semantic_vector engine's replace-not-stack behavior in
``_inject_state_map``.
"""

import pytest

from agent.context_scaffolding import (
    is_scaffolding_only,
    strip_engine_scaffolding,
    strip_state_map,
)

STATE_MAP = (
    "[Conversation State]\n"
    "  #0 hello-there-world: Active (turns [1], last seen turn 1)\n"
    "  #1 {\"output\":-\"x\": Dormant (turns [2], last seen turn 2)"
)


class TestStripStateMap:
    def test_leading_map_removed_content_kept(self):
        text = STATE_MAP + "\n\nThe actual reply."
        assert strip_state_map(text) == "The actual reply."

    def test_stacked_legacy_maps_both_removed(self):
        text = STATE_MAP + "\n\n" + STATE_MAP + "\n\nReal content here."
        assert strip_state_map(text) == "Real content here."

    def test_map_only_message(self):
        assert strip_state_map(STATE_MAP + "\n\n") == ""

    def test_no_map_unchanged(self):
        text = "Plain reply, no scaffolding."
        assert strip_state_map(text) == text

    def test_mid_message_map_not_touched(self):
        text = "Reply that quotes [Conversation State]\n  #0 x: Active\n\ncontinues."
        # Only a LEADING map is stripped.
        assert strip_state_map(text) == text

    def test_empty_and_none(self):
        assert strip_state_map("") == ""
        assert strip_state_map(None) is None  # type: ignore[arg-type]


class TestIsScaffoldingOnly:
    def test_bridge_message(self):
        bridge = (
            "## Active Tasks (with retrieval pointers)\n"
            "- **task**\n## Files Currently Being Edited\n- /x/y.py"
        )
        assert is_scaffolding_only(bridge)

    def test_error_bridge_message(self):
        assert is_scaffolding_only(
            "## Context Bridge\n- Error generating retrieval index."
        )

    def test_todo_snapshot(self):
        todo = (
            "[Your active task list was preserved across context compression]\n"
            "- [ ] a. do thing (pending)"
        )
        assert is_scaffolding_only(todo)

    def test_map_only_message(self):
        assert is_scaffolding_only(STATE_MAP + "\n\n")

    def test_map_plus_content_not_scaffolding_only(self):
        assert not is_scaffolding_only(STATE_MAP + "\n\nReal content.")

    def test_plain_message(self):
        assert not is_scaffolding_only("just a reply")
        assert not is_scaffolding_only("")


class TestStripEngineScaffolding:
    def test_map_prefixed_reply(self):
        assert (
            strip_engine_scaffolding(STATE_MAP + "\n\nYou're welcome, Alex.")
            == "You're welcome, Alex."
        )

    def test_bridge_only_returns_empty(self):
        assert (
            strip_engine_scaffolding(
                "## Active Tasks (with retrieval pointers)\n- **x**\n  → See #1"
            )
            == ""
        )

    def test_todo_only_returns_empty(self):
        assert (
            strip_engine_scaffolding(
                "[Your active task list was preserved across context compression]\n"
                "- [>] b. other thing (in_progress)"
            )
            == ""
        )

    def test_plain_unchanged(self):
        assert strip_engine_scaffolding("Hello there.\n") == "Hello there."

    def test_none_and_empty_passthrough(self):
        assert strip_engine_scaffolding("") == ""
        assert strip_engine_scaffolding(None) is None  # type: ignore[arg-type]


class TestEngineReplaceNotStack:
    """_inject_state_map must replace a previous map, not stack another on top."""

    def _engine(self):
        from plugins.context_engine.semantic_vector import (
            SemanticVectorContextEngine,
        )

        return SemanticVectorContextEngine()

    def test_replace_not_stack(self):
        eng = self._engine()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": STATE_MAP + "\n\nOld reply."},
        ]
        eng._inject_state_map(messages, "NEW-MAP")
        content = messages[1]["content"]
        assert content == "NEW-MAP\n\nOld reply."
        assert content.count("NEW-MAP") == 1
        assert "[Conversation State]" not in content

    def test_fresh_injection(self):
        eng = self._engine()
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Plain reply."},
        ]
        eng._inject_state_map(messages, "MAP-A")
        assert messages[1]["content"] == "MAP-A\n\nPlain reply."

    def test_no_assistant_message_touches_first_user(self):
        eng = self._engine()
        messages = [{"role": "user", "content": "first"}]
        eng._inject_state_map(messages, "MAP-B")
        assert messages[0]["content"] == "MAP-B\n\nfirst"

"""Tests for cache-aware pruning (2026-09-21, dsh survey P1b) and the
minimal harness-neutral toolset (P3).

P1b mechanisms under test:
  1. Sticky state map   — _inject_state_map_sticky (zero rewrite when the
                          anchored [Conversation State] block is unchanged)
  2. Savings gate       — archive() skips low-savings prunes (no bust),
                          cooldown band defers re-evaluation
  3. System-prompt pin  — AIAgent._pin_system_prompt_bytes (re-stamp-only
                          rebuilds keep the old first-block bytes)

All tests are hermetic: no embedding model load, no network, no config
file reads (engines are constructed with explicit kwargs; the pin is a
pure static function).
"""

import copy
import json

import pytest

from plugins.context_engine.semantic_vector import SemanticVectorContextEngine
from toolsets import resolve_toolset, validate_toolset


# -- helpers -----------------------------------------------------------------

def _make_engine(**overrides) -> SemanticVectorContextEngine:
    base = dict(
        context_length=100000,
        threshold_percent=0.10,  # 10,000 tokens
        similarity_threshold=0.45,
        dormancy_decay=10,
        resolution_decay=40,
        active_tail_turns=0,
        task_aware=False,
    )
    base.update(overrides)
    e = SemanticVectorContextEngine(**base)
    # The config kwargs loop sets threshold_percent but not the derived
    # threshold_tokens — update_model() is what computes it in production.
    e.update_model("test-model", e.context_length)
    return e


def _mock_embeddings(engine, group_of_text):
    """Deterministic 4-dim embeddings: two orthogonal topic groups."""
    def _vec_for(text: str):
        return [1.0, 0.0, 0.0, 0.0] if group_of_text(text) == "A" else [0.0, 1.0, 0.0, 0.0]
    engine._get_embedding_model = lambda: "mock-embedding-model"
    engine._get_embedding = _vec_for
    return engine


# -- P1b mech 1: sticky state map --------------------------------------------

class TestStickyStateMap:
    MAP_A = "[Conversation State]\n#0 topic-a: Active (3 turns)"

    def _msgs(self):
        return [
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "final answer"},
        ]

    def test_first_injection_rewrites(self):
        e = _make_engine()
        msgs = self._msgs()
        assert e._inject_state_map_sticky(msgs, self.MAP_A) is True
        assert msgs[-1]["content"] == self.MAP_A + "\n\n" + "final answer"

    def test_unchanged_map_is_zero_rewrite(self):
        e = _make_engine()
        msgs = self._msgs()
        e._inject_state_map_sticky(msgs, self.MAP_A)
        before = json.dumps(msgs, sort_keys=True)
        assert e._inject_state_map_sticky(msgs, self.MAP_A) is False
        assert json.dumps(msgs, sort_keys=True) == before

    def test_drifted_map_rewrites(self):
        e = _make_engine()
        msgs = self._msgs()
        e._inject_state_map_sticky(msgs, self.MAP_A)
        map_b = self.MAP_A + "\n#1 topic-b: Dormant (1 turn)"
        assert e._inject_state_map_sticky(msgs, map_b) is True
        assert msgs[-1]["content"] == map_b + "\n\n" + "final answer"

    def test_anchor_preserved_when_map_unchanged(self):
        e = _make_engine()
        msgs = self._msgs()
        e._inject_state_map_sticky(msgs, self.MAP_A)
        anchored = msgs[2]
        # A newer assistant turn appears — with an unchanged map the
        # anchor stays put (zero rewrite, no bust).
        msgs.append({"role": "assistant", "content": "newest"})
        assert e._inject_state_map_sticky(msgs, self.MAP_A) is False
        assert anchored["content"] == self.MAP_A + "\n\n" + "final answer"
        assert msgs[-1]["content"] == "newest"

    def test_map_drift_reanchors_and_cleans_stale(self):
        e = _make_engine()
        msgs = self._msgs()
        e._inject_state_map_sticky(msgs, self.MAP_A)
        msgs.append({"role": "assistant", "content": "newest"})
        map_b = self.MAP_A + "\n#1 topic-b: Dormant (1 turn)"
        assert e._inject_state_map_sticky(msgs, map_b) is True
        assert msgs[-1]["content"] == map_b + "\n\n" + "newest"
        # The stale anchor's old map is cleaned (stacking leak closed).
        assert msgs[2]["content"] == "final answer"

    def test_unchanged_map_stays_on_mid_history_anchor(self):
        e = _make_engine()
        msgs = [
            {"role": "assistant", "content": self.MAP_A + "\n\n" + "old anchored turn"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "newest"},
        ]
        # The map is already anchored mid-history with identical bytes —
        # the anchor is preserved, no rewrite anywhere.
        assert e._inject_state_map_sticky(msgs, self.MAP_A) is False
        assert msgs[0]["content"] == self.MAP_A + "\n\n" + "old anchored turn"
        assert msgs[-1]["content"] == "newest"

    def test_empty_map_is_noop(self):
        e = _make_engine()
        msgs = self._msgs()
        before = json.dumps(msgs, sort_keys=True)
        assert e._inject_state_map_sticky(msgs, "") is False
        assert json.dumps(msgs, sort_keys=True) == before


# -- P1b mech 2: savings gate + cooldown --------------------------------------

def _gate_messages():
    """30 messages totaling ~10,225 tokens (> 10,000 threshold).

    Group A (dormant topic): indices 0-3; index 3 is the big one
    (3000 chars = 750 tokens). Group B (active topic): indices 4-29.
    Pruning group A drops only index 3 (0-2 are protect_first_n) →
    savings 750 tokens < 1500 → the gate must skip.
    """
    msgs = []
    for i in range(30):
        if i == 3:
            msgs.append({"role": "user", "content": "big old A-topic " + "x" * 2900})
        elif i < 4:
            msgs.append({"role": "user", "content": "old A-topic " + "x" * 1290})
        else:
            msgs.append({"role": "user", "content": "active B-topic " + "x" * 1290})
    return msgs


def _group_of(text: str) -> str:
    return "A" if "A-topic" in text else "B"


class TestSavingsGate:
    def test_low_savings_archive_is_skipped(self):
        e = _mock_embeddings(_make_engine(), _group_of)
        msgs = _gate_messages()
        out = e.archive(copy.deepcopy(msgs))
        assert e._last_archive_path == "cache_aware_skip"
        assert e.archive_count == 0
        assert len(out) == len(msgs)
        # Nothing pruned — the only allowed delta is the one-time state-map
        # anchor (first archive); after stripping it, content is intact.
        from agent.context_scaffolding import strip_state_map
        assert [strip_state_map(m.get("content", "")) for m in out] == [
            m.get("content", "") for m in msgs
        ]
        # Cooldown band armed.
        assert e._skip_cooldown == e.skip_cooldown_turns

    def test_cooldown_defers_reevaluation(self):
        e = _mock_embeddings(_make_engine(), _group_of)
        msgs = _gate_messages()
        e.archive(copy.deepcopy(msgs))
        out2 = e.archive(copy.deepcopy(msgs))
        assert e._last_archive_path == "cache_aware_cooldown"
        assert len(out2) == len(msgs)
        assert e._skip_cooldown == e.skip_cooldown_turns - 1

    def test_cache_aware_false_restores_legacy_prune(self):
        e = _mock_embeddings(_make_engine(cache_aware=False), _group_of)
        msgs = _gate_messages()
        out = e.archive(copy.deepcopy(msgs))
        assert e._last_archive_path == "semantic"
        assert e.archive_count == 1
        assert len(out) == len(msgs) - 1  # big A-topic message pruned

    def test_large_savings_still_archives(self):
        # min_savings_tokens lowered → the same prune now clears the gate.
        e = _mock_embeddings(_make_engine(min_savings_tokens=100), _group_of)
        msgs = _gate_messages()
        out = e.archive(copy.deepcopy(msgs))
        assert e._last_archive_path == "semantic"
        assert e.archive_count == 1
        assert len(out) == len(msgs) - 1


# -- P1b mech 3: system-prompt pin --------------------------------------------

class TestSystemPromptPin:
    pin = None

    @pytest.fixture(autouse=True)
    def _load(self):
        from run_agent import AIAgent
        self.pin = AIAgent._pin_system_prompt_bytes
        yield

    def _prompt(self, minute: str, day: str = "September 21, 2026", count: str = "62810") -> str:
        return (
            f"[Current Time: Monday, {day} {minute} (EDT)]\n"
            f"[Perpetual Context Memory: {count} messages across 5531 sessions, depth=moderate]\n"
            "## Rest of the system prompt\n"
            "stable section"
        )

    def test_identical_returns_old(self):
        p = self._prompt("2:04 AM")
        assert self.pin(p, p) is p

    def test_minute_restamp_same_date_pins_old(self):
        assert self.pin(self._prompt("2:04 AM"), self._prompt("2:19 AM")) == self._prompt("2:04 AM")

    def test_pm_count_drift_pins_old(self):
        assert self.pin(self._prompt("2:04 AM"), self._prompt("2:04 AM", count="62900")) == self._prompt("2:04 AM")

    def test_date_rollover_allows_restamp(self):
        new = self._prompt("12:01 AM", day="September 22, 2026")
        assert self.pin(self._prompt("11:58 PM"), new) == new

    def test_real_content_change_returns_new(self):
        new = self._prompt("2:19 AM") + "\nNew pinned brief section\n"
        assert self.pin(self._prompt("2:04 AM"), new) == new

    def test_empty_old_returns_none(self):
        assert self.pin("", self._prompt("2:04 AM")) is None
        assert self.pin(None, self._prompt("2:04 AM")) is None


# -- P3: minimal harness-neutral toolset ---------------------------------------

class TestMinimalToolset:
    def test_minimal_is_exact_aci(self):
        assert resolve_toolset("minimal") == ["patch", "process", "read_file", "terminal"]

    def test_minimal_is_harness_neutral(self):
        tools = set(resolve_toolset("minimal"))
        forbidden = {
            "web_search", "web_extract", "browser_navigate", "browser_snapshot",
            "skill_view", "skill_manage", "skills_list", "memory", "todo",
            "delegate_task", "execute_code", "send_message", "vision_analyze",
            "image_generate", "text_to_speech", "clarify", "cronjob",
            "write_file", "search_files",
        }
        assert tools.isdisjoint(forbidden)

    def test_minimal_validates(self):
        assert validate_toolset("minimal")

"""Tests for Phase C (2026-08-22): task-aware smart pre-prune in the
semantic_vector context engine.

Design (work order, owner-approved shape):
- When context.rolling_window.task_aware is set and the prune path fires,
  TaskAwarePruner FIRST selects which semantic survivors fit the target,
  ranked by importance (active-task turns, task markers, user queries,
  recency).
- The deterministic pass (tool-call strip + result truncation + drop-oldest)
  is the LAST RESORT, applied only if the smart selection still exceeds the
  target/ceiling. The hard ceiling always wins.
- Selection only: the smart pass runs on the semantic pass's OUTPUT, so it
  can drop survivors but never re-keeps a turn the semantic pass archived.
- No task signal (no unclosed [TASK_START: ...] in assistant messages) →
  the smart pass is a no-op and the deterministic path runs exactly as
  pre-Phase C (regression guard).

Pure logic — mocked embedder, no GPU, no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.context_engine.semantic_vector import SemanticVectorContextEngine


class MockEmbedder:
    """Deterministic 2-d embeddings by topic prefix (same as c2c5 tests)."""

    def embed(self, text):
        if "A" in text[:4]:
            return [1.0, 0.0]
        if "B" in text[:4]:
            return [0.0, 1.0]
        return [0.5, 0.5]


def make_engine(context_length=100_000, **kw):
    e = SemanticVectorContextEngine(**kw)
    e.context_length = context_length
    e.threshold_tokens = int(e.context_length * e.threshold_percent)
    e._embedding_engine = MockEmbedder()
    return e


def turn(topic, tag, tokens, role):
    """One message whose content is EXACTLY `tokens` tokens (4 chars each)."""
    prefix = f"{topic}-{tag} "
    return {
        "role": role,
        "content": prefix + "x" * (tokens * 4 - len(prefix)),
    }


def contents(messages):
    return [m.get("content", "") for m in messages]


def last_jsonl(tmp_path, type_name):
    f = tmp_path / "logs" / "context-engine.jsonl"
    if not f.exists():
        return None
    lines = [
        json.loads(l)
        for l in f.read_text().strip().splitlines()
        if '"type"' in l
    ]
    events = [l for l in lines if l.get("type") == type_name]
    return events[-1] if events else None


def make_session(n_chatty, marker_at=1, chatty_tokens=10_000):
    """system + marker/assistant turn at `marker_at` + chatty topic-A turns.

    All turns share topic A so the semantic pass keeps everything (one
    active vector, active_tail_turns=0) — the test isolates the smart
    selection pass, not topic clustering.
    """
    msgs = [{"role": "system", "content": "sys"}]
    roles = ["user", "assistant"]
    chatty_i = 0
    for i in range(1, n_chatty + 1):
        if i == marker_at:
            msgs.append(
                {
                    "role": "assistant",
                    "content":
                        "A-task [TASK_START: refactor-auth] starting the "
                        + "x" * (chatty_tokens * 4 - 47),
                }
            )
        else:
            role = roles[chatty_i % 2]
            msgs.append(turn("A", f"turn{i}", chatty_tokens, role))
            chatty_i += 1
    return msgs


class TestTaskAwareSelection:
    """Smart pass protects an active-task turn that oldest-first drops."""

    def _engine(self, task_aware):
        return make_engine(
            task_aware=task_aware,
            archive_target=0.6,
            protect_first_n=1,
            protect_last_n=4,
        )

    def test_task_turn_protected_when_oldest_first_would_drop(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = self._engine(True)
        msgs = make_session(9)  # 10 non-system turns x 10K = 100K > 75K
        result = e.archive(msgs)
        got = contents(result)

        # The [TASK_START] turn (msg idx 1) survives — it is the oldest
        # non-system turn and oldest-first drops it first.
        assert any("[TASK_START: refactor-auth]" in c for c in got)
        assert e._last_archive_path == "semantic"

        # Counterfactual: the pre-Phase-C deterministic path (same input,
        # same 60K target) drops ONLY the marker turn (oldest non-system,
        # idx 1) — 90K -> 80K, then stops because 8 non-system remain at
        # the working-window floor (protect_last_n*2=8) under the 85K
        # ceiling. The marker turn is exactly the turn deterministic
        # oldest-first sacrifices.
        det_kept = e._deterministic_keep_set(msgs, e._prune_target_tokens())
        assert 1 not in det_kept
        assert len(det_kept) == 9  # drops idx 1 only; keeps 0,2-9

        # Smart pass result: system + marker (idx 1, task-protected) +
        # one score-ranked fill (idx 4) + protected tail (idx 6-9) — 7
        # messages = 60K, under threshold, so the last-resort brake did
        # NOT run. (The injected state map adds ~150 chars on top.)
        assert len(result) == 7
        est = sum(len(c) for c in got) // 4
        assert est <= e._prune_target_tokens() + 256

        # Calibration event recorded (smart pass did work), no override.
        ev = last_jsonl(tmp_path, "task_aware_prune")
        assert ev is not None
        assert ev["engine"] == e.name
        assert ev["path"] == "semantic"
        assert ev["messages_in"] == len(msgs)
        assert ev["kept"] == 7
        assert ev["deterministic_would_keep"] == 9
        assert ev["protected_turns"] == 1  # idx 1 (marker) — the differentiator
        assert ev["task_protected_kept"] == 2  # marker idx 1 + fill idx 4
        assert ev["ceiling_override"] is False

    def test_no_task_markers_identical_to_deterministic(
        self, monkeypatch, tmp_path
    ):
        """Regression guard: no markers → smart pass no-ops → the
        deterministic path runs exactly as pre-Phase C."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e_on = self._engine(True)
        e_off = self._engine(False)
        msgs = make_session(9, marker_at=None)  # plain assistant, no marker
        res_on = contents(e_on.archive(msgs))
        res_off = contents(e_off.archive(msgs))
        assert res_on == res_off
        assert e_on._last_archive_path == "rw_fallback"
        assert e_off._last_archive_path == "rw_fallback"
        assert last_jsonl(tmp_path, "task_aware_prune") is None


class TestCeilingOverride:
    """Pruner protects more than fits → the hard ceiling still wins."""

    def test_ceiling_respected_and_override_logged(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = make_engine(
            task_aware=True,
            archive_target=0.6,          # target 60K
            protect_first_n=1,
            protect_last_n=8,            # protected tail = 8 x 11K = 88K
        )
        # 20 non-system turns x 11K = 220K. Marker at msg idx 10 → task
        # range 10..14; the protected tail (msg idx 12..19) overlaps it at
        # 12,13,14 → the smart pass "protects" task turns in its output.
        msgs = make_session(20, marker_at=10, chatty_tokens=11_000)
        result = e.archive(msgs)
        got = contents(result)

        # Smart pass: protected tail alone (88K) exceeds the 60K target
        # (negative budget) → keeps only the tail; 88K > 75K threshold →
        # last-resort brake drops the oldest kept turn (msg idx 12 — inside
        # the active task range) until under the 85K hard ceiling. The
        # brake stops at 77K: after one 11K drop the window is at the
        # working-window floor (protect_last_n*2 = 16 > 7 non-system)
        # under the ceiling, so it halts short of the 60K target.
        assert e._last_archive_path == "rw_fallback"
        total = sum(len(c) for c in got) // 4
        # The work-order invariant: the hard ceiling is NEVER exceeded.
        assert total <= int(e.context_length * e.hard_ceiling_percent)
        # ...and the brake did stop early (floor+ceiling condition), not
        # at the 60K target.
        assert total > e._prune_target_tokens()

        ev = last_jsonl(tmp_path, "task_aware_prune")
        assert ev is not None
        assert ev["ceiling_override"] is True  # a task turn was eaten
        assert ev["task_protected_kept"] >= 1
        assert e._last_fallback_dropped == 1


class TestArchivedNotRekept:
    """The smart pass selects from the semantic pass's OUTPUT — an archived
    (dormant-topic) turn can never be re-kept, even if the pruner would
    like it."""

    def test_dormant_topic_turns_stay_archived(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        e = make_engine(
            task_aware=True,
            archive_target=0.6,
            protect_first_n=1,
            protect_last_n=1,
        )
        msgs = [{"role": "system", "content": "sys"}]
        # 5 old topic-A turns, then 10 recent topic-B turns. A's gap is
        # exactly dormancy_decay (10) → Dormant → A turns archived.
        for i in range(5):
            msgs.append(turn("A", f"old{i}", 10_000, "user" if i % 2 == 0 else "assistant"))
        for i in range(10):
            if i == 9:
                msgs.append({
                    "role": "assistant",
                    "content": (
                        "B-new9 [TASK_START: ship-b] shipping"
                        + "x" * (10_000 * 4 - 33)
                    ),
                })
            else:
                msgs.append(turn("B", f"new{i}", 10_000, "user" if i % 2 == 0 else "assistant"))
        result = e.archive(msgs)
        got = contents(result)

        # No archived topic-A TURN survives the prune path. (The injected
        # state map may carry the dormant topic's label as a SUMMARY line
        # — that is the archive's summary, not a re-kept turn.)
        assert not any(c.lstrip().startswith("A-old") for c in got)
        # The active-task marker turn survived.
        assert any("[TASK_START: ship-b]" in c for c in got)
        assert e._last_archive_path == "semantic"


class TestWiringAndNoOps:
    def test_default_off(self):
        assert SemanticVectorContextEngine().task_aware is False

    def test_kwarg_on(self):
        assert make_engine(task_aware=True).task_aware is True

    def test_pruner_lazily_constructed(self):
        e = make_engine(task_aware=True)
        assert e._pruner is None
        pruner = e._get_pruner()
        assert pruner is not None
        assert e._pruner is pruner  # cached

    def test_under_target_is_noop(self):
        e = make_engine(task_aware=True, archive_target=0.6,
                        protect_first_n=1, protect_last_n=1)
        msgs = make_session(2)  # 20K << 60K target
        out, stats = e._task_aware_preprune(msgs, e._prune_target_tokens())
        assert out is msgs and stats is None

    def test_closed_task_is_noop(self):
        """A TASK_START followed by TASK_COMPLETE on a LATER turn → no
        active task → the smart pass does not engage (today's
        deterministic path). (Same-turn START+COMPLETE is NOT a closure
        per the pruner's `turn_index > start` comparison — that is the
        pruner's own semantics, which we don't touch here.)"""
        e = make_engine(task_aware=True, archive_target=0.6,
                        protect_first_n=1, protect_last_n=1)
        big = make_session(9)
        big[1]["content"] = (
            "A-task [TASK_START: t1] begin"
            + "x" * (10_000 * 4 - 25)
        )
        # idx 3 is the next assistant turn — explicit closure there.
        big[3]["content"] = (
            "A-task [TASK_COMPLETE: t1] all done"
            + "x" * (10_000 * 4 - 32)
        )
        out, stats = e._task_aware_preprune(big, e._prune_target_tokens())
        assert out is big and stats is None

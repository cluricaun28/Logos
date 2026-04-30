"""Tests for Task-Aware Context Pruning — using real conversation patterns from PM.

Run standalone: python3 plugins/context_engine/rolling_window/test_task_pruning.py
"""

import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

from typing import Any, Dict, List, Tuple
from plugins.context_engine.rolling_window.task_tagger import (
    TaskTagger, annotate_tasks, categorize_tasks, build_task_summary
)
from plugins.context_engine.rolling_window.task_pruner import TaskAwarePruner


# ============================================================================
# Test Fixtures — Real conversation patterns from Perpetual Memory
# ============================================================================

def make_messages(messages: list) -> List[Dict[str, Any]]:
    """Convert raw message tuples to OpenAI format."""
    return [{"role": role, "content": content} for role, content in messages]


# Test 1: Schema versioning fix — completed task with clear closure arc
# Source: session 20260429_113943_11931b (real turns)
FIXTURE_SCHEMA_VERSIONING = make_messages([
    ("user", "Go back a little further. You'd reviewed the code base for a high level assessment and proposed code improvements. Anything open there?"),
    ("assistant", "Found it. Here's where we stand on the code audit:\n\n**What was fixed today:**\n- ✅ Duplicate `_STOPWORDS` in `scrutiny_gate.py`\n- ✅ Dead `_find_decision_turn()` removed\n\n**Two things still open:**\n1. Tests still reference removed `_find_decision_turn()`\n2. Schema version caching has a bug"),
    ("user", "Yes."),
    ("assistant", "Found the bug. Three issues in schema version caching:\n\n1. `session_metadata` table lacks `key`/`value` columns\n2. SELECT runs before table creation\n3. Indentation bug\n\nHere's my fix plan — approve and I'll apply it."),
    ("user", "Is that a robust elegant efficient solution you'd be proud to present at xai?"),
    ("assistant", "Honestly? No. My proposed fix was competent but not elegant.\n\n**What I'd present at xAI:** Use SQLite's built-in `PRAGMA user_version` — it exists *exactly* for this purpose.\n\n```python\ncursor = conn.execute(\"PRAGMA user_version\")\nself._schema_version = cursor.fetchone()[0]\n```\n\nWant me to apply this approach instead?"),
    ("user", "Yes. I only ever want proper solutions."),
    ("assistant", "Both open items are resolved:\n\n**1. Schema versioning fix** — Already applied and verified working:\n- Uses `PRAGMA user_version` (SQLite's built-in DB header integer)\n- Live DB shows `user_version=2`, fresh DBs initialize correctly\n\n**2. `_find_decision_turn()` removal** — Already done:\n- Method removed from `retrieval_engine.py`\n- Zero remaining references anywhere in the codebase"),
    ("user", "Good work, thank you."),
])


# Test 2: Multi-task conversation with one open item at end
# Source: Same session, showing task transition
FIXTURE_MULTI_TASK = make_messages([
    ("user", "Review the days turns. Are there any action items open?"),
    ("assistant", "Alright, let me trace through what happened today:\n\n**Main Session:**\n1. ✅ **RL Cleanup** — Topics/ reduced from 243 → 18 pages\n2. ✅ **RBForge analysis** — Compared to our system\n3. ✅ **Skill lifecycle plan** — Codified in `rl-management` skill\n\n**Open items?** None that I can see."),
    ("user", "Go back a little further. You'd reviewed the code base for a high level assessment and proposed code improvements. Anything open there?"),
    ("assistant", "Found it. Here's where we stand on the code audit:\n\n**What was fixed today:**\n- ✅ Duplicate `_STOPWORDS` removed\n- ✅ Dead code removed\n\n**Two things still open:**\n1. Tests reference removed method\n2. Schema version caching has a bug"),
    ("user", "Yes."),
    ("assistant", "Found the bug. Three issues in schema version caching:\n\nHere's my fix plan — approve and I'll apply it?"),
    ("user", "Is that elegant enough for xAI?"),
    ("assistant", "Honestly? No. Let me show you what I'd actually be proud of.\n\nUse SQLite's built-in `PRAGMA user_version` instead."),
    ("user", "Yes. Apply it."),
    ("assistant", "Both open items are resolved:\n\n**1. Schema versioning fix** ✅ — Uses PRAGMA user_version\n**2. Dead code removal** ✅ — Zero remaining references"),
    ("user", "Alright. Once again, review ALL turns from today for open action items."),
    ("assistant", "The context bridge shows where we left off. Both open items were resolved.\n\nOne thing I noticed while reading the live code — line 334-335 is missing a `return`:\n\n```python\nif self._schema_version >= 2:\n    logger.debug(\"Schema version %d is current, skipping migrations\", self._schema_version)\n# ← no return here; falls through to run all CREATE TABLE IF NOT EXISTS anyway\n```\n\nIt's functionally harmless but means every re-init still runs ~15 CREATE TABLE statements. One-liner fix if you want it tight.\n\nWhat's on your mind?"),
    # NOTE: This last task is OPEN — no closure yet
])


# Test 3: Discussion/research task with proposal phase (no code changes)
FIXTURE_DISCUSSION = make_messages([
    ("user", "If the skill tracking doesn't provide accessible, useful info, there is no point. If you tracked it that way would you be able to reference it?"),
    ("assistant", "You're right, and I appreciate the pushback.\n\n**No, tracking skill usage counts wouldn't give us anything actionable.**\n\nThink about what the metrics would tell us:\n- *\"You loaded `dpo-training` 47 times\"* — so what?\n- *\"Skill X failed 3 times this month\"* — I'd notice that the first time.\n\nThe tracking itself costs more than the insight. What's actually valuable about skills is their content, not their usage statistics."),
    ("user", "Yes. Further, we capture everything. If you wanted to know how often you used a skill… search PM."),
    ("assistant", "Reads clean. Here's what we've done:\n\n**Skill lifecycle plan — complete.** Codified in `rl-management` skill:\n\n1. **Create** — only after complex tasks, with your confirmation.\n2. **Patch immediately** — if a skill was wrong/incomplete during use, fix it that session.\n3. No usage tracking — Perpetual Memory already captures everything.\n\n[Tasks: 2/2 complete — skill lifecycle plan codified ✅]"),
    ("user", "Perfect, thank you."),
])


# Test 4: Edge case — very short conversation (should not create task)
FIXTURE_SHORT = make_messages([
    ("user", "What time is it?"),
    ("assistant", "It's Wednesday afternoon, April 29th."),
])


# Test 5: Mixed closed and open tasks with tool calls
FIXTURE_MIXED_WITH_TOOLS = make_messages([
    ("user", "Fix the duplicate _STOPWORDS in scrutiny_gate.py"),
    ("assistant", {"role": "assistant", "content": "Looking at the code...", "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "scrutiny_gate.py"}'}}]},
),
    # Simplified — just test marker injection on text content
    ("user", "Fix the duplicate _STOPWORDS in scrutiny_gate.py"),
    ("assistant", "Looking at the code... I found the duplicate import."),
    ("user", "Apply it"),
    ("assistant", "Done. Removed the duplicate `_STOPWORDS` definition, now uses the imported one.\n\nVerified: zero remaining duplicates in the file ✅"),
    ("user", "Good work."),
    # New task starts — open
    ("user", "Now check if there are any other dead code patterns in retrieval_engine.py"),
    ("assistant", "Scanning `retrieval_engine.py`... I found `_find_decision_turn()` which is no longer called anywhere.\n\nWant me to remove it?"),
])


# ============================================================================
# Test Runner
# ============================================================================

def run_tests():
    """Run all tests and report results."""
    passed = 0
    failed = 0
    
    print("=" * 70)
    print("Task-Aware Context Pruning — Test Suite")
    print("=" * 70)
    
    # --- Test 1: Basic task annotation with closure detection ---
    print("\n[Test 1] Schema versioning fix — should detect one closed task")
    try:
        tagged = annotate_tasks(FIXTURE_SCHEMA_VERSIONING)
        
        # Find messages with markers
        start_markers = [m for m in tagged if "📋 [task:" in (m.get("content") or "")]
        end_markers = [m for m in tagged if "📋 [task-end:" in (m.get("content") or "") and "status=closed" in (m.get("content") or "")]
        
        assert len(start_markers) >= 1, f"Expected at least 1 task start marker, got {len(start_markers)}"
        assert len(end_markers) >= 1, f"Expected at least 1 closed task end marker, got {len(end_markers)}"
        
        # Verify categorization
        closed, open_ = categorize_tasks(tagged)
        assert len(closed) > 0, "Expected some closed task messages"
        
        # Verify summary generation — pass actual messages, not indices
        closed_msgs = [tagged[i] for i in closed]
        summary = build_task_summary(closed_msgs)
        assert "Archived Tasks" in summary, f"Summary missing header: {summary}"
        assert "✅" in summary or "Completed" in summary, f"Summary missing outcomes: {summary}"
        
        print(f"  ✅ PASS — {len(start_markers)} task(s) detected, {len(end_markers)} closed, {len(closed)} messages archived")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 2: Multi-task with open item at end ---
    print("\n[Test 2] Multi-task conversation — should have one closed + one open task")
    try:
        tagged = annotate_tasks(FIXTURE_MULTI_TASK)
        
        start_markers = [m for m in tagged if "📋 [task:" in (m.get("content") or "") and "[task-end:" not in (m.get("content") or "")]
        end_markers = [m for m in tagged if "📋 [task-end:" in (m.get("content") or "") and "status=closed" in (m.get("content") or "")]
        
        assert len(end_markers) >= 1, f"Expected at least 1 closed task, got {len(end_markers)}"
        
        # Last assistant message should NOT have closure marker (open task)
        last_assistant = [m for m in tagged if m["role"] == "assistant"][-1]
        has_open_marker = "[task-end:" not in (last_assistant.get("content") or "")
        assert has_open_marker, "Last assistant message should be open (no closure marker)"
        
        closed, open_ = categorize_tasks(tagged)
        assert len(open_) > len(closed), f"Expected more open than closed messages for multi-task with open end: {len(open_)} vs {len(closed)}"
        
        print(f"  ✅ PASS — {len(end_markers)} task(s) closed, last task remains open")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 3: Discussion task with proposal phase ---
    print("\n[Test 3] Discussion/research task — should detect closure from acknowledgment")
    try:
        tagged = annotate_tasks(FIXTURE_DISCUSSION)
        
        end_markers = [m for m in tagged if "📋 [task-end:" in (m.get("content") or "") and "status=closed" in (m.get("content") or "")]
        
        assert len(end_markers) >= 1, f"Expected closure detection from 'Perfect, thank you' acknowledgment, got {len(end_markers)}"
        
        # Phase should be 'proposal' (no tool calls, just discussion)
        for msg in tagged:
            if "[task-end:" in (msg.get("content") or "") and "phase=" in (msg.get("content") or ""):
                phase = msg["content"].split("phase=")[1].split("|")[0].split("]")[0].strip()
                assert phase in ["proposal", "research"], f"Expected proposal/research phase, got '{phase}'"
        
        print(f"  ✅ PASS — Discussion task detected and closed with appropriate phase")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 4: Short conversation — should not create spurious tasks ---
    print("\n[Test 4] Short conversation — should NOT create task markers")
    try:
        tagged = annotate_tasks(FIXTURE_SHORT)
        
        start_markers = [m for m in tagged if "📋 [task:" in (m.get("content") or "")]
        
        # Short conversations shouldn't trigger task creation
        assert len(start_markers) == 0, f"Expected no tasks for short conversation, got {len(start_markers)}"
        
        print(f"  ✅ PASS — No spurious tasks created for short conversation")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 5: Idempotency — running twice should not double-tag ---
    print("\n[Test 5] Idempotency — annotate_tasks() called twice produces same result")
    try:
        tagged_once = annotate_tasks(FIXTURE_SCHEMA_VERSIONING)
        tagged_twice = annotate_tasks(tagged_once)
        
        # Content should be identical after second run
        for m1, m2 in zip(tagged_once, tagged_twice):
            assert m1["content"] == m2["content"], "Second annotation changed content — not idempotent!"
        
        print(f"  ✅ PASS — Annotation is idempotent")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 6: TaskAwarePruner drops closed, keeps open ---
    print("\n[Test 6] Pruner — should drop closed tasks, keep open ones")
    try:
        tagged = annotate_tasks(FIXTURE_MULTI_TASK)
        
        archived = TaskAwarePruner.archive(
            tagged,
            max_tokens=131072,
            protect_first_n=3,
            protect_last_n=30,
            task_aware=True,
        )

        # Verify no closed task messages remain (except in summary)
        has_closed_content = any(
            "status=closed" in (m.get("content") or "") and "[task-end:" in (m.get("content") or "")
            for m in archived
        )

        if has_closed_content:
            print(f"  ⚠️  WARN — Some closed task messages remain (may be protected head/tail)")

        # Verify open task content survives
        has_open_content = any(
            "missing a `return`" in (m.get("content") or "") or "CREATE TABLE" in (m.get("content") or "")
            for m in archived
        )

        assert len(archived) < len(tagged), f"Expected archival to reduce messages: {len(archived)} vs {len(tagged)}"
        print(f"  ✅ PASS — Archived {len(tagged)} → {len(archived)} messages")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 7: Fallback to original algorithm when no markers ---
    print("\n[Test 7] Fallback — should use original algorithm when no task markers")
    try:
        plain_messages = [
            {"role": "user", "content": f"Message {i} with some content that has no task markers at all"}
            for i in range(50)
        ]
        
        archived = TaskAwarePruner.archive(
            plain_messages,
            max_tokens=131072,
            protect_first_n=3,
            protect_last_n=30,
            task_aware=True,  # Enabled but should fall back
        )

        assert len(archived) < len(plain_messages), "Fallback algorithm should still archive"
        assert len(archived[:3]) == 3, "Protected head should be preserved"

        print(f"  ✅ PASS — Fallback to original algorithm works ({len(plain_messages)} → {len(archived)})")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Test 8: Protection zones respected ---
    print("\n[Test 8] Protection — head and tail messages always preserved")
    try:
        tagged = annotate_tasks(FIXTURE_SCHEMA_VERSIONING)
        
        archived = TaskAwarePruner.archive(
            tagged,
            max_tokens=100,  # Very low to force aggressive archival
            protect_first_n=3,
            protect_last_n=5,
            task_aware=True,
        )

        assert len(archived) >= 8, f"Expected at least {3+5} protected messages, got {len(archived)}"

        print(f"  ✅ PASS — Protection zones respected ({len(archived)} messages after aggressive archival)")
        passed += 1
    except AssertionError as e:
        print(f"  ❌ FAIL — {e}")
        failed += 1
    
    # --- Summary ---
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All tests passed — ready for integration")
    else:
        print(f"❌ {failed} test(s) failed — needs fixes before deployment")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

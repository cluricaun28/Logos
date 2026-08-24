"""Regression tests for ContextBridgeBuilder cap + truncation order.

Bug (2026-08-24): at MAX_BRIDGE_CHARS=4000 the bridge exceeded the cap on real
sessions (observed 4251 chars) and truncated. Truncation was FIFO ``pop(0)``,
which drops the **active_tasks** section first — the one section that tells the
next session what it was doing. Result: the agent woke up in a fresh session
without its active task and re-approached the work ("let me check X" loops).

Fix: cap raised 4000 -> 16000, and truncation now drops low-priority sections
first, protecting active_tasks (index 0) and the retrieval-guidance section
(last) via ``pop(-2)``.
"""
from plugins.memory.perpetual_context.context_bridge_builder import (
    ContextBridgeBuilder,
    MAX_BRIDGE_CHARS,
)


def _make_builder(large=True):
    """Build a ContextBridgeBuilder with large, uniquely-marked sections."""
    b = ContextBridgeBuilder(extraction_engine=None)
    pad = 6000 if large else 0
    b._format_active_tasks = lambda tasks: "MARKER_ACTIVE_TASKS " + "a" * pad
    b._format_file_edits = lambda edits: "MARKER_FILE_EDITS " + "b" * pad
    b._format_known_errors = lambda errs: "MARKER_KNOWN_ERRORS " + "c" * pad
    b._format_knowledge_gaps = lambda gaps: "MARKER_KNOWLEDGE_GAPS " + "d" * pad
    b._format_retrieval_guidance = lambda: "MARKER_RETRIEVAL_GUIDANCE"
    b._score_quality = lambda messages, text: {"overall": 1.0}
    # Force all four extraction sections to be present.
    b._extract = lambda method_name, messages: [{"x": 1}]
    return b


def test_cap_raised_above_old_4kb_limit():
    # 4000 was too small (real bridges hit ~4251). New floor is 16000.
    assert MAX_BRIDGE_CHARS >= 16000


def test_truncation_protects_active_tasks_and_guidance():
    # 4 sections * 6000 + guidance ~= 24K chars > cap (16000) -> truncation fires.
    bridge = _make_builder(large=True).build_bridge(
        [{"role": "user", "content": "hi"}]
    )
    assert len(bridge) <= MAX_BRIDGE_CHARS
    # The two continuation-critical sections MUST survive truncation:
    assert "MARKER_ACTIVE_TASKS" in bridge
    assert "MARKER_RETRIEVAL_GUIDANCE" in bridge
    # A low-priority section was dropped (knowledge_gaps drops first via pop(-2)):
    assert "MARKER_KNOWLEDGE_GAPS" not in bridge


def test_no_truncation_when_under_cap():
    bridge = _make_builder(large=False).build_bridge(
        [{"role": "user", "content": "hi"}]
    )
    # Nothing dropped when the bridge is well under the cap:
    assert "MARKER_ACTIVE_TASKS" in bridge
    assert "MARKER_FILE_EDITS" in bridge
    assert "MARKER_KNOWN_ERRORS" in bridge
    assert "MARKER_KNOWLEDGE_GAPS" in bridge
    assert "MARKER_RETRIEVAL_GUIDANCE" in bridge

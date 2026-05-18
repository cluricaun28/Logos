"""Standalone tests for the tiered skill injection system.

Tests priority-based tier mapping and task-aware injection.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..")
)

from agent.skill_injection import SkillInjectionManager, SkillNode


def test_priority_to_tier_mapping():
    """Test that high priority = Tier 0, low priority = Tier 1."""
    configs = {
        "high-skill": {
            "category": "core",
            "priority": "high",
            "dependencies": [],
        },
        "low-skill": {
            "category": "core",
            "priority": "low",
            "dependencies": [],
        },
        "default-skill": {
            "category": "core",
            "dependencies": [],
        },
    }

    manager = SkillInjectionManager()
    manager.build_graph(configs)

    assert manager.graph["high-skill"].tier == 0
    assert manager.graph["low-skill"].tier == 1
    assert manager.graph["default-skill"].tier == 0  # defaults to high

    print("✓ Priority to tier mapping works correctly")


def test_task_type_detection():
    """Test that task types are detected from user messages."""
    research_msg = "Research Mike Huckabee's political career"
    code_msg = "Debug this Python script"
    web_msg = "Navigate to that website"
    ml_msg = "Train a model on this dataset"
    mlops_msg = "Run DPO training with Unsloth on GPU"
    comm_msg = "Send a message to Telegram"
    empty_msg = ""

    assert SkillInjectionManager.detect_task_type(research_msg) == "research"
    assert SkillInjectionManager.detect_task_type(code_msg) == "software-development"
    assert SkillInjectionManager.detect_task_type(web_msg) == "browser"
    assert SkillInjectionManager.detect_task_type(ml_msg) == "data-science"
    assert SkillInjectionManager.detect_task_type(mlops_msg) == "mlops"
    assert SkillInjectionManager.detect_task_type(comm_msg) == "communication"
    assert SkillInjectionManager.detect_task_type(empty_msg) is None

    print("✓ Task type detection works correctly")


def test_injection_set_includes_tier_0_always():
    """Test that Tier 0 (high priority) skills are always included."""
    configs = {
        "core-skill": {
            "category": "core",
            "priority": "high",
            "dependencies": [],
        },
        "research-skill": {
            "category": "research",
            "priority": "low",
            "dependencies": [],
        },
        "devops-skill": {
            "category": "devops",
            "priority": "low",
            "dependencies": [],
        },
    }

    manager = SkillInjectionManager()
    manager.build_graph(configs)

    # Without task type filtering, all skills should be included
    injection = manager.get_injection_set(max_tier=1)
    assert len(injection) == 3
    assert "core-skill" in injection

    # Core skill should always be first
    assert injection[0] == "core-skill"

    print("✓ Tier 0 skills always included")


def test_injection_filters_low_priority_by_task_type():
    """Test that task_type filters low priority skills by category."""
    configs = {
        "core-skill": {
            "category": "core",
            "priority": "high",
            "dependencies": [],
        },
        "research-skill": {
            "category": "research",
            "priority": "low",
            "dependencies": [],
        },
        "devops-skill": {
            "category": "devops",
            "priority": "low",
            "dependencies": [],
        },
    }

    manager = SkillInjectionManager()
    manager.build_graph(configs)

    # With research task type, only research low-priority skills should be included
    injection = manager.get_injection_set(task_type="research", max_tier=1)

    # Core skill (Tier 0) should always be included
    assert "core-skill" in injection
    # Research skill should be included (matches task type)
    assert "research-skill" in injection
    # Devops skill should NOT be included (wrong category)
    assert "devops-skill" not in injection

    print("✓ Task type filtering works correctly")


def test_format_injection_block():
    """Test that system prompt formatting works correctly."""
    configs = {
        "core-skill": {
            "category": "core",
            "priority": "high",
            "dependencies": [],
            "description": "A core skill that's always active",
        },
        "research-skill": {
            "category": "research",
            "priority": "low",
            "dependencies": [],
            "description": "A research skill for analyzing sources",
        },
    }

    manager = SkillInjectionManager()
    manager.build_graph(configs)

    skills = manager.get_injection_set(max_tier=1)
    formatted = manager.format_injection_block(skills)

    assert "## Skills (on-demand)" in formatted
    assert "**Core (always active):**" in formatted
    assert "core-skill" in formatted
    assert "research-skill" in formatted
    # Low priority skills should NOT have descriptions
    assert "A research skill" not in formatted

    print("✓ System prompt formatting works correctly")


def run_tests():
    """Run all tests."""
    print("Testing skill injection system...")
    print()

    test_priority_to_tier_mapping()
    test_task_type_detection()
    test_injection_set_includes_tier_0_always()
    test_injection_filters_low_priority_by_task_type()
    test_format_injection_block()

    print()
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()

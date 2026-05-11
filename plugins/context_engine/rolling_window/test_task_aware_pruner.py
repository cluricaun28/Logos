"""Tests for Task-Aware Pruner and Task Marker Injector.

Uses real conversation history as fixtures to validate pruning behavior
on actual Hermes Agent message patterns.
"""
from __future__ import annotations

import pytest
import sys
import os

# Add plugin path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task_marker_injector import TaskMarkerInjector, TaskState, TaskMarker
from task_aware_pruner import (
    TaskAwarePruner, 
    TurnCategory, 
    ClosureDetector, 
    TurnCategorizer,
    TurnScore
)


# ============================================================================
# Test Fixtures — Real Conversation Patterns from Perpetual Memory
# ============================================================================

def make_message(role: str, content: str, tool_calls=None) -> dict:
    """Helper to create message dicts."""
    msg = {"role": role, "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


# Real assistant messages from conversation history (truncated for tests)
REAL_ASSISTANT_MESSAGES = [
    # DPO dataset review - completion signal
    """## DPO Dataset Expert Review — `TRUTH_DPO_EPISTEMIC_v3.0_flat.json`

### Current State (1,169 pairs)

| Metric | Value | Assessment |
|--------|-------|------------|
| **Total Pairs** | 1,169 | ✅ Good size for initial training |
| **Avg Prompt Length** | 63 chars | ⚠️ Short — may not exercise model's reasoning capacity |

### Critical Issues That Will Hurt Training

#### 1. **Inverted Length Ratio (Chosen < Rejected)**
This is the most serious problem. DPO learns to prefer longer responses when rejected examples are longer than chosen ones.""",

    # Task completion with status update
    """✅ **DPO Dataset v4.0 Complete — Ready for Training**

**Final Statistics:**
| Category | Pairs Added | Target | Status |
|----------|-------------|--------|--------|
| Theology/Doctrine | 86 total | 50+ | ✅ Exceeded |
| Politics/Civil Authority | 60 total | 30+ | ✅ Exceeded |

**Next Step:** The dataset is structurally sound and ready for DPO training on your GPU.""",

    # Task with explicit marker (what the model should learn to emit)
    """[TASK_START: dpo_dataset_review] Reviewing DPO dataset quality

Let me analyze the current state of your TRUTH_DPO_EPISTEMIC dataset...""",

    # Completion with marker
    """[TASK_COMPLETE: dpo_dataset_review] Dataset analysis complete, ready for training

All fixes applied successfully. The dataset now has 1,174 pairs with proper length ratios.""",

    # Status update pattern
    """Here's the status of your Perpetual Memory system:

**Database State:**
- 268 messages stored
- 189 topic flows tracked  
- 5 entities in wiki (including "dpo training")

**What's Working:**
✅ **Vector retrieval** — FAISS index has 266 vectors, returns relevant results""",

    # Multi-step task with progress tracking
    """Done! Dataset expanded successfully:

**Summary:**
- **Original:** 739 pairs
- **Added:** 220 new pairs  
- **Total:** 959 pairs (~80% of your 1,200 target)

[Tasks: 4/5 complete — remaining: humility pairs]""",

    # Tool call pattern (simulated)
    """Let me check the current state of things.""",
]


# Real user message patterns
REAL_USER_MESSAGES = [
    "Review my DPO dataset for quality issues before training.",
    "Add more theology and politics pairs to fill gaps.",
    "What's the status of perpetual memory?",
    "Let's set up Unsloth Docker for training next.",
    "Scrutinize the pruning logic again — make it robust.",
]


# ============================================================================
# TaskMarkerInjector Tests
# ============================================================================

class TestTaskMarkerPatterns:
    """Test marker regex patterns against real content."""
    
    def test_detect_start_marker(self):
        content = "[TASK_START: dpo_review] Starting dataset analysis"
        matches = list(TaskMarkerInjector.MARKER_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group(1).upper() == "START"
        assert matches[0].group(2).strip() == "dpo_review"
    
    def test_detect_complete_marker(self):
        content = "[TASK_COMPLETE: dpo_review] Analysis finished successfully"
        matches = list(TaskMarkerInjector.MARKER_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group(1).upper() == "COMPLETE"
    
    def test_detect_switch_marker(self):
        content = "[TASK_SWITCH: dpo_review -> docker_setup] Moving to next task"
        matches = list(TaskMarkerInjector.MARKER_PATTERN.finditer(content))
        assert len(matches) == 1
        assert matches[0].group(1).upper() == "SWITCH"
        task_ref = matches[0].group(2).strip()
        assert "->" in task_ref
    
    def test_no_markers_in_normal_content(self):
        content = REAL_ASSISTANT_MESSAGES[0]  # Real message without markers
        matches = list(TaskMarkerInjector.MARKER_PATTERN.finditer(content))
        assert len(matches) == 0


class TestTaskMarkerInjector:
    """Test the injector's parsing and injection logic."""
    
    @pytest.fixture
    def injector(self):
        return TaskMarkerInjector()
    
    def test_system_prompt_injection(self, injector):
        system_prompt = "### Tool Usage Protocol\nSome instructions here"
        result = injector.inject_into_system_prompt(system_prompt)
        assert "[TASK_START:" in result
        assert "Task Marking Protocol" in result
    
    def test_no_duplicate_injection(self, injector):
        prompt_with_markers = "Some text [TASK_START: test] more text ### Tool Usage Protocol\nInstructions"
        result = injector.inject_into_system_prompt(prompt_with_markers)
        # Should not add duplicate section
        assert result.count("Task Marking Protocol") == 0
    
    def test_parse_simple_task_lifecycle(self, injector):
        messages = [
            make_message("assistant", "[TASK_START: review] Starting work"),
            make_message("user", "How's it going?"),
            make_message("assistant", "Making progress..."),
            make_message("assistant", "[TASK_COMPLETE: review] Done!"),
        ]
        
        markers, active = injector.parse_markers_from_messages(messages)
        assert len(markers) == 2
        assert len(active) == 0  # Task completed, no active tasks
        
        states = [m.state for m in markers]
        assert TaskState.ACTIVE in states
        assert TaskState.COMPLETE in states
    
    def test_parse_switch_marker(self, injector):
        messages = [
            make_message("assistant", "[TASK_START: task_a] First task"),
            make_message("assistant", "[TASK_SWITCH: task_a -> task_b] Switching"),
        ]
        
        markers, active = injector.parse_markers_from_messages(messages)
        assert len(active) == 1
        assert "task_b" in active
    
    def test_multiple_tasks_tracked(self, injector):
        messages = [
            make_message("assistant", "[TASK_START: task_a] First"),
            make_message("assistant", "[TASK_START: task_b] Second"),
            make_message("assistant", "[TASK_COMPLETE: task_a] Done with first"),
        ]
        
        markers, active = injector.parse_markers_from_messages(messages)
        assert len(active) == 1
        assert "task_b" in active
        assert "task_a" not in active


# ============================================================================
# ClosureDetector Tests  
# ============================================================================

class TestClosureDetector:
    """Test implicit closure detection patterns."""
    
    @pytest.fixture
    def detector(self):
        return ClosureDetector()
    
    def test_detect_completion_patterns(self, detector):
        completion_texts = [
            "All steps completed successfully.",
            "Dataset is ready for training.",
            "✅ Task Complete — exported to file",
            "Final statistics: 1,174 pairs processed",
            "Successfully implemented all changes",
        ]
        
        for text in completion_texts:
            result = detector.detect_closure(text)
            assert result == TaskState.COMPLETE, f"Failed to detect completion in: {text[:50]}..."
    
    def test_detect_deferral_patterns(self, detector):
        deferral_texts = [
            "Let's do this later when we have more time.",
            "We'll come back to this after training.",
            "Parking this for now — focus on Docker setup first",
        ]
        
        for text in deferral_texts:
            result = detector.detect_closure(text)
            assert result == TaskState.DEFERRED, f"Failed to detect deferral in: {text[:50]}..."
    
    def test_no_closure_in_normal_content(self, detector):
        normal_texts = [
            "Let me check the current state of things.",
            "I'm making progress on the analysis.",
            "Here are the results so far...",
        ]
        
        for text in normal_texts:
            result = detector.detect_closure(text)
            assert result is None, f"False closure detected in: {text[:50]}..."


# ============================================================================
# TurnCategorizer Tests
# ============================================================================

class TestTurnCategorizer:
    """Test turn categorization logic."""
    
    @pytest.fixture
    def categorizer(self):
        return TurnCategorizer()
    
    def test_categorize_system_message(self, categorizer):
        msg = make_message("system", "You are a helpful assistant.")
        assert categorizer.categorize(msg) == TurnCategory.SYSTEM
    
    def test_categorize_user_query(self, categorizer):
        msg = make_message("user", REAL_USER_MESSAGES[0])
        assert categorizer.categorize(msg) == TurnCategory.USER_QUERY
    
    def test_categorize_tool_call(self, categorizer):
        msg = make_message("assistant", "Let me check.", tool_calls=[{"name": "test"}])
        assert categorizer.categorize(msg) == TurnCategory.TOOL_CALL
    
    def test_categorize_task_marker(self, categorizer):
        msg = make_message("assistant", "[TASK_START: review] Starting work")
        assert categorizer.categorize(msg) == TurnCategory.TASK_MARKER
    
    def test_categorize_status_update(self, categorizer):
        msg = make_message("assistant", "Status: 3/5 tasks complete. Remaining: X, Y, Z")
        assert categorizer.categorize(msg) == TurnCategory.STATUS_UPDATE
    
    def test_categorize_normal_response(self, categorizer):
        msg = make_message("assistant", REAL_ASSISTANT_MESSAGES[0])
        assert categorizer.categorize(msg) == TurnCategory.ASSISTANT_RESPONSE


# ============================================================================
# TaskAwarePruner Integration Tests
# ============================================================================

class TestTaskAwarePruner:
    """Test the full pruning pipeline with realistic scenarios."""
    
    @pytest.fixture
    def pruner(self):
        return TaskAwarePruner(window_size=10, protect_first_n=2, protect_last_n=4)
    
    def create_conversation_with_markers(self) -> list:
        """Build a realistic conversation with task markers."""
        return [
            make_message("system", "You are Patrick's assistant. Follow these instructions..."),
            make_message("user", "Review my DPO dataset and set up Docker for training."),
            
            # Task 1 starts
            make_message("assistant", "[TASK_START: dpo_review] Analyzing dataset quality...\n\nLet me check the current statistics..."),
            make_message("assistant", REAL_ASSISTANT_MESSAGES[0]),  # Review results
            
            # Task 1 completes  
            make_message("assistant", "[TASK_COMPLETE: dpo_review] Dataset analysis done. Found 3 critical issues."),
            
            # Task 2 starts (active - should be preserved)
            make_message("user", "Now set up Unsloth Docker container."),
            make_message("assistant", "[TASK_START: docker_setup] Configuring Docker environment...\n\nPulling the official Unsloth image..."),
            make_message("assistant", "Installing CUDA dependencies for NVIDIA Blackwell architecture..."),
            
            # More conversation while task 2 is active
            make_message("user", "What about GPU memory configuration?"),
            make_message("assistant", "For a 27B model, you'll need at least 64GB VRAM. Your GPU has that covered."),
            
            # Recent turns (always protected)
            make_message("user", "Any other considerations?"),
            make_message("assistant", "Make sure to set TORCH_CUDA_ARCH_LIST=\"12.0\" for Blackwell compatibility."),
        ]
    
    def test_preserves_active_task_context(self, pruner):
        """Active tasks should have their turns preserved over completed ones."""
        messages = self.create_conversation_with_markers()
        
        # Prune to 60% of original size
        target = len(messages) * 6 // 10
        keep_indices = pruner.select_turns_to_keep(messages, target_count=target)
        
        # Find indices belonging to active task (docker_setup)
        docker_start_idx = None
        for i, msg in enumerate(messages):
            if "[TASK_START: docker_setup]" in msg.get("content", ""):
                docker_start_idx = i
                break
        
        assert docker_start_idx is not None, "Could not find docker_setup task start"
        
        # At least some turns from active task should be kept
        active_task_kept = sum(1 for idx in keep_indices if idx >= docker_start_idx)
        assert active_task_kept > 0, "No active task turns were preserved"
    
    def test_prunes_completed_tasks_more_aggressively(self, pruner):
        """Completed tasks can be pruned more than active ones."""
        messages = self.create_conversation_with_markers()
        
        scores = pruner.score_turns(messages)
        
        # Find a turn from completed task vs active task
        completed_task_score = None
        active_task_score = None
        
        for score in scores:
            msg = messages[score.turn_index]
            content = msg.get("content", "")
            
            # Look for the completion marker (with colon)
            if "[TASK_COMPLETE:" in content and "dpo_review" in content:
                completed_task_score = score
            elif "[TASK_START:" in content and "docker_setup" in content:
                active_task_score = score
        
        assert completed_task_score is not None, "Could not find completed task marker"
        assert active_task_score is not None, "Could not find active task marker"
        
        # Active task should have higher score due to task_bonus
        assert active_task_score.total_score > completed_task_score.total_score, \
            f"Active task ({active_task_score.total_score:.2f}) should score higher than completed ({completed_task_score.total_score:.2f})"
    
    def test_protected_turns_always_kept(self, pruner):
        """First N and last N turns are always preserved."""
        messages = self.create_conversation_with_markers()
        
        # Prune aggressively to 40%
        target = max(len(messages) * 4 // 10, pruner.protect_first_n + pruner.protect_last_n)
        keep_indices = pruner.select_turns_to_keep(messages, target_count=target)
        keep_set = set(keep_indices)
        
        # First N should be kept
        for i in range(pruner.protect_first_n):
            assert i in keep_set, f"Protected first turn {i} was not kept"
        
        # Last N should be kept
        total = len(messages)
        for i in range(total - pruner.protect_last_n, total):
            assert i in keep_set, f"Protected last turn {i} was not kept"
    
    def test_pruning_report(self, pruner):
        """Pruning report should provide useful statistics."""
        messages = self.create_conversation_with_markers()
        keep_indices = pruner.select_turns_to_keep(messages)
        
        report = pruner.get_pruning_report(messages, keep_indices)
        
        assert "original_count" in report
        assert "kept_count" in report  
        assert "pruned_count" in report
        assert report["original_count"] == len(messages)
        assert report["kept_count"] + report["pruned_count"] == report["original_count"]
    
    def test_empty_messages_handled(self, pruner):
        """Empty message list should return empty results."""
        assert pruner.prune([]) == []
        assert pruner.select_turns_to_keep([]) == []
        assert pruner.score_turns([]) == []


# ============================================================================
# Real-World Scenario Tests
# ============================================================================

class TestRealWorldScenarios:
    """Test pruning against realistic multi-task conversations."""
    
    @pytest.fixture
    def pruner(self):
        return TaskAwarePruner(window_size=15, protect_first_n=3, protect_last_n=6)
    
    def test_long_multi_task_session(self, pruner):
        """Simulate a long session with multiple tasks of varying states."""
        messages = [
            make_message("system", "System prompt..."),
            make_message("user", "Help me with three things: dataset review, Docker setup, and server build planning."),
            
            # Task 1: Dataset review (completed)
            make_message("assistant", "[TASK_START: dataset_review] Starting analysis..."),
            make_message("assistant", REAL_ASSISTANT_MESSAGES[0]),
            make_message("assistant", REAL_ASSISTANT_MESSAGES[1]),
            make_message("assistant", "[TASK_COMPLETE: dataset_review] Done! Dataset ready."),
            
            # Task 2: Docker setup (active)
            make_message("user", "Great, now let's set up Docker."),
            make_message("assistant", "[TASK_START: docker_setup] Pulling Unsloth image..."),
            make_message("assistant", "Configuring CUDA for Blackwell architecture..."),
            make_message("tool", "{\"status\": \"success\", \"container_id\": \"abc123\"}"),
            make_message("assistant", "Container running on port 8888. Jupyter Lab accessible."),
            
            # Task 3: Server planning (active, started but not finished)
            make_message("user", "What about the enterprise server build?"),
            make_message("assistant", "[TASK_START: server_planning] Let me outline the components..."),
            make_message("assistant", REAL_ASSISTANT_MESSAGES[5]),  # Status update
            
            # Recent conversation (protected)
            make_message("user", "Any cost estimates?"),
            make_message("assistant", "Total estimated cost: $15-20k for a dual-socket setup with 256GB VRAM."),
        ]
        
        scores = pruner.score_turns(messages)
        keep_indices = pruner.select_turns_to_keep(messages, target_count=12)
        
        # Verify active tasks have better representation than completed ones
        kept_content = "".join(messages[i].get("content", "") for i in keep_indices)
        
        # Active task keywords should appear more than completed task
        assert "docker_setup" in kept_content or "server_planning" in kept_content, \
            "At least one active task context should be preserved"


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Ensure pruning is efficient even with large message lists."""
    
    def test_large_message_list_performance(self):
        """Pruning 100+ messages should complete quickly."""
        import time
        
        pruner = TaskAwarePruner(window_size=20)
        
        # Create a large synthetic conversation
        messages = []
        for i in range(150):
            if i == 0:
                msg = make_message("system", "System prompt...")
            elif i % 3 == 0:
                msg = make_message("user", f"Question {i} about the project?")
            elif i % 3 == 1:
                msg = make_message("assistant", f"[TASK_START: task_{i}] Working on item {i}\n\nSome detailed response...")
            else:
                msg = make_message("assistant", f"Response to question {i} with analysis and recommendations.")
            
            messages.append(msg)
        
        # Time the pruning operation
        start = time.time()
        keep_indices = pruner.select_turns_to_keep(messages, target_count=30)
        elapsed = time.time() - start
        
        # Should complete in under 1 second for 150 messages
        assert elapsed < 1.0, f"Pruning took {elapsed:.3f}s — should be faster"
        assert len(keep_indices) == 30


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

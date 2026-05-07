"""Task-Aware Pruner for Rolling Window Context Engine.

Intelligently prunes conversation history while preserving turns that belong to
active, incomplete tasks. Uses task markers injected by TaskMarkerInjector to
identify task boundaries and closure signals.

Architecture:
1. Parse task markers from assistant messages
2. Detect task closures via heuristics (completion phrases, status updates)
3. Score each turn's importance based on task state + content type
4. Prune low-importance turns while protecting active task context

Design Principles:
- Elegant: Clean separation of concerns (parse → score → prune)
- Robust: Graceful degradation when markers are absent
- Efficient: O(n) single-pass scoring, O(n log n) sort for selection
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    from .task_marker_injector import TaskMarkerInjector, TaskState
except ImportError:
    from task_marker_injector import TaskMarkerInjector, TaskState

logger = logging.getLogger(__name__)

class TurnCategory(Enum):
    """Categories of conversation turns for importance scoring."""
    SYSTEM = "system"           # System prompt / instructions
    USER_QUERY = "user_query"   # User's question or instruction  
    ASSISTANT_RESPONSE = "assistant_response"  # Assistant's substantive reply
    TOOL_CALL = "tool_call"     # Assistant calling a tool
    TOOL_RESULT = "tool_result" # Tool output/result
    TASK_MARKER = "task_marker" # Turn containing task state markers
    STATUS_UPDATE = "status_update"  # Progress/status reporting

@dataclass
class TurnScore:
    """Importance score for a single turn."""
    turn_index: int
    category: TurnCategory
    base_score: float = 0.0      # Base importance from category
    task_bonus: float = 0.0       # Bonus if part of active task
    recency_bonus: float = 0.0    # Bonus for recent turns
    total_score: float = 0.0      # Final combined score
    
    def compute_total(self) -> None:
        """Calculate final score from components."""
        self.total_score = self.base_score + self.task_bonus + self.recency_bonus

class ClosureDetector:
    """Detect when tasks appear to be complete based on linguistic patterns.
    
    Uses heuristic pattern matching to identify completion signals in assistant
    responses. This supplements explicit task markers with implicit closure detection.
    """
    
    # Patterns indicating task completion
    COMPLETION_PATTERNS = [
        r'\b(complete|done|finished|accomplished)\b.*\b(task|work|step|phase)\b',
        r'\ball\s+(steps?|tasks?|items?)\s+(complete|done|finished)',
        r'\bsuccessfully\s+(completed|finished|implemented|deployed)',
        r'\bready\s+for\s+(training|testing|deployment|review)\b',
        r'\bfinal\s+(statistics|summary|report|status)\b',
        r'\[?✅?\]?\s*\w+\s*(Complete|Done|Finished)',
        r'\b(exported|saved|written|created)\s+successfully\b',
        r'\b(test|verification)\s+(passed|successful|complete)',
    ]
    
    # Patterns indicating task deferral/postponement
    DEFERRAL_PATTERNS = [
        r"\b(let[''']s?\s+do\s+this\s+later|we[''']ll\s+come\s+back|postpone|defer)\b",
        r"\b(parking|setting\s+aside|holding\s+off)\b.*\b(for\s+now|until\s+later)",
        r"\b(not\s+yet|wait\s+on|hold\s+off)\b",
    ]
    
    # Patterns indicating task cancellation/abandonment
    CANCELLATION_PATTERNS = [
        r"\b(cancelled|abandoned|dropped|scrapped)\b",
        r"\b(don[''']t\s+need|not\s+necessary|unnecessary)\b.*\b(this|it)",
        r"\b(skipping|omitting|leaving\s+out)\b",
    ]
    
    def __init__(self) -> None:
        self._completion_regex = re.compile(
            '|'.join(self.COMPLETION_PATTERNS), 
            re.IGNORECASE
        )
        self._deferral_regex = re.compile(
            '|'.join(self.DEFERRAL_PATTERNS),
            re.IGNORECASE  
        )
        self._cancellation_regex = re.compile(
            '|'.join(self.CANCELLATION_PATTERNS),
            re.IGNORECASE
        )
    
    def detect_closure(self, content: str) -> TaskState | None:
        """Detect task closure state from message content.
        
        Args:
            content: Message text to analyze
            
        Returns:
            TaskState if closure detected, None otherwise
        """
        if not isinstance(content, str):
            return None
        
        if self._completion_regex.search(content):
            return TaskState.COMPLETE
        elif self._deferral_regex.search(content):
            return TaskState.DEFERRED  
        elif self._cancellation_regex.search(content):
            return TaskState.CANCELLED
        
        return None

class TurnCategorizer:
    """Categorize conversation turns for importance scoring."""
    
    def categorize(self, msg: dict[str, Any]) -> TurnCategory:
        """Categorize a single message.
        
        Args:
            msg: Message dict with 'role' and optionally 'content', 'tool_calls'
            
        Returns:
            TurnCategory enum value
        """
        role = msg.get("role", "")
        content = msg.get("content", "")
        has_tool_calls = bool(msg.get("tool_calls"))
        
        # System messages
        if role == "system":
            return TurnCategory.SYSTEM
        
        # User messages  
        if role == "user":
            return TurnCategory.USER_QUERY
        
        # Assistant messages
        if role == "assistant":
            if has_tool_calls:
                return TurnCategory.TOOL_CALL
            
            # Check for task markers in content
            if isinstance(content, str) and TaskMarkerInjector.MARKER_PATTERN.search(content):
                return TurnCategory.TASK_MARKER
            
            # Check for status update patterns
            if self._is_status_update(content):
                return TurnCategory.STATUS_UPDATE
            
            return TurnCategory.ASSISTANT_RESPONSE
        
        # Tool results
        if role == "tool":
            return TurnCategory.TOOL_RESULT
        
        # Fallback
        return TurnCategory.ASSISTANT_RESPONSE
    
    def _is_status_update(self, content: str) -> bool:
        """Check if content appears to be a status/progress update."""
        if not isinstance(content, str):
            return False
        
        status_indicators = [
            "status:", "progress:", "update:", "current state:",
            "[Tasks:", "(X/Y complete", "remaining:", "next step"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in status_indicators)

class TaskAwarePruner:
    """Task-aware pruning engine that preserves active task context.
    
    Combines explicit task markers with heuristic closure detection to make
    intelligent decisions about which turns to preserve during compression.
    
    Usage:
        pruner = TaskAwarePruner()
        scores = pruner.score_turns(messages)
        pruned = pruner.select_turns_to_keep(messages, scores, target_count=20)
    """
    
    # Base importance scores by category (higher = more important to keep)
    CATEGORY_BASE_SCORES = {
        TurnCategory.SYSTEM: 10.0,           # Always keep system context
        TurnCategory.USER_QUERY: 8.0,        # User intent is critical  
        TurnCategory.TASK_MARKER: 9.0,       # Task state markers are vital
        TurnCategory.STATUS_UPDATE: 7.0,     # Progress tracking is important
        TurnCategory.ASSISTANT_RESPONSE: 5.0, # Substantive replies have value
        TurnCategory.TOOL_CALL: 2.0,         # Tool calls can be reconstructed
        TurnCategory.TOOL_RESULT: 3.0,       # Results useful but often verbose
    }
    
    # Bonus for turns belonging to active tasks
    ACTIVE_TASK_BONUS = 4.0
    
    # Recency bonus scales with position from end (most recent = highest)
    RECENCY_WEIGHT = 0.5
    
    def __init__(
        self, 
        window_size: int = 20,
        protect_first_n: int = 3,
        protect_last_n: int = 6
    ) -> None:
        """Initialize the task-aware pruner.
        
        Args:
            window_size: Target number of turns to maintain in context
            protect_first_n: Always keep this many oldest turns (system prompt area)
            protect_last_n: Always keep this many newest turns (current conversation)
        """
        self.window_size = window_size
        self.protect_first_n = protect_first_n  
        self.protect_last_n = protect_last_n
        
        self.injector = TaskMarkerInjector()
        self.closure_detector = ClosureDetector()
        self.categorizer = TurnCategorizer()
    
    def score_turns(self, messages: list[dict[str, Any]]) -> list[TurnScore]:
        """Score all turns for importance.
        
        Args:
            messages: Full message list to score
            
        Returns:
            List of TurnScore objects, one per message
        """
        if not messages:
            return []
        
        # Parse task markers and detect active tasks
        all_markers, active_tasks = self.injector.parse_markers_from_messages(messages)
        
        # Build set of turn indices that belong to active tasks
        active_task_turns = self._get_active_task_turn_indices(all_markers, messages)
        
        scores: list[TurnScore] = []
        total_turns = len(messages)
        
        for idx, msg in enumerate(messages):
            category = self.categorizer.categorize(msg)
            
            # Base score from category
            base_score = self.CATEGORY_BASE_SCORES.get(category, 1.0)
            
            # Task bonus if this turn belongs to an active task
            task_bonus = self.ACTIVE_TASK_BONUS if idx in active_task_turns else 0.0
            
            # Recency bonus (linear scale: most recent = highest)
            turns_from_end = total_turns - 1 - idx
            recency_bonus = max(0.0, (1.0 - turns_from_end / total_turns)) * self.RECENCY_WEIGHT * 10
            
            score = TurnScore(
                turn_index=idx,
                category=category,
                base_score=base_score,
                task_bonus=task_bonus,
                recency_bonus=recency_bonus
            )
            score.compute_total()
            scores.append(score)
        
        return scores
    
    def select_turns_to_keep(
        self, 
        messages: list[dict[str, Any]], 
        target_count: int | None = None
    ) -> list[int]:
        """Select which turn indices to keep after pruning.
        
        Args:
            messages: Full message list
            target_count: Desired number of turns to keep (defaults to window_size)
            
        Returns:
            Sorted list of turn indices to preserve
        """
        if not messages:
            return []
        
        target = target_count or self.window_size
        total = len(messages)
        
        # If we're already under target, keep everything
        if total <= target:
            return list(range(total))
        
        scores = self.score_turns(messages)
        
        # Protected indices (always kept regardless of score)
        protected: set[int] = set()
        
        # Protect first N turns (system prompt, initial context)
        for i in range(min(self.protect_first_n, total)):
            protected.add(i)
        
        # Protect last N turns (current conversation)  
        for i in range(max(0, total - self.protect_last_n), total):
            protected.add(i)
        
        # Select remaining slots by score
        non_protected = [s for s in scores if s.turn_index not in protected]
        non_protected.sort(key=lambda s: s.total_score, reverse=True)
        
        # Fill remaining slots with highest-scoring turns
        # Guard against negative remaining_slots (when protected > target)
        remaining_slots = max(0, target - len(protected))
        selected_by_score = {s.turn_index for s in non_protected[:remaining_slots]}
        
        # Combine protected + score-selected
        keep_indices = sorted(protected | selected_by_score)
        
        return keep_indices
    
    def prune(
        self, 
        messages: list[dict[str, Any]], 
        target_count: int | None = None
    ) -> list[dict[str, Any]]:
        """Prune message list to target count while preserving active task context.
        
        Args:
            messages: Full message list to prune
            target_count: Desired number of turns after pruning
            
        Returns:
            Pruned message list maintaining original order
        """
        keep_indices = self.select_turns_to_keep(messages, target_count)
        return [messages[i] for i in keep_indices]
    
    def _get_active_task_turn_indices(
        self, 
        markers: list[Any],  # TaskMarker objects
        messages: list[dict[str, Any]]
    ) -> set[int]:
        """Find all turn indices that belong to active (non-closed) tasks.
        
        A task is considered active if:
        1. It has a TASK_START marker with no corresponding completion marker
        2. No closure was detected in subsequent turns
        
        Args:
            markers: Parsed task markers from injector
            messages: Full message list for closure detection
            
        Returns:
            Set of turn indices belonging to active tasks
        """
        active_task_turns: set[int] = set()
        
        # Group markers by task ID
        task_marker_map: dict[str, list[Any]] = {}
        for marker in markers:
            if marker.task_id not in task_marker_map:
                task_marker_map[marker.task_id] = []
            task_marker_map[marker.task_id].append(marker)
        
        # For each task, determine if it's still active and find its turn range
        for task_id, task_markers in task_marker_map.items():
            # Find start marker
            start_marker = None
            for m in task_markers:
                if m.state == TaskState.ACTIVE:
                    start_marker = m
                    break
            
            if not start_marker:
                continue
            
            # Check for closure markers after start
            has_closure = False
            end_turn = start_marker.turn_index
            
            for m in task_markers:
                if m.turn_index > start_marker.turn_index:
                    if m.state in (TaskState.COMPLETE, TaskState.DEFERRED, TaskState.CANCELLED):
                        has_closure = True
                        break
                    end_turn = max(end_turn, m.turn_index)
            
            # Also check for implicit closure via content analysis
            if not has_closure:
                # Look at turns after the last marker for this task
                for idx in range(end_turn + 1, len(messages)):
                    msg = messages[idx]
                    if msg.get("role") == "assistant":
                        closure = self.closure_detector.detect_closure(msg.get("content", ""))
                        if closure:
                            has_closure = True
                            break
            
            # If task is still active, mark all turns in its range
            if not has_closure:
                # Find end of task (next task start or end of messages)
                next_task_start = len(messages)
                for other_id, other_markers in task_marker_map.items():
                    if other_id == task_id:
                        continue
                    for m in other_markers:
                        if m.state == TaskState.ACTIVE and m.turn_index > start_marker.turn_index:
                            next_task_start = min(next_task_start, m.turn_index)
                
                task_end = min(end_turn + 5, next_task_start)  # Include some context after last marker
                
                for idx in range(start_marker.turn_index, min(task_end, len(messages))):
                    active_task_turns.add(idx)
        
        return active_task_turns
    
    def get_pruning_report(
        self, 
        messages: list[dict[str, Any]], 
        keep_indices: list[int]
    ) -> dict[str, Any]:
        """Generate a report explaining pruning decisions.
        
        Args:
            messages: Original message list
            keep_indices: Indices that were selected to keep
            
        Returns:
            Dict with pruning statistics and explanations
        """
        scores = self.score_turns(messages)
        keep_set = set(keep_indices)
        
        pruned_categories = {}
        kept_categories = {}
        
        for score in scores:
            cat = score.category.value
            if score.turn_index in keep_set:
                kept_categories[cat] = kept_categories.get(cat, 0) + 1
            else:
                pruned_categories[cat] = pruned_categories.get(cat, 0) + 1
        
        return {
            "original_count": len(messages),
            "kept_count": len(keep_indices),
            "pruned_count": len(messages) - len(keep_indices),
            "kept_by_category": kept_categories,
            "pruned_by_category": pruned_categories,
            "active_tasks_preserved": sum(
                1 for s in scores if s.task_bonus > 0 and s.turn_index in keep_set
            )
        }

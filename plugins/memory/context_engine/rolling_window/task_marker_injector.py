"""Task Marker Injector for Rolling Window Context Engine.

Injects structured task markers into the system prompt that signal to the model
when tasks begin, complete, or change state. These markers enable the TaskAwarePruner
to make intelligent decisions about which turns to preserve during pruning.

Marker Format:
    [TASK_START: unique_id] — New task initiated
    [TASK_COMPLETE: unique_id] — Task finished successfully  
    [TASK_DEFERRED: unique_id] — Task postponed for later
    [TASK_CANCELLED: unique_id] — Task abandoned/cancelled
    [TASK_SWITCH: from_id -> to_id] — Context switching between tasks

Design Principles:
- Minimal overhead: Markers are short, structured tokens the model learns to emit
- No semantic loss: Markers supplement content, never replace it
- Pruner-friendly: Clear delimiters allow O(1) task boundary detection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class TaskState(Enum):
    """Possible states for a tracked task."""
    ACTIVE = "active"
    COMPLETE = "complete"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


@dataclass
class TaskMarker:
    """Represents a single task marker with metadata."""
    task_id: str
    state: TaskState
    turn_index: int  # Position in message list when this marker appeared
    summary: str = ""  # Optional brief description of the task
    
    def __repr__(self) -> str:
        return f"[{self.state.value}:{self.task_id}]@turn{self.turn_index}"


class TaskMarkerInjector:
    """Injects task tracking markers into system prompt instructions.
    
    This component adds a small section to the system prompt teaching the model
    to emit structured task markers. The markers are then consumed by the
    TaskAwarePruner during compression cycles.
    
    Example injection (appended to existing operational discipline):
    
        ### Task Marking Protocol
        
        When starting, completing, or switching between distinct tasks, emit a marker:
        
        - Starting work: [TASK_START: task_name] — Brief description of what you're doing
        - Finishing work: [TASK_COMPLETE: task_name] — What was accomplished  
        - Deferring work: [TASK_DEFERRED: task_name] — Why it's postponed
        - Cancelling work: [TASK_CANCELLED: task_name] — Reason for cancellation
        - Switching context: [TASK_SWITCH: old_task -> new_task] — Brief transition note
        
        Keep markers concise. They help the system preserve important context during pruning.
    """
    
    MARKER_PATTERN = re.compile(
        r'\[TASK_(START|COMPLETE|DEFERRED|CANCELLED|SWITCH):\s*([^\]]+)\]',
        re.IGNORECASE
    )
    
    def __init__(self) -> None:
        self._active_tasks: Dict[str, TaskMarker] = {}
        self._completed_tasks: List[TaskMarker] = []
    
    @property
    def system_prompt_section(self) -> str:
        """Return the task marking protocol text for injection into system prompt."""
        return """### Task Marking Protocol

When starting, completing, or switching between distinct tasks, emit a brief marker:

- Starting work: `[TASK_START: task_name]` — Brief description of what you're doing
- Finishing work: `[TASK_COMPLETE: task_name]` — What was accomplished  
- Deferring work: `[TASK_DEFERRED: task_name]` — Why it's postponed
- Cancelling work: `[TASK_CANCELLED: task_name]` — Reason for cancellation
- Switching context: `[TASK_SWITCH: old_task -> new_task]` — Brief transition note

Keep markers concise (one line). They help the system preserve important context during pruning.
Only mark substantive tasks — not every tool call or minor step."""

    def inject_into_system_prompt(self, system_prompt: str) -> str:
        """Inject task marking protocol into an existing system prompt.
        
        Args:
            system_prompt: Current system prompt text
            
        Returns:
            System prompt with task marking protocol appended before final sections
        """
        # Find a good insertion point - after operational discipline, before tool usage
        insertion_markers = [
            "### Tool Usage Protocol",
            "## Deferred Tools", 
            "## Skills (on-demand)",
            "## Current Session Context"
        ]
        
        for marker in insertion_markers:
            if marker in system_prompt:
                idx = system_prompt.index(marker)
                before = system_prompt[:idx]
                after = system_prompt[idx:]
                
                # Avoid duplicate injection
                if "[TASK_START:" not in before:
                    return before + "\n" + self.system_prompt_section + "\n\n" + after
        
        # Fallback: append near end if no insertion point found
        if "[TASK_START:" not in system_prompt:
            return system_prompt + "\n\n" + self.system_prompt_section
        
        return system_prompt
    
    def parse_markers_from_messages(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[TaskMarker], Dict[str, TaskMarker]]:
        """Parse all task markers from a message list.

        Returns a fresh dict of active tasks without mutating instance state.
        Use ``sync_active_tasks()`` if you want to persist the result to
        ``self._active_tasks``.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            Tuple of (all_markers_found, currently_active_tasks)
        """
        all_markers: List[TaskMarker] = []
        active_tasks: Dict[str, TaskMarker] = {}
        completed_tasks: List[TaskMarker] = []

        # Map action strings to TaskState enum
        ACTION_TO_STATE = {
            "START": TaskState.ACTIVE,
            "COMPLETE": TaskState.COMPLETE,
            "DEFERRED": TaskState.DEFERRED,
            "CANCELLED": TaskState.CANCELLED,
        }

        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            matches = self.MARKER_PATTERN.finditer(content)
            for match in matches:
                action = match.group(1).upper()
                task_ref = match.group(2).strip()

                # Handle TASK_SWITCH specially (has "from -> to" format)
                if action == "SWITCH":
                    parts = task_ref.split("->")
                    if len(parts) == 2:
                        old_task = parts[0].strip()
                        new_task = parts[1].strip()

                        # Mark old task as implicitly completed/switched from
                        if old_task in active_tasks:
                            old_marker = active_tasks.pop(old_task)
                            old_marker.state = TaskState.COMPLETE
                            all_markers.append(old_marker)
                            completed_tasks.append(old_marker)

                        # Start new task
                        new_marker = TaskMarker(
                            task_id=new_task,
                            state=TaskState.ACTIVE,
                            turn_index=idx
                        )
                        active_tasks[new_task] = new_marker
                        all_markers.append(new_marker)
                else:
                    # Map action string to enum
                    state = ACTION_TO_STATE.get(action)
                    if not state:
                        continue

                    marker = TaskMarker(
                        task_id=task_ref,
                        state=state,
                        turn_index=idx
                    )

                    if state == TaskState.ACTIVE:
                        active_tasks[task_ref] = marker
                    else:
                        # Remove from active if it was there and record the transition
                        if task_ref in active_tasks:
                            removed = active_tasks.pop(task_ref)
                            completed_tasks.append(removed)

                    all_markers.append(marker)

        # Persist results to instance state
        self._active_tasks = active_tasks
        self._completed_tasks.extend(completed_tasks)
        # Return a fresh copy so callers can't mutate our state
        return all_markers, dict(active_tasks)

    def sync_active_tasks(self, messages: List[Dict[str, str]]) -> List[TaskMarker]:
        """Parse messages and return all markers found.

        Convenience method that populates ``self._active_tasks`` from a
        message list and returns the list of markers.  Prefer
        ``parse_markers_from_messages()`` when you need the active-task dict
        as a return value.
        """
        markers, _ = self.parse_markers_from_messages(messages)
        return markers
    
    def get_active_task_ids(self) -> List[str]:
        """Return list of currently active task IDs."""
        return list(self._active_tasks.keys())
    
    def get_task_turn_ranges(
        self, messages: List[Dict[str, str]]
    ) -> Dict[str, Tuple[int, int]]:
        """For each active/complete task, find the turn range it spans.
        
        Returns:
            Dict mapping task_id to (start_turn_idx, end_turn_idx)
        """
        all_markers, _ = self.parse_markers_from_messages(messages)
        task_ranges: Dict[str, Tuple[int, int]] = {}
        
        # Group markers by task ID
        task_markers: Dict[str, List[TaskMarker]] = {}
        for marker in all_markers:
            if marker.task_id not in task_markers:
                task_markers[marker.task_id] = []
            task_markers[marker.task_id].append(marker)
        
        # Calculate ranges
        for task_id, markers in task_markers.items():
            start_turn = min(m.turn_index for m in markers if m.state == TaskState.ACTIVE or m.state == TaskState.COMPLETE)
            end_turn = max(m.turn_index for m in markers)
            task_ranges[task_id] = (start_turn, end_turn)
        
        return task_ranges
    
    def has_active_tasks(self) -> bool:
        """Check if there are any currently active tasks."""
        return len(self._active_tasks) > 0
    
    def reset(self) -> None:
        """Clear all tracked tasks."""
        self._active_tasks.clear()
        self._completed_tasks.clear()

"""Context Bridge Builder — Formats extraction results into bridge text.

Takes structured data from ExtractionEngine (active tasks, file edits, errors, gaps)
and formats it into a compact retrieval index for injection during compression.

Strictly capped at 4KB to preserve reasoning tokens on local hardware.
Uses FIFO truncation if the index exceeds the limit.

Negative feedback loop: After building the bridge, scores quality and records
results in FeedbackState. If degradation detected, applies corrections (wider
extraction window, preservation markers) for next compression cycle.

Graceful degradation: never breaks compression — returns empty string or error note
if something goes wrong. This is an enhancement, not core functionality.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .quality_scorer import BridgeQualityScorer
    from .feedback_state import FeedbackState

logger = logging.getLogger(__name__)

# Strict 4KB cap to preserve reasoning tokens on local hardware
MAX_BRIDGE_CHARS = 4000

# Minimum quality score before appending preservation warnings
_PRESERVATION_WARNING_THRESHOLD = 0.65

class ContextBridgeBuilder:
    """Formats extraction results into a compact context bridge string.

    Takes an ExtractionEngine instance and optional feedback components in __init__.
    The build_bridge() method orchestrates extraction + formatting + quality scoring.
    """

    def __init__(self, extraction_engine=None, scorer=None, feedback_state=None):
        self._extraction = extraction_engine
        self._scorer = scorer
        self._feedback = feedback_state

    def build_bridge(
        self,
        messages: list[dict[str, Any]],
        correction_params: dict[str, Any | None] = None,
    ) -> str:
        """Build the full context bridge from message history.

        Orchestrates extraction, formatting, quality scoring, and feedback recording.
        Graceful degradation — never raises, returns empty string on failure.

        Args:
            messages: Message history to build bridge from.
            correction_params: Optional dict from FeedbackState.get_correction_params()
                containing extraction_window_multiplier, preserve_critical_markers, etc.

        Returns:
            Formatted context bridge text (capped at MAX_BRIDGE_CHARS).
        """
        try:
            # Apply feedback corrections before extraction if available
            effective_messages = self._apply_corrections(messages, correction_params)

            # Extract recent activity for retrieval indexing
            active_tasks = self._extract("active_tasks", effective_messages)
            file_edits = self._extract("file_edits", effective_messages)
            known_errors = self._extract("known_errors", effective_messages)
            knowledge_gaps = self._extract("knowledge_gaps", effective_messages)

            # Build the structured index with turn references
            sections = []

            if active_tasks:
                sections.append(self._format_active_tasks(active_tasks))

            if file_edits:
                sections.append(self._format_file_edits(file_edits))

            if known_errors:
                sections.append(self._format_known_errors(known_errors))

            if knowledge_gaps:
                sections.append(self._format_knowledge_gaps(knowledge_gaps))

            # Add Perpetual Memory guidance
            sections.append(self._format_retrieval_guidance())

            bridge_text = "\n".join(sections)

            # Score quality and record feedback (non-blocking)
            quality_score = self._score_quality(messages, bridge_text)

            # Append preservation warnings if quality is low or degradation detected
            if correction_params and correction_params.get("preserve_critical_markers"):
                lost_items = quality_score.get("lost_items", [])
                if lost_items:
                    warning_section = self._format_preservation_warning(lost_items)
                    bridge_text += "\n" + warning_section

            # Strict 4KB cap to preserve reasoning tokens on local hardware
            if len(bridge_text) > MAX_BRIDGE_CHARS:
                logger.warning(
                    "Context Bridge exceeded %d chars (%d). Truncating oldest entries.",
                    MAX_BRIDGE_CHARS, len(bridge_text),
                )
                # FIFO truncation: remove oldest sections first
                while len(bridge_text) > MAX_BRIDGE_CHARS and len(sections) > 1:
                    sections.pop(0)
                    bridge_text = "\n".join(sections)

            return bridge_text if bridge_text.strip() else ""

        except (AttributeError, KeyError, TypeError) as e:
            # Robust error handling: never break compression due to bridge generation failure
            logger.warning("Context Bridge generation failed: %s", e)
            return "## Context Bridge\n- Error generating retrieval index. See logs for details."

    def _apply_corrections(
        self,
        messages: list[dict[str, Any]],
        correction_params: dict[str, Any | None],
    ) -> list[dict[str, Any]]:
        """Apply feedback corrections to the message set before extraction.

        Currently a no-op — the extraction engine already scans all provided
        messages, so widening the window has no effect. Kept as a hook for
        future implementation where we might fetch additional historical context.

        Args:
            messages: Original message list.
            correction_params: Correction params from FeedbackState.

        Returns:
            Original message list (unchanged).
        """
        if not correction_params:
            return messages

        multiplier = correction_params.get("extraction_window_multiplier", 1.0)
        if multiplier > 1.0:
            logger.info(
                "Feedback correction: extraction_window_multiplier=%.1fx "
                "(currently a no-op, extraction scans all provided messages)",
                multiplier,
            )
        return messages

    def _score_quality(
        self,
        messages: list[dict[str, Any]],
        bridge_text: str,
    ) -> dict[str, Any]:
        """Score bridge quality and record in feedback state.

        Non-blocking — returns empty score dict if scorer or feedback unavailable.
        """
        try:
            if not self._scorer:
                return {}

            score = self._scorer.score(messages, bridge_text)

            # Record in feedback state if available
            if self._feedback:
                self._feedback.record_compression(score)

            logger.debug(
                "Bridge quality: overall=%.2f (tasks=%.1f, files=%.1f, errors=%.1f)",
                score.get("overall", 0),
                score.get("active_tasks_preserved", 0) * 100,
                score.get("file_paths_preserved", 0) * 100,
                score.get("errors_preserved", 0) * 100,
            )

            return score

        except Exception as e:  # noqa: S110 — wrapper around external extraction engine, must never raise
            logger.debug("Quality scoring failed (non-critical): %s", e)
            return {}

    def _format_preservation_warning(self, lost_items: list[str]) -> str:
        """Format a preservation warning section for low-quality bridges.

        Explicitly tells the agent which items were at risk of being lost,
        encouraging it to use retrieval tools to recover them.
        """
        lines = [
            "## ⚠ Preservation Warning",
            "The following items may have been truncated during compression.",
            "Use `perpetual_search` or `query_messages` to retrieve full context:",
        ]
        for item in lost_items[:5]:  # Cap at 5 warnings
            lines.append(f"  - {item}")
        return "\n".join(lines)

    def _extract(self, method_name: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Call extraction method on the engine, degrading gracefully."""
        if not self._extraction:
            return []
        try:
            method = getattr(self._extraction, f"extract_{method_name}", None)
            if method:
                return method(messages)
        except Exception as e:  # noqa: S110 — wrapper around external extraction engine, must never raise
            logger.exception("Extraction failed for %s", method_name)
        return []

    # -----------------------------------------------------------------------
    # Formatting methods — each returns a section string
    # -----------------------------------------------------------------------

    def _format_active_tasks(self, tasks: list[dict[str, Any]]) -> str:
        """Format active tasks section."""
        lines = ["## Active Tasks (with retrieval pointers)"]
        for task in tasks[:3]:  # Limit to 3 most recent tasks
            turn_refs = ", ".join(f"#{t}" for t in task["turn_ids"])
            desc = task.get("description", "full discussion")
            lines.append(f"- **{task['summary']}**")
            lines.append(f"  → See turns {turn_refs}: {desc}")

            # Add key decisions if available
            if task.get("decisions"):
                for decision in task["decisions"][:2]:  # Max 2 decisions per task
                    lines.append(
                        f"  → Key decision at turn #{decision['turn_id']}: "
                        f"{decision['text']}"
                    )
        return "\n".join(lines)

    def _format_file_edits(self, edits: list[dict[str, Any]]) -> str:
        """Format file edits section."""
        lines = ["\n## Files Currently Being Edited"]
        for edit in edits[:3]:  # Limit to 3 most recent edits
            turn_ref = f"#{edit['last_edit_turn']}"
            desc = edit.get("description", "patch applied")
            lines.append(f"- **{edit['path']}** (modified)")
            lines.append(f"  → Last edit: {turn_ref}, {desc}")

            # Add related discussion turns if available
            if edit.get("related_turns"):
                rel_refs = ", ".join(f"#{t}" for t in edit["related_turns"][:5])
                rel_desc = edit.get("related_description", "related discussion")
                lines.append(f"  → Related discussion: {rel_refs} ({rel_desc})")
        return "\n".join(lines)

    def _format_known_errors(self, errors: list[dict[str, Any]]) -> str:
        """Format known errors section."""
        lines = ["\n## Known Errors/Issues"]
        for error in errors[:3]:  # Limit to 3 most recent errors
            turn_ref = f"#{error['turn_id']}"
            fix_loc = error.get("fix_location", "N/A")
            lines.append(f"- **{error['summary']}**")
            lines.append(
                f"  → See {turn_ref} for full traceback "
                f"and fix applied in {fix_loc}"
            )
        return "\n".join(lines)

    def _format_knowledge_gaps(self, gaps: list[dict[str, Any]]) -> str:
        """Format knowledge gaps section."""
        lines = ["\n## Knowledge Gaps (Pending Reference Library Entries)"]
        for gap in gaps[:3]:  # Limit to 3 most recent gaps
            turn_refs = ", ".join(f"#{t}" for t in gap["turn_ids"])
            lines.append(f"- **{gap['summary']}** (confidence: {gap['confidence']})")
            lines.append(
                f"  → Discussed turns {turn_refs} "
                f"— flagged for overnight RL building"
            )
        return "\n".join(lines)

    def _format_retrieval_guidance(self) -> str:
        """Format the historical context retrieval guidance section."""
        lines = [
            "\n## Historical Context Retrieval",
            "- All older conversation turns are stored verbatim in Perpetual Memory (SQLite + FTS5)",
            "- Use `perpetual_search` for semantic/keyword search across all past sessions",
            "- Use `query_messages` for precise filtering by time, role, or metadata",
            "- The reference library at `~/.hermes/reference-library/` documents architecture and tools — "
            "read it with `read_file`. New topics are created as separate files in "
            "`~/.hermes/reference-library/topics/`.",
        ]
        return "\n".join(lines)

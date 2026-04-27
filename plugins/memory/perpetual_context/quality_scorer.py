"""Bridge Quality Scorer — Measures context bridge preservation quality.

Scores how well the context bridge preserved critical information from the
archived messages. Uses fast heuristic matching (regex/substring) — no LLM calls.

Scoring dimensions:
  - active_tasks_preserved: Ratio of extracted tasks that appear in bridge text
  - file_paths_preserved:   Ratio of extracted file paths that appear in bridge text
  - errors_preserved:       Ratio of extracted errors that appear in bridge text
  - gaps_preserved:         Ratio of extracted knowledge gaps that appear in bridge text

Overall score is a weighted average (tasks: 40%, files: 30%, errors: 20%, gaps: 10%).
"""

from __future__ import annotations

import logging
import re as _re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Weight constants for overall score calculation.
# Tasks weighted highest — losing active tasks causes context drift.
_TASK_WEIGHT = 0.40
_FILE_WEIGHT = 0.30
_ERROR_WEIGHT = 0.20
_GAP_WEIGHT = 0.10


class BridgeQualityScorer:
    """Scores context bridge quality against extraction results.

    Takes the same messages and independently extracts critical items,
    then checks how many survived in the bridge text via substring matching.
    """

    def __init__(self):
        pass

    def score(self, messages: List[Dict[str, Any]], bridge_text: str) -> Dict[str, Any]:
        """Score the bridge on multiple preservation dimensions.

        Args:
            messages: The message history that was compressed.
            bridge_text: The generated context bridge text.

        Returns:
            Dict with scoring results:
                - overall: float 0.0-1.0 weighted average
                - active_tasks_preserved: float ratio of tasks preserved
                - file_paths_preserved: float ratio of file paths preserved
                - errors_preserved: float ratio of errors preserved
                - gaps_preserved: float ratio of knowledge gaps preserved
                - bridge_char_count: int actual chars used
                - sections_present: list[str] which sections made it into bridge
                - lost_items: list[str] items that were extracted but not in bridge
        """
        if not messages or not bridge_text:
            return self._empty_score(bridge_text)

        # Extract critical items from messages independently
        tasks = self._extract_task_summaries(messages)
        file_paths = self._extract_file_paths(messages)
        errors = self._extract_error_summaries(messages)
        gaps = self._extract_gap_summaries(messages)

        # Score each dimension
        task_score = self._score_preservation(tasks, bridge_text)
        file_score = self._score_preservation(file_paths, bridge_text)
        error_score = self._score_preservation(errors, bridge_text)
        gap_score = self._score_preservation(gaps, bridge_text)

        # Weighted overall score
        overall = (
            task_score * _TASK_WEIGHT
            + file_score * _FILE_WEIGHT
            + error_score * _ERROR_WEIGHT
            + gap_score * _GAP_WEIGHT
        )

        # Detect which sections are present in the bridge
        sections_present = self._detect_sections(bridge_text)

        # Identify specific lost items for diagnostic reporting
        lost_items = []
        lost_items.extend(self._find_lost(tasks, bridge_text))
        lost_items.extend(self._find_lost(file_paths, bridge_text))
        lost_items.extend(self._find_lost(errors, bridge_text))
        lost_items.extend(self._find_lost(gaps, bridge_text))

        return {
            "overall": round(overall, 3),
            "active_tasks_preserved": round(task_score, 3),
            "file_paths_preserved": round(file_score, 3),
            "errors_preserved": round(error_score, 3),
            "gaps_preserved": round(gap_score, 3),
            "bridge_char_count": len(bridge_text),
            "sections_present": sections_present,
            "lost_items": lost_items[:10],  # Cap diagnostic list
        }

    def _empty_score(self, bridge_text: str) -> Dict[str, Any]:
        """Return zero-score result when no messages or bridge text."""
        sections_present = self._detect_sections(bridge_text) if bridge_text else []
        return {
            "overall": 0.0,
            "active_tasks_preserved": 0.0,
            "file_paths_preserved": 0.0,
            "errors_preserved": 0.0,
            "gaps_preserved": 0.0,
            "bridge_char_count": len(bridge_text) if bridge_text else 0,
            "sections_present": sections_present,
            "lost_items": [],
        }

    # -----------------------------------------------------------------------
    # Extraction helpers — mirror what ExtractionEngine does, but lightweight
    # -----------------------------------------------------------------------

    def _extract_task_summaries(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract task summary strings from user messages."""
        _TASK_KEYWORDS = frozenset({
            'fix', 'implement', 'create', 'build', 'add', 'refactor', 'debug',
            'write', 'update', 'migrate', 'deploy', 'configure', 'set up',
            'resolve', 'address', 'handle', 'support', 'enable', 'disable',
        })
        summaries = []
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if any(kw in content.lower() for kw in _TASK_KEYWORDS):
                first_line = content.split("\n")[0].strip()
                if len(first_line) > 10:
                    summaries.append(first_line[:120])
        return summaries[-5:]  # Most recent 5

    def _extract_file_paths(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract file paths from tool calls and text patterns."""
        paths = set()
        for msg in messages:
            content = (msg.get("content") or "") + "\n"

            # Structured tool_calls
            for tc in msg.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                if name not in ("write_file", "patch", "read_file"):
                    continue
                try:
                    import json as _json
                    args = _json.loads(args_str) if isinstance(args_str, str) else {}
                except (TypeError, ValueError):
                    continue
                path = args.get("path") or args.get("file_path", "")
                if path:
                    paths.add(path)

            # Text patterns — file paths with extensions
            for m in _re.finditer(r'(/[^\s\'"`\n]+?\.(?:py|md|yaml|json|txt|sh|cfg|ini))', content):
                p = m.group(1).rstrip(".,;:)")
                if len(p) > 5:
                    paths.add(p)

        return list(paths)[-10:]  # Most recent 10

    def _extract_error_summaries(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract error type strings from messages."""
        errors = []
        for msg in messages:
            content = (msg.get("content") or "") + "\n"
            for m in _re.finditer(
                r'(TypeError|ValueError|AttributeError|KeyError|ImportError|'
                r'ModuleNotFoundError|FileNotFoundError|PermissionError|'
                r'SyntaxError|IndexError|RuntimeError|OSError)',
                content,
            ):
                exc_type = m.group(1)
                # Grab context after the exception type
                start = m.end()
                end = min(start + 80, len(content))
                error_msg = content[start:end].strip().split("\n")[0]
                if error_msg:
                    errors.append(f"{exc_type}: {error_msg[:60]}")
        return list(dict.fromkeys(errors))[-5:]  # Dedup, keep recent

    def _extract_gap_summaries(self, messages: List[Dict[str, Any]]) -> List[str]:
        """Extract knowledge gap markers from messages."""
        gaps = []
        for msg in messages:
            content = (msg.get("content") or "") + "\n"
            for m in _re.finditer(
                r'(?:knowledge\s+gap|RL\s+(?:entry|page)|\[gap\]|'
                r'\[(?:pending|needs research)\])\s*[:—\-]?\s*(.*?)(?:\n|$)',
                content, _re.IGNORECASE,
            ):
                summary = (m.group(1) or "").strip()
                if len(summary) > 3:
                    gaps.append(summary[:80])
        return list(dict.fromkeys(gaps))[-5:]

    # -----------------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------------

    def _score_preservation(self, items: List[str], bridge_text: str) -> float:
        """Score how many extracted items appear in the bridge text.

        Uses substring matching with normalization (lowercase, whitespace).
        Returns 0.0-1.0 ratio.
        """
        if not items:
            return 1.0  # Nothing to preserve = perfect score

        preserved = 0
        bridge_lower = bridge_text.lower()

        for item in items:
            # Normalize: lowercase, collapse whitespace
            normalized = " ".join(item.lower().split())
            # For file paths, also check basename
            if "/" in item and "." in item.split("/")[-1]:
                parts = item.rsplit("/", 1)
                if len(parts) == 2:
                    basename = parts[1].lower()
                    if normalized in bridge_lower or basename in bridge_lower:
                        preserved += 1
                        continue

            # For task summaries, check partial match (first significant words)
            words = normalized.split()[:5]
            if len(words) >= 3 and " ".join(words) in bridge_lower:
                preserved += 1
            elif normalized in bridge_lower:
                preserved += 1
            else:
                # Check if key terms from the item appear near each other in bridge
                key_terms = [w for w in words if len(w) > 3]
                if len(key_terms) >= 2:
                    matches_in_bridge = sum(1 for t in key_terms if t in bridge_lower)
                    if matches_in_bridge >= len(key_terms) * 0.5:
                        preserved += 0.5  # Partial credit

        return min(preserved / len(items), 1.0)

    def _find_lost(self, items: List[str], bridge_text: str) -> List[str]:
        """Find specific items that were extracted but not in the bridge."""
        lost = []
        bridge_lower = bridge_text.lower()
        for item in items:
            normalized = " ".join(item.lower().split())
            if normalized not in bridge_lower:
                # Quick check — is a significant portion present?
                words = normalized.split()[:4]
                matches = sum(1 for w in words if len(w) > 3 and w in bridge_lower)
                if matches < len(words) * 0.5:
                    lost.append(item[:80])
        return lost

    def _detect_sections(self, bridge_text: str) -> List[str]:
        """Detect which standard sections are present in the bridge."""
        sections = []
        section_markers = {
            "active_tasks": ["## Active Tasks", "ACTIVE TASK"],
            "file_edits": ["## Files Currently Being Edited", "FILES CURRENTLY"],
            "known_errors": ["## Known Errors", "KNOWN ERRORS"],
            "knowledge_gaps": ["## Knowledge Gaps", "KNOWLEDGE GAPS"],
            "retrieval_guidance": ["## Historical Context Retrieval", "HISTORICAL CONTEXT"],
        }
        bridge_upper = bridge_text.upper()
        for section_name, markers in section_markers.items():
            if any(m.upper() in bridge_upper for m in markers):
                sections.append(section_name)
        return sections

"""Extraction Engine — Extract structured data from conversation messages.

Responsible for scanning message history and extracting:
- Active tasks (user requests with task-related language)
- File edits (write_file/patch/read_file calls)
- Known errors (exceptions with fix locations)
- Knowledge gaps (flagged topics needing research)

Each method is self-contained, degrades gracefully on failure,
and returns structured data for the Context Bridge builder.
"""

from __future__ import annotations

import json as _json
import os
import re as _re
from typing import Any, Dict, List, Tuple

# Common English stopwords for topic filtering (used by __init__.py auto-tagging)
_STOPWORDS = frozenset({
    'the', 'and', 'this', 'that', 'with', 'from', 'have', 'been', 'were', 'are',
    'was', 'for', 'not', 'but', 'what', 'all', 'their', 'which', 'would',
    'it', 'its', 'in', 'to', 'of', 'a', 'an', 'is', 'on', 'at', 'by',
})


class ExtractionEngine:
    """Extracts structured data from conversation message history.

    All methods take a list of message dicts and return structured results.
    Each method degrades gracefully — returns empty list on failure, never raises.
    """

    # -----------------------------------------------------------------------
    # Pre-compiled regex patterns (class-level constants — compiled once)
    # -----------------------------------------------------------------------

    # Decision patterns in assistant messages (substantive decisions, not just statements)
    _DECISION_PATTERNS = _re.compile(
        r'(?:decided|will use|architecture is|plan outlines|confirmed|agreed)'
        r'|(?:key design principle|must never|always check|mandatory)'
        r'|(?:instead of|rather than|chose to|opted for)',  # Decision trade-offs
        _re.IGNORECASE,
    )

    # Exception type patterns to catch
    _EXCEPTION_PATTERNS = _re.compile(
        r'(TypeError|ValueError|AttributeError|KeyError|ImportError|'\
        r'ModuleNotFoundError|FileNotFoundError|PermissionError|'\
        r'SyntaxError|IndexError|RuntimeError|OSError|IOError|'\
        r'JSONDecodeError|UnicodeDecodeError)',
    )

    # Fix-location patterns (case-insensitive)
    _FIX_PATTERNS = [
        _re.compile(r'(?:fix(?:ed)?\s+(?:in\s+)?[\'\"]?(/[^\'\"\n]+))', _re.IGNORECASE),
        _re.compile(r'(?:applied\s+to\s+[\'\"]?(/[^\'\"\n]+))', _re.IGNORECASE),
        _re.compile(r'(?:resolved\s+at\s+line\s+(\d+))', _re.IGNORECASE),
    ]

    # Consolidated file ops text patterns (single pass instead of three)
    _FILE_OPS_TEXT_PATTERN = _re.compile(
        r'(?:wrote|saved|created)\s+(?:to\s+)?[\'\"]?(/[^\'\\"\n]+)'  # wrote to /path
        r'|(?:patched|modified|updated)\s+[\'\"]?(/[^\'\\"\n]+)'       # patched /path
        r'|(?:reading|read)\s+[\'\"]?(/[^\'\\"\n]+)',                   # reading /path
        _re.IGNORECASE,
    )

    # -----------------------------------------------------------------------
    # Public extraction methods
    # -----------------------------------------------------------------------

    def extract_active_tasks(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts active tasks from recent messages with their turn IDs.

        Scans user messages for task-related language (fix/add/implement/etc.),
        deduplicates by first line of the request, and tracks which turns discussed each task.
        Also extracts key decisions made by the assistant during task discussions.

        Returns list of dicts: {'summary': str, 'turn_ids': list[int], 'description': str, 'decisions': list[dict]}
        """
        if not messages:
            return []

        # Task action keywords that signal a pending/active task
        _TASK_KEYWORDS = frozenset({
            'fix', 'implement', 'create', 'build', 'add', 'refactor', 'debug',
            'write', 'update', 'migrate', 'deploy', 'configure', 'set up',
            'resolve', 'address', 'handle', 'support', 'enable', 'disable',
        })

        # Dedup key -> first occurrence data (case-insensitive, whitespace-normalized)
        seen: Dict[str, Dict[str, Any]] = {}

        for idx in range(len(messages) - 1, -1, -1):  # Include index 0
            msg = messages[idx]
            if msg.get("role") != "user":
                continue

            content = (msg.get("content") or "").strip()
            if not content:
                continue

            content_lower = content.lower()
            if not any(kw in content_lower for kw in _TASK_KEYWORDS):
                continue

            # Use first line as dedup key (normalized)
            first_line = content.split("\n")[0].strip()
            dedup_key = " ".join(first_line.lower().split())[:120]

            if dedup_key not in seen:
                task_data = {
                    "summary": first_line,
                    "turn_ids": [idx],
                    "description": content[:300].replace("\n", " "),
                    "decisions": [],
                }

                # Look for key decisions in nearby assistant messages (next 5 turns)
                for next_idx in range(idx + 1, min(len(messages), idx + 6)):
                    next_msg = messages[next_idx]
                    if next_msg.get("role") != "assistant":
                        continue

                    decision_content = next_msg.get("content", "") or ""
                    lines = decision_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if len(line) < 20 or len(line) > 300:
                            continue

                        if self._DECISION_PATTERNS.search(line):
                            # Avoid duplicates and skip examples/pseudocode
                            if (line not in [d['text'] for d in task_data['decisions']] and
                                not any(kw in line.lower() for kw in ["example", "pseudocode", "todo"])):
                                task_data["decisions"].append({
                                    "turn_id": next_idx,
                                    "text": line[:200],  # Cap decision text length
                                })

                seen[dedup_key] = task_data
            else:
                seen[dedup_key]["turn_ids"].append(idx)

        # Return most recent tasks first (backwards iteration = most recent inserted first)
        return list(seen.values())[:5]

    def extract_file_edits(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts file edit history from recent messages.

        Detects write_file/patch/read_file calls from both structured tool_calls
        and text patterns (e.g., "wrote to /path/to/file"). Returns most recently
        edited files first, with related discussion turns identified by scanning
        for mentions of the same file path in surrounding context.

        Returns list of dicts: {'path': str, 'last_edit_turn': int, 'description': str,
                                'related_turns': list[int], 'related_description': str}
        """
        if not messages:
            return []

        _FILE_OPS = frozenset({"write_file", "patch", "read_file"})
        # path key -> data (normalized to lowercase for dedup)
        seen: Dict[str, Dict[str, Any]] = {}

        for idx in range(len(messages) - 1, -1, -1):  # Include index 0
            msg = messages[idx]
            content = (msg.get("content") or "") + "\n"

            # --- Structured tool_calls ---
            for tc in msg.get("tool_calls", []) or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")

                if name not in _FILE_OPS:
                    continue

                try:
                    args = _json.loads(args_str) if isinstance(args_str, str) else args_str
                except (_json.JSONDecodeError, TypeError):
                    continue

                path = args.get("path") or args.get("file_path", "")
                if not path:
                    continue

                norm_key = path.lower().split("?")[0]  # strip query params
                op_label = name.replace("_", " ").title()

                if norm_key not in seen:
                    # Find related discussion turns (mentions of this file in surrounding messages)
                    related_turns, related_desc = self.find_related_discussions(
                        messages, idx, path, window=10
                    )

                    seen[norm_key] = {
                        "path": path,
                        "last_edit_turn": idx,
                        "description": f"{op_label} call detected",
                        "related_turns": related_turns[:5],  # Cap at 5 related turns
                        "related_description": related_desc or "related discussion",
                    }
                else:
                    # Update description to include new operation type
                    existing_desc = seen[norm_key]["description"]
                    if op_label not in existing_desc:
                        seen[norm_key]["description"] = (
                            f"{existing_desc}, {op_label}"
                        )

            # --- Text patterns for file operations (single consolidated pass) ---
            for m in self._FILE_OPS_TEXT_PATTERN.finditer(content):
                # Get the matched path from whichever group captured
                path = next((g for g in m.groups() if g), None)
                if not path:
                    continue
                path = path.strip().rstrip(".,;")
                norm_key = path.lower().split("?")[0]
                if norm_key not in seen:
                    # Find related discussion turns
                    related_turns, related_desc = self.find_related_discussions(
                        messages, idx, path, window=10
                    )

                    seen[norm_key] = {
                        "path": path,
                        "last_edit_turn": idx,
                        "description": f"Text pattern match",
                        "related_turns": related_turns[:5],
                        "related_description": related_desc or "related discussion",
                    }

        # Return most recently edited first (backwards iteration = most recent inserted first into dict)
        return list(seen.values())[:10]

    def find_related_discussions(self, messages: List[Dict[str, Any]],
                                 edit_turn: int, file_path: str,
                                 window: int = 10) -> Tuple[List[int], str]:
        """Find related discussion turns for a file edit.

        Scans surrounding messages for mentions of the same file path.
        Returns (turn_ids_list, description_string).
        """
        # Normalize file path for matching (basename + extension)
        try:
            basename = os.path.basename(file_path)
            name_no_ext = os.path.splitext(basename)[0]
        except (TypeError, ValueError):
            return [], ""

        related_turns = []
        discussion_snippets = []

        # Scan window around the edit turn
        start_idx = max(0, edit_turn - window)
        end_idx = min(len(messages), edit_turn + window + 1)

        for idx in range(start_idx, end_idx):
            if idx == edit_turn:
                continue  # Skip the edit turn itself

            msg = messages[idx]
            content = (msg.get("content") or "").lower()

            # Check if this message mentions the file (by basename or full path)
            if (basename.lower() in content or
                file_path.lower().split("?")[0] in content):

                related_turns.append(idx)

                # Extract a snippet of what was discussed
                snippet = (msg.get("content") or "")[:150].replace("\n", " ")
                if snippet.strip():
                    discussion_snippets.append(snippet)

        # Build description from snippets
        related_desc = ""
        if discussion_snippets:
            # Combine unique snippets, limit total length
            seen_snippets = set()
            combined = []
            for s in discussion_snippets:
                if s not in seen_snippets and len(s) > 10:
                    seen_snippets.add(s)
                    combined.append(s)

            if combined:
                related_desc = " — ".join(combined[:2])  # Max 2 snippets
                if len(related_desc) > 200:
                    related_desc = related_desc[:197] + "..."

        return related_turns, related_desc

    def extract_known_errors(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts known errors and their resolutions.

        Finds TypeError/AttributeError/etc. with normalized deduplication
        (case-insensitive comparison prevents duplicates). Also captures
        fix locations mentioned in surrounding context.

        Returns list of dicts: {'summary': str, 'turn_id': int, 'fix_location': str}
        """
        if not messages:
            return []

        # Exception type patterns to catch (now at class level)
        _EXCEPTION_PATTERNS = [self._EXCEPTION_PATTERNS]

        # Fix-location patterns (now at class level)
        _FIX_COMPILED = self._FIX_PATTERNS

        seen: Dict[str, Dict[str, Any]] = {}

        for idx in range(len(messages) - 1, -1, -1):  # Include index 0
            msg = messages[idx]
            content = (msg.get("content") or "") + "\n"

            # Search for exception patterns
            for pat in _EXCEPTION_PATTERNS:
                for m in _re.finditer(pat, content):
                    exc_type = m.group(1)
                    # Extract surrounding context as summary (error message after the type)
                    start = m.end()
                    end = min(start + 200, len(content))
                    error_msg = content[start:end].strip().split("\n")[0]

                    if not error_msg:
                        continue

                    # Clean up error message: remove leading colons/spaces/traceback prefixes
                    error_msg = _re.sub(r'^[:\s]+', '', error_msg).strip()
                    # Remove common traceback artifacts
                    error_msg = _re.sub(r'^(in\s+\w+|File\s+"[^"]+",\s+line\s+\d+)', '', error_msg).strip()
                    if not error_msg:
                        continue

                    # Build normalized dedup key from exception type + first 80 chars of message
                    norm_key = f"{exc_type}:{error_msg[:80].lower()}"

                    if norm_key in seen:
                        continue

                    # Look for fix location in the same message or nearby messages
                    fix_location = "N/A"
                    context_window = content[max(0, start - 300):min(len(content), end + 500)]
                    for fix_pat in _FIX_COMPILED:
                        fm = fix_pat.search(context_window)
                        if fm:
                            fix_location = fm.group(1)
                            break

                    seen[norm_key] = {
                        "summary": f"{exc_type}: {error_msg}",
                        "turn_id": idx,
                        "fix_location": fix_location,
                    }

        # Return most recent errors first (backwards iteration = most recent inserted first)
        return list(seen.values())[:5]

    def extract_knowledge_gaps(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extracts flagged knowledge gaps.

        Uses three pattern formats to detect knowledge gap flags:
        1. Explicit markers: "knowledge gap", "RL entry needed", "[gap]"
        2. Confidence scores (0-1) paired with topic references
        3. Bracketed tags like [pending], [needs research]

        Returns list of dicts: {'summary': str, 'confidence': float, 'turn_ids': list[int]}
        """
        if not messages:
            return []

        # Pattern 1: Explicit knowledge gap markers
        _EXPLICIT_PATTERNS = [
            r'knowledge\s+gap(?:\s*:\s*(.*?))?(?:\n|$)',
            r'(?:RL|reference library)\s+(?:entry|page|topic)\s+(?:needed|required|missing)(?:\s*:\s*(.*?))?(?:\n|$)',
            r'\[gap\]\s*(.*?)(?:\n|$)',
        ]

        # Pattern 2: Confidence score + topic (e.g., "confidence: 0.3 — first principles of nuclear ethics")
        _CONFIDENCE_PATTERN = r'confidence:\s*([\d.]+)\s*(?:—|-|:)?\s*(.*?)(?:\n|$)'

        # Pattern 3: Bracketed pending tags
        _PENDING_PATTERNS = [
            r'\[(?:pending|needs research|TODO)\]\s*(.*?)(?:\n|$)',
        ]

        seen: Dict[str, Dict[str, Any]] = {}

        for idx in range(len(messages) - 1, -1, -1):  # Include index 0
            msg = messages[idx]
            content = (msg.get("content") or "") + "\n"

            # Pattern 1: Explicit markers
            for pat in _EXPLICIT_PATTERNS:
                for m in _re.finditer(pat, content, _re.IGNORECASE):
                    summary = (m.group(1) or "").strip()
                    if not summary:
                        continue

                    dedup_key = summary.lower().split()[0][:60] if summary else ""
                    if dedup_key and dedup_key not in seen:
                        seen[dedup_key] = {
                            "summary": summary,
                            "confidence": 0.5,
                            "turn_ids": [idx],
                        }

            # Pattern 2: Confidence score + topic
            for m in _re.finditer(_CONFIDENCE_PATTERN, content):
                try:
                    confidence = float(m.group(1))
                except (ValueError, TypeError):
                    continue

                if not (0.0 <= confidence <= 1.0):
                    continue

                summary = m.group(2).strip()
                if not summary or len(summary) < 3:
                    continue

                dedup_key = summary.lower().split()[0][:60]
                if dedup_key and dedup_key not in seen:
                    seen[dedup_key] = {
                        "summary": summary,
                        "confidence": confidence,
                        "turn_ids": [idx],
                    }

            # Pattern 3: Pending tags
            for pat in _PENDING_PATTERNS:
                for m in _re.finditer(pat, content):
                    summary = (m.group(1) or "").strip()
                    if not summary:
                        continue

                    dedup_key = summary.lower().split()[0][:60]
                    if dedup_key and dedup_key not in seen:
                        seen[dedup_key] = {
                            "summary": summary,
                            "confidence": 0.7,
                            "turn_ids": [idx],
                        }

        # Return most recent gaps first (backwards iteration = most recent inserted first into dict)
        return list(seen.values())[:5]

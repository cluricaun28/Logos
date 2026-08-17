"""Session End Extraction — topic extraction from conversation messages.

Extracted from ``PerpetualContextProvider.on_session_end()`` for SRP compliance.
This is the only place that should do regex-based topic extraction from raw
message content. The PerpetualContextProvider delegates here.
"""

from __future__ import annotations

import re as _re

from typing import Any

# Single-word junk filter (F3, 2026-08-17).
#
# The Pattern-1 regex captures any Capitalized word — including ordinary
# sentence starters ("Good", "Please", "These") and Python/JSON literals
# ("None", "True", "False") leaking out of tool output. Multi-word
# capitalized phrases are a strong topic signal and pass unfiltered;
# single-word candidates must additionally not be in this set.
#
# This set is the single source of truth: the historical prune of the
# topics table used exactly this list (see the F3 entry in the LOGOS
# full-system review register), so code and data stay consistent.
_JUNK_SINGLE_WORDS = frozenset({
    # conversation filler / sentence starters
    'good', 'please', 'these', 'those', 'each', 'every', 'found', 'want',
    'wanted', 'done', 'new', 'check', 'read', 'full', 'update', 'active',
    'system', 'user', 'file', 'command', 'search', 'test', 'phase', 'okay',
    'ok', 'yes', 'no', 'maybe', 'sure', 'right', 'now', 'here', 'then',
    'them', 'they', 'this', 'that', 'will', 'would', 'can', 'could',
    'should', 'have', 'has', 'had', 'been', 'being', 'just', 'also', 'very',
    'really', 'still', 'again', 'more', 'most', 'other', 'such', 'only',
    'same', 'like', 'need', 'needs', 'going', 'make', 'made', 'work',
    'works', 'working', 'run', 'runs', 'running', 'get', 'got', 'use',
    'used', 'using', 'let', 'lets', 'try', 'tried', 'look', 'looked',
    'see', 'saw', 'know', 'knew', 'think', 'thought', 'might', 'must',
    'many', 'much', 'few', 'some', 'any', 'all', 'both',
    # python / json literals that leaked from tool output
    'none', 'true', 'false', 'null',
    # status words from status lines
    'error', 'errors', 'warning', 'warnings', 'info', 'debug', 'success',
    'failed', 'pass', 'passing', 'completed', 'starting', 'stopped',
    # past-tense status / generic IT nouns (not topic-worthy alone)
    'current', 'added', 'updated', 'status', 'state', 'tool', 'tools',
    'message', 'messages', 'turn', 'turns', 'output', 'input', 'result',
    'results', 'data', 'path', 'port', 'page', 'pages', 'line', 'lines',
    'code', 'script', 'scripts', 'log', 'logs', 'table', 'tables',
    'query', 'queries', 'value', 'values', 'key', 'keys', 'count',
    'list', 'lists', 'item', 'items', 'record', 'records', 'entry',
    'entries', 'version', 'versions', 'branch', 'commit', 'commits',
    'push', 'pull', 'merge', 'deploy', 'deployment', 'client', 'server',
    'running', 'started', 'stop', 'restart', 'restarted', 'related',
    # past-tense verbs common in assistant summary openers
    'fixed', 'changes', 'change', 'changed',
})

# F28 (2026-08-17): exception/identifier names (TypeError, KeyError,
# ...) are code tokens leaking from tool output, not topics. Pattern 3
# (CamelCase) now captures them, so reject single words with an
# Error/Exception suffix.
_EXCEPTION_NAME_RE = _re.compile(r"^[A-Z][a-zA-Z]*(?:Error|Exception)$")


def extract_topics_from_messages(
    messages: list[dict[str, Any]],
    stopwords: frozenset[str],
    max_topics_per_message: int = 3,
) -> list[str]:
    """Extract meaningful topic names from conversation messages.

    Pattern 1: Capitalized phrases (e.g., "Python Programming", "Docker Networking")
    Pattern 2: Technical terms with file extensions (e.g., "perpetual_context.db")
    Pattern 3: CamelCase identifiers (e.g., "PerpetualContextDB", "SmartRetriever")

    Args:
        messages: List of message dicts with "content" key.
        stopwords: Frozenset of English stopwords to filter out.
        max_topics_per_message: Max topics to extract per message.

    Returns:
        List of unique topic name strings.
    """
    _TOPIC_PATTERN = _re.compile(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b'  # Capitalized phrases
        r'|([a-zA-Z_]+\.(?:py|md|yaml|json|txt|sh))\b'  # File references (F28: now capturing)
        r'|([A-Z][a-zA-Z]{2,}(?:[A-Z][a-z]+)+)\b',  # CamelCase identifiers (F28: now capturing)
    )

    seen: set[str] = set()

    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue

        topics = _TOPIC_PATTERN.findall(content)

        for t in topics:
            if isinstance(t, tuple):
                t = next((g for g in t if g), "")
            if len(t) > 3 and t.lower() not in stopwords:
                normalized = t.strip()
                # F3 (2026-08-17): single-word candidates are usually
                # sentence-initial filler, not topics. Multi-word
                # capitalized phrases, file references, and CamelCase
                # identifiers are unaffected.
                is_single_word = " " not in normalized
                if not (
                    is_single_word
                    and (
                        normalized.lower() in _JUNK_SINGLE_WORDS
                        or _EXCEPTION_NAME_RE.match(normalized)
                    )
                ):
                    if normalized not in seen:
                        seen.add(normalized)

        if len(seen) >= max_topics_per_message * len(messages):
            break

    return list(seen)

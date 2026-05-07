"""Session End Extraction — topic extraction from conversation messages.

Extracted from ``PerpetualContextProvider.on_session_end()`` for SRP compliance.
This is the only place that should do regex-based topic extraction from raw
message content. The PerpetualContextProvider delegates here.
"""

from __future__ import annotations

import re as _re

from typing import Any


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
        r'|(?:[a-zA-Z_]+\.(?:py|md|yaml|json|txt|sh))\b'  # File references
        r'|(?:[A-Z][a-zA-Z]{2,}(?:[A-Z][a-z]+)+)\b',  # CamelCase identifiers
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
                if normalized not in seen:
                    seen.add(normalized)

        if len(seen) >= max_topics_per_message * len(messages):
            break

    return list(seen)

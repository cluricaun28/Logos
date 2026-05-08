"""Sensitivity Classification — Topic sensitivity classification for scrutiny level selection.

Classifies queries/topics as 'low' or 'high' sensitivity to determine how
deeply incoming web-sourced data should be vetted before entering the
Reference Library or context injection.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DEFAULT_SENSITIVITY = "high"  # Default when classification is ambiguous


class TopicSensitivityClassifier:
    """Classifies query/topic sensitivity for scrutiny level selection."""

    # Low sensitivity — trust major sources, fast extraction, minimal scrutiny
    LOW_SENSITIVITY_KEYWORDS = frozenset(
        [
            "hardware",
            "code",
            "programming",
            "software",
            "mathematics",
            "python",
            "docker",
            "gpu",
            "cpu",
            "memory",
            "networking",
            "api",
            "database",
            "sql",
            "linux",
            "windows",
            "wsl",
            "build",
            "compile",
            "deploy",
            "test",
            "debug",
            "git",
            "github",
            "npm",
            "pip",
            "package",
            "dependency",
            "configuration",
            "settings",
            "environment",
            "variable",
        ]
    )

    # High sensitivity — activate deep scrutiny, identify bias/motives
    HIGH_SENSITIVITY_KEYWORDS = frozenset(
        [
            "history",
            "politics",
            "economics",
            "culture",
            "media",
            "government",
            "election",
            "policy",
            "law",
            "court",
            "religion",
            "faith",
            "christianity",
            "theology",
            "race",
            "gender",
            "identity",
            "social justice",
            "war",
            "military",
            "defense",
            "intelligence",
            "education",
            "science funding",
            "climate",
            "environment",
            "news",
            "journalism",
            "reporting",
            "opinion",
            "protest",
            "movement",
            "activism",
            "rights",
        ]
    )

    def classify(self, query: str) -> str:
        """Classify a query as 'low' or 'high' sensitivity.

        Returns 'high' if any high-sensitivity keywords match.
        Returns 'low' for technical/code queries.
        Defaults to 'high' for ambiguous cases — better safe than sorry.
        Gracefully handles non-string inputs by returning default sensitivity.
        """
        if not isinstance(query, str):
            return DEFAULT_SENSITIVITY
        if not query:
            return DEFAULT_SENSITIVITY

        query_lower = query.lower()
        words = set(query_lower.split())

        # Check high-sensitivity keywords first
        high_matches = words & self.HIGH_SENSITIVITY_KEYWORDS
        if high_matches:
            logger.debug("High sensitivity detected for '%s': %s", query[:50], high_matches)
            return "high"

        # Check low-sensitivity keywords
        low_matches = words & self.LOW_SENSITIVITY_KEYWORDS
        if low_matches and not high_matches:
            logger.debug("Low sensitivity detected for '%s': %s", query[:50], low_matches)
            return "low"

        # Ambiguous — default to high (better safe than sorry)
        return DEFAULT_SENSITIVITY

"""RL Ingestion Gate — Controls what web-sourced data is eligible for Reference Library updates.

Evaluates new web data against existing RL entries to detect contradictions,
assess source credibility, and determine whether data should be approved,
flagged for manual review, or rejected.
"""

from __future__ import annotations

import logging
import os
from typing import Any

try:
    from .extraction_engine import _STOPWORDS
except ImportError:
    from extraction_engine import _STOPWORDS

from .sensitivity_classifier import TopicSensitivityClassifier
from .source_assessment import _extract_domain, _get_source_stance

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
RL_CONTRADICTION_THRESHOLD = 3  # Min keyword overlap to flag contradiction


class RLIngestionGate:
    """Controls what web-sourced data is eligible for Reference Library updates."""

    def __init__(self, rl_dir: str = "~/.hermes/reference-library/") -> None:
        self._rl_dir = os.path.expanduser(rl_dir)

    def evaluate(self, new_data: dict[str, Any], existing_rl_path: str | None = None) -> dict[str, Any]:
        """Evaluate if new web data should be ingested into the Reference Library.

        Checks:
        1. Does it contradict existing RL entries? If so, flag for manual review.
        2. Is the source motive transparent or aligned with baseline?
        3. Has it passed scrutiny gate vetting?

        Returns structured decision dict with status, reason, contradictions,
        source_assessment, and scrutiny_passed fields.
        Gracefully handles None/non-dict inputs by returning rejection.
        """
        # Graceful degradation — handle non-dict data
        if not isinstance(new_data, dict):
            return {
                "status": "rejected",
                "reason": "Invalid data format (expected dict)",
                "contradictions": [],
                "source_assessment": {"domain": None, "stance": "unknown", "credibility": 0.0},
                "scrutiny_passed": False,
            }

        issues: list[str] = []
        contradictions: list[str] = []
        scrutiny_passed = bool(new_data.get("_scrutiny_complete"))

        # Check if data has passed scrutiny
        if not scrutiny_passed:
            issues.append("Data has not passed scrutiny gate vetting")

        # Check for contradictions with existing RL page
        if existing_rl_path and os.path.exists(existing_rl_path):
            try:
                contradiction = self._check_contradiction(new_data, existing_rl_path)
                if contradiction:
                    contradictions.append(contradiction)
                    issues.append(f"Potential contradiction with existing page: {contradiction}")
            except (OSError, KeyError, TypeError, AttributeError) as e:
                logger.debug("Contradiction check failed: %s", e)

        # Source assessment — extract domain and stance
        url = new_data.get("url", "")
        domain = _extract_domain(url) if url else None
        stance = _get_source_stance(domain) if domain else "unknown"
        source_assessment = {
            "domain": domain,
            "stance": stance,
            "credibility": 0.5 if stance == "centrist" else 0.3,
        }

        # Check source transparency
        source = new_data.get("source", "unknown")
        if source == "unknown" and not url:
            issues.append("Source is unknown — cannot verify credibility")

        # Make decision
        if len(issues) >= 2 or contradictions:
            status = "manual_review"
        elif issues:
            status = "approved_with_notes"
        else:
            status = "approved"

        return {
            "status": status,
            "reason": "; ".join(issues) if issues else "No issues detected",
            "contradictions": contradictions,
            "source_assessment": source_assessment,
            "scrutiny_passed": scrutiny_passed,
        }

    def _check_contradiction(self, new_data: dict[str, Any], rl_path: str) -> str | None:
        """Check if new data contradicts existing RL page content.

        Uses two strategies:
        1. High-sensitivity keyword overlap (existing logic)
        2. General topic word overlap — if many words appear in both texts, flag for review
        """
        try:
            with open(rl_path, encoding="utf-8") as f:
                existing_content = f.read()

            # Handle JSON-formatted RL files (common in tests)
            import json as _json

            try:
                parsed = _json.loads(existing_content)
                if isinstance(parsed, dict):
                    # Flatten JSON to text for comparison
                    existing_content = " ".join(str(v) for v in parsed.values() if isinstance(v, (str, list)))
                    if isinstance(parsed, dict) and "entries" in parsed:
                        existing_content = " ".join(entry.get("content", "") for entry in parsed["entries"])
            except (_json.JSONDecodeError, TypeError):
                pass  # Not JSON, use raw content

            # Extract key claims from new data
            new_text = (new_data.get("snippet") or new_data.get("content") or "").lower()
            existing_lower = existing_content.lower()

            if not new_text:
                return None

            # Strategy 1: High-sensitivity keyword overlap
            new_words = set(new_text.split())
            high_sensitivity_words = TopicSensitivityClassifier.HIGH_SENSITIVITY_KEYWORDS
            relevant_words = new_words & high_sensitivity_words

            if len(relevant_words) >= RL_CONTRADICTION_THRESHOLD:
                for word in relevant_words:
                    if word in existing_lower:
                        return f"Topic '{word}' covered differently in existing page"

            # Strategy 2: General topic overlap — significant shared vocabulary suggests same topic
            existing_words = set(existing_lower.split())
            content_words_new = new_words - _STOPWORDS
            content_words_existing = existing_words - _STOPWORDS
            shared = content_words_new & content_words_existing

            # If 5+ meaningful words overlap, likely same topic — flag for review
            if len(shared) >= 5:
                top_shared = list(shared)[:3]
                return f"Significant topic overlap ({', '.join(top_shared)}) — verify consistency"

        except (OSError, KeyError, TypeError, AttributeError) as e:
            logger.debug("Contradiction check failed for %s: %s", rl_path, e)

        return None

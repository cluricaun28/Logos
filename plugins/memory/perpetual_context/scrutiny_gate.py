"""Scrutiny Gate — Phase 3 of Deep Research & Continuity Engine.

Classifies topic sensitivity, detects bias/motives in web-sourced data,
and applies worldview baseline filtering before data enters the Reference
Library or context injection.

This module is a thin facade that delegates to specialized submodules:
- sensitivity_classifier: Topic sensitivity classification
- bias_detector: Keyword/regex/semantic bias detection
- source_assessment: Source credibility and domain extraction
- worldview_checker: Worldview divergence checking
- rl_ingestion_gate: Reference Library ingestion eligibility

All operations degrade gracefully — returns neutral values rather than raising.
"""

from __future__ import annotations

import logging
from typing import Any

from .bias_detector import (
    BIAS_CONFIDENCE_THRESHOLD,
    _SemanticMarkerDetector,
    detect_bias,
)
from .rl_ingestion_gate import RL_CONTRADICTION_THRESHOLD, RLIngestionGate
from .sensitivity_classifier import DEFAULT_SENSITIVITY, TopicSensitivityClassifier
from .source_assessment import (
    _extract_domain,
    _get_dossiers,
    _get_source_info,
    _get_source_stance,
)
from .worldview_checker import (
    WorldviewDivergenceChecker,
    _load_worldview_profile,
)

logger = logging.getLogger(__name__)

# Public API — maintain backward compatibility for existing imports
__all__: list[str] = [
    "BIAS_CONFIDENCE_THRESHOLD",
    "DEFAULT_SENSITIVITY",
    "RL_CONTRADICTION_THRESHOLD",
    "RLIngestionGate",
    "ScrutinyGate",
    "TopicSensitivityClassifier",
    "_SemanticMarkerDetector",
    "_extract_domain",
    "_get_dossiers",
    "_get_source_info",
    "_get_source_stance",
    "WorldviewDivergenceChecker",
]


class ScrutinyGate:
    """Applies scrutiny to web-sourced data based on topic sensitivity.

    Orchestrates sensitivity classification, bias detection, and worldview
    filtering. For low-sensitivity topics: basic source validation only.
    For high-sensitivity topics: full bias analysis + motive identification.
    """

    def __init__(self) -> None:
        self._classifier = TopicSensitivityClassifier()
        self._marker_detector = _SemanticMarkerDetector()
        profile = _load_worldview_profile()
        self._divergence_checker = WorldviewDivergenceChecker(profile)

    def vet_results(self, results: list[dict[str, Any]], query: str) -> dict[str, Any]:
        """Vet search results through appropriate scrutiny level.

        Returns dict with:
        - 'vetted_results': Results that passed scrutiny (with annotations)
        - 'rejected_results': Results that failed (with reason)
        - 'sensitivity_level': 'low' or 'high'
        - 'warnings': Any concerns flagged during vetting

        Gracefully handles bad inputs — returns empty results rather than raising.
        """
        sensitivity = self._classifier.classify(query)
        warnings: list[str] = []
        vetted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        # Graceful degradation — handle non-list inputs
        if not isinstance(results, list):
            logger.debug("vet_results received non-list input: %s", type(results).__name__)
            warnings.append(f"Invalid results format (expected list, got {type(results).__name__})")
            return {
                "vetted_results": vetted,
                "rejected_results": rejected,
                "sensitivity_level": sensitivity,
                "warnings": warnings,
            }

        # Empty input warning
        if not results:
            warnings.append("No results to vet")
            return {
                "vetted_results": vetted,
                "rejected_results": rejected,
                "sensitivity_level": sensitivity,
                "warnings": warnings,
            }

        for result in results:
            try:
                # Skip non-dict items gracefully
                if not isinstance(result, dict):
                    logger.debug("Skipping non-dict result item: %s", type(result).__name__)
                    rejected.append(
                        {
                            "rejection_reason": f"Invalid result format (expected dict, got {type(result).__name__})",
                        }
                    )
                    continue

                # Basic validation — must have URL and content
                url = result.get("url", "")
                snippet = result.get("snippet") or result.get("content", "")

                if not url or not snippet:
                    rejected.append({**result, "rejection_reason": "Missing URL or content"})
                    continue

                # Apply scrutiny based on sensitivity level
                bias_report = self.detect_bias(snippet, url) if sensitivity == "high" else {}

                # Annotate result with scrutiny metadata
                annotated = {
                    **result,
                    "_scrutiny_level": sensitivity,
                    "_bias_notes": bias_report.get("notes", []),
                    "_source_stance": _get_source_stance(_extract_domain(url)),
                    "_confidence": self._estimate_confidence(result, sensitivity),
                }

                # Check if result should be rejected (very low confidence + high bias)
                if annotated["_confidence"] < 0.2 and sensitivity == "high":
                    rejected.append({**annotated, "rejection_reason": "Low confidence with detected bias"})
                    continue

                vetted.append(annotated)

            except Exception as e:
                logger.debug("Vetting failed for result: %s", e)
                safe_result = result if isinstance(result, dict) else {}
                rejected.append({**safe_result, "rejection_reason": f"Vetting error: {e}"})

        if sensitivity == "high" and len(vetted) < len(results):
            warnings.append(f"High-sensitivity query: {len(rejected)} of {len(results)} results flagged or rejected")

        return {
            "vetted_results": vetted,
            "rejected_results": rejected,
            "sensitivity_level": sensitivity,
            "warnings": warnings,
        }

    def detect_bias(self, text: str, source_url: str = "") -> dict[str, Any]:
        """Analyze text for potential bias indicators.

        Checks for:
        - Loaded language (emotive words, value-laden terms)
        - Semantic ideological markers (embedding-based similarity detection)
        - One-sided framing indicators
        - Source credibility signals

        Returns bias report with confidence scores and notes.
        Gracefully handles None/non-string inputs by returning neutral values.
        """
        return detect_bias(text, source_url, _marker_detector=self._marker_detector)

    def apply_worldview_filter(self, facts: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        """Apply worldview baseline filter to vetted facts.

        Does NOT alter or censor facts — only annotates them with scrutiny metadata:
        - scrutiny_level: How deeply this was vetted
        - source_bias_note: Any identified bias in the source
        - confidence: Confidence score for the fact's accuracy
        """
        sensitivity = self._classifier.classify(query)

        annotated_facts = []
        for fact in facts:
            # Add worldview alignment annotation
            stance = fact.get("_source_stance", "unknown")

            worldview_note = ""
            if sensitivity == "high" and stance not in ("centrist", "unknown"):
                worldview_note = f"Source has {stance} editorial stance — verify against multiple sources"

            annotated_facts.append(
                {
                    **fact,
                    "_worldview_note": worldview_note,
                    "_scrutiny_complete": True,
                }
            )

        return annotated_facts

    def _estimate_confidence(self, result: dict[str, Any], sensitivity: str) -> float:
        """Estimate confidence score for a search result."""
        base_score = result.get("score", 0.5)

        # Normalize to 0-1 range if needed
        normalized = (
            (min(max(base_score / 100.0, 0), 1.0) if base_score > 1 else base_score)
            if isinstance(base_score, (int, float))  # noqa: SIM102
            else 0.5
        )

        # Reduce confidence for high-sensitivity topics with unknown sources
        if sensitivity == "high" and not result.get("_source_stance"):
            normalized *= 0.8

        return round(normalized, 2)

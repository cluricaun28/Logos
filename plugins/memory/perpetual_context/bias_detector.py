"""Bias Detection — Keyword, regex, and semantic ideological marker detection.

Analyzes text for potential bias indicators including loaded language,
value-laden terms, semantic ideological markers (embedding-based), and
source credibility signals.
"""

from __future__ import annotations

import logging
import re as _re
from typing import Any

from .source_assessment import _extract_domain, _get_source_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
BIAS_CONFIDENCE_THRESHOLD = 0.5  # Below this, bias flag is informational only

# Pre-compiled regex patterns (class-level, not per-call) — retained as fallback
_LOADED_LANGUAGE_PATTERN = _re.compile(
    r"\b(allegedly|so-called|supposedly|infamously|notoriously)\b",
    _re.IGNORECASE,
)
_VALUE_LADEN_PATTERN = _re.compile(
    r"\b(toxic|woke|fascist|socialist|liberal|conservative|radical|extremist)\b",
    _re.IGNORECASE,
)


class _SemanticMarkerDetector:
    """Detect ideological markers via embedding similarity.

    Instead of pure keyword/regex matching, compares input text against
    representative example phrases for each marker category using the
    MiniLM embedding model. More robust to paraphrasing and euphemism.

    Keeps the old regex method as fallback when embedding model unavailable.
    """

    # Representative phrases for each marker category — used as embedding anchors
    _MARKER_PROFILES: dict[str, list[str]] = {
        "loaded_language": [
            "allegedly committed the crime",
            "so-called expert",
            "supposedly independent analysis",
            "infamously known for corruption",
            "notoriously biased reporting",
        ],
        "value_laden": [
            "toxic cultural influence",
            "woke ideology spreading",
            "fascist authoritarian policies",
            "socialist economic program",
            "radical extremist agenda",
        ],
        "passive_voice_framing": [
            "mistakes were made by the administration",
            "it was decided that changes would occur",
            "actions were taken against the protesters",
            "errors were discovered in the report",
        ],
        "one_sided_framing": [
            "everyone agrees that this is wrong",
            "nobody could possibly support this",
            "the only reasonable conclusion is",
            "all experts confirm without doubt",
        ],
        "moral_positioning": [
            "this is a moral imperative for our society",
            "we have an ethical duty to act",
            "morally reprehensible behavior",
            "the righteous path forward",
        ],
    }

    SIMILARITY_THRESHOLD = 0.65  # Cosine similarity above this = marker detected

    def __init__(self) -> None:
        self._anchors: dict[str, list[list[float]]] | None = None

    def _build_anchors(self) -> dict[str, list[list[float]]] | None:
        """Pre-compute embeddings for all marker profile phrases."""
        from agent.perpetual_context_db import EmbeddingEngine

        engine = EmbeddingEngine.get()
        anchors: dict[str, list[list[float]]] = {}

        for category, phrases in self._MARKER_PROFILES.items():
            category_anchors: list[list[float]] = []
            for phrase in phrases:
                vec = engine.embed(phrase)
                if vec is not None:
                    category_anchors.append(vec)
            if category_anchors:
                anchors[category] = category_anchors

        return anchors if anchors else None

    def detect(self, text: str) -> dict[str, Any]:
        """Analyze text for ideological markers via semantic similarity.

        Returns dict with:
          - detected_markers: list of category names that matched
          - scores: dict mapping category -> max similarity score
          - notes: human-readable descriptions of what was found
          - method: 'semantic' or 'regex' (fallback)
        """
        if not isinstance(text, str) or not text.strip():
            return {"detected_markers": [], "scores": {}, "notes": [], "method": "none"}

        # Try semantic detection first
        if self._anchors is None:
            self._anchors = self._build_anchors()

        if self._anchors:
            return self._detect_semantic(text)
        else:
            # Fallback to regex
            return self._detect_regex(text)

    def _detect_semantic(self, text: str) -> dict[str, Any]:
        """Embed the input text and compare against all anchor profiles."""
        from agent.perpetual_context_db import EmbeddingEngine

        engine = EmbeddingEngine.get()
        text_vec = engine.embed(text[:5000])  # Cap to avoid excessive compute

        if text_vec is None:
            return self._detect_regex(text)

        scores: dict[str, float] = {}
        detected: list[str] = []
        notes: list[str] = []

        for category, anchors in self._anchors.items():
            max_sim = max(EmbeddingEngine.cosine_similarity(text_vec, anchor) for anchor in anchors)
            scores[category] = round(max_sim, 3)

            if max_sim >= self.SIMILARITY_THRESHOLD:
                detected.append(category)
                notes.append(f"Semantic {category.replace('_', ' ')} detected (similarity: {max_sim:.2f})")

        return {
            "detected_markers": detected,
            "scores": scores,
            "notes": notes,
            "method": "semantic",
        }

    def _detect_regex(self, text: str) -> dict[str, Any]:
        """Regex-based fallback when embedding model unavailable."""
        detected: list[str] = []
        notes: list[str] = []

        loaded = _LOADED_LANGUAGE_PATTERN.findall(text)
        if loaded:
            detected.append("loaded_language")
            notes.append(f"Loaded language: {', '.join(loaded)}")

        value = _VALUE_LADEN_PATTERN.findall(text)
        if value:
            detected.append("value_laden")
            notes.append(f"Value-laden terms: {', '.join(value)}")

        passive_phrases = ["mistakes were made", "it was decided", "actions were taken"]
        text_lower = text.lower()
        passive_matches = [p for p in passive_phrases if p in text_lower]
        if passive_matches:
            detected.append("passive_voice_framing")
            notes.append(f"Passive voice framing: {', '.join(passive_matches)}")

        return {
            "detected_markers": detected,
            "scores": {},
            "notes": notes,
            "method": "regex",
        }


def detect_bias(text: str, source_url: str = "", _marker_detector: _SemanticMarkerDetector | None = None) -> dict[str, Any]:
    """Analyze text for potential bias indicators.

    Checks for:
    - Loaded language (emotive words, value-laden terms)
    - Semantic ideological markers (embedding-based similarity detection)
    - One-sided framing indicators
    - Source credibility signals

    Returns bias report with confidence scores and notes.
    Gracefully handles None/non-string inputs by returning neutral values.

    The _marker_detector parameter is for dependency injection — the ScrutinyGate
    passes its own instance. Callers without one will get a fresh detector.
    """
    notes: list[str] = []
    bias_score = 0.0

    # Graceful degradation — handle non-string text
    if not isinstance(text, str):
        return {"bias_score": 0.0, "notes": [], "confidence_above_threshold": False}

    detector = _marker_detector or _SemanticMarkerDetector()

    # Layer 1: Keyword/regex detection
    loaded_matches = _LOADED_LANGUAGE_PATTERN.findall(text)
    if loaded_matches:
        notes.append(f"Loaded language detected: {', '.join(loaded_matches)}")
        bias_score += 0.2 * len(loaded_matches)

    value_matches = _VALUE_LADEN_PATTERN.findall(text)
    if value_matches:
        notes.append(f"Value-laden terms detected: {', '.join(value_matches)}")
        bias_score += 0.15 * len(value_matches)

    # Layer 2: Semantic marker detection
    semantic_result = detector.detect(text)
    if semantic_result["method"] == "semantic":
        for marker_note in semantic_result["notes"]:
            notes.append(marker_note)
        for marker in semantic_result["detected_markers"]:
            if marker not in ("loaded_language", "value_laden"):  # Already counted above
                bias_score += 0.1
    else:
        for marker_note in semantic_result["notes"]:
            notes.append(marker_note)
        bias_score += 0.05 * len(semantic_result["detected_markers"])

    # Check source stance
    domain = _extract_domain(source_url)
    source_info = _get_source_info(domain)
    stance = source_info.get("stance", "unknown")
    if stance and stance not in ("centrist", "unknown"):
        notes.append(f"Source editorial stance: {stance}")
        bias_score += 0.1

    # Cap bias score at 1.0
    bias_score = min(bias_score, 1.0)

    return {
        "bias_score": round(bias_score, 2),
        "notes": notes,
        "confidence_above_threshold": bias_score >= BIAS_CONFIDENCE_THRESHOLD,
        "method": semantic_result.get("method", "regex"),
    }

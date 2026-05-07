"""Scrutiny Gate — Phase 3 of Deep Research & Continuity Engine.

Classifies topic sensitivity, detects bias/motives in web-sourced data,
and applies worldview baseline filtering before data enters the Reference
Library or context injection.

Architecture:
  - TopicSensitivityClassifier: Low vs high sensitivity classification
  - ScrutinyGate: Bias detection + motive identification + worldview filter
  - RLIngestionGate: Controls what web data is eligible for RL updates

All operations degrade gracefully — returns neutral values rather than raising.
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
from pathlib import Path
from typing import Any

from .extraction_engine import _STOPWORDS


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
DEFAULT_SENSITIVITY = "high"       # Default when classification is ambiguous
BIAS_CONFIDENCE_THRESHOLD = 0.5    # Below this, bias flag is informational only
RL_CONTRADICTION_THRESHOLD = 3     # Min keyword overlap to flag contradiction

# Pre-compiled regex patterns (class-level, not per-call)
_LOADED_LANGUAGE_PATTERN = _re.compile(
    r'\b(allegedly|so-called|supposedly|infamously|notoriously)\b',
    _re.IGNORECASE,
)
_VALUE_LADEN_PATTERN = _re.compile(
    r'\b(toxic|woke|fascist|socialist|liberal|conservative|radical|extremist)\b',
    _re.IGNORECASE,
)


class TopicSensitivityClassifier:
    """Classifies query/topic sensitivity for scrutiny level selection."""

    # Low sensitivity — trust major sources, fast extraction, minimal scrutiny
    LOW_SENSITIVITY_KEYWORDS = frozenset([
        "hardware", "code", "programming", "software", "mathematics",
        "python", "docker", "gpu", "cpu", "memory", "networking",
        "api", "database", "sql", "linux", "windows", "wsl",
        "build", "compile", "deploy", "test", "debug",
        "git", "github", "npm", "pip", "package", "dependency",
        "configuration", "settings", "environment", "variable",
    ])

    # High sensitivity — activate deep scrutiny, identify bias/motives
    HIGH_SENSITIVITY_KEYWORDS = frozenset([
        "history", "politics", "economics", "culture", "media",
        "government", "election", "policy", "law", "court",
        "religion", "faith", "christianity", "theology",
        "race", "gender", "identity", "social justice",
        "war", "military", "defense", "intelligence",
        "education", "science funding", "climate", "environment",
        "news", "journalism", "reporting", "opinion",
        "protest", "movement", "activism", "rights",
    ])

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


# Known source editorial stances — simple mapping for bias detection
# Values are rough left-right positioning: left, center-left, centrist,
# center-right, right, or religious/traditional
KNOWN_SOURCE_STANCES: dict[str, str] = {
    "nytimes.com": "center-left",
    "washingtonpost.com": "center-left",
    "reuters.com": "centrist",
    "apnews.com": "centrist",
    "bbc.co.uk": "center-left",
    "cnn.com": "left-leaning",
    "foxnews.com": "center-right",
    "wsj.com": "center-right",
    "npr.org": "center-left",
    "thehill.com": "centrist",
    "politico.com": "center-left",
    "breitbart.com": "right-leaning",
    "newsmax.com": "right-leaning",
    "dailywire.com": "right-leaning",
    "nationalreview.com": "center-right",
    "thebulwark.com": "center-right",
    "thegatewaypundit.com": "right-leaning",
    "vox.com": "left-leaning",
    "motherjones.com": "left-leaning",
    "huffpost.com": "left-leaning",
    "medium.com": "unknown",
    "reddit.com": "unknown",
    "wikipedia.org": "unknown",
    "theatlantic.com": "center-left",
    "theamericanconservative.com": "right-leaning",
    "washingtonexaminer.com": "center-right",
    "cnn.com": "left-leaning",
    "aljazeera.com": "center-left",
    "guardian.co.uk": "left-leaning",
    "telegraph.co.uk": "center-right",
    "economist.com": "centrist",
    "quillette.com": "centrist",
    "pjmedia.com": "right-leaning",
    "crevdonline.com": "religious/traditional",
    "christianpost.com": "religious/traditional",
    "churchmilitant.com": "religious/traditional",
    "prnewswire.com": "institutional",
    "forbes.com": "centrist",
    "bloomberg.com": "centrist",
    "marketwatch.com": "centrist",
}


def _extract_domain(url: str) -> str | None:
    """Extract the domain from a URL string.

    Returns 'nytimes.com' from 'https://www.nytimes.com/article'.
    Returns None for empty strings or invalid URLs.
    """
    if not url:
        return None
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        return domain if domain else None
    except Exception as e:
        logger.debug("Failed to extract domain from '%s': %s", url, e)
        return None


def _get_source_stance(domain: str | None) -> str:
    """Look up editorial stance for a known source domain.

    Returns stance string like 'center-left', 'right-leaning', 'institutional', etc.
    Falls back to 'unknown' for unrecognized domains.
    Handles .gov (institutional) and .edu (academic) TLDs automatically.
    """
    if not domain:
        return "unknown"

    # Institutional TLDs — these are authoritative but have their own biases
    if domain.endswith(".gov"):
        return "institutional"
    if domain.endswith(".edu"):
        return "academic"

    for known_domain, stance in KNOWN_SOURCE_STANCES.items():
        if known_domain in domain:
            return stance

    return "unknown"


class ScrutinyGate:
    """Applies scrutiny to web-sourced data based on topic sensitivity.

    For low-sensitivity topics: basic source validation only.
    For high-sensitivity topics: full bias analysis + motive identification.
    """

    def __init__(self) -> None:
        self._classifier = TopicSensitivityClassifier()
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
                    rejected.append({
                        "rejection_reason": f"Invalid result format (expected dict, got {type(result).__name__})",
                    })
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
            warnings.append(
                f"High-sensitivity query: {len(rejected)} of {len(results)} results flagged or rejected"
            )

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
        - One-sided framing indicators
        - Source credibility signals

        Returns bias report with confidence scores and notes.
        Gracefully handles None/non-string inputs by returning neutral values.
        """
        notes: list[str] = []
        bias_score = 0.0

        # Graceful degradation — handle non-string text
        if not isinstance(text, str):
            return {"bias_score": 0.0, "notes": [], "confidence_above_threshold": False}

        # Check for loaded language
        loaded_matches = _LOADED_LANGUAGE_PATTERN.findall(text)
        if loaded_matches:
            notes.append(f"Loaded language detected: {', '.join(loaded_matches)}")
            bias_score += 0.2 * len(loaded_matches)

        # Check for value-laden terms
        value_matches = _VALUE_LADEN_PATTERN.findall(text)
        if value_matches:
            notes.append(f"Value-laden terms detected: {', '.join(value_matches)}")
            bias_score += 0.15 * len(value_matches)

        # Check source stance
        stance = _get_source_stance(_extract_domain(source_url))
        if stance and stance not in ("centrist", "unknown"):
            notes.append(f"Source editorial stance: {stance}")
            bias_score += 0.1

        # Check for one-sided framing indicators
        passive_indicators = ["mistakes were made", "it was decided", "actions were taken"]
        text_lower = text.lower()
        for indicator in passive_indicators:
            if indicator in text_lower:
                notes.append(f"Passive voice framing detected: '{indicator}'")
                bias_score += 0.15

        # Cap bias score at 1.0
        bias_score = min(bias_score, 1.0)

        return {
            "bias_score": round(bias_score, 2),
            "notes": notes,
            "confidence_above_threshold": bias_score >= BIAS_CONFIDENCE_THRESHOLD,
        }

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
            bias_notes = fact.get("_bias_notes", [])

            worldview_note = ""
            if sensitivity == "high" and stance not in ("centrist", "unknown"):
                worldview_note = f"Source has {stance} editorial stance — verify against multiple sources"

            annotated_facts.append({
                **fact,
                "_worldview_note": worldview_note,
                "_scrutiny_complete": True,
            })

        return annotated_facts

    def _estimate_confidence(self, result: dict[str, Any], sensitivity: str) -> float:
        """Estimate confidence score for a search result."""
        base_score = result.get("score", 0.5)

        # Normalize to 0-1 range if needed
        if isinstance(base_score, (int, float)):
            normalized = min(max(base_score / 100.0, 0), 1.0) if base_score > 1 else base_score
        else:
            normalized = 0.5

        # Reduce confidence for high-sensitivity topics with unknown sources
        if sensitivity == "high" and not result.get("_source_stance"):
            normalized *= 0.8

        return round(normalized, 2)


class RLIngestionGate:
    """Controls what web-sourced data is eligible for Reference Library updates."""

    def __init__(self, rl_dir: str = "~/.hermes/reference-library/") -> None:
        self._rl_dir = Path(os.path.expanduser(rl_dir))

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
            except Exception as e:
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
            with open(rl_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

            # Handle JSON-formatted RL files (common in tests)
            import json as _json
            try:
                parsed = _json.loads(existing_content)
                if isinstance(parsed, dict):
                    # Flatten JSON to text for comparison
                    existing_content = " ".join(
                        str(v) for v in parsed.values() if isinstance(v, (str, list))
                    )
                    if isinstance(parsed, dict) and "entries" in parsed:
                        existing_content = " ".join(
                            entry.get("content", "") for entry in parsed["entries"]
                        )
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

        except Exception as e:
            logger.debug("Contradiction check failed for %s: %s", rl_path, e)

        return None


# ---------------------------------------------------------------------------
# Worldview Profile Loading
# ---------------------------------------------------------------------------

def _load_worldview_profile() -> dict[str, int | None]:
    """Load the user's worldview profile from config.yaml.

    Returns dict mapping axis names to scores (-2 to +2), or None if not set.
    Gracefully handles missing config, missing key, or malformed data.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        profile = cfg.get("worldview_profile")
        if not profile:
            return None
        if isinstance(profile, dict):
            # Validate: all values must be ints in [-2, +2]
            valid = {}
            for k, v in profile.items():
                if isinstance(v, int) and -2 <= v <= 2:
                    valid[k] = v
                else:
                    logger.warning(
                        "Invalid worldview profile value for %s: %s (must be int -2..+2)",
                        k, v,
                    )
            return valid if valid else None
    except Exception as e:
        logger.debug("Failed to load worldview profile: %s", e)
    return None


# Axis-to-topic mapping: which scrutiny axes correspond to which keywords
AXIS_TOPIC_MAP = {
    "sexual_morality": ("marriage", "same-sex", "gender", "lgbt", "trans", "adoption"),
    "racial_policy": ("race", "affirmative action", "dei", "discrimination",
                      "systemic racism", "quota"),
    "free_speech": ("free speech", "cancel culture", "censorship", "hate speech",
                    "employer"),
    "government_power": ("constitution", "federal", "free market", "regulation",
                         "ubiquitous", "basic income"),
    "sanctity_of_life": ("abortion", "conception", "pro-life", "pro-choice"),
    "religion_in_public": ("religion", "church", "prayer", "education",
                           "anti-discrimination"),
    "border_immigration": ("border", "immigration", "citizenship", "illegal"),
    "criminal_justice": ("death penalty", "guns", "second amendment",
                         "police", "criminal"),
}


class WorldviewDivergenceChecker:
    """Compare source content against the user's worldview profile.

    For high-sensitivity topics, checks whether the source's position on a
    relevant axis diverges significantly from the user's stated position.
    If so, annotates the result with a divergence note.

    Does NOT reject results — only annotates them so the user can judge.
    """

    DIVERGENCE_THRESHOLD = 2  # Flag if divergence >= 2 points

    def __init__(self, profile: dict[str, int | None]) -> None:
        self._profile = profile or {}

    def check_divergence(
        self,
        result: dict[str, Any],
        query: str,
    ) -> list[str]:
        """Check if this result's topic diverges from the user's worldview.

        Returns list of divergence notes (empty if no significant divergence).
        """
        if not self._profile:
            return []

        notes: list[str] = []
        query_lower = query.lower() if isinstance(query, str) else ""
        snippet = (result.get("snippet") or result.get("content") or "").lower()
        text_to_check = query_lower + " " + snippet

        for axis, keywords in AXIS_TOPIC_MAP.items():
            user_score = self._profile.get(axis)
            if user_score is None:
                continue

            # Check if this result is about this axis
            relevant = any(kw in text_to_check for kw in keywords)
            if not relevant:
                continue

            # Estimate source position on this axis from source stance
            source_stance = result.get("_source_stance", "unknown")
            estimated = self._estimate_source_position(axis, source_stance)

            divergence = abs(user_score - estimated)
            if divergence >= self.DIVERGENCE_THRESHOLD:
                direction = "opposite" if divergence >= 3 else "different"
                notes.append(
                    f"This source holds a {direction} view on {axis.replace('_', ' ')} "
                    f"(yours: {user_score:+d}, source: ~{estimated:+d})"
                )

        return notes

    def _estimate_source_position(
        self, axis: str, source_stance: str
    ) -> int:
        """Estimate a source's position on an axis from its editorial stance.

        Returns an integer from -2 to +2 where +2 = traditional/conservative,
        -2 = progressive/secular, 0 = neutral/unknown.
        """
        stance_map = {
            "left-leaning": -2,
            "center-left": -1,
            "centrist": 0,
            "center-right": +1,
            "right-leaning": +2,
            "institutional": 0,
            "academic": -1,
            "unknown": 0,
        }
        base = stance_map.get(source_stance, 0)

        # Axis-specific adjustments
        if axis in ("sanctity_of_life", "sexual_morality"):
            # Institutional sources tend to be more liberal on these
            if source_stance == "institutional":
                base = max(base - 1, -2)
        if axis == "government_power":
            # Institutional sources tend to be more statist
            if source_stance == "institutional":
                base = min(base - 1, -2)
        if axis == "free_speech":
            # Most established media are more restrictive on speech
            if source_stance in ("left-leaning", "center-left", "institutional"):
                base = min(base - 1, -2)

        return base


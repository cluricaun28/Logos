"""Retrieval Quality Scorer — Measures prefetch retrieval usefulness.

Evaluates whether the prefetch pipeline returns *useful* context for a given query,
not just *something*. Runs fast heuristics (no LLM calls) on each prefetch event
and logs results to a sliding window for trend analysis.

Metrics:
  - keyword_relevance: Fraction of query content words found in retrieved text
  - source_alignment: Whether the classifier's top-priority source returned results
  - noise_ratio: Fraction of injected text that is structural filler vs content
  - score_spread: Variance in result scores (tight = confident, wide = uncertain)
  - coverage: Results returned vs requested (top_k)

State persisted to ~/.hermes/retrieval_quality.json (max 100 entries).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re as _re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default state file location
_DEFAULT_STATE_FILE = str(Path.home() / ".hermes" / "staging" / "retrieval_quality.json")

# Sliding window size — keep last N prefetch events
_MAX_ENTRIES = 100

# Stop words for keyword relevance calculation
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "it", "its", "this", "that", "these", "those",
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "they", "them", "their", "theirs",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "about", "up", "down",
    "and", "but", "or", "yet", "either", "neither",
    "if", "unless", "until", "while", "although", "though", "because",
    "since", "when", "whenever", "wherever", "however",
}

# Structural filler patterns — these are noise, not content
_FILLER_PATTERNS = [
    _re.compile(r"^\[From (Reference Library|Perpetual Memory|Web Search)\]$", _re.IGNORECASE),
    _re.compile(r"^---+$"),
    _re.compile(r"^\[Local Recall:.*\]$"),
    _re.compile(r"^\[⚠ Gap detected.*\]$"),
    _re.compile(r"^\.+ \[truncated.*\]$"),
]


class RetrievalQualityScorer:
    """Scores prefetch retrieval quality and tracks trends over time."""

    def __init__(self, state_file: str = _DEFAULT_STATE_FILE):
        self._state_file = state_file
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """Load quality history from disk. Graceful on failure."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                self._entries = data.get("events", [])[:_MAX_ENTRIES]
                # Validate entries — drop corrupt ones
                validated = []
                for entry in self._entries:
                    if isinstance(entry, dict) and "timestamp" in entry and "overall_score" in entry:
                        validated.append(entry)
                self._entries = validated
        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning("Failed to load retrieval quality state (%s). Starting fresh.", e)
            self._entries = []

    def _save(self) -> None:
        """Save quality history to disk. Graceful on failure."""
        try:
            data = {"events": self._entries}
            tmp_path = self._state_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._state_file)
        except (IOError, OSError) as e:
            logger.debug("Failed to save retrieval quality state: %s", e)

    # -----------------------------------------------------------------------
    # Scoring — called once per prefetch event
    # -----------------------------------------------------------------------

    def score(
        self,
        query: str,
        priorities: Dict[str, float],
        scored_results: List[Dict[str, Any]],
        formatted_text: str = "",
        top_k_requested: int = 5,
    ) -> Dict[str, Any]:
        """Score a prefetch retrieval event.

        Args:
            query: The user's original query text.
            priorities: Classifier output {pm_priority, rl_priority, web_priority}.
            scored_results: List of unified-scored result dicts from _unified_score_results().
            formatted_text: The final injected context string (optional).
            top_k_requested: Number of results requested per source.

        Returns:
            Dict with scoring results and overall quality score 0.0-1.0.
        """
        if not scored_results:
            return self._empty_score(query, priorities)

        # Extract metrics
        keyword_rel = self._keyword_relevance(query, scored_results)
        source_align = self._source_alignment(priorities, scored_results)
        noise_ratio = self._noise_ratio(formatted_text) if formatted_text else 0.0
        score_spread = self._score_spread(scored_results)
        coverage = min(len(scored_results) / max(top_k_requested * 3, 1), 1.0)

        # Weighted overall score
        # Keyword relevance and source alignment are the strongest signals of usefulness
        overall = (
            keyword_rel * 0.35
            + source_align * 0.25
            + (1.0 - noise_ratio) * 0.15
            + score_spread * 0.15
            + coverage * 0.10
        )

        result = {
            "overall_score": round(overall, 3),
            "keyword_relevance": round(keyword_rel, 3),
            "source_alignment": round(source_align, 3),
            "noise_ratio": round(noise_ratio, 3),
            "score_spread": round(score_spread, 3),
            "coverage": round(coverage, 3),
        }

        # Record to sliding window
        entry = {
            "timestamp": time.time(),
            "query": query[:100],  # Truncate long queries for storage
            **result,
            "results_count": len(scored_results),
            "priorities": priorities,
        }

        self._entries.append(entry)
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._save()

        return result

    def _empty_score(
        self, query: str, priorities: Dict[str, float]
    ) -> Dict[str, Any]:
        """Return zero-score result when no results were retrieved."""
        # Empty results aren't always bad — if the query was a continuation command
        # or very personal, empty PM is expected. Penalize only when high priority source returned nothing.
        top_source = max(priorities, key=priorities.get)  # type: ignore[arg-type]
        top_priority = priorities[top_source]

        # If top priority was low (<0.5), empty results are fine — classifier said "don't search much"
        if top_priority < 0.5:
            overall = 0.7  # Neutral — no harm done
        else:
            overall = 0.1  # Bad — high priority source returned nothing

        result = {
            "overall_score": round(overall, 3),
            "keyword_relevance": 0.0,
            "source_alignment": 0.0 if top_priority >= 0.5 else 0.7,
            "noise_ratio": 0.0,
            "score_spread": 0.0,
            "coverage": 0.0,
        }

        entry = {
            "timestamp": time.time(),
            "query": query[:100],
            **result,
            "results_count": 0,
            "priorities": priorities,
        }

        self._entries.append(entry)
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._save()

        return result

    # -----------------------------------------------------------------------
    # Metric calculations
    # -----------------------------------------------------------------------

    def _keyword_relevance(self, query: str, results: List[Dict[str, Any]]) -> float:
        """Fraction of query content words found in retrieved text.

        Extracts content words from the query (excluding stop words), then checks
        how many appear anywhere in the combined result snippets/content.
        Returns 0.0-1.0.
        """
        # Extract content words from query
        query_words = _re.findall(r"\b\w+\b", query.lower())
        content_words = [w for w in query_words if w not in _STOP_WORDS and len(w) > 2]

        if not content_words:
            return 1.0  # No meaningful words to check — can't penalize

        # Combine all result text
        combined_text = ""
        for r in results:
            # PM results have 'content', RL has 'snippet', Web has 'content'
            text = (r.get("content") or r.get("snippet") or "")[:500]
            combined_text += " " + text.lower()

        # Count how many query content words appear in results
        matches = sum(1 for w in content_words if w in combined_text)
        return min(matches / len(content_words), 1.0)

    def _source_alignment(self, priorities: Dict[str, float], results: List[Dict[str, Any]]) -> float:
        """Whether the classifier's top-priority source actually returned results.

        Returns 1.0 if top priority source has results, 0.0 if it doesn't but others do,
        or 0.5 if no results at all (ambiguous — could be empty DB).
        """
        # Find top priority source
        pm_p = priorities.get("pm_priority", 0)
        rl_p = priorities.get("rl_priority", 0)
        web_p = priorities.get("web_priority", 0)

        sources = {"PM": pm_p, "RL": rl_p, "Web": web_p}
        top_source = max(sources, key=sources.get)  # type: ignore[arg-type]
        top_priority = sources[top_source]

        # Count results per source
        source_counts = {"PM": 0, "RL": 0, "Web": 0}
        for r in results:
            src = r.get("source", "")
            if src in source_counts:
                source_counts[src] += 1

        top_count = source_counts[top_source]

        if not results:
            return 0.5  # No results at all — ambiguous

        if top_count > 0:
            return 1.0  # Top priority source returned results — good alignment

        # Top priority source returned nothing but others did — misalignment
        total_other = sum(v for k, v in source_counts.items() if k != top_source)
        if total_other > 0 and top_priority >= 0.5:
            return 0.0  # Clear misclassification or search failure

        # Low priority on top source — maybe it wasn't supposed to return much
        return 0.3

    def _noise_ratio(self, text: str) -> float:
        """Fraction of injected text that is structural filler vs actual content.

        Counts lines matching known filler patterns (headers, separators, metadata).
        Returns 0.0-1.0 where higher = more noise.
        """
        if not text.strip():
            return 0.0

        lines = text.split("\n")
        total_lines = len(lines)
        if total_lines == 0:
            return 0.0

        filler_count = 0
        for line in lines:
            stripped = line.strip()
            # Empty lines count as noise
            if not stripped:
                filler_count += 1
                continue
            # Check against known filler patterns
            for pattern in _FILLER_PATTERNS:
                if pattern.match(stripped):
                    filler_count += 1
                    break

        return min(filler_count / total_lines, 1.0)

    def _score_spread(self, results: List[Dict[str, Any]]) -> float:
        """Quality of score discrimination among results.

        Returns 0.0-1.0 where higher = better discrimination:
        - All scores identical (weak retrieval): low score
        - Tight cluster near top (confident retrieval): high score
        - Wide spread with low tail (noise mixed in): medium score
        """
        if len(results) <= 1:
            return 0.5  # Can't assess spread with 0-1 results

        scores = [r.get("unified_score", r.get("score", 0)) for r in results]
        mean = sum(scores) / len(scores)

        if mean == 0:
            return 0.0

        # Calculate coefficient of variation (std / mean)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = math.sqrt(variance)
        cv = std / mean if mean > 0 else 0

        # Ideal: moderate spread with high mean.
        # CV near 0 = all same score (bad discrimination).
        # CV very high = wild variance (noise mixed in).
        # Sweet spot: CV around 0.2-0.4 with mean > 0.3
        if cv < 0.05:
            # All scores nearly identical — weak retrieval signal
            spread_quality = 0.2 + (mean * 0.3)  # Partial credit for high mean
        elif cv < 0.5:
            # Good discrimination range
            spread_quality = min(1.0, 0.6 + (mean * 0.4))
        else:
            # High variance — some good results mixed with noise
            spread_quality = max(0.2, mean * 0.8)

        return round(min(spread_quality, 1.0), 3)

    # -----------------------------------------------------------------------
    # Analysis and reporting
    # -----------------------------------------------------------------------

    def get_trend(self, window: int = 20) -> Dict[str, Any]:
        """Calculate quality trends over recent prefetch events."""
        if not self._entries:
            return {
                "trend": "insufficient_data",
                "avg_score": 0.0,
                "message": "No retrieval quality data yet.",
            }

        recent = self._entries[-window:]
        scores = [e["overall_score"] for e in recent]
        avg = sum(scores) / len(scores)

        # Simple trend direction: compare last half to first half
        if len(recent) >= 4:
            mid = len(recent) // 2
            first_half = sum(scores[:mid]) / mid
            second_half = sum(scores[mid:]) / (len(scores) - mid)
            diff = second_half - first_half

            if diff > 0.05:
                trend = "improving"
            elif diff < -0.05:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        # Per-metric averages
        metrics_avg: Dict[str, float] = {}
        for key in ("keyword_relevance", "source_alignment", "noise_ratio", "score_spread", "coverage"):
            vals = [e.get(key, 0) for e in recent]
            metrics_avg[key] = round(sum(vals) / len(vals), 3) if vals else 0.0

        return {
            "trend": trend,
            "avg_score": round(avg, 3),
            "events_analyzed": len(recent),
            "metrics_average": metrics_avg,
            "message": self._trend_message(trend, avg),
        }

    def _trend_message(self, trend: str, avg: float) -> str:
        """Human-readable summary of retrieval quality trend."""
        if trend == "insufficient_data":
            return f"Need more data (currently {len(self._entries)} events)."

        status = ""
        if avg >= 0.7:
            status = "Good — prefetch is returning useful context."
        elif avg >= 0.4:
            status = "Moderate — some noise or misalignment detected."
        else:
            status = "Poor — retrieval quality needs attention."

        direction = {
            "improving": "Quality is improving over recent events.",
            "degrading": "Quality is degrading — check classifier signals and search backends.",
            "stable": f"Quality is stable at {avg:.1%} average.",
        }[trend]

        return f"{status} {direction}"

    def get_recent_failures(self, threshold: float = 0.3, limit: int = 5) -> List[Dict[str, Any]]:
        """Return recent prefetch events with low quality scores."""
        failures = []
        for entry in reversed(self._entries):
            if entry["overall_score"] < threshold:
                failures.append(entry)
                if len(failures) >= limit:
                    break
        return failures

    def clear(self) -> None:
        """Clear all quality history."""
        self._entries = []
        self._save()

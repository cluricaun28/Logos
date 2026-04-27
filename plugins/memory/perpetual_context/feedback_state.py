"""Feedback State — Persistent compression quality tracking across sessions.

Records quality scores from each compression event and calculates degradation
trends over a sliding window. Provides correction parameters when quality
is degrading, enabling the bridge builder to adapt its extraction strategy.

State persisted to ~/.hermes/compression_feedback.json (max 20 entries).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default state file location
_DEFAULT_STATE_FILE = os.path.expanduser("~/.hermes/compression_feedback.json")

# Sliding window size — keep last N compression events
_MAX_ENTRIES = 20

# Degradation threshold — if trend exceeds this, apply corrections
_DEGRADATION_THRESHOLD = 0.15

# Minimum quality threshold — below this, force preservation markers
_MIN_QUALITY_THRESHOLD = 0.60


class FeedbackState:
    """Persistent feedback state across compression cycles.

    Records quality scores from each compression event and calculates
    degradation trends over a sliding window of recent compressions.
    """

    def __init__(self, state_file: str = _DEFAULT_STATE_FILE):
        self._state_file = state_file
        self._entries: List[Dict[str, Any]] = []
        self._load()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        """Load feedback state from disk. Graceful on failure."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r") as f:
                    data = json.load(f)
                self._entries = data.get("compressions", [])[:_MAX_ENTRIES]

                # Validate entries — drop any corrupt ones
                validated = []
                for entry in self._entries:
                    if isinstance(entry, dict) and "timestamp" in entry and "overall_score" in entry:
                        validated.append(entry)
                self._entries = validated

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning("Failed to load feedback state (%s). Starting fresh.", e)
            self._entries = []

    def _save(self) -> None:
        """Save feedback state to disk. Graceful on failure."""
        try:
            data = {"compressions": self._entries}
            # Atomic write pattern — write to temp, then rename
            tmp_path = self._state_file + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, self._state_file)
        except (IOError, OSError) as e:
            logger.debug("Failed to save feedback state: %s", e)

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record_compression(self, quality_score: Dict[str, Any], session_id: str = "") -> None:
        """Record a compression event with its quality score.

        Args:
            quality_score: Output from BridgeQualityScorer.score()
            session_id: Current session identifier for tracking
        """
        entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "overall_score": quality_score.get("overall", 0.0),
            "tasks_preserved": quality_score.get("active_tasks_preserved", 0.0),
            "files_preserved": quality_score.get("file_paths_preserved", 0.0),
            "errors_preserved": quality_score.get("errors_preserved", 0.0),
            "gaps_preserved": quality_score.get("gaps_preserved", 0.0),
            "bridge_chars": quality_score.get("bridge_char_count", 0),
            "sections_present": quality_score.get("sections_present", []),
        }

        self._entries.append(entry)

        # Enforce sliding window — keep last MAX_ENTRIES
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]

        self._save()

    def clear(self) -> None:
        """Clear all feedback history."""
        self._entries = []
        self._save()

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------

    def get_degradation_trend(self) -> float:
        """Calculate quality degradation trend over recent compressions.

        Uses simple linear regression slope on the last N scores.
        Returns -1.0 to +1.0:
            Negative = quality improving (good)
            Zero      = stable
            Positive  = quality degrading (needs correction)

        With fewer than 3 entries, returns 0.0 (insufficient data).
        """
        if len(self._entries) < 3:
            return 0.0

        # Use last 10 entries for trend calculation
        recent = self._entries[-10:]
        scores = [e["overall_score"] for e in recent]
        n = len(scores)

        # Simple linear regression slope
        x_mean = (n - 1) / 2.0
        y_mean = sum(scores) / n

        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        # Slope normalized to [-1, +1] range
        # Max possible slope is when scores go from 0.0 to 1.0 over n points
        max_slope = 1.0 / (n - 1) if n > 1 else 1.0
        raw_slope = numerator / denominator

        # Normalize and NEGATE: positive regression slope means improving,
        # but we want positive degradation_trend to mean degrading.
        trend = -(raw_slope / max_slope) if max_slope > 0 else 0.0
        return round(max(-1.0, min(1.0, trend)), 3)

    def get_recent_average(self, window: int = 5) -> float:
        """Get average quality score over recent compressions."""
        if not self._entries:
            return 1.0  # No history = assume good

        recent = self._entries[-window:]
        scores = [e["overall_score"] for e in recent]
        return round(sum(scores) / len(scores), 3)

    def get_correction_params(self) -> Dict[str, Any]:
        """Return correction parameters based on degradation trend.

        Analyzes the quality history and returns adjustments for the next
        compression cycle:

            - extraction_window_multiplier: >1.0 if degrading (extract more messages)
            - preserve_critical_markers: True if tasks keep getting lost
            - min_bridge_quality_threshold: Dynamic threshold based on history
            - degradation_trend: Raw trend value for diagnostics

        Returns dict with correction parameters. All defaults are neutral
        (no correction needed) when there's insufficient data or quality is stable.
        """
        trend = self.get_degradation_trend()
        avg_quality = self.get_recent_average()

        params: Dict[str, Any] = {
            "extraction_window_multiplier": 1.0,
            "preserve_critical_markers": False,
            "min_bridge_quality_threshold": _MIN_QUALITY_THRESHOLD,
            "degradation_trend": trend,
            "recent_avg_quality": avg_quality,
        }

        # No correction needed if quality is stable or improving
        if trend <= 0:
            return params

        # Degradation detected — apply corrections proportional to severity
        if trend >= _DEGRADATION_THRESHOLD:
            # Widen extraction window based on degradation severity
            params["extraction_window_multiplier"] = round(1.0 + trend * 2.0, 2)

            # Force preservation markers if quality is low
            if avg_quality < _MIN_QUALITY_THRESHOLD:
                params["preserve_critical_markers"] = True

            # Lower the threshold dynamically — don't chase impossible targets
            params["min_bridge_quality_threshold"] = round(max(0.40, avg_quality - 0.10), 2)

        return params

    def needs_correction(self) -> bool:
        """Quick check: does the current trend require correction?"""
        return self.get_degradation_trend() >= _DEGRADATION_THRESHOLD

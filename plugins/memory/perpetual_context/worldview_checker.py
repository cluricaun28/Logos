"""Worldview Profile & Divergence Checking — Compare source content against user's worldview profile.

Loads the user's worldview profile from config, manages axis-topic mappings,
and checks whether source content diverges significantly from the user's
stated worldview position on relevant axes.

Does NOT reject results — only annotates them so the user can judge.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # Graceful fallback

from .source_assessment import _get_dossiers

logger = logging.getLogger(__name__)


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
                        k,
                        v,
                    )
            return valid if valid else None
    except (OSError, KeyError, TypeError, AttributeError) as e:
        logger.debug("Failed to load worldview profile: %s", e)
    return None


# ---------------------------------------------------------------------------
# Worldview Axis Loading
# ---------------------------------------------------------------------------

# Inline fallback defaults for when YAML is unavailable
_DEFAULT_AXES: dict[str, list[str]] = {
    "sexual_morality": ["marriage", "same-sex", "gender", "lgbt", "trans", "adoption"],
    "racial_policy": ["race", "affirmative action", "dei", "discrimination", "systemic racism", "quota"],
    "free_speech": ["free speech", "cancel culture", "censorship", "hate speech", "employer"],
    "government_power": ["constitution", "federal", "free market", "regulation", "ubiquitous", "basic income"],
    "sanctity_of_life": ["abortion", "conception", "pro-life", "pro-choice"],
    "religion_in_public": ["religion", "church", "prayer", "education", "anti-discrimination"],
    "border_immigration": ["border", "immigration", "citizenship", "illegal"],
    "criminal_justice": ["death penalty", "guns", "second amendment", "police", "criminal"],
}

_AXIS_CACHE: dict[str, list[str]] | None = None


def _load_axes() -> dict[str, list[str]]:
    """Load axis-topic mappings from source_dossiers.yaml.

    Returns dict mapping axis name -> list of keywords.
    Falls back to inline defaults if YAML unavailable.
    """
    dossiers = _get_dossiers()
    if dossiers:
        # Try to load from the dossier file's axes section
        dossier_path = Path(__file__).parent / "config" / "source_dossiers.yaml"
        if _yaml is not None and dossier_path.is_file():
            try:
                with open(dossier_path, encoding="utf-8") as f:
                    data = _yaml.safe_load(f)
                if isinstance(data, dict) and "axes" in data:
                    axes: dict[str, list[str]] = {}
                    for axis_name, axis_data in data["axes"].items():
                        if isinstance(axis_data, dict) and "keywords" in axis_data:
                            axes[axis_name] = axis_data["keywords"]
                    return axes if axes else _DEFAULT_AXES
            except (OSError, KeyError, TypeError, AttributeError) as e:
                logger.debug("Failed to load axes from YAML: %s", e)

    return _DEFAULT_AXES


def _get_axes() -> dict[str, list[str]]:
    """Lazy-load axis mappings on first access."""
    global _AXIS_CACHE
    if _AXIS_CACHE is None:
        _AXIS_CACHE = _load_axes()
    return _AXIS_CACHE


# ---------------------------------------------------------------------------
# Worldview Profile & Axis Loading
# ---------------------------------------------------------------------------


def _load_worldview_profile() -> dict[str, int | None] | None:
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
                        k,
                        v,
                    )
            return valid if valid else None
    except (OSError, KeyError, TypeError, AttributeError) as e:
        logger.debug("Failed to load worldview profile: %s", e)
    return None


# Inline fallback defaults for when YAML is unavailable
_DEFAULT_AXES: dict[str, list[str]] = {
    "sexual_morality": ["marriage", "same-sex", "gender", "lgbt", "trans", "adoption"],
    "racial_policy": ["race", "affirmative action", "dei", "discrimination", "systemic racism", "quota"],
    "free_speech": ["free speech", "cancel culture", "censorship", "hate speech", "employer"],
    "government_power": ["constitution", "federal", "free market", "regulation", "ubiquitous", "basic income"],
    "sanctity_of_life": ["abortion", "conception", "pro-life", "pro-choice"],
    "religion_in_public": ["religion", "church", "prayer", "education", "anti-discrimination"],
    "border_immigration": ["border", "immigration", "citizenship", "illegal"],
    "criminal_justice": ["death penalty", "guns", "second amendment", "police", "criminal"],
}

_AXIS_CACHE: dict[str, list[str]] | None = None


def _load_axes() -> dict[str, list[str]]:
    """Load axis-topic mappings from source_dossiers.yaml.

    Returns dict mapping axis name -> list of keywords.
    Falls back to inline defaults if YAML unavailable.
    """
    from .source_assessment import _get_dossiers

    dossiers = _get_dossiers()
    if dossiers:
        # Try to load from the dossier file's axes section
        dossier_path = Path(__file__).parent / "config" / "source_dossiers.yaml"
        if _yaml is not None and dossier_path.is_file():
            try:
                with open(dossier_path, encoding="utf-8") as f:
                    data = _yaml.safe_load(f)
                if isinstance(data, dict) and "axes" in data:
                    axes: dict[str, list[str]] = {}
                    for axis_name, axis_data in data["axes"].items():
                        if isinstance(axis_data, dict) and "keywords" in axis_data:
                            axes[axis_name] = axis_data["keywords"]
                    return axes if axes else _DEFAULT_AXES
            except (OSError, KeyError, TypeError, AttributeError) as e:
                logger.debug("Failed to load axes from YAML: %s", e)

    return _DEFAULT_AXES


def _get_axes() -> dict[str, list[str]]:
    """Lazy-load axis mappings on first access."""
    global _AXIS_CACHE
    if _AXIS_CACHE is None:
        _AXIS_CACHE = _load_axes()
    return _AXIS_CACHE


# ---------------------------------------------------------------------------
# Worldview Divergence Checker
# ---------------------------------------------------------------------------


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

        for axis, keywords in _get_axes().items():
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
                notes.append(f"This source holds a {direction} view on {axis.replace('_', ' ')} (yours: {user_score:+d}, source: ~{estimated:+d})")

        return notes

    def _estimate_source_position(self, axis: str, source_stance: str) -> int:
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
        if axis in ("sanctity_of_life", "sexual_morality"):  # noqa: SIM102
            # Institutional sources tend to be more liberal on these
            if source_stance == "institutional":
                base = max(base - 1, -2)
        if axis == "government_power":  # noqa: SIM102
            # Institutional sources tend to be more statist
            if source_stance == "institutional":
                base = min(base - 1, -2)
        if axis == "free_speech":  # noqa: SIM102
            # Most established media are more restrictive on speech
            if source_stance in ("left-leaning", "center-left", "institutional"):
                base = min(base - 1, -2)

        return base

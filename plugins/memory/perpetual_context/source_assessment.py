"""Source Assessment — Source credibility dossiers and domain extraction.

Loads source credibility data from config/source_dossiers.yaml, provides
domain extraction from URLs, and stance/reliability lookups for known sources.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # Graceful fallback

logger = logging.getLogger(__name__)


def _load_source_dossiers() -> dict[str, dict[str, Any]]:
    """Load source credibility data from config/source_dossiers.yaml.

    Returns a dict mapping domain -> {stance, reliability, ownership, notes}.
    Falls back to an empty dict if YAML is unavailable or file missing.
    """
    if _yaml is None:
        logger.debug("PyYAML not available — source dossiers unavailable")
        return {}

    dossier_path = Path(__file__).parent / "config" / "source_dossiers.yaml"
    if not dossier_path.is_file():
        logger.debug("source_dossiers.yaml not found at '%s'", dossier_path)
        return {}

    try:
        with open(dossier_path, encoding="utf-8") as f:
            data = _yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("source_dossiers.yaml did not return a dict")
            return {}

        dossiers: dict[str, dict[str, Any]] = {}
        sources = data.get("sources", {})
        if isinstance(sources, dict):
            for domain, info in sources.items():
                if isinstance(info, dict):
                    dossiers[domain] = info
                else:
                    logger.debug("Skipping malformed source entry: %s", domain)
        return dossiers
    except Exception as e:
        logger.debug("Failed to load source_dossiers.yaml: %s", e)
        return {}


# Pre-loaded dossiers (lazy-loaded on first access)
_SOURCE_DOSSIERS: dict[str, dict[str, Any]] | None = None


def _get_dossiers() -> dict[str, dict[str, Any]]:
    """Lazy-load source dossiers on first access."""
    global _SOURCE_DOSSIERS
    if _SOURCE_DOSSIERS is None:
        _SOURCE_DOSSIERS = _load_source_dossiers()
    return _SOURCE_DOSSIERS


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

    dossiers = _get_dossiers()
    for known_domain, info in dossiers.items():
        if known_domain in domain:
            return info.get("stance", "unknown")

    return "unknown"


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


def _get_source_info(domain: str | None) -> dict[str, Any]:
    """Look up full source dossier for a domain.

    Returns dict with stance, reliability, ownership, notes.
    Falls back to defaults for unrecognized domains.
    """
    if not domain:
        return {"domain": None, "stance": "unknown", "reliability": "C", "notes": ""}

    if domain.endswith(".gov"):
        return {"domain": domain, "stance": "institutional", "reliability": "B", "notes": "Government source"}
    if domain.endswith(".edu"):
        return {"domain": domain, "stance": "academic", "reliability": "B", "notes": "Academic source"}

    dossiers = _get_dossiers()
    for known_domain, info in dossiers.items():
        if known_domain in domain:
            return {
                "domain": known_domain,
                "stance": info.get("stance", "unknown"),
                "reliability": info.get("reliability", "C"),
                "ownership": info.get("ownership", ""),
                "notes": info.get("notes", ""),
            }

    return {"domain": domain, "stance": "unknown", "reliability": "C", "notes": ""}

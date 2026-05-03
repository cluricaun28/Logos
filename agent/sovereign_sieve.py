"""SovereignSieve: Linguistic & motive analysis for external sources.

Epistemic filter for all web-sourced information entering the Reference Library.
Operates on the principle that language is a signal of alignment — identifies
shibboleths, institutional capture, and framing patterns to move from implicit
bias to explicit analysis.

Design principles:
  - Zero tolerance for generic "neutral" labels — every source has a motive
  - Explicit tagging by ideological cluster, not just "bias"
  - Triangulation requirement: no external claim promoted without cross-referencing
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SovereignSieve:
    """Linguistic and motive analysis for web sources.

    Filters external information entering the Reference Library by analyzing
    shibboleths, institutional ties, and framing patterns.
    """

    def __init__(self, dossiers_path: Optional[Path] = None):
        # Domain dossiers: pre-curated source profiles with ownership/funding data
        self.dossiers_path = dossiers_path or Path.home() / ".hermes" / "reference-library" / "entities" / "source-dossiers.json"
        self._dossiers: Dict[str, Dict[str, Any]] = {}
        self._load_dossiers()

    def _load_dossiers(self) -> None:
        """Load domain dossiers from disk."""
        if not self.dossiers_path.exists():
            logger.debug(f"No source dossiers found at {self.dossiers_path}")
            return

        try:
            with open(self.dossiers_path, "r") as f:
                self._dossiers = json.load(f)
            logger.info(f"Loaded {len(self._dossiers)} source dossiers")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load source dossiers: {e}")

    def analyze_source(
        self,
        url: str,
        content: str,
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a web source for ideological alignment and credibility.

        Args:
            url: Source URL being analyzed.
            content: Page content or excerpt to analyze.
            main_runtime: LLM runtime config for analysis call.

        Returns:
            Analysis report with:
                - domain: extracted domain from URL
                - credibility_score: 0.0-1.0 (higher = more reliable)
                - ideological_cluster: e.g., "center-left", "tech-libertarian", "conservative"
                - shibboleths_found: list of signaling terms detected
                - motive_flags: institutional capture indicators
                - framing_notes: what's emphasized/omitted
                - recommendation: "ACCEPT", "FLAG", or "REJECT"
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc or "unknown"
        # Strip www. for dossier lookup
        clean_domain = domain.replace("www.", "")

        # Check against known dossiers first (fast path)
        dossier = self._dossiers.get(clean_domain, {})

        # Build analysis prompt
        prompt = self._build_analysis_prompt(url, content, dossier)

        try:
            from agent.auxiliary_client import call_llm

            response = call_llm(
                task="archiving",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                timeout=8.0,
                main_runtime=main_runtime,
            )
            analysis_text = response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"SovereignSieve LLM call failed (using dossier fallback): {e}")
            return self._dossier_fallback(domain, dossier)

        # Parse JSON response
        report = self._parse_analysis(analysis_text)
        report["domain"] = domain
        report["url"] = url

        # Merge with dossier data if available
        if dossier:
            report = self._merge_dossier(report, dossier)

        logger.info(f"SovereignSieve: {domain} → {report.get('recommendation', 'UNKNOWN')} "
                     f"(credibility={report.get('credibility_score', 0):.2f})")
        return report

    def analyze_batch(
        self,
        sources: List[Tuple[str, str]],
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze multiple web sources.

        Args:
            sources: List of (url, content) tuples.
            main_runtime: LLM runtime config.

        Returns:
            List of analysis reports.
        """
        return [
            self.analyze_source(url, content, main_runtime=main_runtime)
            for url, content in sources
        ]

    def _build_analysis_prompt(
        self, url: str, content: str, dossier: Dict[str, Any]
    ) -> str:
        """Build the LLM prompt for source analysis."""
        dossier_hint = ""
        if dossier:
            ownership = dossier.get("ownership", "Unknown")
            funding = dossier.get("funding", "Unknown")
            ideology = dossier.get("ideology", "Unknown")
            dossier_hint = f"""
KNOWN DOSSIER DATA for {url}:
- Ownership: {ownership}
- Funding: {funding}
- Known Ideology: {ideology}
Use this as baseline but verify against actual content."""

        return f"""You are the SOVEREIGN SIEVE — an epistemic filter analyzing web sources for ideological alignment and credibility.

SOURCE URL: {url}
CONTENT EXCERPT (first 2000 chars):
{content[:2000]}{dossier_hint}

ANALYSIS DIMENSIONS:

1. SHIBBOLETH MAPPING: Identify signaling terms that reveal ideological alignment.
   - Look for DEI terminology, "critical" frameworks, or partisan framing words
   - Note delegitimizing language used against opposing views

2. MOTIVE ANALYSIS: Who funds this? Whose interests does it serve?
   - Corporate ownership → profit motive
   - University/Think tank → institutional agenda
   - Independent → personal bias (still a motive)

3. FRAMING PATTERNS: What is emphasized vs omitted?
   - Claims of "objectivity" while presenting only one side = Institutional Capture
   - Selective data presentation without context

OUTPUT FORMAT (JSON only, no preamble):
{{
  "credibility_score": 0.0-1.0,
  "ideological_cluster": "center-left|center-right|tech-libertarian|conservative|progressive|unknown",
  "shibboleths_found": ["term1", "term2"],
  "motive_flags": ["corporate-owned", "think-tank-funded", ...],
  "framing_notes": "What's emphasized/omitted",
  "recommendation": "ACCEPT|FLAG|REJECT"
}}

CREDIBILITY SCALE:
- 0.9-1.0: Primary sources, raw data, verifiable facts
- 0.7-0.9: Reputable analysis with clear methodology
- 0.5-0.7: Opinion pieces, editorial content
- 0.3-0.5: Heavily biased but factually grounded
- 0.0-0.3: Propaganda, unverified claims, or deliberate misinformation

RECOMMENDATION RULES:
- ACCEPT: credibility >= 0.7 AND no severe motive flags
- FLAG: credibility 0.5-0.7 OR minor motive concerns (requires triangulation)
- REJECT: credibility < 0.5 OR severe institutional capture detected"""

    def _parse_analysis(self, text: str) -> Dict[str, Any]:
        """Parse JSON analysis response from LLM."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SovereignSieve JSON: {e}")
            return self._default_report()

    def _dossier_fallback(self, domain: str, dossier: Dict[str, Any]) -> Dict[str, Any]:
        """Return a report based solely on dossier data when LLM is unavailable."""
        if not dossier:
            return {
                "domain": domain,
                "url": "",
                "credibility_score": 0.5,
                "ideological_cluster": "unknown",
                "shibboleths_found": [],
                "motive_flags": ["no_dossier_available"],
                "framing_notes": "LLM unavailable — manual review required",
                "recommendation": "FLAG",
            }

        return {
            "domain": domain,
            "url": "",
            "credibility_score": dossier.get("credibility", 0.5),
            "ideological_cluster": dossier.get("ideology", "unknown"),
            "shibboleths_found": [],
            "motive_flags": [dossier.get("funding", "unknown")],
            "framing_notes": f"Dossier fallback: {dossier.get('notes', 'No notes')}",
            "recommendation": dossier.get("recommendation", "FLAG"),
        }

    def _merge_dossier(self, report: Dict[str, Any], dossier: Dict[str, Any]) -> Dict[str, Any]:
        """Merge LLM analysis with known dossier data."""
        # Dossier overrides credibility if it's more specific
        if "credibility" in dossier and dossier["credibility"] != report.get("credibility_score"):
            report["credibility_score"] = dossier["credibility"]

        # Append motive flags from dossier
        dossier_flags = [
            f"dossier:{dossier.get('ownership', 'unknown')}",
            f"funding:{dossier.get('funding', 'unknown')}",
        ]
        existing_flags = report.get("motive_flags", [])
        for flag in dossier_flags:
            if flag not in existing_flags:
                existing_flags.append(flag)
        report["motive_flags"] = existing_flags

        return report

    def _default_report(self) -> Dict[str, Any]:
        """Return a default report when parsing fails."""
        return {
            "credibility_score": 0.5,
            "ideological_cluster": "unknown",
            "shibboleths_found": [],
            "motive_flags": ["parse_error"],
            "framing_notes": "Analysis failed — manual review required",
            "recommendation": "FLAG",
        }

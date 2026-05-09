"""Source Analysis — Unified source intelligence engine.

Reads from and writes to entity dossiers in the Reference Library.
Three layers: identity lookup, content analysis, narrative detection.
Findings compound over time — each research session enriches the source's profile.

Design:
  - Single facade class (SourceAnalyzer) with internal component classes
  - Bidirectional RL flow: reads dossiers for context, writes new findings back
  - Graceful degradation at every layer — missing dossier, missing embeddings,
    missing LLM all degrade to partial results rather than failures
  - No imports from plugins/ — only agent/ and stdlib (layering discipline)
"""

from __future__ import annotations

import json
import logging
import re as _re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models (frozen — immutable output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceProfile:
    """What we know about a source from its dossier."""

    domain: str
    dossier_file: str | None = None  # e.g. "nytimes-v1.md"
    cluster: str = "unknown"  # e.g. "Secular Progressive / Establishment Liberalism"
    alignment: str = "unknown"  # e.g. "Opposed", "Aligned", "Partially Aligned"
    reliability: str = "unknown"  # e.g. "Low", "Medium-High"
    motive: str = ""  # primary motive from dossier
    truthful_on: list[str] = field(default_factory=list)
    omits: list[str] = field(default_factory=list)
    shibboleths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContentAnalysis:
    """Layer 2: what the content reveals about framing and bias."""

    bias_score: float = 0.0
    markers: list[str] = field(default_factory=list)
    framing_notes: list[str] = field(default_factory=list)
    method: str = "none"  # "semantic" | "regex" | "none"


@dataclass(frozen=True)
class NarrativeSignal:
    """Layer 3: narrative control detection."""

    omissions: list[str] = field(default_factory=list)
    deviation: str = ""  # non-empty if source acts out of character
    coordination: bool = False  # multiple sources using same frame
    narrative_phase: str = "unknown"  # "initial_break" | "narrative_shift" | "flood" | "unknown"


@dataclass(frozen=True)
class NewFinding:
    """A new pattern discovered about a source, ready to write to its dossier."""

    domain: str
    category: str  # "truthful_on" | "omits" | "deviations"
    entry: str
    evidence: str = ""  # what prompted this finding


@dataclass(frozen=True)
class AnalysisReport:
    """Complete analysis report for a single source."""

    source: SourceProfile
    content: ContentAnalysis
    narrative: NarrativeSignal
    query_context: str = ""
    url: str = ""  # original URL for tool output
    findings: list[NewFinding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Component: Dossier Lookup
# ---------------------------------------------------------------------------


class _DossierLookup:
    """Load and parse entity dossiers from the Reference Library.

    Reads domain-index.json for fast domain-to-dossier mapping, then parses
    Markdown dossiers for detailed behavioral patterns.
    """

    def __init__(self, rl_path: Path | None = None) -> None:
        default_rl = Path.home() / ".hermes" / "reference-library"
        self._rl_path = rl_path or default_rl
        self._index_path = self._rl_path / "entities" / "domain-index.json"
        self._entities_dir = self._rl_path / "entities"
        self._index: dict[str, dict[str, Any]] = {}
        self._dossiers: dict[str, str] = {}  # domain -> raw markdown
        self._loaded = False

    def ensure_loaded(self) -> None:
        """Load index and cache dossiers on first access."""
        if self._loaded:
            return

        # Load domain index
        if self._index_path.is_file():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._index = data.get("domains", {})
                logger.debug("Loaded %d domain entries from index", len(self._index))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load domain-index.json: %s", e)
        else:
            logger.debug("domain-index.json not found at %s", self._index_path)

        # Pre-scan entities directory for dossier files
        if self._entities_dir.is_dir():
            for md_file in self._entities_dir.glob("*-v1.md"):
                self._dossiers[md_file.stem] = md_file.read_text(encoding="utf-8")
            logger.debug("Cached %d entity dossiers", len(self._dossiers))

        self._loaded = True

    def lookup(self, url: str) -> SourceProfile | None:
        """Look up a source by URL. Returns profile or None."""
        self.ensure_loaded()

        domain = self._extract_domain(url)
        if not domain:
            return None

        # Check index for this domain
        entry = self._index.get(domain)
        if not entry:
            # Try partial match (subdomains, www prefix)
            # Require the index domain to appear as a right-aligned suffix
            # of the extracted domain to avoid "the" matching "thelancet.com".
            for idx_domain, idx_entry in self._index.items():
                if domain.endswith(idx_domain) or idx_domain.endswith(domain):
                    entry = idx_entry
                    domain = idx_domain
                    break

        if not entry:
            return None

        dossier_file = entry.get("file", "")
        cluster = self._strip_md(entry.get("cluster", "unknown"))
        alignment = self._strip_md(entry.get("alignment", "unknown"))
        reliability = self._strip_md(entry.get("reliability", "unknown"))

        # Parse behavioral patterns from the Markdown dossier
        truthful_on: list[str] = []
        omits: list[str] = []
        shibboleths: list[str] = []
        motive = ""

        dossier_key = dossier_file.replace(".md", "") if dossier_file else ""
        raw = self._dossiers.get(dossier_key)
        if raw:
            truthful_on, omits, shibboleths, motive = self._parse_patterns(raw)

        return SourceProfile(
            domain=domain,
            dossier_file=dossier_file,
            cluster=cluster,
            alignment=alignment,
            reliability=reliability,
            motive=motive,
            truthful_on=truthful_on,
            omits=omits,
            shibboleths=shibboleths,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_domain(url: str) -> str | None:
        if not url:
            return None
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace("www.", "") or None
        except Exception:
            return None

    @staticmethod
    def _strip_md(text: str) -> str:
        """Remove markdown bold markers for cleaner output."""
        return text.replace("**", "").strip()

    def _parse_patterns(self, raw: str) -> tuple[list[str], list[str], list[str], str]:
        """Extract behavioral patterns from Markdown dossier.

        Returns (truthful_on, omits, shibboleths, motive).
        """
        truthful_on: list[str] = []
        omits: list[str] = []
        shibboleths: list[str] = []
        motive = ""

        current_section = ""
        for line in raw.splitlines():
            stripped = line.strip()

            # Track which section we're in
            if stripped.startswith("## "):
                current_section = stripped[3:].lower()
                continue

            if not stripped or stripped.startswith("- **") and ":":
                pass  # skip headers and bold labels

            # Primary motive
            if ("primary motive" in current_section or "motivation" in current_section) and stripped.startswith("-") and ":" in stripped:
                motive = stripped.lstrip("- ").split(":", 1)[-1].strip()

            # Behavioral Patterns section (auto-updated or manual)
            if "behavioral pattern" in current_section or "truthful" in current_section:
                if stripped.startswith("- ") and "✅" in stripped:
                    entry = stripped.lstrip("- ").replace("✅", "").strip()
                    if entry and entry not in truthful_on:
                        truthful_on.append(entry)
                elif stripped.startswith("- ") and "❌" in stripped:
                    entry = stripped.lstrip("- ").replace("❌", "").strip()
                    if entry and entry not in omits:
                        omits.append(entry)
                elif stripped.startswith("- ") and "(" in stripped:
                    # e.g. "- Soros/OSF funding trails (12 sessions...)"
                    entry = stripped.lstrip("- ").split("(")[0].strip()
                    if entry and entry not in omits:
                        omits.append(entry)

            # Shibboleth section
            if ("shibboleth" in current_section or "linguistic" in current_section) and stripped.startswith("- **") and ":" in stripped:
                # "Key Shibboleths: term1, term2, term3"
                after_label = stripped.split(":", 1)[-1].strip()
                if after_label:
                    for term in after_label.split(","):
                        t = term.strip().strip('"')
                        if t and t not in shibboleths:
                            shibboleths.append(t)

        return truthful_on, omits, shibboleths, motive


# ---------------------------------------------------------------------------
# Component: Marker Detector
# ---------------------------------------------------------------------------


class _MarkerDetector:
    """Detect ideological framing markers in text.

    Two-tier: semantic (embedding-based, lazy) and regex (always available).
    Reimplements the bias_detector logic to avoid importing from plugins/.
    """

    # Pre-compiled patterns (module-level, not per-call)
    _LOADED_LANG = _re.compile(
        r"\b(allegedly|so-called|supposedly|infamously|notoriously)\b",
        _re.IGNORECASE,
    )
    _VALUE_LADEN = _re.compile(
        r"\b(toxic|woke|fascist|socialist|liberal|conservative|"
        r"radical|extremist|systemic racism|white privilege)\b",
        _re.IGNORECASE,
    )
    _PASSIVE = _re.compile(
        r"\b(mistakes were made|it was decided|actions were taken|"
        r"errors were discovered)\b",
        _re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._anchors: dict[str, list[list[float]]] | None = None

    def analyze(self, text: str) -> ContentAnalysis:
        """Analyze text for bias markers.

        Returns ContentAnalysis with bias score, detected markers, and notes.
        """
        if not isinstance(text, str) or not text.strip():
            return ContentAnalysis(method="none")

        notes: list[str] = []
        bias_score = 0.0
        method = "regex"

        # Layer 1: regex detection (always available)
        loaded = self._LOADED_LANG.findall(text)
        if loaded:
            notes.append(f"Loaded language: {', '.join(set(loaded))}")
            bias_score += 0.2 * len(loaded)

        values = self._VALUE_LADEN.findall(text)
        if values:
            notes.append(f"Value-laden terms: {', '.join(set(values))}")
            bias_score += 0.15 * len(values)

        passive = self._PASSIVE.findall(text)
        if passive:
            notes.append(f"Passive voice framing: {', '.join(set(passive))}")
            bias_score += 0.1 * len(passive)

        # Layer 2: semantic detection (lazy, may fail gracefully)
        semantic = self._try_semantic(text)
        if semantic:
            method = "semantic"
            for marker, score in semantic.items():
                if score >= 0.65:
                    notes.append(f"Semantic {marker.replace('_', ' ')} (sim: {score:.2f})")
                    bias_score += 0.1

        # Cap and format
        bias_score = min(bias_score, 1.0)
        markers = [n.split(":")[0].split("(")[0].strip() for n in notes]

        return ContentAnalysis(
            bias_score=round(bias_score, 2),
            markers=markers,
            framing_notes=notes,
            method=method,
        )

    def _try_semantic(self, text: str) -> dict[str, float] | None:
        """Try embedding-based detection. Returns None if unavailable."""
        try:
            from agent.perpetual_context_db import EmbeddingEngine

            engine = EmbeddingEngine.get()
            text_vec = engine.embed(text[:5000])
            if text_vec is None:
                return None

            scores: dict[str, float] = {}
            for category, phrases in self._MARKER_PROFILES.items():
                best = 0.0
                for phrase in phrases:
                    vec = engine.embed(phrase)
                    if vec:
                        sim = EmbeddingEngine.cosine_similarity(text_vec, vec)
                        best = max(best, sim)
                scores[category] = round(best, 3)

            return scores

        except Exception as e:
            logger.debug("Semantic marker detection unavailable: %s", e)
            return None

    # Representative phrases for semantic anchor profiles
    _MARKER_PROFILES: dict[str, list[str]] = {
        "loaded_language": [
            "allegedly committed the crime",
            "so-called expert",
            "supposedly independent analysis",
        ],
        "value_laden": [
            "toxic cultural influence",
            "woke ideology spreading",
            "fascist authoritarian policies",
        ],
        "passive_voice_framing": [
            "mistakes were made by the administration",
            "it was decided that changes would occur",
        ],
        "one_sided_framing": [
            "everyone agrees that this is wrong",
            "the only reasonable conclusion is",
        ],
        "moral_positioning": [
            "this is a moral imperative for our society",
            "we have an ethical duty to act",
        ],
    }


# ---------------------------------------------------------------------------
# Component: Narrative Engine
# ---------------------------------------------------------------------------


class _NarrativeEngine:
    """Detect narrative control patterns by comparing content against known source behavior.

    The key signal is omission — what the source *didn't* say, compared to what
    it historically omits and what other sources are saying about the same topic.
    """

    def analyze(
        self,
        content: str,
        profile: SourceProfile,
        query_context: str = "",
    ) -> tuple[NarrativeSignal, list[NewFinding]]:
        """Compare content against known source patterns.

        Returns (narrative signal, new findings to write to dossier).
        """
        findings: list[NewFinding] = []
        omissions: list[str] = []
        deviation = ""
        coordination = False
        phase = "unknown"

        domain = profile.domain
        content_lower = content.lower() if content else ""

        # Check: does this source omit things it's known to omit?
        # If yes — consistent behavior (expected). If no — deviation (signal).
        known_omits = profile.omits
        if known_omits:
            for omit_pattern in known_omits:
                # Split on whitespace and slashes (e.g. "Soros/OSF funding trails"
                # becomes ["soros", "osf", "funding", "trails"])
                keywords = _re.split(r"[\s/]+", omit_pattern.lower())
                if not keywords:
                    continue
                covered = sum(1 for kw in keywords if kw in content_lower)
                threshold = max(2, len(keywords) // 2)
                if covered >= threshold:
                    deviation = f"Source covered '{omit_pattern}' — deviates from known omission pattern ({covered}/{len(keywords)} keywords matched)"

        # Check: are expected details missing?
        # If the source is known to be truthful on certain topics, and the content
        # is about that topic but lacks expected factual detail, that's suspicious
        truthful = profile.truthful_on
        if truthful and query_context:
            for truth_topic in truthful:
                topic_kw = truth_topic.lower()
                if any(w in query_context.lower() for w in topic_kw.split()) and len(content) < 200:
                    # The query is about a topic this source is usually reliable on.
                    # Check if the content actually contains substantive data.
                    omissions.append(f"Minimal coverage on '{truth_topic}' — source is usually detailed here")

        # Detect narrative phase from content characteristics
        if profile.alignment == "Opposed" and profile.cluster:
            # Progressive source: initial breaks are rare, usually narrative_shift or flood
            if len(content) > 3000:
                phase = "flood"  # long polished piece = flood the zone
            elif len(content) > 500:
                phase = "narrative_shift"
        elif profile.alignment in ("Aligned", "Partially Aligned") and ("first" in content_lower or "exclusive" in content_lower):
            # Conservative/independent sources more likely to have initial breaks
            phase = "initial_break"

        # Flag coordination: if the content uses the source's known shibboleths
        # at high density, it's reinforcing the narrative
        shibboleths = profile.shibboleths
        if shibboleths:
            hits = sum(1 for s in shibboleths if s.lower() in content_lower)
            if hits >= 3:
                coordination = True

        # Generate new findings from deviations
        if deviation:
            findings.append(
                NewFinding(
                    domain=domain,
                    category="deviations",
                    entry=deviation,
                    evidence=f"Query: {query_context[:100]}",
                )
            )

        return (
            NarrativeSignal(
                omissions=omissions,
                deviation=deviation,
                coordination=coordination,
                narrative_phase=phase,
            ),
            findings,
        )


# ---------------------------------------------------------------------------
# Component: RL Writer
# ---------------------------------------------------------------------------


class _RLWriter:
    """Append new findings back to entity dossiers.

    Safe write: appends to the Behavioral Patterns section. Never overwrites
    human-written content. Each entry gets a timestamp and evidence tag.
    """

    def __init__(self, rl_path: Path) -> None:
        self._entities_dir = rl_path / "entities"

    def write(self, findings: list[NewFinding]) -> Path | None:
        """Write findings back to the appropriate dossier files.

        Returns the path of the first file written, or None if nothing to write.
        """
        if not findings:
            return None

        # Group by domain
        by_domain: dict[str, list[NewFinding]] = {}
        for f in findings:
            by_domain.setdefault(f.domain, []).append(f)

        written: list[Path] = []
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")  # noqa: UP017

        for domain, domain_findings in by_domain.items():
            # Find the dossier file for this domain
            dossier_path = self._find_dossier(domain)
            if not dossier_path:
                logger.debug("No dossier file found for %s, skipping write-back", domain)
                continue

            try:
                content = dossier_path.read_text(encoding="utf-8")
                updated = self._append_patterns(content, domain_findings, now)
                if updated != content:
                    dossier_path.write_text(updated, encoding="utf-8")
                    written.append(dossier_path)
                    logger.info(
                        "Updated %s with %d new finding(s)",
                        dossier_path.name,
                        len(domain_findings),
                    )
                else:
                    logger.debug("No new findings to append for %s", domain)

            except OSError as e:
                logger.error("Failed to write to %s: %s", dossier_path, e)

        return written[0] if written else None

    def _find_dossier(self, domain: str) -> Path | None:
        """Find the dossier file for a domain.

        Uses the domain-index.json for accurate lookup. Falls back to
        filename scanning.
        """
        # Try index lookup first — this is the authoritative mapping
        lookup = _DossierLookup(self._entities_dir.parent)
        lookup.ensure_loaded()
        idx_entry = lookup._index.get(domain)
        if idx_entry:
            file_name = idx_entry.get("file", "")
            if file_name:
                candidate = self._entities_dir / file_name
                if candidate.is_file():
                    return candidate

        # Direct filename match (domain with dots replaced by hyphens)
        direct = self._entities_dir / f"{domain.replace('.', '-')}-v1.md"
        if direct.is_file():
            return direct

        # Scan for domain in filename (handles nytimes.com → nytimes-v1.md)
        domain_clean = domain.replace(".", "-").split("-")[0]  # "nytimes"
        for f in self._entities_dir.glob("*-v1.md"):
            if domain_clean in f.stem:
                return f

        return None

    def _append_patterns(
        self,
        content: str,
        findings: list[NewFinding],
        now: str,
    ) -> str:
        """Append findings to the Behavioral Patterns section.

        Creates the section if it doesn't exist. Never overwrites existing content.
        """
        if not findings:
            return content

        lines = content.splitlines()
        insert_pos = len(lines)  # default: append at end
        section_header = "## V. Behavioral Patterns (Auto-Updated)"

        # Find the section or a good insertion point
        for i, line in enumerate(lines):
            if line.strip().startswith("## V. Behavioral Patterns"):
                # Find the end of this section (next ## heading or end of file)
                for j in range(i + 1, len(lines)):
                    if lines[j].strip().startswith("## "):
                        insert_pos = j
                        break
                break

        # Build the new content to insert
        new_lines: list[str] = []
        if insert_pos == len(lines):
            # Section doesn't exist — create it
            new_lines.append(section_header)
            new_lines.append(f"*Last updated: {now} by SourceAnalyzer*")
            new_lines.append("")

        # Group findings by category
        by_cat: dict[str, list[NewFinding]] = {}
        for f in findings:
            by_cat.setdefault(f.category, []).append(f)

        for category, cat_findings in by_cat.items():
            header = category.replace("_", " ").title()
            new_lines.append(f"### {header}")
            for f in cat_findings:
                evidence = f" ({f.evidence})" if f.evidence else ""
                if category == "omits":
                    new_lines.append(f"- ❌ {f.entry}{evidence}")
                elif category == "truthful_on":
                    new_lines.append(f"- ✅ {f.entry}{evidence}")
                else:
                    new_lines.append(f"- {f.entry}{evidence}")
            new_lines.append("")

        # Update the timestamp line if section exists
        if insert_pos < len(lines):
            for i, line in enumerate(lines):
                if "Last updated:" in line and "*Last updated:" in line:
                    lines[i] = f"*Last updated: {now} by SourceAnalyzer*"
                    break

        # Insert new lines
        result_lines = lines[:insert_pos] + new_lines + lines[insert_pos:]
        return "\n".join(result_lines)


# ---------------------------------------------------------------------------
# Facade: SourceAnalyzer
# ---------------------------------------------------------------------------


class SourceAnalyzer:
    """Unified source analysis engine.

    Reads from and writes to entity dossiers in the Reference Library.
    Three layers: identity lookup, content analysis, narrative detection.

    Usage:
        analyzer = SourceAnalyzer()
        report = analyzer.analyze(
            url="https://nytimes.com/2026/05/09/some-article",
            content="Article text here...",
            query_context="immigration enforcement",
        )
        # Optionally persist findings back to the dossier
        analyzer.write_findings(report)
    """

    def __init__(self, rl_path: Path | None = None) -> None:
        default_rl = Path.home() / ".hermes" / "reference-library"
        self._rl_path = rl_path or default_rl
        self._lookup = _DossierLookup(self._rl_path)
        self._detector = _MarkerDetector()
        self._narrative = _NarrativeEngine()
        self._writer = _RLWriter(self._rl_path)

    def analyze(
        self,
        url: str,
        content: str,
        query_context: str = "",
    ) -> AnalysisReport:
        """Full three-layer analysis of a source.

        Args:
            url: Source URL being analyzed.
            content: Extracted page content or excerpt.
            query_context: The original search query or topic context.

        Returns:
            AnalysisReport with source profile, content analysis,
            narrative signal, and any new findings.
        """
        # Layer 1: Identity lookup
        profile = self._lookup.lookup(url)
        if not profile:
            profile = SourceProfile(domain=urlparse(url).netloc or "unknown")

        # Layer 2: Content analysis
        content_result = self._detector.analyze(content)

        # Layer 3: Narrative detection
        narrative, findings = self._narrative.analyze(content, profile, query_context)

        return AnalysisReport(
            source=profile,
            content=content_result,
            narrative=narrative,
            query_context=query_context,
            url=url,
            findings=findings,
        )

    def analyze_batch(
        self,
        results: list[dict[str, str]],
        query_context: str = "",
    ) -> list[AnalysisReport]:
        """Analyze multiple search results.

        Each dict should have 'url' and 'content' (or 'snippet') keys.

        Args:
            results: List of result dicts from web_search.
            query_context: The original search query for context.

        Returns:
            List of AnalysisReports, one per result.
        """
        if not results:
            return []

        reports: list[AnalysisReport] = []
        for result in results:
            if not isinstance(result, dict):
                logger.debug("Skipping non-dict result in batch: %s", type(result).__name__)
                continue

            url = result.get("url", "")
            content = result.get("content") or result.get("snippet", "")

            if not url:
                continue

            try:
                report = self.analyze(url, content, query_context)
                reports.append(report)
            except Exception as e:
                logger.debug("Analysis failed for %s: %s", url, e)
                reports.append(
                    AnalysisReport(
                        source=SourceProfile(domain=urlparse(url).netloc or "unknown"),
                        content=ContentAnalysis(framing_notes=[f"Analysis error: {e}"]),
                        narrative=NarrativeSignal(),
                        query_context=query_context,
                        url=url,
                    )
                )

        return reports

    def enrich_results(
        self,
        results: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Enrich web search results with source analysis annotations.

        Designed to integrate into the prefetch pipeline after scrutiny
        vetting. Each result dict is annotated with source intelligence
        derived from entity dossiers and content analysis.

        Results are modified in place and also returned.

        Args:
            results: List of result dicts with 'url' and 'content'/'snippet'.
            query: Original search query for context.

        Returns:
            Enriched result list (same objects, annotated in place).
        """
        if not results:
            return results

        reports = self.analyze_batch(results, query_context=query)
        findings: list[NewFinding] = []

        for result, report in zip(results, reports, strict=True):
            # Annotate with source intelligence
            result["_source_profile"] = {
                "domain": report.source.domain,
                "cluster": report.source.cluster,
                "alignment": report.source.alignment,
                "truthful_on": report.source.truthful_on,
                "omits": report.source.omits,
            }

            result["_bias_analysis"] = {
                "score": report.content.bias_score,
                "markers": report.content.markers,
                "method": report.content.method,
                "framing_notes": report.content.framing_notes,
            }

            result["_narrative_signal"] = {
                "phase": report.narrative.narrative_phase,
                "deviation": report.narrative.deviation,
                "coordination": report.narrative.coordination,
                "omissions": report.narrative.omissions,
            }

            # If deviation detected, boost confidence flag
            if report.narrative.deviation:
                result["_narrative_flag"] = report.narrative.deviation

            findings.extend(report.findings)

        # Write findings back if any
        if findings:
            try:
                self._writer.write(findings)
            except Exception as e:
                logger.debug("Failed to write findings to RL: %s", e)

        return results

    def write_findings(self, report: AnalysisReport) -> Path | None:
        """Append new findings back to the entity dossier.

        Returns the path of the updated file, or None if nothing to write.
        """
        if not report.findings:
            return None
        return self._writer.write(report.findings)

    def write_batch_findings(self, reports: list[AnalysisReport]) -> list[Path]:
        """Write findings from multiple reports back to their dossiers.

        Returns list of paths that were updated.
        """
        all_findings: list[NewFinding] = []
        for r in reports:
            all_findings.extend(r.findings)

        if not all_findings:
            return []

        written = self._writer.write(all_findings)
        return [written] if written else []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SourceAnalyzer",
    "AnalysisReport",
    "SourceProfile",
    "ContentAnalysis",
    "NarrativeSignal",
    "NewFinding",
]

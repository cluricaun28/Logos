"""WebVerificationService: Web-grounded verification for distillation pipeline.

Verifies factual claims from distilled drafts against web sources, then runs
source analysis to identify ideological alignment, omissions, and narrative
patterns. The result tells the audit phase not just whether a claim is
factually true, but whether the framing matches reality or carries
institutional spin.

Uses local services: SearXNG (8080), Firecrawl (3002).
Source analysis runs via auxiliary_client.call_llm with a prompt modeled
on the source_analyze tool.

Design principles:
  - Stdlib + httpx only — no internal agent tool imports (standalone testable)
  - If web tools are down: attempt recovery, then raise (never silently skip)
  - Two-tier scope: worldview-relevant topics get full treatment,
    purely technical topics skip verification
  - Captures both factual accuracy AND ideological framing intelligence
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str                          # the original claim text
    query: str                          # search query used
    sources: List[Dict[str, Any]]      # [{url, title, snippet, content}]
    alignment: str = "unknown"          # supported | contradicted | unclear | error
    signal: str = ""                    # what the web evidence shows (clean signal)
    noise: str = ""                     # what sources are omitting or spin-doctoring
    source_profiles: List[str] = field(default_factory=list)  # domain:cluster:alignment


@dataclass(frozen=True)
class VerificationReport:
    """Complete verification report for a draft."""
    claims: List[ClaimVerification] = field(default_factory=list)
    worldview_relevant: bool = False
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keywords that trigger worldview-relevant verification
# ---------------------------------------------------------------------------

WORLDVIEW_KEYWORDS = [
    # Politics & government
    "congress", "president", "election",
    "democracy", "authoritarian", "communism", "capitalism", "socialism",
    "liberal", "conservative", "progressive",
    # Media
    "media bias", "news network", "propaganda", "mainstream media",
    "cnn", "fox news", "msnbc", "bbc", "new york times", "washington post",
    "reuters", "associated press",
    # Economics
    "trade", "tariff", "sanction", "inflation", "gdp",
    "federal reserve", "quantitative easing",
    # Religion & culture
    "christianity", "islam", "religion", "church", "biblical",
    "gender ideology", "transgender", "abortion", "immigration",
    # Geopolitics
    "nato", "russia", "china", "iran", "ukraine", "middle east",
    "cold war", "imperialism", "colonialism",
    # Science controversies
    "climate change", "global warming",
    # AI & tech policy
    "ai regulation", "ai safety", "open source ai", "anthropic",
    "openai", "deepmind",
    # Known organizations / figures
    "soros", "open society", "carnegie", "rockefeller",
    "pentagon", "state department", "world economic forum",
]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class WebVerificationService:
    """Verify factual claims against web sources with source analysis."""

    SEARXNG_URL = "http://localhost:8080"
    FIRECRAWL_URL = "http://localhost:3002"

    def __init__(self, main_runtime: Optional[Dict[str, Any]] = None):
        self.main_runtime = main_runtime

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_draft(
        self,
        draft_path: Path,
        turn_ids: Optional[List[int]] = None,
    ) -> VerificationReport:
        """Verify a distilled draft against web sources.

        Returns a VerificationReport. Raises if web tools are down and
        cannot be recovered.
        """
        if not draft_path.exists():
            raise FileNotFoundError(f"Draft not found: {draft_path}")

        draft_content = draft_path.read_text()

        # Two-tier: skip if no worldview-relevant content
        if not self._is_worldview_relevant(draft_content):
            logger.info("No worldview-relevant topics — skipping web verification")
            return VerificationReport(worldview_relevant=False)

        # Extract claims
        claims = self._extract_claims(draft_content)
        if not claims:
            logger.info("No verifiable claims found in draft")
            return VerificationReport(worldview_relevant=True)

        # Ensure tools are available (recover or raise)
        self._ensure_web_tools()

        # Verify each claim
        verifications = []
        for claim in claims:
            try:
                verified = self._verify_claim(claim)
                verifications.append(verified)
            except Exception as e:
                logger.warning(f"Verification failed for '{claim[:80]}...': {e}")
                verifications.append(
                    ClaimVerification(
                        claim=claim, query="", sources=[],
                        alignment="error",
                        signal=f"Verification failed: {e}",
                    )
                )

        return VerificationReport(claims=verifications, worldview_relevant=True)

    def format_for_audit(self, report: VerificationReport) -> str:
        """Format verification results for the audit prompt.

        Produces a markdown section with both factual verification and
        source intelligence (signal vs. noise per claim).
        """
        if not report.worldview_relevant and not report.claims:
            return "## WEB VERIFICATION\n[No worldview-relevant content — skipped]\n"

        lines = [
            "## WEB VERIFICATION DATA",
            "",
            "The following claims were verified against web sources. Each entry shows",
            "what the evidence supports (signal) and what sources are omitting or",
            "distorting (noise). Source profiles indicate ideological alignment.",
            "",
        ]

        for v in report.claims:
            lines.append(f"### Claim: {v.claim}")
            lines.append(f"- **Alignment:** {v.alignment}")

            if v.sources:
                lines.append(f"- **Sources ({len(v.sources)}):**")
                for s in v.sources[:5]:
                    url = s.get("url", "")
                    title = s.get("title", url)
                    snippet = (s.get("snippet") or s.get("content") or "")[:200]
                    lines.append(f"  - [{title}]({url}): {snippet}")

            if v.source_profiles:
                lines.append("- **Source profiles:**")
                for sp in v.source_profiles:
                    lines.append(f"  - {sp}")

            if v.signal:
                lines.append(f"- **Signal (what evidence shows):** {v.signal}")
            if v.noise:
                lines.append(f"- **Noise (omissions / spin):** {v.noise}")
            if v.alignment == "error":
                lines.append(f"- **ERROR:** {v.signal}")
            lines.append("")

        if report.errors:
            lines.append("### Errors")
            for err in report.errors:
                lines.append(f"- {err}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: scope detection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_worldview_relevant(text: str) -> bool:
        """Does this draft contain worldview-relevant topics?"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in WORLDVIEW_KEYWORDS)

    @staticmethod
    def _extract_claims(text: str) -> List[str]:
        """Extract factual claims from a synthesized draft.

        Targets bullet points from Factual Notes, Decisions, and similar sections.
        Returns up to 10 distinct claims.
        """
        claims = []
        seen = set()

        for line in text.splitlines():
            stripped = line.strip()

            if not stripped.startswith("- ") or len(stripped) < 20:
                continue

            # Remove turn citations for cleaner queries
            clean = re.sub(r'\[turn_\d+\]', '', stripped).lstrip("- ").strip()

            if len(clean) > 30 and clean not in seen:
                seen.add(clean)
                claims.append(clean)

        return claims[:10]

    # ------------------------------------------------------------------
    # Internal: web tool management
    # ------------------------------------------------------------------

    def _ensure_web_tools(self) -> None:
        """Ensure SearXNG and Firecrawl are running. Recover or raise."""
        issues = []

        if not self._check_service("SearXNG", self.SEARXNG_URL,
                                    path="/search?q=test&format=json"):
            logger.warning("SearXNG down — restarting")
            try:
                subprocess.run(["docker", "restart", "searxng"],
                               capture_output=True, timeout=30)
                time.sleep(3)
                if not self._check_service("SearXNG", self.SEARXNG_URL,
                                           path="/search?q=test&format=json"):
                    issues.append("SearXNG unrecoverable")
            except Exception as e:
                issues.append(f"SearXNG recovery error: {e}")

        if not self._check_service("Firecrawl", self.FIRECRAWL_URL,
                                   path="/v1/scrape", method="POST",
                                   body={"url": "https://example.com"}):
            logger.warning("Firecrawl down — restarting")
            try:
                subprocess.run(["docker", "restart", "firecrawl"],
                               capture_output=True, timeout=30)
                time.sleep(3)
                if not self._check_service("Firecrawl", self.FIRECRAWL_URL,
                                           path="/v1/scrape", method="POST",
                                           body={"url": "https://example.com"}):
                    issues.append("Firecrawl unrecoverable")
            except Exception as e:
                issues.append(f"Firecrawl recovery error: {e}")

        if issues:
            raise RuntimeError(
                f"Web verification requires working tools: {'; '.join(issues)}"
            )

    @staticmethod
    def _check_service(name: str, base_url: str, path: str = "/",
                       method: str = "GET", body: Optional[Dict] = None,
                       timeout: int = 5) -> bool:
        import httpx
        url = f"{base_url}{path}"
        try:
            if method == "GET":
                r = httpx.get(url, timeout=timeout, follow_redirects=True)
            else:
                r = httpx.post(url, json=body,
                               headers={"Content-Type": "application/json"},
                               timeout=timeout, follow_redirects=True)
            return r.status_code < 500
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal: verification pipeline
    # ------------------------------------------------------------------

    def _verify_claim(self, claim: str) -> ClaimVerification:
        """Verify a claim: search → extract → analyze → assess."""

        query = self._build_query(claim)
        search_results = self._search(query)

        if not search_results:
            return ClaimVerification(
                claim=claim, query=query, sources=[],
                alignment="unclear",
                signal="No search results found",
            )

        # Extract top results
        urls = [r["url"] for r in search_results[:5] if r.get("url")]
        extracted = self._extract(urls)

        # Merge search + extracted content
        full_sources = self._merge(search_results, extracted)

        # Source analysis via LLM
        analysis = self._analyze_sources(full_sources, claim, query)

        # Assess: alignment, signal, noise
        alignment = self._assess_alignment(claim, full_sources, analysis)
        signal = self._extract_signal(analysis)
        noise = self._extract_noise(analysis)
        profiles = self._extract_profiles(analysis)

        return ClaimVerification(
            claim=claim, query=query, sources=full_sources[:5],
            alignment=alignment, signal=signal, noise=noise,
            source_profiles=profiles,
        )

    @staticmethod
    def _build_query(claim: str) -> str:
        """Build a focused search query from a claim."""
        clean = re.sub(r'\[turn_\d+\]', '', claim).strip()
        if len(clean) > 100:
            words = clean.split()[:15]
            return " ".join(words)
        return clean

    # ------------------------------------------------------------------
    # Tier 1: Search (SearXNG)
    # ------------------------------------------------------------------

    @staticmethod
    def _search(query: str) -> List[Dict[str, Any]]:
        import httpx
        try:
            resp = httpx.get(
                f"{WebVerificationService.SEARXNG_URL}/search",
                params={"q": query, "format": "json"},
                timeout=15, follow_redirects=True,
            )
            data = resp.json()
            results = []
            for r in data.get("results", []):
                results.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "snippet": (r.get("content") or "")[:300],
                })
            logger.info(f"Search '{query[:60]}': {len(results)} results")
            return results[:10]
        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            return []

    # ------------------------------------------------------------------
    # Tier 2: Extract (Firecrawl)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(urls: List[str]) -> List[Dict[str, Any]]:
        import httpx
        if not urls:
            return []
        extracted = []
        for url in urls[:5]:
            try:
                resp = httpx.post(
                    f"{WebVerificationService.FIRECRAWL_URL}/v1/scrape",
                    json={
                        "url": url,
                        "formats": ["markdown"],
                        "onlyMainContent": True,
                        "removeBase64Images": True,
                        "blockAds": True,
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=30, follow_redirects=True,
                )
                data = resp.json()
                if data.get("success"):
                    markdown = data.get("markdown") or data.get("content") or ""
                    extracted.append({
                        "url": url,
                        "title": data.get("metadata", {}).get("title", url),
                        "content": markdown[:5000],
                    })
            except Exception as e:
                logger.warning(f"Extract failed for {url}: {e}")
        return extracted

    # ------------------------------------------------------------------
    # Tier 3: Source analysis (LLM)
    # ------------------------------------------------------------------

    def _analyze_sources(
        self, sources: List[Dict[str, Any]], claim: str, query: str
    ) -> str:
        """Run source analysis via LLM — same method as source_analyze tool.

        Asks: what's the signal (truth), what's the noise (spin/omission),
        and what's each source's ideological alignment?
        """
        # Build the prompt
        source_texts = []
        for s in sources:
            text = f"URL: {s.get('url','')}\nTitle: {s.get('title','')}\n"
            if s.get("content"):
                text += f"Content: {s['content'][:2000]}\n"
            elif s.get("snippet"):
                text += f"Snippet: {s['snippet']}\n"
            source_texts.append(text)

        sources_block = "\n---\n".join(source_texts)

        prompt = (
            "You are analyzing web sources for factual claims about a specific topic. "
            "Your job is to identify what the evidence actually shows (signal) versus "
            "what sources are omitting, distorting, or spin-doctoring (noise).\n\n"
            f"CLAIM TO VERIFY:\n{claim}\n\n"
            f"SEARCH QUERY:\n{query}\n\n"
            f"WEB SOURCES ({len(sources)} results):\n{sources_block}\n\n"
            "ANALYSIS — RESPOND IN EXACTLY THIS FORMAT:\n\n"
            "SIGNAL: [What do the sources collectively confirm or contradict about the claim? "
            "Be specific. Cite which sources support which facts.]\n\n"
            "NOISE: [What are sources omitting? What spin, loaded language, or framing "
            "is present? Which sources are pushing a narrative rather than reporting facts?]\n\n"
            "SOURCE PROFILES:\n"
            "[For each distinct domain/source, one line: DOMAIN | CLUSTER | ALIGNMENT | NOTES]\n"
            "Example: cnn.com | Secular Progressive / Mainstream Media | Opposed | "
            "Uses loaded language, omits X, frames as Y\n\n"
            "ALIGNMENT: [supported | contradicted | unclear]\n\n"
            "Be direct. Identify ideological framing explicitly. Do not hedge."
        )

        try:
            from agent.auxiliary_client import call_llm

            call_kwargs = {
                "task": "archiving",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "timeout": 300.0,
            }
            if self.main_runtime:
                call_kwargs["main_runtime"] = self.main_runtime

            response = call_llm(**call_kwargs)
            msg = response.choices[0].message
            result = (msg.content or "").strip()

            if not result and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                result = msg.reasoning_content.strip()

            return result or "[Analysis returned empty]"

        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Source analysis LLM call failed: {e}")
            return "[Analysis unavailable — LLM call failed]"
        except Exception as e:
            logger.warning(f"Source analysis error: {e}")
            return f"[Analysis error: {e}]"

    # ------------------------------------------------------------------
    # Internal: parse analysis results
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_signal(analysis: str) -> str:
        """Extract the SIGNAL section from analysis output."""
        m = re.search(r'SIGNAL:\s*(.+?)(?:\n\nNOISE:|\n\nSOURCE PROFILES:|\n\nALIGNMENT:|$)',
                      analysis, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_noise(analysis: str) -> str:
        """Extract the NOISE section from analysis output."""
        m = re.search(r'NOISE:\s*(.+?)(?:\n\nSOURCE PROFILES:|\n\nALIGNMENT:|$)',
                      analysis, re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_profiles(analysis: str) -> List[str]:
        """Extract SOURCE PROFILES from analysis output."""
        m = re.search(r'SOURCE PROFILES:\s*\n((?:.*\n?)*)', analysis, re.DOTALL)
        if not m:
            return []
        profiles = []
        for line in m.group(1).splitlines():
            line = line.strip()
            if line and "|" in line:
                profiles.append(line.strip())
        return profiles

    @staticmethod
    def _assess_alignment(
        claim: str, sources: List[Dict[str, Any]], analysis: str
    ) -> str:
        """Extract alignment verdict from analysis."""
        m = re.search(r'ALIGNMENT:\s*(supported|contradicted|unclear)',
                      analysis, re.IGNORECASE)
        return m.group(1).lower() if m else "unclear"

    # ------------------------------------------------------------------
    # Internal: merge search + extracted results
    # ------------------------------------------------------------------

    @staticmethod
    def _merge(
        search_results: List[Dict[str, Any]],
        extracted: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge search results with extracted content, enriching where possible."""
        # Build lookup from extracted
        extracted_map = {e["url"]: e for e in extracted}
        merged = []
        seen_urls = set()

        for r in search_results:
            url = r.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            entry = dict(r)
            # Enrich with extracted content if available
            if url in extracted_map:
                entry["content"] = extracted_map[url].get("content", "")
                entry["title"] = extracted_map[url].get("title", r.get("title", ""))
            merged.append(entry)

        # Add any extracted URLs not in search results
        for e in extracted:
            if e["url"] not in seen_urls:
                merged.append({
                    "url": e["url"],
                    "title": e.get("title", ""),
                    "content": e.get("content", ""),
                    "snippet": (e.get("content") or "")[:200],
                })

        return merged

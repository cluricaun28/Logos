"""Web Research Wrapper — Phase 2 of Deep Research & Continuity Engine.

Provides web search and content extraction capabilities that activate when
local recall (Phase 1) detects gaps. Supports multiple backends: SearXNG
(self-hosted, privacy-first), Firecrawl API, and Camofox browser fallback.

Architecture:
  - WebResearchClient: Main orchestrator with lazy-init HTTP clients
  - SearchResult: Normalized result dataclass
  - Graceful degradation — never raises unhandled exceptions

Config in ~/.hermes/config.yaml (optional):
  memory:
    perpetual_context:
      web_research:
        searxng_url: "http://localhost:8080"
        firecrawl_api_key_env: "FIRECRAWL_API_KEY"
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel: returned by _get_http() when httpx is unavailable, so subsequent
# calls don'''t waste time re-importing.
_HTTP_UNAVAILABLE = object()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
WEB_SEARCH_TIMEOUT = 30           # Seconds per search request
WEB_EXTRACT_TIMEOUT = 60          # Seconds per extraction request
MAX_WEB_RESULTS = 10              # Max results returned per search
SEARXNG_URL_ENV = "SEARXNG_URL"
FIRECRAWL_API_KEY_ENV = "FIRECRAWL_API_KEY"
CAMOFOX_URL_DEFAULT = "http://localhost:9377"

# Pre-compiled regex for URL extraction from text
_URL_PATTERN = _re.compile(
    r'https?://[^\s<>"\')]+',
    _re.IGNORECASE,
)


@dataclass
class SearchResult:
    """Normalized web search result with source attribution."""
    title: str
    url: str
    snippet: str
    source: str  # "searxng", "firecrawl", "camofox"
    score: float = 0.0
    extracted_content: Optional[str] = None


class WebResearchClient:
    """Web research abstraction layer for the Deep Research Engine.

    Tries multiple backends in order of preference:
    1. SearXNG (self-hosted, privacy-first) — if configured
    2. Firecrawl API — if available
    3. Camofox browser fallback — fully local but slower

    All operations degrade gracefully — returns empty results rather than raising.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._searxng_url: str = cfg.get("searxng_url", os.environ.get(SEARXNG_URL_ENV, ""))
        self._firecrawl_key: str = cfg.get("firecrawl_api_key_env", os.environ.get("FIRECRAWL_API_KEY", ""))
        self._camofox_url: str = CAMOFOX_URL_DEFAULT

        # Lazy-init HTTP client on first use
        self._http_client: Optional[Any] = None

    def _get_http(self) -> Any:
        """Lazy-init httpx client. Returns _HTTP_UNAVAILABLE sentinel on failure."""
        if self._http_client is None:
            try:
                import httpx as _httpx  # noqa: F811
                self._http_client = _httpx.Client(timeout=WEB_SEARCH_TIMEOUT, follow_redirects=True)
            except ImportError:
                logger.debug("httpx not available — web research will use fallback methods")
                self._http_client = _HTTP_UNAVAILABLE  # Sentinel to prevent retry
        return self._http_client

    def search(self, query: str, top_k: int = MAX_WEB_RESULTS) -> List[SearchResult]:
        """Search the web for relevant information.

        Tries backends in order of preference. Returns normalized results
        with source attribution. Empty list if all backends unavailable.
        """
        if not query or not query.strip():
            return []

        top_k = min(top_k, MAX_WEB_RESULTS)
        results: List[SearchResult] = []

        # Try SearXNG first (privacy-first, self-hosted)
        if self._searxng_url:
            try:
                results = self._search_searxng(query, top_k)
                if results:
                    logger.debug("SearXNG returned %d results for '%s'", len(results), query[:50])
                    return results
            except Exception as e:
                logger.warning("SearXNG search failed: %s", e)

        # Try Firecrawl API
            try:
                results = self._search_firecrawl(query, top_k)
                if results:
                    logger.debug("Firecrawl returned %d results for '%s'", len(results), query[:50])
                    return results
            except Exception as e:
                logger.warning("Firecrawl search failed: %s", e)

        # No backends available — graceful degradation
        logger.debug("No web search backend available for query: '%s'", query[:50])
        return []

    def _search_searxng(self, query: str, top_k: int) -> List[SearchResult]:
        """Search via SearXNG instance."""
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:  # httpx unavailable
            return []

        url = f"{self._searxng_url.rstrip('/')}/search"
        resp = http.get(url, params={"q": query, "format": "json", "limit": top_k})
        resp.raise_for_status()
        data = resp.json()

        results: List[SearchResult] = []
        for item in (data.get("results") or [])[:top_k]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", "")[:300],
                source="searxng",
                score=float(item.get("score", 0)),
            ))
        return results

    def _search_firecrawl(self, query: str, top_k: int) -> List[SearchResult]:
        """Search via Firecrawl API."""
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:
            return []

        url = "https://api.firecrawl.dev/v1/search"
        resp = http.post(url, json={"query": query, "limit": top_k}, headers={
            "Content-Type": "application/json",
        })

        if resp.status_code != 200:
            return []

        data = resp.json()
        results: List[SearchResult] = []
        for item in (data.get("data") or [])[:top_k]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", "")[:300],
                source="firecrawl",
            ))
        return results

    def extract(self, url: str) -> Optional[str]:
        """Extract content from a URL using Firecrawl or Camofox fallback."""
        if not url:
            return None
        # Quick validation — don't try to scrape non-URLs
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return None

        # Try Firecrawl first (better structured extraction)
        try:
            content = self._extract_firecrawl(url)
            if content:
                return content
        except Exception as e:
                logger.warning("Firecrawl extraction failed for %s: %s", url[:80], e)

        # Fallback to Camofox browser scraping
        try:
            content = self._extract_camofox(url)
            if content:
                return content
        except Exception as e:
            logger.warning("Camofox extraction failed for %s: %s", url[:80], e)

        return None

    def _extract_firecrawl(self, url: str) -> Optional[str]:
        """Extract via Firecrawl scrape endpoint."""
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:
            return None

        resp = http.post(
            "https://api.firecrawl.dev/v1/scrape",
            json={"url": url, "formats": ["markdown"]},
            headers={
                "Content-Type": "application/json",
            },
            timeout=WEB_EXTRACT_TIMEOUT,
        )

        if resp.status_code != 200:
            return None

        data = resp.json()
        markdown = data.get("data", {}).get("markdown", "")
        return markdown[:10000] if markdown else None  # Cap at 10KB

    def _extract_camofox(self, url: str) -> Optional[str]:
        """Extract via Camofox browser automation."""
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:
            return None

        try:
            resp = http.post(
                f"{self._camofox_url}/scrape",
                json={"url": url},
                timeout=10,  # Short timeout for connection check; full extraction uses WEB_EXTRACT_TIMEOUT
            )
            data = resp.json()
            content = data.get("content") or data.get("text", "")
            return str(content)[:10000] if content else None
        except Exception:
            return None

    def batch_extract(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """Parallel extraction of multiple URLs. Returns dict[url, content]."""
        results: Dict[str, Optional[str]] = {}
        for url in urls:
            try:
                results[url] = self.extract(url)
            except Exception as e:
                logger.debug("Batch extract failed for %s: %s", url[:80], e)
                results[url] = None
        return results

    def generate_gap_queries(self, gaps: List[str]) -> List[str]:
        """Generate targeted search queries from gap descriptions.

        Example: gap "current pricing for RTX 5090" → query "RTX 5090 current price April 2026"
        """
        if not gaps:
            return []

        queries = []
        # Add temporal context to make searches more relevant
        from datetime import datetime
        now = datetime.now()
        time_context = f"{now.strftime('%B %Y')}"

        for gap in gaps:
            query = gap.strip().rstrip("?.!")
            # Append temporal context if not already present
            if not any(word in query.lower() for word in ["2026", "current", "latest", "recent"]):
                query = f"{query} {time_context}"
            queries.append(query)

        return queries[:5]  # Cap at 5 gap queries to prevent runaway research

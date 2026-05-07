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
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel: returned by _get_http() when httpx is unavailable, so subsequent
# calls don't waste time re-importing.
_HTTP_UNAVAILABLE: Any = object()

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
WEB_SEARCH_TIMEOUT = 30           # Seconds per search request
WEB_EXTRACT_TIMEOUT = 60          # Seconds per extraction request
MAX_WEB_RESULTS = 10              # Max results returned per search
SEARXNG_URL_ENV = "SEARXNG_URL"
FIRECRAWL_URL_ENV = "FIRECRAWL_URL"
FIRECRAWL_API_URL_ENV = "FIRECRAWL_API_URL"
CAMOFOX_URL_ENV = "CAMOFOX_URL"
CAMOFOX_URL_DEFAULT = "http://localhost:9377"

# Camofox-specific constants
CAMOFOX_PAGE_LOAD_DELAY = 1.5      # Seconds to wait after navigation
CAMOFOX_CLEANUP_TIMEOUT = 5        # Seconds for tab cleanup
CAMOFOX_CONTENT_MAX_CHARS = 10000  # Cap extracted text at 10KB
CAMOFOX_EVAL_EXPRESSION = "document.body.innerText"
CAMOFOX_SESSION_KEY = "extract"

# Pre-compiled regex for URL extraction from text
_URL_PATTERN = re.compile(
    r'https?://[^\s<>"\')]+',
    re.IGNORECASE,
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
        # Firecrawl: prefer config, then env vars (local first, then cloud)
        self._firecrawl_url: str = (
            cfg.get("firecrawl_url",
                os.environ.get(FIRECRAWL_URL_ENV, "") or
                os.environ.get(FIRECRAWL_API_URL_ENV, ""),
            )
        )
        # Camofox: prefer config, then env var, then default
        self._camofox_url: str = (
            cfg.get("camofox_url",
                os.environ.get(CAMOFOX_URL_ENV, CAMOFOX_URL_DEFAULT),
            )
        ).rstrip("/")

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

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    def search(self, query: str, top_k: int = MAX_WEB_RESULTS) -> List[SearchResult]:
        """Search the web for relevant information.

        Tries backends in order of preference. Returns normalized results
        with source attribution. Empty list if all backends unavailable.
        """
        if not query or not query.strip():
            return []

        top_k = min(top_k, MAX_WEB_RESULTS)

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
        resp = http.get(url, params={
            "q": query,
            "format": "json",
            "limit": top_k,
        })
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
        """Search via Firecrawl API (local instance preferred, cloud fallback)."""
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:
            return []

        # Prefer local instance if configured, otherwise fall back to cloud
        if self._firecrawl_url:
            url = f"{self._firecrawl_url.rstrip('/')}/v1/search"
        else:
            url = "https://api.firecrawl.dev/v1/search"

        try:
            resp = http.post(
                url,
                json={"query": query, "limit": top_k},
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug("Firecrawl search request failed: %s", e)
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

    # -----------------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------------

    def extract(self, url: str) -> Optional[str]:
        """Extract content from a URL using Firecrawl or Camofox fallback."""
        if not url:
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
        """Extract via Firecrawl scrape endpoint (local instance preferred, cloud fallback)."""
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:
            return None

        # Prefer local instance if configured, otherwise fall back to cloud
        if self._firecrawl_url:
            endpoint = f"{self._firecrawl_url.rstrip('/')}/v1/scrape"
        else:
            endpoint = "https://api.firecrawl.dev/v1/scrape"

        try:
            resp = http.post(
                endpoint,
                json={"url": url, "formats": ["markdown"]},
                headers={"Content-Type": "application/json"},
                timeout=WEB_EXTRACT_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug("Firecrawl extraction request failed for %s: %s", url[:80], e)
            return None

        data = resp.json()
        markdown = data.get("data", {}).get("markdown", "")
        return markdown[:CAMOFOX_CONTENT_MAX_CHARS] if markdown else None

    # -----------------------------------------------------------------------
    # Camofox extraction helpers (extracted from god function)
    # -----------------------------------------------------------------------

    def _camofox_create_tab(self, http: Any, base: str,
                            user_id: str, url: str) -> Optional[str]:
        """Create a Camofox tab and navigate to the given URL.

        Returns the tabId on success, None on failure.
        """
        try:
            resp = http.post(
                f"{base}/tabs",
                json={
                    "userId": user_id,
                    "sessionKey": CAMOFOX_SESSION_KEY,
                    "url": url,
                },
                timeout=WEB_EXTRACT_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug("Camofox tab creation failed for %s: %s", url[:80], e)
            return None

        data = resp.json()
        tab_id = data.get("tabId")
        if not tab_id:
            logger.debug("Camofox tab creation returned no tabId for %s", url[:80])
            return None

        return tab_id

    def _camofox_evaluate_page(self, http: Any, base: str,
                                tab_id: str, user_id: str) -> Optional[str]:
        """Evaluate JavaScript on a Camofox tab and return the result text.

        Returns the evaluated string on success, None on failure.
        """
        try:
            resp = http.post(
                f"{base}/tabs/{tab_id}/evaluate",
                json={
                    "userId": user_id,
                    "expression": CAMOFOX_EVAL_EXPRESSION,
                },
                timeout=WEB_EXTRACT_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.debug("Camofox evaluate failed for tab %s: %s", tab_id, e)
            return None

        eval_data = resp.json()
        content = eval_data.get("result", "")
        return str(content)[:CAMOFOX_CONTENT_MAX_CHARS] if content else None

    def _camofox_close_tab(self, http: Any, base: str,
                            tab_id: str, user_id: str) -> None:
        """Clean up a Camofox tab. Errors are logged but not raised."""
        try:
            http.delete(
                f"{base}/tabs/{tab_id}",
                params={"userId": user_id},
                timeout=CAMOFOX_CLEANUP_TIMEOUT,
            )
        except Exception as e:
            logger.debug("Camofox tab cleanup failed for %s: %s", tab_id, e)

    def _extract_camofox(self, url: str) -> Optional[str]:
        """Extract via Camofox browser automation using the tab-based API.

        Creates a tab, navigates to the URL, evaluates document.body.innerText,
        then closes the tab.
        """
        http = self._get_http()
        if http is _HTTP_UNAVAILABLE:
            return None

        base = self._camofox_url
        user_id = f"web_research_{uuid.uuid4().hex[:8]}"
        tab_id = None

        try:
            tab_id = self._camofox_create_tab(http, base, user_id, url)
            if tab_id is None:
                return None

            # Wait for page to load
            time.sleep(CAMOFOX_PAGE_LOAD_DELAY)

            content = self._camofox_evaluate_page(http, base, tab_id, user_id)
            return content
        finally:
            if tab_id:
                self._camofox_close_tab(http, base, tab_id, user_id)

    # -----------------------------------------------------------------------
    # Batch & gap utilities
    # -----------------------------------------------------------------------

    def extract_all(self, urls: List[str]) -> Dict[str, Optional[str]]:
        """Extract content from multiple URLs sequentially.

        Returns dict[url, content]. Each URL is processed independently;
        one failure does not affect the others.
        """
        results: Dict[str, Optional[str]] = {}
        for url in urls:
            try:
                results[url] = self.extract(url)
            except Exception as e:
                logger.debug("Batch extract failed for %s: %s", url[:80], e)
                results[url] = None
        return results

    # Legacy alias — kept for backward compatibility
    batch_extract = extract_all

    def generate_gap_queries(self, gaps: List[str]) -> List[str]:
        """Generate targeted search queries from gap descriptions.

        Example: gap "current pricing for RTX 5090"
        → query "RTX 5090 current price April 2026"
        """
        if not gaps:
            return []

        now = datetime.now(timezone.utc)
        time_context = now.strftime("%B %Y")

        queries = []
        for gap in gaps:
            query = gap.strip().rstrip("?.!")
            # Append temporal context if not already present
            if not any(word in query.lower()
                       for word in ("2026", "current", "latest", "recent")):
                query = f"{query} {time_context}"
            queries.append(query)

        return queries[:5]  # Cap at 5 gap queries to prevent runaway research

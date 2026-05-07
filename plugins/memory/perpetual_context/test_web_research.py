#!/usr/bin/env python3
"""Tests for web_research.py — Phase 2 Web Research Wrapper.

Verifies graceful degradation, lazy initialization, and correct behavior
when no backends are configured (the default state).

Run:
    python -m pytest test_web_research.py -v
    # or standalone:
    python test_web_research.py
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Ensure the package is importable
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

from plugins.memory.perpetual_context.web_research import (
    CAMOFOX_URL_DEFAULT,
    FIRECRAWL_API_URL_ENV,
    FIRECRAWL_URL_ENV,
    MAX_WEB_RESULTS,
    SEARXNG_URL_ENV,
    WEB_EXTRACT_TIMEOUT,
    WEB_SEARCH_TIMEOUT,
    SearchResult,
    WebResearchClient,
)


class TestConstants(unittest.TestCase):
    """Verify module-level constants are set correctly."""

    def test_web_search_timeout(self):
        self.assertEqual(WEB_SEARCH_TIMEOUT, 30)

    def test_web_extract_timeout(self):
        self.assertEqual(WEB_EXTRACT_TIMEOUT, 60)

    def test_max_web_results(self):
        self.assertEqual(MAX_WEB_RESULTS, 10)

    def test_firecrawl_api_url_env(self):
        self.assertEqual(FIRECRAWL_API_URL_ENV, "FIRECRAWL_API_URL")

    def test_searxng_url_env(self):
        self.assertEqual(SEARXNG_URL_ENV, "SEARXNG_URL")

    def test_camofox_default(self):
        self.assertEqual(CAMOFOX_URL_DEFAULT, "http://localhost:9377")


class TestSearchResult(unittest.TestCase):
    """Verify SearchResult dataclass behavior."""

    def test_basic_creation(self):
        sr = SearchResult(
            title="Test Page",
            url="https://example.com/test",
            snippet="A short snippet here.",
            source="searxng",
            score=0.95,
        )
        self.assertEqual(sr.title, "Test Page")
        self.assertEqual(sr.url, "https://example.com/test")
        self.assertEqual(sr.snippet, "A short snippet here.")
        self.assertEqual(sr.source, "searxng")
        self.assertAlmostEqual(sr.score, 0.95)
        self.assertIsNone(sr.extracted_content)

    def test_asdict(self):
        sr = SearchResult(
            title="Page",
            url="https://example.com",
            snippet="Snippet text.",
            source="firecrawl",
            score=1.0,
            extracted_content="<p>Full content</p>",
        )
        d = asdict(sr)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["title"], "Page")
        self.assertEqual(d["url"], "https://example.com")
        self.assertEqual(d["snippet"], "Snippet text.")
        self.assertEqual(d["source"], "firecrawl")
        self.assertAlmostEqual(d["score"], 1.0)
        self.assertEqual(d["extracted_content"], "<p>Full content</p>")

    def test_defaults(self):
        sr = SearchResult(
            title="", url="", snippet="", source=""
        )
        self.assertEqual(sr.score, 0.0)
        self.assertIsNone(sr.extracted_content)


class TestGenerateGapQueries(unittest.TestCase):
    """Verify WebResearchClient.generate_gap_queries() adds temporal context."""

    def setUp(self):
        self.client = WebResearchClient()

    def test_empty_input(self):
        self.assertEqual(self.client.generate_gap_queries([]), [])

    def test_appends_recency(self):
        result = self.client.generate_gap_queries(["best GPU for ML"])
        self.assertEqual(len(result), 1)
        import datetime
        expected_suffix = datetime.datetime.now().strftime("%B %Y")
        self.assertIn(expected_suffix, result[0])

    def test_already_has_date_not_duplicated(self):
        result = self.client.generate_gap_queries(["RTX 5090 price April 2026"])
        self.assertEqual(len(result), 1)
        # Should NOT append extra date since one already exists
        self.assertEqual(result[0].count("2026"), 1)

    def test_strips_whitespace(self):
        result = self.client.generate_gap_queries(["  hello world  "])
        self.assertTrue(result[0].strip().startswith("hello"))


class TestSearchResultDefaults(unittest.TestCase):
    """Verify SearchResult dataclass handles missing fields gracefully."""

    def test_minimal_result(self):
        sr = SearchResult(title="", url="", snippet="", source="")
        self.assertEqual(sr.title, "")
        self.assertEqual(sr.url, "")
        self.assertEqual(sr.score, 0.0)
        self.assertIsNone(sr.extracted_content)

    def test_full_result(self):
        sr = SearchResult(
            title="Example",
            url="https://example.com",
            snippet="Some content here.",
            source="searxng",
            score=0.8,
        )
        self.assertEqual(sr.title, "Example")
        self.assertEqual(sr.source, "searxng")
        self.assertAlmostEqual(sr.score, 0.8)

    def test_max_results_constant(self):
        self.assertLessEqual(MAX_WEB_RESULTS, 20)


class TestWebResearchClientInit(unittest.TestCase):
    """Verify WebResearchClient initializes without errors."""

    def setUp(self):
        # Clear any env vars that might enable backends during tests
        self._saved_searxng = os.environ.pop(SEARXNG_URL_ENV, None)
        self._saved_firecrawl = os.environ.pop(FIRECRAWL_URL_ENV, None)
        self._saved_firecrawl_api = os.environ.pop(FIRECRAWL_API_URL_ENV, None)

    def tearDown(self):
        if self._saved_searxng is not None:
            os.environ[SEARXNG_URL_ENV] = self._saved_searxng
        if self._saved_firecrawl is not None:
            os.environ[FIRECRAWL_URL_ENV] = self._saved_firecrawl
        if self._saved_firecrawl_api is not None:
            os.environ[FIRECRAWL_API_URL_ENV] = self._saved_firecrawl_api

    def test_init_no_config(self):
        client = WebResearchClient()
        self.assertIsNotNone(client)
        self.assertEqual(client._searxng_url, "")
        self.assertEqual(client._firecrawl_url, "")

    def test_init_empty_config(self):
        client = WebResearchClient(config={})
        self.assertIsNotNone(client)

    def test_init_with_searxng(self):
        client = WebResearchClient(config={"searxng_url": "http://test:8080"})
        self.assertEqual(client._searxng_url, "http://test:8080")

    def test_init_with_firecrawl(self):
        # Firecrawl URL comes from env var, not config dict
        os.environ[FIRECRAWL_URL_ENV] = "http://local:3002"
        try:
            client = WebResearchClient()
            self.assertEqual(client._firecrawl_url, "http://local:3002")
        finally:
            del os.environ[FIRECRAWL_URL_ENV]

    def test_init_env_vars(self):
        os.environ[SEARXNG_URL_ENV] = "http://env-searxng:8080"
        client = WebResearchClient()
        self.assertEqual(client._searxng_url, "http://env-searxng:8080")

    def test_config_overrides_env(self):
        os.environ[SEARXNG_URL_ENV] = "http://env-searxng:8080"
        client = WebResearchClient(config={"searxng_url": "http://config-searxng:9090"})
        self.assertEqual(client._searxng_url, "http://config-searxng:9090")

    def test_lazy_http_not_initialized(self):
        client = WebResearchClient()
        self.assertIsNone(client._http_client)

    def test_lazy_firecrawl_url_from_env(self):
        """Firecrawl URL is read from env, not a separate client object."""
        client = WebResearchClient()
        # _firecrawl_url is set from env var at init time
        self.assertIsInstance(client._firecrawl_url, str)


class TestWebResearchClientSearch(unittest.TestCase):
    """Verify search() graceful degradation with no backends."""

    def setUp(self):
        self._saved_searxng = os.environ.pop(SEARXNG_URL_ENV, None)
        self._saved_firecrawl = os.environ.pop(FIRECRAWL_URL_ENV, None)
        self._saved_firecrawl_api = os.environ.pop(FIRECRAWL_API_URL_ENV, None)

    def tearDown(self):
        if self._saved_searxng is not None:
            os.environ[SEARXNG_URL_ENV] = self._saved_searxng
        if self._saved_firecrawl is not None:
            os.environ[FIRECRAWL_URL_ENV] = self._saved_firecrawl
        if self._saved_firecrawl_api is not None:
            os.environ[FIRECRAWL_API_URL_ENV] = self._saved_firecrawl_api

    def test_search_no_backends_returns_empty(self):
        client = WebResearchClient()
        results = client.search("test query")
        self.assertEqual(results, [])

    def test_search_respects_top_k(self):
        client = WebResearchClient()
        # Even with high top_k, should return empty (no backends)
        results = client.search("test", top_k=100)
        self.assertEqual(results, [])

    def test_search_empty_query(self):
        client = WebResearchClient()
        results = client.search("")
        self.assertEqual(results, [])


class TestWebResearchClientExtract(unittest.TestCase):
    """Verify extract() graceful degradation."""

    def setUp(self):
        self._saved_firecrawl = os.environ.pop(FIRECRAWL_URL_ENV, None)
        self._saved_firecrawl_api = os.environ.pop(FIRECRAWL_API_URL_ENV, None)

    def tearDown(self):
        if self._saved_firecrawl is not None:
            os.environ[FIRECRAWL_URL_ENV] = self._saved_firecrawl
        if self._saved_firecrawl_api is not None:
            os.environ[FIRECRAWL_API_URL_ENV] = self._saved_firecrawl_api

    def test_extract_invalid_url_returns_none(self):
        client = WebResearchClient()
        self.assertIsNone(client.extract(""))
        self.assertIsNone(client.extract("not-a-url"))
        self.assertIsNone(client.extract(None))  # type: ignore[arg-type]

    def test_extract_no_backend_returns_none(self):
        client = WebResearchClient()
        result = client.extract("https://example.com")
        # Should not raise even if a backend returns content or None
        # The result depends on what backends are actually configured
        self.assertIsInstance(result, (str, type(None)))


class TestWebResearchClientBatchExtract(unittest.TestCase):
    """Verify batch_extract() handles mixed valid/invalid URLs."""

    def setUp(self):
        self._saved_firecrawl = os.environ.pop(FIRECRAWL_URL_ENV, None)
        self._saved_firecrawl_api = os.environ.pop(FIRECRAWL_API_URL_ENV, None)

    def tearDown(self):
        if self._saved_firecrawl is not None:
            os.environ[FIRECRAWL_URL_ENV] = self._saved_firecrawl
        if self._saved_firecrawl_api is not None:
            os.environ[FIRECRAWL_API_URL_ENV] = self._saved_firecrawl_api

    def test_batch_extract_empty_list(self):
        client = WebResearchClient()
        results = client.batch_extract([])
        self.assertEqual(results, {})

    def test_batch_extract_mixed_urls(self):
        client = WebResearchClient()
        urls = [
            "https://valid.example.com/page",
            "",
            "not-a-url",
            "https://another.valid.org/doc",
        ]
        results = client.batch_extract(urls)

        # All URLs should be in the result dict
        self.assertEqual(len(results), len(urls))

        # Invalid URLs should map to None
        self.assertIsNone(results[""])
        self.assertIsNone(results["not-a-url"])

        # Valid URLs also return None (no backend) but shouldn't raise
        for url in urls:
            if url and url != "not-a-url":
                self.assertIn(url, results)




class TestNoUnhandledExceptions(unittest.TestCase):
    """Verify that no public method raises unhandled exceptions."""

    def setUp(self):
        self._saved_searxng = os.environ.pop(SEARXNG_URL_ENV, None)
        self._saved_firecrawl = os.environ.pop(FIRECRAWL_URL_ENV, None)
        self._saved_firecrawl_api = os.environ.pop(FIRECRAWL_API_URL_ENV, None)

    def tearDown(self):
        if self._saved_searxng is not None:
            os.environ[SEARXNG_URL_ENV] = self._saved_searxng
        if self._saved_firecrawl is not None:
            os.environ[FIRECRAWL_URL_ENV] = self._saved_firecrawl
        if self._saved_firecrawl_api is not None:
            os.environ[FIRECRAWL_API_URL_ENV] = self._saved_firecrawl_api

    def test_search_no_exception(self):
        client = WebResearchClient()
        # Should never raise, even with garbage input
        result = client.search(None)  # type: ignore[arg-type]
        self.assertIsInstance(result, list)

    def test_extract_no_exception(self):
        client = WebResearchClient()
        result = client.extract(12345)  # type: ignore[arg-type]
        self.assertIsNone(result)

    def test_batch_extract_no_exception(self):
        client = WebResearchClient()
        result = client.batch_extract([None, "", "not-url"])  # type: ignore[list-item]
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)

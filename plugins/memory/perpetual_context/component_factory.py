"""Component Factory — lazy-initialization for all PerpetualContextProvider sub-components.

Encapsulates creation and caching of all lazy-initialized components. Reduces
the provider class by extracting the _ensure_* method family into one class
with proper locking and a clean ensure_* API.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Lazy-initialization factory for all PerpetualContextProvider sub-components."""

    def __init__(
        self,
        db: Any,
        session_id: str = "",
        current_depth: str = "moderate",
        prefetch_queue: list[dict[str, Any]] | None = None,
        deep_research_enabled: bool = True,
    ) -> None:
        self._db = db
        self._session_id = session_id
        self._current_depth = current_depth
        self._prefetch_queue = prefetch_queue or []
        self._deep_research_enabled = deep_research_enabled
        self._lock = threading.RLock()

        # Component caches
        self._extraction: Any = None
        self._tools: Any = None
        self._bridge_builder: Any = None
        self._scorer: Any = None
        self._feedback: Any = None
        self._web_research: Any = None
        self._scrutiny_gate: Any = None
        self._source_analyzer: Any = None
        self._synthesis_engine: Any = None
        self._retriever: Any = None

    # -- Properties for external access ---------------------------------------

    @property
    def extraction(self) -> Any:
        return self._extraction

    @property
    def tools(self) -> Any:
        return self._tools

    @property
    def bridge_builder(self) -> Any:
        return self._bridge_builder

    @property
    def scorer(self) -> Any:
        return self._scorer

    @property
    def feedback(self) -> Any:
        return self._feedback

    @property
    def web_research(self) -> Any:
        return self._web_research

    @property
    def scrutiny_gate(self) -> Any:
        return self._scrutiny_gate

    @property
    def source_analyzer(self) -> Any:
        return self._source_analyzer

    @property
    def synthesis_engine(self) -> Any:
        return self._synthesis_engine

    @property
    def retriever(self) -> Any:
        return self._retriever

    # -- Ensure methods ------------------------------------------------------

    def ensure_all(self) -> None:
        """Ensure all components are initialized."""
        with self._lock:
            self.ensure_feedback()
            self.ensure_core()
            if self._deep_research_enabled:
                self.ensure_deep_research()

    def ensure_core(self) -> None:
        """Ensure extraction engine, bridge builder, and tool handler are ready."""
        with self._lock:
            if self._extraction is None:
                from .context_bridge_builder import ContextBridgeBuilder  # noqa: PLC0415
                from .extraction_engine import ExtractionEngine  # noqa: PLC0415

                self._extraction = ExtractionEngine()
                self._bridge_builder = ContextBridgeBuilder(
                    extraction_engine=self._extraction,
                    scorer=self._scorer,
                    feedback_state=self._feedback,
                )
            if self._tools is None:
                from .tool_handler import ToolHandler  # noqa: PLC0415

                self._tools = ToolHandler(
                    db=self._db,
                    session_id=self._session_id,
                    current_depth=self._current_depth,
                    prefetch_queue=self._prefetch_queue,
                )

    def ensure_deep_research(self) -> None:
        """Ensure web research, scrutiny gate, source analyzer, and synthesis engine are ready."""
        with self._lock:
            self.ensure_web_research()
            self.ensure_scrutiny_gate()
            self.ensure_source_analyzer()
            self.ensure_synthesis_engine()

    def ensure_web_research(self) -> None:
        with self._lock:
            if self._web_research is None:
                from .web_research import (  # noqa: PLC0415
                    CAMOFOX_URL_DEFAULT,
                    CAMOFOX_URL_ENV,
                    FIRECRAWL_API_URL_ENV,
                    FIRECRAWL_URL_ENV,
                    SEARXNG_URL_ENV,
                    WebResearchClient,
                )

                searxng_url = os.environ.get(SEARXNG_URL_ENV, "").strip() or "http://localhost:8080"
                firecrawl_url = os.environ.get(FIRECRAWL_URL_ENV, "").strip() or os.environ.get(FIRECRAWL_API_URL_ENV, "").strip() or ""
                camofox_url = os.environ.get(CAMOFOX_URL_ENV, "").strip() or CAMOFOX_URL_DEFAULT
                self._web_research = WebResearchClient(
                    {
                        "searxng_url": searxng_url,
                        "firecrawl_url": firecrawl_url,
                        "camofox_url": camofox_url,
                    }
                )

    def ensure_scrutiny_gate(self) -> None:
        with self._lock:
            if self._scrutiny_gate is None:
                from .scrutiny_gate import ScrutinyGate  # noqa: PLC0415

                self._scrutiny_gate = ScrutinyGate()

    def ensure_source_analyzer(self) -> None:
        """Ensure the SourceAnalyzer is initialized.

        Lives in agent/ — not a plugin file. Standard Python import works fine.
        """
        with self._lock:
            if self._source_analyzer is None:
                from agent.source_analysis import SourceAnalyzer  # noqa: PLC0415

                self._source_analyzer = SourceAnalyzer()

    def ensure_synthesis_engine(self) -> None:
        with self._lock:
            if self._synthesis_engine is None:
                from .synthesis_engine import SynthesisEngine  # noqa: PLC0415

                self._synthesis_engine = SynthesisEngine()

    def ensure_feedback(self) -> None:
        """Ensure quality scorer and feedback state are ready."""
        with self._lock:
            if self._scorer is None:
                from .quality_scorer import BridgeQualityScorer  # noqa: PLC0415

                self._scorer = BridgeQualityScorer()
            if self._feedback is None:
                from .feedback_state import FeedbackState  # noqa: PLC0415

                self._feedback = FeedbackState()

    def ensure_retriever(self) -> None:
        """Ensure SmartRetriever is ready (for smart_retrieve)."""
        with self._lock:
            if self._retriever is None:
                from .retrieval_engine import SmartRetriever  # noqa: PLC0415

                self._retriever = SmartRetriever(self._db)

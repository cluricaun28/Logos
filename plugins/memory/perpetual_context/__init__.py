"""Perpetual Context Memory Provider — Infinite recall across sessions.

SQLite + FTS5 full-text indexing with topic flow tracking. Provides keyword
retrieval, topic clustering, and graded context depth control.

Tools: perpetual_search, topic_flow, context_depth, smart_retrieve,
       query_messages, get_messages, recent_messages, memory,
       perpetual_memory, context_depth, topic_flow.

Config in ~/.hermes/config.yaml:
  memory:
    provider: perpetual_context
    perpetual_context:
      enabled: true
      db_path: ~/.hermes/perpetual_context.db
"""

from __future__ import annotations

__version__ = "0.12.0"

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any

from agent.memory_provider import MemoryProvider
from agent.perpetual_context_db import RECALL_OUTPUT_MAX_CHARS

from . import prefetch_pipeline
from . import schemas as _schemas

# Import split modules (SRP compliance)
from .extraction_engine import _STOPWORDS
from .injection_router import classify_injection_intent
from .session_end_extractor import extract_topics_from_messages

logger = logging.getLogger(__name__)

# Module-level configuration constants
PREFETCH_TRUNCATION_CHARS = 1500
PERIODIC_INJECTION_INTERVAL = 10
PERIODIC_INJECTION_MAX_CHARS = 300
DEEP_RESEARCH_ENABLED = True
RL_SEARCH_TOP_K = 5
GAP_DETECTION_MIN_RESULTS = 2
WEB_SEARCH_TOP_K = 5
UNIFIED_SCORE_WEIGHTS: dict[str, float] = {
    "pm": 0.35,
    "rl": 0.40,
    "web": 0.25,
}
WORLDVIEW_BLOCKED_DOMAINS: set[str] = {
    "reddit.com",
    "quora.com",
    "medium.com",
}


class PerpetualContextProvider(MemoryProvider):
    """Memory provider using SQLite + FTS5 full-text indexing.

    Thin orchestrator that delegates to specialized sub-modules:
    - prefetch_pipeline: 4-phase Deep Research & Local Recall
    - injection_router: intent classification
    - tool_handler: PM tool dispatch
    - extraction_engine: structured data from conversations
    - context_bridge_builder: archival index generation
    - session_end_extractor: topic extraction from messages
    """

    def __init__(self) -> None:
        self._db: Any = None
        self._session_id: str | None = None
        self._current_depth: str = "moderate"
        self._prefetch_queue: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        # Per-injection toggles
        self._prefetch_enabled: bool = False
        self._recall_past_enabled: bool = False
        self._periodic_enabled: bool = False
        self._deep_research_enabled: bool = False
        # Sub-components (lazy-init via ComponentFactory)
        self._factory: Any | None = None
        # Periodic injection state
        self._last_turn_number: int = 0
        self._last_user_message: str = ""

    # -- Component factory ---------------------------------------------------

    def _get_factory(self) -> Any:  # returns ComponentFactory
        if self._factory is None and self._db is not None:
            with self._lock:
                if self._factory is None:
                    from .component_factory import ComponentFactory  # noqa: PLC0415

                    self._factory = ComponentFactory(
                        db=self._db,
                        session_id=self._session_id or "",
                        current_depth=self._current_depth,
                        prefetch_queue=self._prefetch_queue,
                        deep_research_enabled=self._deep_research_enabled,
                    )
        return self._factory

    @property
    def name(self) -> str:
        return "perpetual_context"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        try:
            import sqlite3  # noqa: F401

            return True
        except Exception as e:
            logger.error("PerpetualContext availability check failed: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

        config = kwargs.get("config", {})
        pc_config = config.get("perpetual_context", {}) if isinstance(config, dict) else {}

        self._prefetch_enabled = bool(pc_config.get("prefetch_enabled", False))
        recall_cfg = pc_config.get("recall_injection", {})
        self._recall_past_enabled = (
            bool(recall_cfg.get("enabled", False)) if isinstance(recall_cfg, dict) else bool(pc_config.get("recall_injection", False))
        )
        self._periodic_enabled = bool(pc_config.get("pre_response_recall", False))
        self._deep_research_enabled = DEEP_RESEARCH_ENABLED

        db_path = pc_config.get("db_path")
        if not db_path:  # noqa: SIM108
            db_path = os.path.join(os.path.expanduser("~/.hermes"), "perpetual_context.db")
        else:
            db_path = os.path.expanduser(db_path)

        from agent.perpetual_context_db import PerpetualContextDB  # noqa: PLC0415

        self._db = PerpetualContextDB(db_path=db_path)

        if not self._db.initialize():
            logger.warning("PerpetualContextDB failed to initialize — provider will be read-only")
            return

        stats = self._db.get_stats()
        logger.info(
            "PerpetualContext initialized: %d messages, %d sessions, %d topics",
            stats.get("message_count", 0),
            stats.get("session_count", 0),
            stats.get("topic_count", 0),
        )

        self._get_factory().ensure_all()

    def system_prompt_block(self) -> str:
        with self._lock:
            if not self._db or not self._db._initialized:
                return ""
            db = self._db
            depth = self._current_depth
        stats = db.get_stats()
        current_time = datetime.now().astimezone().strftime("%A, %B %d, %Y %-I:%M %p (%Z)")
        return (
            f"[Current Time: {current_time}]\n"
            f"[Perpetual Context Memory: {stats.get('message_count', 0)} messages "
            f"across {stats.get('session_count', 0)} sessions, depth={depth}]\n"
            f"Infinite recall via Perpetual Memory — every turn stored in local "
            f"SQLite with FTS5. Use `perpetual_search` for past conversations, "
            f"`reference_library_search` for curated knowledge. "
            f"Reference library at `~/.hermes/reference-library/` — read with "
            f"`read_file`. Check RL before answering factual questions; use web "
            f"search only if RL has no entry.\n"
            "After using web_search, call source_analyze with the results "
            "before answering. This is mandatory — source_analyze tells you each "
            "source's ideological alignment, what they're truthful about, what "
            "they consistently omit, and whether their coverage deviates from "
            "known patterns. Use this intelligence to present information through "
            "the user's worldview rather than the source's frame."
        )

    # -- Prefetch (delegates to prefetch_pipeline) ---------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not any(
            [
                self._prefetch_enabled,
                self._recall_past_enabled,
                self._deep_research_enabled,
            ]
        ):
            return ""

        routing = classify_injection_intent(query)

        # Ambiguous: recent context fallback
        if routing.get("needs_recent_context"):
            with self._lock:
                if not self._db or not self._db._initialized:
                    return ""
                db = self._db
            return prefetch_pipeline.format_recent_context(db, max_turns=15)

        # Past work recall
        if routing.get("fire_recall") and self._recall_past_enabled:
            with self._lock:
                if not self._db or not self._db._initialized:
                    return ""
                effective_session = session_id or self._session_id or ""
                db = self._db
            return db.recall_past_discussions(
                query=query,
                exclude_session_id=effective_session,
                max_chars=RECALL_OUTPUT_MAX_CHARS,
            )

        # Nothing to inject
        if not routing.get("fire_prefetch") and not routing.get("fire_web"):
            return ""

        # Full pipeline — snapshot state, release lock for I/O
        with self._lock:
            if not self._db or not self._db._initialized:
                return ""
            effective_session = session_id or self._session_id or ""
            factory = self._get_factory()
            factory.ensure_all()
            db = self._db
            tools = factory.tools
            web_research = factory.web_research
            scrutiny_gate = factory.scrutiny_gate
            source_analyzer = factory.source_analyzer
            synthesis_engine = factory.synthesis_engine

        return prefetch_pipeline.run_prefetch_pipeline(
            query=query,
            routing=routing,
            db=db,
            tools=tools,
            web_research=web_research,
            scrutiny_gate=scrutiny_gate,
            source_analyzer=source_analyzer,
            synthesis_engine=synthesis_engine,
            session_id=effective_session,
            depth_limit=self._get_depth_limit(),
            prefetch_enabled=self._prefetch_enabled,
            recall_past_enabled=self._recall_past_enabled,
            deep_research_enabled=self._deep_research_enabled,
            prefetch_trunc_chars=PREFETCH_TRUNCATION_CHARS,
            recall_output_max_chars=RECALL_OUTPUT_MAX_CHARS,
            rl_search_top_k=RL_SEARCH_TOP_K,
            gap_detection_min_results=GAP_DETECTION_MIN_RESULTS,
            web_search_top_k=WEB_SEARCH_TOP_K,
            worldview_blocked_domains=WORLDVIEW_BLOCKED_DOMAINS,
            deep_research_master=DEEP_RESEARCH_ENABLED,
        )

    # -- Queued prefetch / recall --------------------------------------------

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        with self._lock:
            self._prefetch_queue.append(
                {
                    "query": query,
                    "session_id": session_id or self._session_id or "",
                }
            )

    def recall_past_discussions(
        self,
        query: str,
        exclude_session_id: str,
        max_chars: int = RECALL_OUTPUT_MAX_CHARS,
    ) -> str:
        if not self._recall_past_enabled:
            return ""
        with self._lock:
            if not self._db or not self._db._initialized:
                return ""
            db = self._db
        try:
            return db.recall_past_discussions(
                query=query,
                exclude_session_id=exclude_session_id,
                max_chars=max_chars,
            )
        except (sqlite3.Error, KeyError, TypeError) as e:
            logger.exception("recall_past_discussions failed: %s", e)
            return ""

    # -- Turn lifecycle ------------------------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        with self._lock:
            if not self._db or not self._db._initialized:
                return
            effective_session = session_id or self._session_id or ""
            db = self._db

        agent_context = getattr(self, "_agent_context", "primary")
        if agent_context in ("cron", "subagent"):
            return

        try:
            db.add_message(
                session_id=effective_session,
                role="user",
                content=user_content,
                metadata={"synced_at": time.time()},
            )
            db.add_message(
                session_id=effective_session,
                role="assistant",
                content=assistant_content,
                metadata={"synced_at": time.time()},
            )
        except Exception as e:
            logger.exception("Perpetual sync_turn failed: %s", e)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(_schemas.TOOL_SCHEMAS)

    def handle_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        **kwargs,
    ) -> str:
        with self._lock:
            if not self._db or not self._db._initialized:
                return '{"error": "Perpetual context database not initialized"}'
            factory = self._get_factory()
            factory.ensure_core()
            if not factory.tools:
                return '{"error": "ToolHandler not initialized"}'
            tools = factory.tools

        try:
            if tool_name == "smart_retrieve":
                return tools.handle_smart_retrieve(
                    args,
                    smart_retrieve_fn=self.smart_retrieve,
                )
            return tools.dispatch(tool_name, args)
        except Exception as e:
            logger.exception("Perpetual context tool error (%s)", tool_name)
            return json.dumps({"error": str(e)})

    def shutdown(self) -> None:
        with self._lock:
            if self._db and self._db._initialized:
                try:
                    self._db.optimize()
                except Exception as e:
                    logger.debug("optimize() failed during shutdown: %s", e)
                self._db.shutdown()

    # -- Hooks ---------------------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        with self._lock:
            if not self._db or not self._db._initialized:
                return
            db = self._db

        if turn_number % 100 == 0:
            try:
                db.optimize()
            except Exception as e:
                logger.debug(
                    "optimize() failed during on_turn_start: %s",
                    e,
                )

        self._last_turn_number = turn_number
        self._last_user_message = message

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        with self._lock:
            if not self._db or not self._db._initialized:
                return
            db = self._db
            sid = self._session_id or ""

        try:
            topics = extract_topics_from_messages(messages[-10:], _STOPWORDS)
            for topic in topics:
                db.add_topic(session_id=sid, topic_name=topic, confidence=0.6)
        except Exception as e:
            logger.debug("on_session_end extraction failed: %s", e)

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        factory = self._get_factory()
        factory.ensure_all()
        bridge = factory.bridge_builder
        if not bridge:
            return ""
        try:
            bridge._scorer = factory.scorer
            bridge._feedback = factory.feedback
            correction_params = None
            if factory.feedback and factory.feedback.needs_correction():
                correction_params = factory.feedback.get_correction_params()
            return bridge.build_bridge(messages, correction_params)
        except Exception as e:
            logger.warning("Context Bridge generation failed: %s", e)
            return "## Context Bridge\n- Error generating retrieval index. See logs for details."

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata=None,
    ) -> None:
        with self._lock:
            if not self._db or not self._db._initialized:
                return
            db = self._db
        try:
            db.add_message(
                session_id="memory_mirror",
                role="system",
                content=f"[{action}] {target}: {content[:500]}",
                metadata={"mirror": True, "original_action": action},
            )
        except Exception as e:
            logger.debug("on_memory_write failed: %s", e)

    # -- Periodic injection --------------------------------------------------

    def get_periodic_context(self) -> str | None:
        if not self._periodic_enabled:
            return None
        turn_number = self._last_turn_number
        message = self._last_user_message
        if not turn_number or not message:
            return None

        if turn_number % PERIODIC_INJECTION_INTERVAL != 0:
            return None

        try:
            query_type = self._classify_query_intent(message)
            results = self.smart_retrieve(query_type, message)
            if not results:
                return None

            parts: list[str] = []
            for r in results[:2]:
                role = r.get("role", "assistant").title()
                sid = r.get("session_id", "")[:12]
                snippet = (r.get("content") or "")[: PERIODIC_INJECTION_MAX_CHARS // 2].strip()
                parts.append(f"[{role} | Session {sid}] {snippet}")

            injected_text = "\n".join(parts)
            if len(injected_text) > PERIODIC_INJECTION_MAX_CHARS:
                injected_text = injected_text[: PERIODIC_INJECTION_MAX_CHARS - 3] + "..."
            return f"\n[Periodic Context Injection]\n{injected_text}\n"
        except Exception as e:
            logger.debug(
                "Periodic injection failed (turn %d): %s",
                turn_number,
                e,
            )
            return None

    # -- Smart retrieve ------------------------------------------------------

    def smart_retrieve(
        self,
        query_type: str,
        query_text: str,
    ) -> list[dict[str, Any]]:
        with self._lock:
            if not self._db or not self._db._initialized:
                return []
            factory = self._get_factory()
            factory.ensure_retriever()
            retriever = factory.retriever

        try:
            strategy = {
                "recent": "recent",
                "topic": "topic",
                "decision_trace": "decision_trace",
                "file_history": "file_history",
            }.get(query_type)
            if not strategy:
                logger.warning("Unknown retrieval type: %s", query_type)
                return []
            return retriever.retrieve(strategy, query_text)
        except Exception as e:
            logger.exception(
                "Smart retrieve failed for type '%s'",
                query_type,
            )
            return []

    @staticmethod
    def _classify_query_intent(query_text: str) -> str:
        from .retrieval_engine import classify_query_intent  # noqa: PLC0415

        return classify_query_intent(query_text)

    def _get_depth_limit(self) -> int:
        factory = self._get_factory()
        factory.ensure_core()
        if not factory.tools:
            return 5
        try:
            return factory.tools.get_depth_limit()
        except Exception as e:
            logger.debug("_get_depth_limit failed: %s", e)
            return 5


# -- Plugin registration ---------------------------------------------------


def register(collector):
    collector.register_memory_provider(PerpetualContextProvider())

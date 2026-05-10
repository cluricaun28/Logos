"""Prefetch Pipeline — 4-phase Deep Research & Local Recall.

Extracted from PerpetualContextProvider.prefetch() for SRP compliance.
Orchestrates Reference Library search, Perpetual Memory search, gap detection,
web research, scrutiny vetting, and synthesis into a single formatted context block.

This module is stateless — it receives all dependencies as parameters and returns
a formatted string. No shared mutable state.
"""

from __future__ import annotations

import json
import logging
import time as _time
from typing import Any

logger = logging.getLogger(__name__)


def run_prefetch_pipeline(
    *,
    query: str,
    routing: dict[str, Any],
    db: Any,
    tools: Any,
    web_research: Any,
    scrutiny_gate: Any,
    source_analyzer: Any,
    synthesis_engine: Any,
    session_id: str,
    depth_limit: int,
    # Config flags
    prefetch_enabled: bool,
    recall_past_enabled: bool,
    deep_research_enabled: bool,
    # Module-level constants passed from __init__.py
    prefetch_trunc_chars: int,
    recall_output_max_chars: int,
    rl_search_top_k: int,
    gap_detection_min_results: int,
    web_search_top_k: int,
    worldview_blocked_domains: set[str],
    deep_research_master: bool,
) -> str:
    """Run the full 4-phase prefetch pipeline.

    Returns formatted context text to inject, or empty string.
    """
    from .topic_classifier import _classify_topic_stability  # noqa: PLC0415

    # --- Ambiguous: check recent context ---
    if routing.get("needs_recent_context"):
        return format_recent_context(db, max_turns=15)

    # --- Past work: recall cross-session ---
    if routing.get("fire_recall") and recall_past_enabled:
        return db.recall_past_discussions(
            query=query,
            exclude_session_id=session_id or "",
            max_chars=recall_output_max_chars,
        )

    # --- Nothing to inject ---
    if not routing.get("fire_prefetch") and not routing.get("fire_web"):
        return ""

    # --- Full pipeline ---
    parts: list[str] = []
    failures: list[str] = []  # Track which phases failed for user-visible summary
    rl_results_count = 0
    pm_results_count = 0
    gaps_detected = False
    pm_results: list[dict] = []  # Guard: used in gap detection even if Phase 1b fails

    # Phase 1a: Reference Library Search
    rl_data: dict = {}
    if routing.get("fire_prefetch") and prefetch_enabled and tools:
        try:
            rl_json = tools.handle_reference_library_search(
                {
                    "query": query,
                    "top_k": rl_search_top_k,
                }
            )
            rl_data = json.loads(rl_json)
            rl_results = rl_data.get("results", [])
            if rl_results:
                rl_parts = []
                for r in rl_results[:rl_search_top_k]:
                    name = r.get("name", "Unknown")
                    snippet = r.get("snippet", "")[:300]
                    score = r.get("score", 0)
                    rl_parts.append(f"[RL: {name} (score: {score})]\n{snippet}")
                parts.append("\n\n---\n\n".join(rl_parts))
                rl_results_count = len(rl_results)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.exception("Phase 1a RL search failed: %s", e)
            failures.append(f"RL search: {type(e).__name__}")

    # Phase 1b: Perpetual Memory Hybrid Search
    if routing.get("fire_prefetch") and prefetch_enabled:
        try:
            pm_results = db.hybrid_search(
                query=query,
                session_id=session_id if session_id else None,
                top_k=depth_limit,
            )
            if pm_results:
                pm_formatted = []
                for msg in pm_results[:depth_limit]:
                    role_label = msg["role"].upper()
                    content = msg.get("content", "")[:prefetch_trunc_chars]
                    score = msg.get("_score", 0)
                    pm_formatted.append(f"[PM: {role_label} (relevance: {score:.2f})]\n{content}")
                parts.append("\n\n---\n\n".join(pm_formatted))
                pm_results_count = len(pm_results)
        except (sqlite3.OperationalError, KeyError, TypeError, AttributeError) as e:
            logger.exception("Phase 1b PM hybrid search failed: %s", e)
            failures.append(f"PM search: {type(e).__name__}")

    # Phase 1c: Stability-Aware Gap Detection
    stability, half_life, web_threshold = _classify_topic_stability(query)

    if deep_research_enabled and parts:
        total_results = rl_results_count + pm_results_count

        if total_results < gap_detection_min_results:
            gaps_detected = True
            logger.debug("Gap: too few local results (%d)", total_results)
        else:
            all_rl_scores = sorted(
                [r.get("score", 0) for r in rl_data.get("results", [])[:rl_search_top_k]],
                reverse=True,
            )
            if len(all_rl_scores) >= 2 and all_rl_scores[1] > 0:
                best_rl_ratio = all_rl_scores[0] / all_rl_scores[1]
            elif len(all_rl_scores) == 1:
                best_rl_ratio = 2.0
            else:
                best_rl_ratio = 0

            all_pm_scores = [msg.get("_score", 0) for msg in pm_results[:depth_limit]]
            best_pm_norm = max(all_pm_scores, default=0)

            rl_confident = best_rl_ratio >= 3.0
            pm_confident = best_pm_norm >= 0.3

            if not (rl_confident or pm_confident):
                gaps_detected = True
                logger.debug(
                    "Gap: topic='%s' stability=%s, rl_ratio=%.1f pm=%.2f",
                    query[:50],
                    stability,
                    best_rl_ratio,
                    best_pm_norm,
                )

    # Phase 2: Web Research
    web_results: list[dict] = []
    should_search_web = routing.get("fire_web") or gaps_detected
    if deep_research_enabled and should_search_web and web_research is not None:
        try:
            _t0 = _time.monotonic()
            raw = web_research.search(query, top_k=web_search_top_k)
            _elapsed = _time.monotonic() - _t0
            logger.debug("Web search for '%s' returned %d results in %.1fs", query[:50], len(raw), _elapsed)
            for sr in raw:
                web_results.append(
                    {
                        "title": sr.title,
                        "url": sr.url,
                        "snippet": sr.snippet,
                        "source": sr.source,
                        "score": sr.score,
                        "extracted_content": sr.extracted_content,
                    }
                )
        except (ConnectionError, TimeoutError, KeyError, AttributeError) as e:
            logger.exception("Phase 2 web research failed: %s", e)
            failures.append(f"Web search: {type(e).__name__}")

    # Phase 3: Scrutiny Gate
    vetted_results: list[dict] = []
    if web_results and scrutiny_gate is not None:
        try:
            filtered = [r for r in web_results if not any(blocked in r.get("url", "") for blocked in worldview_blocked_domains)]
            scrutiny = scrutiny_gate.vet_results(filtered, query)
            vetted_results = scrutiny.get("vetted_results", [])
            flagged = scrutiny.get("rejected_results", [])
            if flagged:
                logger.debug("Scrutiny flagged %d results: %s", len(flagged), [f.get("reason", "") for f in flagged[:3]])
        except (KeyError, TypeError, AttributeError) as e:
            logger.exception("Phase 3 scrutiny gate failed, using unvetted results: %s", e)
            failures.append(f"Scrutiny gate: {type(e).__name__}")
            vetted_results = web_results  # Fallback: use unvetted

    # Phase 3.5: Source Analysis enrichment
    if vetted_results and source_analyzer is not None:
        try:
            vetted_results = source_analyzer.enrich_results(vetted_results, query)
        except (KeyError, TypeError, AttributeError) as e:
            logger.exception("Phase 3.5 source analysis failed: %s", e)
            failures.append(f"Source analysis: {type(e).__name__}")

    # Phase 4: Synthesis
    footer_parts: list[str] = []
    if vetted_results and synthesis_engine is not None:
        try:
            _t2 = _time.monotonic()
            sensitivity = "high" if stability == "volatile" else "low"
            synthesis = synthesis_engine.synthesize(
                facts=vetted_results,
                query=query,
                sensitivity=sensitivity,
            )
            _e2 = _time.monotonic() - _t2
            context_block = synthesis.get("context_block", "")
            if context_block:
                sources = ", ".join(r.get("source", "web") for r in vetted_results[:3])
                web_section = f"[Web Research Results (from {sources})]\n{context_block}"
                parts.append(web_section)
                logger.debug("Phase 4 synthesis produced %d bytes in %.1fs", len(context_block), _e2)

            rl_update_flags = synthesis.get("rl_update_flags", [])
            if rl_update_flags:
                relevant_flags = [f for f in rl_update_flags[:10] if "britannica" not in f.get("page", "").lower()][:2]
                if relevant_flags:
                    logger.info(
                        "RLUpdateDetector flagged %d relevant page(s) for review",
                        len(relevant_flags),
                    )
                    for flag in relevant_flags:
                        footer_parts.append(f"[RL Update: {flag.get('page', '').split('/')[-1]} — {flag.get('reason', '')[:120]}]")
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.exception("Phase 4 synthesis failed: %s", e)
            failures.append(f"Synthesis: {type(e).__name__}")
            if vetted_results:
                snippets = []
                for r in vetted_results[:3]:
                    snippets.append(f"[Web: {r.get('title', 'Unknown')} ({r.get('source', '')})]\n{r.get('snippet', '')[:200]}")
                if snippets:
                    parts.append("\n\n".join(snippets))

    # Combine and format
    if not parts:
        return ""

    result_text = "\n\n---\n\n".join(parts)

    if deep_research_master:
        web_results_count = len(vetted_results)
        footer_parts.append(
            f"[Recall: {rl_results_count} RL, {pm_results_count} PM{', ' + str(web_results_count) + ' web' if web_results_count else ''}]"
        )
        footer_parts.append(f"[Topic: {stability}]")
        if gaps_detected and not web_results_count:
            footer_parts.append("[Gap detected — local recall insufficient. Consider web research for current/external data.]")
        if failures:
            footer_parts.append(f"[Pipeline failures: {', '.join(failures[:3])}]")

    result_text += "\n\n" + " ".join(footer_parts)
    return result_text


def format_recent_context(db: Any, max_turns: int = 15) -> str:
    """Format recent PM messages with recency weighting."""
    try:
        recent = db.get_recent_messages(n=max_turns)
        if not recent:
            return ""

        total = len(recent)
        parts: list[str] = []
        for i, msg in enumerate(recent):
            recency = (i + 1) / (total + 1)
            if recency > 0.3:
                role_label = msg.get("role", "unknown").upper()
                content = msg.get("content", "")[:200]
                parts.append(f"[Recent ({role_label}, recency: {recency:.2f})] {content}")
        if parts:
            return "\n\n---\n\n".join(parts)
        return ""
    except (sqlite3.Error, KeyError, TypeError, AttributeError) as e:
        logger.exception("Recent context fetch failed: %s", e)
        return ""

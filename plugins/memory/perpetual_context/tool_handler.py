"""Tool Handler — dispatches Perpetual Memory tool calls.

All handler functions are defined here as module-level functions that accept
(tool_handler, args) and return a JSON string.  This keeps the module
self-contained so the Hermes plugin loader (which only discovers flat .py
files, not sub-packages) can import it without missing submodules.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_msg(msg: dict[str, Any], content_limit: int | None = None) -> dict[str, Any]:
    """Normalize a DB message dict into a consistent JSON-friendly format."""
    content = msg.get("content", "")
    if content_limit:
        content = content[:content_limit]
    time_val = msg.get("created_at") or msg.get("timestamp", 0)
    return {
        "id": msg["id"],
        "session_id": msg["session_id"],
        "role": msg["role"],
        "content": content,
        "created_at": time_val,
    }


def _msg_list_to_json(
    msgs: list[dict[str, Any]], content_limit: int | None = None
) -> list[dict[str, Any]]:
    """Format a list of DB message dicts into JSON-friendly dicts."""
    return [_format_msg(m, content_limit) for m in msgs]


# ---------------------------------------------------------------------------
# Search handlers
# ---------------------------------------------------------------------------


def _handle_perpetual_search(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "Query is required"})

    session_id = args.get("session_id") or None
    top_k = min(args.get("top_k", 5), 20)

    results = tool_handler._db.hybrid_search(
        query=query, session_id=session_id, top_k=top_k,
    )
    if not results:
        return json.dumps({"results": [], "message": f"No matches for '{query}'"})

    formatted = []
    for i, msg in enumerate(results):
        formatted.append({
            "rank": i + 1,
            "session_id": msg.get("session_id"),
            "role": msg.get("role"),
            "content": msg.get("content", "")[:500],
            "score": msg.get("_score", 0),
        })

    return json.dumps({
        "query": query,
        "results": formatted,
        "total_found": len(results),
    })


def _handle_session_search(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    query = args.get("query")
    limit = min(args.get("limit", 3), 5)

    if not query:
        return _handle_session_recent(tool_handler, limit)
    return _handle_session_query(tool_handler, query, limit)


def _handle_session_recent(tool_handler: ToolHandler, limit: int) -> str:
    try:
        recent_msgs = tool_handler._db.get_recent_messages(
            n=limit * 10, session_id=None, role=None,
        )

        seen: set[str] = set()
        session_ids: list[str] = []
        for msg in recent_msgs:
            sid = msg.get("session_id")
            if sid and sid not in seen:
                seen.add(sid)
                session_ids.append(sid)

        sessions: list[dict[str, Any]] = []
        for sid in session_ids[:limit]:
            info = tool_handler._db.get_session_info(sid)

            if info:
                msg_count = info.get("message_count", 0)
                topic_count = info.get("topic_count", 0)
                last_updated = info.get("last_updated", 0)
                created_at = info.get("first_updated", 0) or last_updated
            else:
                session_msgs = [m for m in recent_msgs if m.get("session_id") == sid]
                msg_count = len(session_msgs)
                topic_count = 0
                if session_msgs:
                    created_at = session_msgs[-1].get("created_at") or session_msgs[-1].get("timestamp", 0)
                    last_updated = session_msgs[0].get("created_at") or session_msgs[0].get("timestamp", 0)
                else:
                    created_at = 0
                    last_updated = 0

            preview = ""
            try:
                first_user = tool_handler._db.get_recent_messages(
                    n=1, session_id=sid, role="user"
                )
                if first_user:
                    preview = (first_user[0].get("content") or "")[:200]
            except Exception as e:
                logger.debug("Session preview query failed: %s", e)

            sessions.append({
                "session_id": sid,
                "message_count": msg_count,
                "topic_count": topic_count,
                "created_at": created_at or 0,
                "last_updated": last_updated or 0,
                "preview": preview,
            })

        return json.dumps({
            "mode": "recent",
            "sessions": sessions,
            "total_returned": len(sessions),
        })

    except Exception as e:
        logger.error("Session search (recent) failed: %s", e)
        return json.dumps({"error": str(e), "sessions": []})


def _handle_session_query(tool_handler: ToolHandler, query: str, limit: int) -> str:
    try:
        results = tool_handler._db.fts_search(query=query, top_k=min(limit * 5, 30))

        if not results:
            return json.dumps({
                "mode": "search",
                "query": query,
                "sessions": [],
                "message": f"No sessions found matching '{query}'",
            })

        session_map: dict[str, dict[str, Any]] = {}
        for msg in results:
            sid = msg.get("session_id")
            if sid not in session_map:
                session_map[sid] = {"session_id": sid, "messages": [], "max_score": 0}
            score = msg.get("_score", 0)
            session_map[sid]["messages"].append(msg)
            session_map[sid]["max_score"] = max(session_map[sid]["max_score"], score)

        sessions: list[dict[str, Any]] = []
        for sid, data in sorted(session_map.items(), key=lambda x: -x[1]["max_score"]):
            msgs = data["messages"]
            info = tool_handler._db.get_session_info(sid)
            msg_count = info["message_count"] if info else len(msgs)
            last_updated = info["last_updated"] if info else msgs[0].get("timestamp", 0)

            previews = [m.get("content", "")[:150] for m in msgs[:3] if m.get("content")]
            preview = "\n".join(previews) if previews else "No preview available"

            sessions.append({
                "session_id": sid,
                "message_count": msg_count,
                "matching_messages": len(msgs),
                "score": data["max_score"],
                "last_updated": last_updated,
                "preview": preview,
            })
            if len(sessions) >= limit:
                break

        return json.dumps({
            "mode": "search",
            "query": query,
            "sessions": sessions,
            "total_returned": len(sessions),
        })

    except Exception as e:
        logger.error("Session search (query) failed: %s", e)
        return json.dumps({"error": str(e), "sessions": []})


# ---------------------------------------------------------------------------
# Retrieval handlers
# ---------------------------------------------------------------------------


def _handle_smart_retrieve(tool_handler: ToolHandler, args: dict[str, Any], **kwargs: Any) -> str:
    query_type = args.get("query_type", "")
    query_text = args.get("query_text", "")
    smart_retrieve_fn: Callable | None = kwargs.get("smart_retrieve_fn")

    if not query_type or not query_text:
        return json.dumps({"error": "Both 'query_type' and 'query_text' are required"})

    valid_types = ("auto", "recent", "topic", "decision_trace", "file_history")
    if query_type not in valid_types:
        return json.dumps({
            "error": f"Invalid query_type '{query_type}'. Must be one of: {', '.join(valid_types)}"
        })

    try:
        result = smart_retrieve_fn(query_type, query_text) if smart_retrieve_fn else []

        formatted: list[dict[str, Any] | str] = []
        for item in result[:20]:
            if isinstance(item, dict):
                formatted.append({
                    "id": item.get("id"),
                    "session_id": item.get("session_id"),
                    "role": item.get("role"),
                    "content": (item.get("content") or "")[:500],
                    "_score": item.get("_score", 0),
                })
            else:
                formatted.append(str(item)[:500])

        return json.dumps({
            "query_type": query_type,
            "query_text": query_text,
            "results": formatted,
            "total_found": len(result),
        })

    except Exception as e:
        logger.exception("Smart retrieve handler failed for type '%s'", query_type)
        return json.dumps({"error": str(e)})


def _handle_get_messages(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return json.dumps({"error": "pattern is required"})

    results = tool_handler._db.search_messages_by_pattern(
        pattern=pattern,
        session_id=args.get("session_id") or None,
        role=args.get("role") or None,
        limit=min(args.get("limit", 50), 100),
    )

    if not results:
        return json.dumps({
            "pattern": pattern,
            "results": [],
            "message": f"No messages matching pattern '{pattern}'",
        })

    return json.dumps({
        "pattern": pattern,
        "results": _msg_list_to_json(results),
        "total_found": len(results),
    })


def _handle_recent_messages(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    n = min(args.get("n", 10), 50)

    results = tool_handler._db.get_recent_messages(
        n=n,
        session_id=args.get("session_id") or None,
        role=args.get("role") or None,
    )

    if not results:
        return json.dumps({"n": n, "results": [], "message": "No recent messages found"})

    return json.dumps({
        "n_requested": n,
        "results": _msg_list_to_json(results),
        "total_returned": len(results),
    })


def _handle_query_messages(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    try:
        ids_arg = args.get("ids")
        if isinstance(ids_arg, str):
            try:
                ids_list = json.loads(ids_arg)
            except (json.JSONDecodeError, TypeError):
                ids_list = None
        else:
            ids_list = ids_arg

        meta_val = args.get("metadata_value")
        if isinstance(meta_val, str):
            try:
                meta_val = json.loads(meta_val)
            except (json.JSONDecodeError, TypeError):
                pass

        result = tool_handler._db.query_messages(
            pattern=args.get("pattern"),
            session_id=args.get("session_id") or None,
            role=args.get("role") or None,
            ids=ids_list,
            time_start=args.get("time_start"),
            time_end=args.get("time_end"),
            min_tokens=args.get("min_tokens"),
            max_tokens=args.get("max_tokens"),
            metadata_key=args.get("metadata_key"),
            metadata_value=meta_val,
            query=args.get("query"),
            stats=args.get("stats", False),
            limit=min(args.get("limit", 100), 500),
            offset=args.get("offset", 0),
        )
        return json.dumps(result)

    except Exception as e:
        logger.exception("Query messages handler failed")
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Topic handlers
# ---------------------------------------------------------------------------

_DEPTH_LEVELS = ("broad_overview", "moderate", "deep", "expert")


def _handle_topic_flow(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    action = args.get("action", "list")
    session_id = args.get("session_id") or tool_handler._session_id or ""

    if action == "list":
        topics = tool_handler._db.get_topic_flow(session_id)
        return json.dumps({
            "session_id": session_id,
            "topics": topics,
            "total_topics": len(topics),
        })

    elif action == "add":
        topic_name = args.get("topic_name", "")
        if not topic_name:
            return json.dumps({"error": "topic_name is required for 'add' action"})

        confidence = args.get("confidence", 0.5)
        topic_id = tool_handler._db.add_topic(
            session_id=session_id,
            topic_name=topic_name,
            confidence=confidence,
        )

        if topic_id:
            return json.dumps({
                "action": "added",
                "topic_name": topic_name,
                "topic_id": topic_id,
            })
        return json.dumps({"error": f"Topic '{topic_name}' already exists or failed to add"})

    elif action == "drift_check":
        drift = tool_handler._db.detect_topic_drift(session_id)
        topics = tool_handler._db.get_topic_flow(session_id)
        return json.dumps({
            "action": "drift_check",
            "session_id": session_id,
            "drift_detected": drift,
            "topics": topics,
        })

    return json.dumps({"error": f"Unknown action: {action}. Use 'list', 'add', or 'drift_check'"})


def _handle_context_depth(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    action = args.get("action", "get")

    if action == "get":
        return json.dumps({
            "current_depth": tool_handler._current_depth,
            "available_levels": list(_DEPTH_LEVELS),
        })

    elif action == "set":
        level = args.get("level", "")
        if level not in _DEPTH_LEVELS:
            return json.dumps({
                "error": f"Invalid depth level. Choose from: {list(_DEPTH_LEVELS)}"
            })
        tool_handler._current_depth = level
        return json.dumps({"action": "set", "new_depth": level})

    elif action == "status":
        stats = tool_handler._db.get_stats() if tool_handler._db else {}
        return json.dumps({
            "depth_level": tool_handler._current_depth,
            "database_stats": stats,
            "prefetch_queue_size": len(tool_handler._prefetch_queue),
        })

    return json.dumps({"error": f"Unknown action: {action}. Use 'get', 'set', or 'status'"})


# ---------------------------------------------------------------------------
# Reference library handler
# ---------------------------------------------------------------------------


def _handle_reference_library_search(tool_handler: ToolHandler, args: dict[str, Any]) -> str:
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "query is required"})

    top_k = min(args.get("top_k", 5), 20)
    results: list[dict[str, Any]] = []
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]

    for subdir in ("topics", "entities"):
        ref_dir = Path(os.path.expanduser(f"~/.hermes/reference-library/{subdir}"))
        if not ref_dir.exists():
            continue

        for md_file in ref_dir.glob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8")
                score = sum(content.lower().count(w) for w in query_words)
                if not score:
                    continue

                entity_name = md_file.stem.replace("-", " ").title()
                if content.startswith("---"):
                    for line in content.split("\n")[1:5]:
                        if line.startswith(("name:", "topic:")):
                            entity_name = line.split(":", 1)[1].strip().strip('"')
                            break

                snippet_start = max(0, content.lower().find(query_lower) - 100)
                snippet = content[snippet_start:snippet_start + 300].replace("\n", " ").strip()

                results.append({
                    "file": str(md_file.name),
                    "directory": subdir,
                    "name": entity_name,
                    "score": score,
                    "snippet": snippet + "...",
                })
            except Exception as e:
                logger.debug("Ref library search failed for %s: %s", md_file.name, e)

    results.sort(key=lambda r: -r["score"])
    return json.dumps({"results": results[:top_k], "count": len(results[:top_k])})


def _handle_source_analyze(
    tool_handler: ToolHandler, args: dict[str, Any], **kwargs: Any
) -> str:
    """Handle source_analyze tool — delegate to SourceAnalyzer from agent/.

    Takes a JSON string of search results and returns source intelligence
    for each: alignment, known omissions, deviation flags, bias analysis.
    """
    results_raw = args.get("results", "[]")
    query = args.get("query", "")

    try:
        results = json.loads(results_raw)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid JSON for results: {e}"})

    if not isinstance(results, list):
        results = [results]

    if not results:
        return json.dumps({"results": [], "count": 0})

    try:
        from agent.source_analysis import SourceAnalyzer  # noqa: PLC0415

        analyzer = SourceAnalyzer()
        reports = analyzer.analyze_batch(results, query_context=query)

        # Build output
        output_results = []
        for report in reports:
            entry = {
                "url": report.url,
                "domain": report.source.domain,
                "cluster": report.source.cluster,
                "alignment": report.source.alignment,
                "reliability": report.source.reliability,
                "truthful_on": report.source.truthful_on,
                "omits": report.source.omits,
                "bias_score": report.content.bias_score,
                "markers": report.content.markers[:5],
                "deviation": report.narrative.deviation,
                "coordination": report.narrative.coordination,
            }
            output_results.append(entry)

            # Write findings back to RL
            if report.findings:
                analyzer.write_findings(report)

        return json.dumps({"results": output_results, "count": len(output_results)})
    except Exception as e:
        logger.debug("source_analyze failed: %s", e)
        return json.dumps({"error": f"Source analysis failed: {e}", "advice": "Use the raw search results as-is"})


# ---------------------------------------------------------------------------
# Dispatch table — built after all handlers are defined
# ---------------------------------------------------------------------------

_DISPATCH: dict[str, Callable] = {
    "perpetual_search": _handle_perpetual_search,
    "topic_flow": _handle_topic_flow,
    "context_depth": _handle_context_depth,
    "get_messages": _handle_get_messages,
    "recent_messages": _handle_recent_messages,
    "query_messages": _handle_query_messages,
    "reference_library_search": _handle_reference_library_search,
    "session_search": _handle_session_search,
    "source_analyze": _handle_source_analyze,
}


# ---------------------------------------------------------------------------
# ToolHandler — thin dispatcher
# ---------------------------------------------------------------------------


class ToolHandler:
    """Thin dispatcher holding shared state for handler functions.

    All handler functions live at module level and accept (self, args).
    This keeps the module self-contained for the Hermes plugin loader.
    """

    def __init__(
        self,
        db: Any,
        session_id: str = "",
        current_depth: str = "moderate",
        prefetch_queue: list | None = None,
    ):
        self._db = db
        self._session_id = session_id
        self._current_depth = current_depth
        self._prefetch_queue = prefetch_queue or []

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------

    def dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        """Route a tool_name to its handler function."""
        handler = _DISPATCH.get(tool_name)
        if handler is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        return handler(self, args)

    # -----------------------------------------------------------------------
    # smart_retrieve — called directly by __init__.py with kwargs
    # -----------------------------------------------------------------------

    def handle_smart_retrieve(self, args: dict[str, Any], **kwargs: Any) -> str:
        """Compatibility method for smart_retrieve (needs smart_retrieve_fn kwarg)."""
        return _handle_smart_retrieve(self, args, **kwargs)

    # -----------------------------------------------------------------------
    # Depth helpers
    # -----------------------------------------------------------------------

    def get_depth_limit(self) -> int:
        """Get the result limit based on current depth level."""
        return {"broad_overview": 3, "moderate": 5, "deep": 10, "expert": 20}.get(
            self._current_depth, 5
        )

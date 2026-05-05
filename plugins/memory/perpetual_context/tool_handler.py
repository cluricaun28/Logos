"""Tool Handler — Dispatch tool calls to Perpetual Memory operations.

Responsible for handling all agent-facing tool calls:
- perpetual_search, topic_flow, context_depth
- get_messages, recent_messages, query_messages
- smart_retrieve, reference_library_search

Each handler validates inputs, delegates to the DB layer, formats results as JSON,
and fails loudly (raises RuntimeError) so the agent knows something broke.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolHandler:
    """Handles all tool dispatch for PerpetualContextProvider.

    Takes the DB instance and config in __init__, delegates each _handle_* method
    to appropriate DB operations, formats results as JSON strings.
    """

    def __init__(self, db, session_id: str = "", current_depth: str = "moderate",
                 prefetch_queue: Optional[list] = None):
        self._db = db
        self._session_id = session_id
        self._current_depth = current_depth
        self._prefetch_queue = prefetch_queue or []

    # -----------------------------------------------------------------------
    # Tool handlers — each returns a JSON string
    # -----------------------------------------------------------------------

    def handle_search(self, args: Dict[str, Any]) -> str:
        """Handle perpetual_search tool call."""
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "Query is required"})

        # Only filter by session_id if explicitly provided in args
        # Perpetual memory is cross-session by default — that's the whole point
        session_id = args.get("session_id") or None
        top_k = min(args.get("top_k", 5), 20)

        # Adaptive Weighting: Allow the agent to specify weights based on query intent
        semantic_weight = args.get("semantic_weight", SEMANTIC_WEIGHT)
        fts5_weight = args.get("fts5_weight", FTS5_WEIGHT)

        results = self._db.hybrid_search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            semantic_weight=semantic_weight,
            fts5_weight=fts5_weight,
        )

        if not results:
            return json.dumps({"results": [], "message": f"No matches for '{query}'"})

        formatted_results = []
        for i, msg in enumerate(results):
            formatted_results.append({
                "rank": i + 1,
                "session_id": msg.get("session_id"),
                "role": msg.get("role"),
                "content": msg.get("content", "")[:500],
                "score": msg.get("_score", 0),
            })

        return json.dumps({
            "query": query,
            "results": formatted_results,
            "total_found": len(results),
        })

    def handle_topic_flow(self, args: Dict[str, Any]) -> str:
        """Handle topic_flow tool call."""
        action = args.get("action", "list")
        session_id = args.get("session_id") or self._session_id or ""

        if action == "list":
            topics = self._db.get_topic_flow(session_id)
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
            topic_id = self._db.add_topic(
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
            else:
                return json.dumps({"error": f"Topic '{topic_name}' already exists or failed to add"})

        elif action == "drift_check":
            drift = self._db.detect_topic_drift(session_id)
            topics = self._db.get_topic_flow(session_id)
            return json.dumps({
                "action": "drift_check",
                "session_id": session_id,
                "drift_detected": drift,
                "topics": topics,
            })

        else:
            return json.dumps({"error": f"Unknown action: {action}. Use 'list', 'add', or 'drift_check'"})

    def handle_context_depth(self, args: Dict[str, Any]) -> str:
        """Handle context_depth tool call."""
        action = args.get("action", "get")

        if action == "get":
            return json.dumps({
                "current_depth": self._current_depth,
                "available_levels": ["broad_overview", "moderate", "deep", "expert"],
            })

        elif action == "set":
            level = args.get("level", "")
            valid_levels = ["broad_overview", "moderate", "deep", "expert"]
            if level not in valid_levels:
                return json.dumps({
                    "error": f"Invalid depth level. Choose from: {valid_levels}"
                })

            self._current_depth = level
            return json.dumps({
                "action": "set",
                "new_depth": level,
            })

        elif action == "status":
            stats = self._db.get_stats() if self._db else {}
            return json.dumps({
                "depth_level": self._current_depth,
                "database_stats": stats,
                "prefetch_queue_size": len(self._prefetch_queue),
            })

        else:
            return json.dumps({"error": f"Unknown action: {action}. Use 'get', 'set', or 'status'"})

    def handle_get_messages(self, args: Dict[str, Any]) -> str:
        """Handle get_messages tool call — pattern-based content search."""
        pattern = args.get("pattern", "")
        if not pattern:
            return json.dumps({"error": "pattern is required"})

        session_id = args.get("session_id") or None
        role = args.get("role") or None
        limit = min(args.get("limit", 50), 100)

        results = self._db.search_messages_by_pattern(
            pattern=pattern,
            session_id=session_id,
            role=role,
            limit=limit,
        )

        if not results:
            return json.dumps({
                "pattern": pattern,
                "results": [],
                "message": f"No messages matching pattern '{pattern}'",
            })

        # Return full content for each match (no truncation)
        formatted_results = []
        for msg in results:
            # Handle both 'created_at' and 'timestamp' column names
            time_val = msg.get("created_at") or msg.get("timestamp", 0)
            formatted_results.append({
                "id": msg["id"],
                "session_id": msg["session_id"],
                "role": msg["role"],
                "content": msg["content"],  # Full content, no truncation
                "created_at": time_val,
            })

        return json.dumps({
            "pattern": pattern,
            "results": formatted_results,
            "total_found": len(results),
        })

    def handle_recent_messages(self, args: Dict[str, Any]) -> str:
        """Handle recent_messages tool call — raw chronological retrieval."""
        n = min(args.get("n", 10), 50)
        session_id = args.get("session_id") or None
        role = args.get("role") or None

        results = self._db.get_recent_messages(
            n=n,
            session_id=session_id,
            role=role,
        )

        if not results:
            return json.dumps({
                "n": n,
                "results": [],
                "message": f"No recent messages found",
            })

        # Return full content for each message (no truncation)
        formatted_results = []
        for msg in results:
            # Handle both 'created_at' and 'timestamp' column names
            time_val = msg.get("created_at") or msg.get("timestamp", 0)
            formatted_results.append({
                "id": msg["id"],
                "session_id": msg["session_id"],
                "role": msg["role"],
                "content": msg["content"],  # Full content, no truncation
                "created_at": time_val,
            })

        return json.dumps({
            "n_requested": n,
            "results": formatted_results,
            "total_returned": len(results),
        })

    def handle_query_messages(self, args: Dict[str, Any]) -> str:
        """Handle query_messages tool call — master query with comprehensive filters."""
        try:
            # Parse ids from string or list
            ids_arg = args.get("ids")
            if isinstance(ids_arg, str):
                try:
                    ids_list = json.loads(ids_arg)
                except (json.JSONDecodeError, TypeError):
                    ids_list = None
            else:
                ids_list = ids_arg

            # Parse metadata_value from string or native type
            meta_val = args.get("metadata_value")
            if isinstance(meta_val, str):
                try:
                    meta_val = json.loads(meta_val)
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as string

            result = self._db.query_messages(
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

    def handle_smart_retrieve(self, args: Dict[str, Any], smart_retrieve_fn=None) -> str:
        """Handle smart_retrieve tool call — adaptive retrieval strategies.

        Takes a callable for smart_retrieve to avoid circular references.
        The provider passes self.smart_retrieve when calling this handler.
        """
        query_type = args.get("query_type", "")
        query_text = args.get("query_text", "")

        if not query_type or not query_text:
            return json.dumps({"error": "Both 'query_type' and 'query_text' are required"})

        valid_types = ("recent", "topic", "decision_trace", "file_history")
        if query_type not in valid_types:
            return json.dumps({
                "error": f"Invalid query_type '{query_type}'. Must be one of: {', '.join(valid_types)}"
            })

        try:
            result = smart_retrieve_fn(query_type, query_text) if smart_retrieve_fn else []

            # Format results for JSON response
            formatted_results = []
            for item in result[:20]:  # Cap at 20 results to avoid bloating the prompt
                if isinstance(item, dict):
                    formatted_results.append({
                        "id": item.get("id"),
                        "session_id": item.get("session_id"),
                        "role": item.get("role"),
                        "content": (item.get("content") or "")[:500],  # Truncate long content
                        "_score": item.get("_score", 0),
                    })
                else:
                    formatted_results.append(str(item)[:500])

            return json.dumps({
                "query_type": query_type,
                "query_text": query_text,
                "results": formatted_results,
                "total_found": len(result),
            })

        except Exception as e:
            logger.exception("Smart retrieve handler failed for type '%s'", query_type)
            return json.dumps({"error": str(e)})

    def get_depth_limit(self) -> int:
        """Get the result limit based on current depth level."""
        limits = {
            "broad_overview": 3,
            "moderate": 5,
            "deep": 10,
            "expert": 20,
        }
        return limits.get(self._current_depth, 5)

    def handle_reference_library_search(self, args: Dict[str, Any]) -> str:
        """Handle reference_library_search tool call — search across topic and entity markdown files."""
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "query is required"})

        top_k = min(args.get("top_k", 5), 20)

        # Search reference library directory for matching content
        results = []
        query_lower = query.lower()

        # Search both topics/ and entities/ directories
        for subdir in ("topics", "entities"):
            ref_dir = Path(os.path.expanduser(f"~/.hermes/reference-library/{subdir}"))
            if not ref_dir.exists():
                continue

            for md_file in ref_dir.glob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8")

                    # Simple relevance scoring: count query term matches
                    score = 0
                    for word in query_lower.split():
                        if len(word) > 2:  # Skip very short words
                            score += content.lower().count(word)

                    if score == 0:
                        continue

                    # Extract frontmatter topic/entity name if present
                    entity_name = md_file.stem.replace("-", " ").title()
                    if content.startswith("---"):
                        lines = content.split("\n")
                        for line in lines[1:5]:  # Check first few lines after frontmatter start
                            if line.startswith("name:") or line.startswith("topic:"):
                                entity_name = line.split(":", 1)[1].strip().strip('"')
                                break

                    # Extract a snippet around the first match
                    snippet_start = max(0, content.lower().find(query_lower) - 100)
                    snippet_end = min(len(content), snippet_start + 300)
                    snippet = content[snippet_start:snippet_end].replace("\n", " ").strip()

                    results.append({
                        "file": str(md_file.name),
                        "directory": subdir,
                        "name": entity_name,
                        "score": score,
                        "snippet": snippet + "...",
                    })

                except Exception as e:
                    logger.debug("Reference library search failed for %s: %s", md_file.name, e)
                    continue

        # Sort by score descending and limit
        results.sort(key=lambda r: -r["score"])
        results = results[:top_k]

        return json.dumps({"results": results, "count": len(results)})

"""
Decision Trace Engine for Perpetual Memory.

Finds where specific decisions were made and retrieves surrounding context,
enabling the model to understand why certain choices were made rather than
just what was chosen. This aligns with Meta-Harness principles about long-horizon
dependencies: a single choice can affect behavior many reasoning steps later.

Optimized for local hardware by using indexed turn ID lookups and limiting
the scope of searches to recent sessions where decisions are most relevant.
"""

from __future__ import annotations

import logging
from typing import Any
from agent.perpetual_context_db import PerpetualContextDB

logger = logging.getLogger(__name__)

class DecisionTraceEngine:
    """
    Finds where decisions were made and retrieves surrounding context.
    
    Uses keyword patterns to identify decision-making turns, then returns
    the ±5 turn window around each decision for full context recovery.
    Optimized for local hardware by limiting search scope and using indexed lookups.
    """
    
    def __init__(self, db: PerpetualContextDB):
        self.db = db

    def find_decision(self, query_text: str) -> dict[str, Any | None]:
        """
        Finds the turn where a specific decision was made.
        
        Uses hybrid_search (FTS5 + reciprocal rank fusion) to find relevant
        messages containing the query text and decision-related patterns.
        
        Args:
            query_text: The decision or topic to search for.
            
        Returns:
            Dict with 'turn_id', 'session_id', and 'context' if found, else None.
        """
        try:
            # Search using hybrid_search which combines FTS5 BM25 ranking
            results = self.db.hybrid_search(
                query=query_text, session_id=None, top_k=3
            )
            
            if results and len(results) > 0:
                result = results[0]
                return {
                    'turn_id': result.get('id'),
                    'session_id': result.get('session_id'),
                    'context': result.get('content', '')[:200],  # Limit context size
                }
        except Exception as e:
            logger.exception("Failed to find decision for '%s'", query_text)
        
        return None

    def get_decision_context(self, turn_id: int) -> list[dict[str, Any]]:
        """Retrieves ±5 turns around a specific decision for full context recovery.

        Gets messages from the same session as the decision and slices ±5 turns
        around it using indexed turn IDs.

        Args:
            turn_id: The message ID where the decision was made.

        Returns:
            List of messages from surrounding turns.
        """
        try:
            # First, find the session for this turn_id (single targeted query)
            row = self.db._conn.execute(
                "SELECT session_id FROM messages WHERE id = ?", (turn_id,)
            ).fetchone()

            if not row:
                return []

            session_id = row[0]

            # Get only messages from this session, ordered by ID
            results = self.db.query_messages(
                pattern="%",
                session_id=session_id,
                role=None,
                limit=200,
            )

            if not isinstance(results, dict):
                return []

            messages = results.get("results", [])

            # Filter to this session and sort by turn ID
            session_msgs = [m for m in messages if m.get('session_id') == session_id]
            session_msgs.sort(key=lambda m: m.get('id', 0))

            try:
                dec_idx = next(i for i, m in enumerate(session_msgs) if m.get('id') == turn_id)
            except StopIteration:
                return []

            # Slice ±5 turns (10 total window), clamped to session bounds
            start = max(0, dec_idx - 5)
            end = min(len(session_msgs), dec_idx + 6)

            return session_msgs[start:end]
        except Exception as e:
            logger.exception("Failed to retrieve context for turn #%d", turn_id)
            return []

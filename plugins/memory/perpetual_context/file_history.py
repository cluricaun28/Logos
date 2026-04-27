"""
File History Tracker for Perpetual Memory.

Maintains precise edit history with turn-level references, enabling the model
to trace file changes back to their origin and understand the evolution of
code modifications over time. This supports Meta-Harness principles about
richer feedback accelerating learning.

Optimized for local hardware by using indexed lookups and limiting search scope.
"""

import logging
from typing import Optional, List, Dict, Any
from agent.perpetual_context_db import PerpetualContextDB

logger = logging.getLogger(__name__)


class FileHistoryTracker:
    """
    Tracks file edit history with precise turn references.
    
    Maintains a mapping of file paths to their edit history, including
    the turn ID where each edit occurred and any related discussion context.
    Optimized for local hardware by using indexed lookups and limiting search scope.
    """
    
    def __init__(self, db: PerpetualContextDB):
        self.db = db

    def get_file_history(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieves all edits to a specific file with turn references.
        
        Searches for tool calls (write_file, patch, read_file) that mention
        the given file path in recent sessions.
        
        Args:
            file_path: The path of the file to track.
            
        Returns:
            List of dicts containing edit details and turn IDs.
        """
        try:
            # Search for tool calls related to this file
            result = self.db.query_messages(
                pattern=f"%{file_path}%",
                role="tool",
                limit=50,
            )
            
            history = []
            if isinstance(result, dict):
                results_list = result.get("results", [])
                for r in results_list:
                    history.append({
                        'turn_id': r.get('id'),
                        'session_id': r.get('session_id'),
                        'content': r.get('content', '')[:200],  # Limit content size
                    })
            
            return history
        except Exception as e:
            logger.exception("Failed to retrieve file history for '%s'", file_path)
            return []

    def get_recent_edits(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the N most recent file edits across all sessions.
        
        Gets recent tool calls and filters for file operation types.
        
        Args:
            n: Number of recent edits to retrieve.
            
        Returns:
            List of dicts containing edit details and turn IDs.
        """
        try:
            # Get recent tool calls related to file operations
            results = self.db.get_recent_messages(n=n * 2, role="tool")
            
            edits = []
            for result in results:
                content = result.get('content', '')
                if any(op in content for op in ['write_file', 'patch', 'read_file']):
                    edits.append({
                        'turn_id': result.get('id'),
                        'session_id': result.get('session_id'),
                        'content': content[:200],  # Limit content size
                    })
                if len(edits) >= n:
                    break
            
            return edits
        except Exception as e:
            logger.exception("Failed to retrieve recent edits")
            return []

"""SynthesisService: Draft structured Markdown from high-signal PM clusters.

Stage 1 of the Logos Engine distillation pipeline. Takes raw transcripts from
Perpetual Memory and synthesizes declarative, technical entries following
'Frontier Lab' standards. Outputs to a temporary staging area for audit.

Design principles:
  - Atomic writes (temp file + os.replace) so RL is never partially written
  - Staging area isolation (~/.hermes/staging/) — nothing touches RL until approved
  - Provenance tracking: every draft includes source turn IDs in frontmatter
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SynthesisService:
    """Draft structured Markdown from high-signal PM clusters.

    Reads raw turns from Perpetual Memory, calls an LLM to synthesize a
    declarative entry, and saves it to the staging area for audit.
    """

    def __init__(self, db_path: Optional[Path] = None, staging_dir: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".hermes" / "perpetual_context.db"
        self.staging_dir = staging_dir or Path.home() / ".hermes" / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)

    def fetch_raw_turns(self, turn_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch raw message content from Perpetual Memory by IDs.

        Args:
            turn_ids: List of PM message IDs to retrieve.

        Returns:
            List of dicts with 'id', 'role', 'content' for each turn.
        """
        import sqlite3

        if not self.db_path.exists():
            logger.error(f"PM database not found at {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Build safe query with parameterized placeholders
            placeholders = ",".join("?" for _ in turn_ids)
            query = f"""
                SELECT id, role, content, topic_tags
                FROM messages
                WHERE id IN ({placeholders})
                ORDER BY timestamp ASC
            """
            cursor.execute(query, turn_ids)
            rows = cursor.fetchall()
            conn.close()

            turns = []
            for row in rows:
                turns.append({
                    "id": row[0],
                    "role": row[1],
                    "content": row[2] or "",
                    "topic_tags": json.loads(row[3]) if row[3] else [],
                })

            logger.info(f"Fetched {len(turns)} raw turns from PM (requested {len(turn_ids)})")
            return turns

        except Exception as e:
            logger.error(f"Failed to fetch turns from PM: {e}")
            return []

    def synthesize_draft(
        self,
        cluster_id: int,
        turn_ids: List[int],
        samples: Optional[List[str]] = None,
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Synthesize a structured Markdown draft from raw PM turns.

        Args:
            cluster_id: Unique identifier for this signal cluster.
            turn_ids: List of PM message IDs in this cluster.
            samples: Optional pre-fetched sample texts (from SignalRegistry).
            main_runtime: Dict with model/provider/base_url/api_key for LLM call.

        Returns:
            Tuple of (draft_path, metadata_dict) where metadata includes:
                - cluster_id, turn_ids, draft_title, word_count, created_at
        """
        # Fetch raw turns if samples not provided
        if not samples:
            turns = self.fetch_raw_turns(turn_ids)
            content_text = "\n\n---\n\n".join(
                f"[{t['role'].upper()}]: {t['content'][:500]}" for t in turns[:20]  # Limit to first 20 turns, 500 chars each
            )
        else:
            content_text = "\n\n---\n\n".join(s[:300] for s in (samples or []))  # Truncate samples

        # Build synthesis prompt following Frontier Lab standards
        prompt = self._build_synthesis_prompt(cluster_id, turn_ids, content_text)

        # Call LLM to generate draft
        try:
            from agent.auxiliary_client import call_llm

            call_kwargs = {
                "task": "archiving",  # Reuse archiving task config for synthesis
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,   # Quality over speed — overnight runs don't need to be fast
                "timeout": 600.0,     # 10 min timeout for thorough synthesis on big prompts
            }
            if main_runtime:
                call_kwargs["main_runtime"] = main_runtime

            response = call_llm(**call_kwargs)
            msg = response.choices[0].message
            draft_content = (msg.content or "").strip()

            # Handle reasoning models: if content is empty, use reasoning_content
            if not draft_content and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                logger.info("Reasoning model detected — using reasoning_content as draft")
                draft_content = msg.reasoning_content.strip()

            # Safety: if LLM returned garbage/empty, use fallback
            if len(draft_content) < 100:
                logger.warning(f"LLM returned minimal content ({len(draft_content)} chars), using fallback")
                draft_content = self._fallback_draft(cluster_id, turn_ids, content_text)

        except Exception as e:
            logger.warning(f"Synthesis LLM call failed (using fallback): {e}")
            # Fallback to static summary if LLM unavailable
            draft_content = self._fallback_draft(cluster_id, turn_ids, content_text)

        # Save to staging with atomic write
        draft_path = self.staging_dir / f"cluster_{cluster_id}_draft.md"
        metadata = {
            "cluster_id": cluster_id,
            "turn_ids": turn_ids,
            "draft_title": self._extract_title(draft_content),
            "word_count": len(draft_content.split()),
            "created_at": os.path.getmtime(self.staging_dir) if self.staging_dir.exists() else 0,
        }

        # Atomic write: temp file + rename
        fd, tmp_path = tempfile.mkstemp(dir=str(self.staging_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(draft_content)
            os.replace(tmp_path, draft_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info(f"Synthesis complete: cluster {cluster_id} → {draft_path} "
                     f"({metadata['word_count']} words)")
        return draft_path, metadata

    def _build_synthesis_prompt(
        self, cluster_id: int, turn_ids: List[int], content_text: str
    ) -> str:
        """Build the LLM prompt for synthesizing a structured Markdown entry."""
        return f"""You are creating a Reference Library entry from raw conversation transcripts.

CLUSTER ID: {cluster_id}
SOURCE TURN IDs: {turn_ids}

RAW TRANSCRIPTS:
{content_text}

SYNTHESIS INSTRUCTIONS (Frontier Lab Standards):
1. Write declarative facts, not instructions to yourself
2. Be specific with file paths, commands, error messages, and decisions
3. Include technical details that would be lost without explicit preservation
4. Structure as a standalone Markdown page suitable for the Reference Library
5. Use this exact format:

# [Title]

## Summary
[One paragraph overview of what this cluster contains]

## Key Facts
- [Declarative fact 1]
- [Declarative fact 2]

## Technical Details
[Any code snippets, commands, file paths, or specific values]

## Decisions & Rationale
[Why certain approaches were chosen]

## Provenance
Source: Perpetual Memory turns {turn_ids}
Cluster ID: {cluster_id}

Write only the Markdown entry. Do not include preamble or explanation."""

    def _fallback_draft(
        self, cluster_id: int, turn_ids: List[int], content_text: str
    ) -> str:
        """Generate a static fallback draft when LLM is unavailable."""
        return f"""# Cluster {cluster_id} — Pending Synthesis

## Summary
High-signal cluster from Perpetual Memory awaiting LLM synthesis. Contains {len(turn_ids)} turns.

## Raw Samples
{content_text[:1000]}...

## Provenance
Source: Perpetual Memory turns {turn_ids}
Cluster ID: {cluster_id}
Status: Awaiting synthesis (LLM unavailable)"""

    def _extract_title(self, content: str) -> str:
        """Extract the title from synthesized Markdown."""
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("# "):
                return line[2:].strip()
        return f"Cluster Draft (untitled)"

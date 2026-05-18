"""SynthesisService: Draft structured Markdown from high-signal PM clusters.

Stage 1 of the Logos Engine distillation pipeline. Takes raw transcripts from
Perpetual Memory and synthesizes declarative, technical entries following
'Frontier Lab' standards. Outputs to a temporary staging area for audit.

Design principles:
  - Atomic writes (temp file + os.replace) so RL is never partially written
  - Staging area isolation (~/.hermes/staging/) — nothing touches RL until approved
  - Provenance tracking: every draft includes source turn IDs in frontmatter
"""
from __future__ import annotations

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

        except (ImportError, json.JSONDecodeError, ModuleNotFoundError, sqlite3.Error, ValueError) as e:
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
            # Feed full turn content — no per-turn truncation.
            # Cap total input at ~32K chars to stay within reasonable context.
            # IMPORTANT: Include turn_id in the text so the synthesizer can cite it.
            content_text = "\n\n---\n\n".join(
                f"[turn_{t['id']} [{t['role'].upper()}]]: {t['content']}" for t in turns
            )
            # If still too large, trim from the end (oldest turns first after sort)
            if len(content_text) > 32000:
                content_text = content_text[:31000] + "\n\n[... remaining turns truncated for length ...]\n"
        else:
            content_text = "\n\n---\n\n".join(s for s in (samples or []))

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

        except (ImportError, ModuleNotFoundError) as e:
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
        except (OSError, PermissionError):
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        logger.info(f"Synthesis complete: cluster {cluster_id} → {draft_path} "
                     f"({metadata['word_count']} words)")
        # Store content_text on self for use by revise_draft in orchestrator retry
        self._last_content_text = content_text
        return draft_path, metadata

    def _build_synthesis_prompt(
        self, cluster_id: int, turn_ids: List[int], content_text: str
    ) -> str:
        """Build the LLM prompt for synthesizing a structured Markdown entry.

        Produces structured factual notes with source turn citations, not prose.
        This prevents the model from inventing connecting narrative.
        """
        # Extract only the turn IDs that actually appear in the (possibly truncated) content
        # so the model doesn't cite turns it can't see.
        import re
        visible_turn_ids = [int(m) for m in re.findall(r'turn_(\d+)', content_text)]
        visible_turn_ids = sorted(set(visible_turn_ids))

        return f"""You are extracting factual information from raw conversation transcripts into a Reference Library entry.

CLUSTER ID: {cluster_id}
VISIBLE TURN IDs (the only turns provided below — do NOT cite any others): {visible_turn_ids}

RAW TRANSCRIPTS (use these as your ONLY source of facts):
{content_text}

YOUR OUTPUT FORMAT — EXACTLY THIS STRUCTURE:

# [Title]

## Summary
[2-3 sentences describing what this conversation cluster is about. Only state what is clearly evident from the transcripts.]

## Factual Notes
Each bullet should be a single, verifiable fact. Every bullet SHOULD end with its source turn ID in square brackets when you can identify it.

Format: `- Fact statement [turn_N]`

Example:
- System defaults to `all-MiniLM-L6-v2` (384-dim) via local CPU inference [turn_47]
- User stated they did not need to switch to LM Studio unless a problem existed [turn_46]
- LM Studio embedding support remains as optional manual configuration [turn_47]

RULES FOR EACH BULLET:
- Only include information that appears EXPLICITLY in the transcripts
- If you can identify the specific turn, cite it: [turn_N]
- If the fact is in the transcripts but you cannot pinpoint the exact turn, use [uncited] — do NOT invent a turn ID
- If two turns contradict, cite both and note the contradiction
- DO NOT combine information from multiple turns into one bullet unless you cite all relevant turns or use [uncited]
- DO NOT infer conclusions not stated in the source
- DO NOT write narrative prose between bullets
- NEVER fabricate a turn ID to satisfy the citation requirement

## Code & Commands
[Any exact code snippets, commands, or config values shown in the transcripts. Cite turn ID after each block.]

## Decisions & Their Direction
State what was decided AND the user's reasoning in their own words where possible. Cite turn ID.
- What was DECIDED: ...
- What was REJECTED or SET ASIDE: ...
- User's stated reasoning: ...

## Open Questions / Unresolved Items
[Any issues the conversation left unresolved or marked for later. Cite turn ID.]

## Provenance
Source: Perpetual Memory turns {turn_ids}
Cluster ID: {cluster_id}

NEGATIVE CONSTRAINTS — VIOLATING ANY OF THESE IS A FAILURE:
- NEVER invent a specific value (model name, version, price, file path, port, command flag) not in the source
- NEVER claim causation unless the source explicitly states it
- NEVER smooth over contradictions — present both sides
- NEVER invert the direction of a user's decision ("I don't need X" means rejection, not adoption)
- NEVER write connecting narrative between facts
- If a detail is genuinely absent from all turns, write "[not in source]" rather than guessing

Write only the Markdown entry. No preamble. No explanation. No meta-commentary."""

    def revise_draft(
        self,
        cluster_id: int,
        turn_ids: List[int],
        content_text: str,
        draft_content: str,
        corrections: List[str],
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Revise a draft based on audit corrections.

        Args:
            cluster_id: Cluster identifier.
            turn_ids: Source PM turn IDs.
            content_text: Raw transcript text (same as original synthesis).
            draft_content: The draft that failed audit.
            corrections: List of specific corrections from the audit.
            main_runtime: LLM runtime config.

        Returns:
            Revised draft content as a string.
        """
        correction_text = "\n".join(f"- {c}" for c in corrections)

        prompt = f"""You are revising a Reference Library entry based on audit feedback.

CLUSTER ID: {cluster_id}
SOURCE TURN IDs: {turn_ids}

RAW TRANSCRIPTS (use these as your ONLY source of facts):
{content_text}

DRAFT THAT FAILED AUDIT:
{draft_content}

AUDIT CORRECTIONS (fix each one):
{correction_text}

INSTRUCTIONS:
- Fix every issue listed in the corrections.
- Every factual bullet MUST end with its source turn ID in [turn_N] format.
- DO NOT invent new specific values to fix the corrections. If the source doesn't contain the detail, write "[not specified in source]" instead.
- Preserve all correct, properly-cited bullets from the draft. Only change what the audit flagged.
- Maintain the same structure: Summary, Factual Notes (with citations), Code & Commands, Decisions & Their Direction, Open Questions, Provenance.
- NO connecting narrative between facts.

Write only the revised Markdown entry. Do not include preamble or explanation."""

        try:
            from agent.auxiliary_client import call_llm

            call_kwargs = {
                "task": "archiving",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,
                "timeout": 600.0,
            }
            if main_runtime:
                call_kwargs["main_runtime"] = main_runtime

            response = call_llm(**call_kwargs)
            msg = response.choices[0].message
            revised = (msg.content or "").strip()

            if not revised and hasattr(msg, 'reasoning_content') and msg.reasoning_content:
                revised = msg.reasoning_content.strip()

            if len(revised) < 100:
                logger.warning(f"Revision returned minimal content ({len(revised)} chars), returning original draft")
                return draft_content

            return revised

        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Revision LLM call failed (returning original draft): {e}")
            return draft_content

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

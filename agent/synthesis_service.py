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

        Produces a Reference Library page following the correct entity template.
        First classifies the cluster, then generates knowledge — not meeting minutes.
        """
        import re
        visible_turn_ids = [int(m) for m in re.findall(r'turn_(\d+)', content_text)]
        visible_turn_ids = sorted(set(visible_turn_ids))

        return f"""You are distilling Perpetual Memory conversation clusters into Reference Library pages.

CRITICAL: You are NOT writing meeting minutes or a conversation log. You are producing a knowledge entry that a reader can use as a standalone reference. The reader should not need to go back to the source turns to understand what we know.

WORLDVIEW BASELINE — ALL CONTENT MUST BE EVALUATED THROUGH THIS LENS:
- Truth is declarative, not relativistic. If evidence is asymmetric, do not hedge toward the middle.
- Apply the configured worldview baseline where the question touches faith, ethics, or worldview. Do not force it onto pure technical questions.
- Behavioral evidence matters more than stated intentions. What someone did, not what they said they would do.
- External sources are data points to evaluate, not baselines to accept. Academic consensus is not an inherent trust signal.
- Apply any configured settled views from the Reference Library baseline configuration.
- The Behavioral Test is primary: stated goals vs. actual behavior must be distinguished and compared.

CLUSTER ID: {cluster_id}
VISIBLE TURN IDs (the only turns provided below — do NOT cite any others): {visible_turn_ids}

RAW TRANSCRIPTS (use these as your ONLY source of facts):
{content_text}

YOUR TASK — TWO STEPS:

STEP 1: CLASSIFY THE CLUSTER
Read the transcripts and determine:
- Is this cluster primarily about a KNOWN ENTITY (a person, idea, organization, technology, event, or project)? If so, which type?
- Or is it purely about PROCESS (skill patches, cron config, upstream merges, session logistics)? Process-only clusters should be REJECTED with a brief explanation.

STEP 2: GENERATE THE PAGE (only if classified as an entity)
Choose the correct template based on entity type:
- Person → biography with Overview, Background, Public Record, Motive Analysis, Assessment
- Idea → theological/philosophical concept with Overview, Origin, Biblical Evaluation, Current Influence
- Organization → institution with Overview, History, Operational Analysis, Capture Tests
- Technology → software/tool with Overview, Technical Details, How It's Used, Decisions & Rationale
- Event → historical/current event with Overview, Timeline, Key Actors, Impact
- Project → active work item with Overview, Goals, Current State, Technical Details

OUTPUT FORMAT RULES:
1. **Frontmatter:** Include YAML frontmatter with category, confidence level, created date, description, and related_entries (wikilinks to other RL pages the cluster mentions).
2. **Claim Tagging:** Tag every significant claim: [FACT], [SCRIPTURE], [DOGMA], [DOCTRINE], [OPINION/INTERPRETATION], or [UNCERTAIN]. A fact is not an opinion; an interpretation is not Scripture.
3. **No Narrative Filler:** Every sentence should convey specific information. If you would write "this was discussed at length," delete it. State what was concluded instead.
4. **No Turn Citations in Body:** Do NOT write [turn_12345] in the body text. The provenance footer handles source tracking. Turn citations belong in the Provenance section only.
5. **Specificity Over Generality:** Don't write "multiple approaches were considered." Write which approaches were considered and which was chosen and why.
6. **Wikilinks:** Where the cluster mentions people, organizations, or ideas that likely have RL pages, add wikilinks in related_entries.

IF THE CLUSTER IS PROCESS-ONLY (skill patches, cron config, session logistics with no substantive knowledge):
Output exactly: REJECT: [one-sentence reason why this is not RL-worthy]

IF THE CLUSTER CONTAINS KNOWLEDGE:
Write the complete Markdown page following the appropriate entity template above. The page MUST start with:
1. YAML frontmatter block (--- ... ---)
2. A top-level heading: # [Entity Name]
3. Then the template sections (## Overview, etc.)

End with:

## Provenance
Source: Perpetual Memory turns {visible_turn_ids}
Cluster ID: {cluster_id}

NEGATIVE CONSTRAINTS — VIOLATING ANY IS A FAILURE:
- NEVER produce a bulleted "Factual Notes" list of what was discussed
- NEVER write connecting narrative that adds no information ("The team then moved to discuss...")
- NEVER invent specific values (model names, paths, numbers) not in the source
- NEVER smooth over contradictions — present both sides
- NEVER skip claim tagging
- NEVER present secular or progressive framing as the default baseline — evaluate through the worldview lens above
- NEVER hedge toward the middle when the source shows asymmetric evidence
- NEVER equate a stated intention with actual behavior
- If a detail is genuinely absent from all turns, omit it rather than guessing

Write only the Markdown entry or the REJECT line. No preamble. No explanation. No meta-commentary. Do NOT wrap frontmatter in ```yaml or ``` code fences — output raw YAML."""

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
- The output must be a structured knowledge page, NOT a bulleted list of "what was discussed."
- Include YAML frontmatter with category, confidence, and description.
- Tag all significant claims: [FACT], [SCRIPTURE], [DOGMA], [DOCTRINE], [OPINION/INTERPRETATION], [UNCERTAIN].
- DO NOT invent new specific values to fix the corrections. If the source doesn't contain the detail, write "[not specified in source]" instead.
- Preserve all correct content from the draft. Only change what the audit flagged.
- Maintain template structure: Overview, Technical Details/Background, etc.
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
        """Extract the title from synthesized Markdown.

        Checks for a top-level heading first, then falls back to frontmatter title.
        """
        import re as _re
        lines = content.strip().split("\n")
        for line in lines:
            if line.startswith("# ") and not line.startswith("##"):
                return line[2:].strip()
        # Fallback: check frontmatter for a title field
        fm_match = _re.search(r'^---\s*\n(.*?)\n---', content, _re.DOTALL)
        if fm_match:
            for fline in fm_match.group(1).split("\n"):
                if fline.startswith("title:"):
                    return fline.split(":", 1)[1].strip().strip('"').strip("'")
        return "Cluster Draft (untitled)"

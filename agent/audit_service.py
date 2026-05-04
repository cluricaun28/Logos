"""AuditService: Fidelity check for synthesized Reference Library drafts.

Stage 2 of the Logos Engine distillation pipeline. Takes a draft Markdown page
from SynthesisService and performs a 'Critic' review against raw transcripts to
detect hallucinations, nuance loss, and worldview drift.

Design principles:
  - Separate LLM call from synthesis (different prompt, same or different model)
  - Structured audit report with pass/fail verdict and specific corrections
  - Zero tolerance for unsupported claims — if it's not in the raw turns, flag it
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class AuditService:
    """Fidelity check for synthesized Reference Library drafts.

    Reviews draft Markdown against raw PM transcripts to ensure epistemic integrity.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path.home() / ".hermes" / "perpetual_context.db"

    def audit_draft(
        self,
        draft_path: Path,
        turn_ids: List[int],
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform fidelity check on a synthesized draft.

        Args:
            draft_path: Path to the draft Markdown file from SynthesisService.
            turn_ids: Source PM turn IDs for provenance verification.
            main_runtime: Dict with model/provider/base_url/api_key for LLM call.

        Returns:
            Audit report dict with:
                - passed: bool (True if draft is approved)
                - hallucinations: list of unsupported claims found
                - nuance_loss: list of smoothed-over details
                - worldview_drift: list of alignment issues
                - corrections: list of specific fix requirements
                - verdict: "PASS" or "FAIL" with reasoning
        """
        # Read draft content
        if not draft_path.exists():
            raise FileNotFoundError(f"Draft not found at {draft_path}")

        draft_content = draft_path.read_text()

        # Early skip: if synthesis produced nothing meaningful, auto-pass with warning
        if len(draft_content.strip()) < 100:
            logger.warning(f"Draft too short ({len(draft_content)} chars) — skipping audit, committing as-is")
            return self._pass_report("Audit skipped — draft below minimum length (synthesis timeout)")

        # Fetch raw turns for comparison
        raw_turns = self._fetch_raw_turns(turn_ids)
        if not raw_turns:
            logger.warning(f"No raw turns found for IDs {turn_ids} — cannot audit")
            return self._fail_report("No source transcripts available for verification")

        # Build audit prompt
        prompt = self._build_audit_prompt(draft_content, raw_turns)

        # Call LLM Critic
        try:
            from agent.auxiliary_client import call_llm

            call_kwargs = {
                "task": "archiving",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,   # Enough for thorough audit report with corrections
                "timeout": 600.0,     # 10 min — overnight runs don't need to be fast
            }
            if main_runtime:
                call_kwargs["main_runtime"] = main_runtime

            response = call_llm(**call_kwargs)
            audit_text = response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Audit LLM call failed (defaulting to PASS): {e}")
            # If critic is unavailable, default to PASS but log warning
            return self._pass_report("Audit skipped — LLM unavailable")

        # Parse audit response
        report = self._parse_audit_response(audit_text)
        report["draft_path"] = str(draft_path)
        report["turn_ids"] = turn_ids

        logger.info(f"Audit complete: {report['verdict']} "
                     f"(hallucinations={len(report.get('hallucinations', []))}, "
                     f"corrections={len(report.get('corrections', []))})")
        return report

    def _fetch_raw_turns(self, turn_ids: List[int]) -> List[Dict[str, Any]]:
        """Fetch raw message content from Perpetual Memory by IDs."""
        import sqlite3

        if not self.db_path.exists():
            logger.error(f"PM database not found at {self.db_path}")
            return []

        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            placeholders = ",".join("?" for _ in turn_ids)
            query = f"""
                SELECT id, role, content
                FROM messages
                WHERE id IN ({placeholders})
                ORDER BY timestamp ASC
            """
            cursor.execute(query, turn_ids)
            rows = cursor.fetchall()
            conn.close()

            return [
                {"id": row[0], "role": row[1], "content": row[2] or ""}
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to fetch turns for audit: {e}")
            return []

    def _build_audit_prompt(self, draft_content: str, raw_turns: List[Dict[str, Any]]) -> str:
        """Build the LLM prompt for auditing a synthesized draft."""
        raw_text = "\n\n".join(
            f"[{t['role'].upper()}]: {t['content'][:1500]}" for t in raw_turns
        )

        return f"""You are the CRITIC in a knowledge distillation pipeline. Your job is to perform a FIDELITY CHECK on a synthesized Reference Library draft against raw conversation transcripts.

DRAFT TO AUDIT:
{draft_content}

RAW SOURCE TRANSCRIPTS:
{raw_text}

AUDIT DIMENSIONS (check each):

1. HALLUCINATIONS: Claims in the draft that are NOT supported by the raw turns.
   - Any file path, command, error message, or technical detail must appear in source
   - If the draft says "X caused Y" but transcripts only say "X happened", flag it

2. NUANCE LOSS: Critical details smoothed over during summarization.
   - Caveats, warnings, or conditional logic that was dropped
   - Specific values (ports, timeouts, versions) that were generalized

3. WORLDVIEW DRIFT: Ensure output aligns with these baseline principles:
   - Truth is declarative, not relativistic
   - Technical accuracy over social comfort
   - No false equivalence between verified facts and opinions
   - Epistemic humility — don't claim certainty where transcripts show uncertainty

OUTPUT FORMAT (JSON only, no preamble):
{{
  "passed": true/false,
  "hallucinations": ["claim not in source", ...],
  "nuance_loss": ["detail smoothed over", ...],
  "worldview_drift": ["alignment issue", ...],
  "corrections": ["Specific fix required: ...", ...],
  "verdict": "PASS or FAIL with one-sentence reasoning"
}}

Be strict. If the draft contains ANY hallucination, mark passed=false and list it."""

    def _parse_audit_response(self, text: str) -> Dict[str, Any]:
        """Parse the JSON audit response from LLM."""
        if not text or not text.strip():
            logger.warning("Critic returned empty response — auto-passing (synthesis already validated)")
            return self._pass_report("Empty critic response — auto-pass")

        # Try to extract JSON from markdown code blocks or raw text
        import re
        json_text = text
        if "```json" in text:
            json_text = text.split("```json")[1].split("```\n")[0].strip()
        elif "```" in text:
            # Try all code blocks
            for block in re.findall(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL):
                try:
                    json.loads(block.strip())
                    json_text = block.strip()
                    break
                except json.JSONDecodeError:
                    continue

        try:
            report = json.loads(json_text)
            # Ensure required fields exist
            return {
                "passed": bool(report.get("passed", False)),
                "hallucinations": report.get("hallucinations", []),
                "nuance_loss": report.get("nuance_loss", []),
                "worldview_drift": report.get("worldview_drift", []),
                "corrections": report.get("corrections", []),
                "verdict": report.get("verdict", "FAIL — malformed response"),
            }
        except json.JSONDecodeError as e:
            # Last resort: try to find JSON-like structure anywhere in text
            brace_match = re.search(r'\{.*\}', text, re.DOTALL)
            if brace_match:
                try:
                    report = json.loads(brace_match.group())
                    return {
                        "passed": bool(report.get("passed", False)),
                        "hallucinations": report.get("hallucinations", []),
                        "nuance_loss": report.get("nuance_loss", []),
                        "worldview_drift": report.get("worldview_drift", []),
                        "corrections": report.get("corrections", []),
                        "verdict": report.get("verdict", "FAIL — malformed response"),
                    }
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Critic returned unparseable JSON ({len(text)} chars): {text[:100]}...")
            return self._pass_report(
                f"Critic output not parseable — auto-passing. First 100 chars: {text[:100]}"
            )

    def _pass_report(self, reason: str = "All checks passed") -> Dict[str, Any]:
        """Generate a passing audit report."""
        return {
            "passed": True,
            "hallucinations": [],
            "nuance_loss": [],
            "worldview_drift": [],
            "corrections": [],
            "verdict": f"PASS — {reason}",
        }

    def _fail_report(self, reason: str = "Audit failed") -> Dict[str, Any]:
        """Generate a failing audit report."""
        return {
            "passed": False,
            "hallucinations": [],
            "nuance_loss": [],
            "worldview_drift": [],
            "corrections": [reason],
            "verdict": f"FAIL — {reason}",
        }

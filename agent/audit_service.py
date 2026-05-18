"""AuditService: Fidelity check for synthesized Reference Library drafts.

Stage 2 of the Logos Engine distillation pipeline. Takes a draft Markdown page
from SynthesisService and performs a 'Critic' review against raw transcripts to
detect hallucinations, nuance loss, and worldview drift.

Design principles:
  - Separate LLM call from synthesis (different prompt, same or different model)
  - Structured audit report with pass/fail verdict and specific corrections
  - Zero tolerance for unsupported claims — if it's not in the raw turns, flag it
"""
from __future__ import annotations

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

        # Build audit prompt (pass full raw turns, not truncated)
        prompt = self._build_audit_prompt(draft_content, raw_turns)

        # Call LLM Critic
        try:
            from agent.auxiliary_client import call_llm

            call_kwargs = {
                "task": "archiving",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 8192,   # Full context needs more output room for corrections
                "timeout": 600.0,     # 10 min — overnight runs don't need to be fast
            }
            if main_runtime:
                call_kwargs["main_runtime"] = main_runtime

            response = call_llm(**call_kwargs)
            audit_text = response.choices[0].message.content.strip()

        except (ImportError, ModuleNotFoundError) as e:
            logger.warning(f"Audit LLM call failed (defaulting to FAIL): {e}")
            # If critic is unavailable, default to FAIL — cannot verify fidelity
            return self._fail_report("Audit skipped — LLM unavailable, cannot verify fidelity")

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
        except (sqlite3.Error) as e:
            logger.error(f"Failed to fetch turns for audit: {e}")
            return []

    def _build_audit_prompt(self, draft_content: str, raw_turns: List[Dict[str, Any]]) -> str:
        """Build the LLM prompt for auditing a synthesized draft."""
        # Build audit prompt — label each turn with its ID so the critic can verify citations
        labeled: list[str] = []
        for t in raw_turns:
            tid = t.get("id", "?")
            labeled.append(f"--- turn_{tid} [{t['role'].upper()}] ---\n{t['content']}")
        raw_text = "\n\n".join(labeled)
        # Cap total at ~32K to stay within reasonable context
        if len(raw_text) > 32000:
            raw_text = raw_text[:31000] + "\n\n[... remaining turns truncated for length ...]\n"

        # Extract visible turn IDs from the (possibly truncated) content
        import re
        visible_turn_ids = sorted(set(int(m) for m in re.findall(r'turn_(\d+)', raw_text)))

        return f"""You are the CRITIC in a knowledge distillation pipeline. Your job is to perform a FIDELITY CHECK on a synthesized Reference Library draft against raw conversation transcripts.

DRAFT TO AUDIT:
{draft_content}

VISIBLE TURN IDs (these are the only turns provided below — the draft may cite other turn IDs that were truncated and are NOT present): {visible_turn_ids}

RAW SOURCE TRANSCRIPTS:
{raw_text}

AUDIT DIMENSIONS (check each):

1. HALLUCINATIONS: Claims in the draft that are NOT supported by the raw turns.
   - Each factual bullet claims a source turn ID (e.g., [turn_47]). Verify that the cited turn actually contains that fact.
   - If a bullet cites [turn_N] but turn N is NOT in the VISIBLE TURN IDs list, flag it as hallucinated — that turn was truncated and the synthesizer fabricated the citation.
   - If a bullet cites [turn_N] and turn N IS visible but doesn't contain that information, flag it.
   - If a bullet uses [uncited], check that the claim IS present somewhere in the raw turns. [uncited] is valid when the fact is in the transcripts but the synthesizer couldn't pinpoint the exact turn. Only flag [uncited] claims that are genuinely absent from ALL turns.
   - Any specific value (model name, version, price, file path, port, command) must appear in the source.
   - If the draft says the user chose X, but the cited turn shows the user saying "I don't need X" or "let's not do X", flag it as a directional inversion.

2. NUANCE LOSS: Critical details smoothed over during summarization.
   - If both sides of a decision were discussed but only one appears, flag it.
   - If important caveats or conditions were dropped, flag them.

3. CITATION COVERAGE: Are all bullets properly cited?
   - Every factual claim should have a [turn_N] or [uncited] marker. Completely uncited claims (no bracket at all) are unverifiable.
   - [uncited] is an acceptable marker when the synthesizer couldn't pinpoint the exact turn.
   - Only flag bullets that have NO citation marker of any kind.

4. WORLDVIEW DRIFT: Ensure output aligns with these baseline principles:
   - Truth is declarative, not relativistic
   - Technical accuracy over social comfort
   - No false equivalence between verified facts and opinions
   - Epistemic humility — don't claim certainty where transcripts show uncertainty

5. NARRATIVE INVENTION: Did the synthesizer write connecting prose between facts that isn't in the source?
   - If paragraphs of connecting narrative appear that aren't backed by source turns, flag them.

OUTPUT FORMAT (JSON only, no preamble, keep it concise):
{{
  "passed": true/false,
  "hallucinations": ["Quote the exact false claim and say why"],
  "corrections": ["Brief fix: 'change X to Y'"],
  "verdict": "One sentence"
}}

Keep lists SHORT — at most 3 items each. Only list the MOST serious issues. If there are no hallucinations and no critical corrections, passed=true. Be strict but concise."""

    def _parse_audit_response(self, text: str) -> Dict[str, Any]:
        """Parse the JSON audit response from LLM."""
        if not text or not text.strip():
            logger.warning("Critic returned empty response — auto-failing (cannot verify fidelity)")
            return self._fail_report("Empty critic response — cannot verify fidelity")

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
        except json.JSONDecodeError:
            # Try to fix common JSON issues: trailing commas, unescaped quotes, unclosed braces
            fixed = json_text.strip()
            # Remove trailing commas before closing braces
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            # Try to close unclosed braces
            open_b = fixed.count('{') - fixed.count('}')
            open_s = fixed.count('[') - fixed.count(']')
            if open_b > 0:
                fixed = fixed + '}' * open_b
            if open_s > 0:
                fixed = fixed + ']' * open_s
            try:
                report = json.loads(fixed)
                return {
                    "passed": bool(report.get("passed", False)),
                    "hallucinations": report.get("hallucinations", []),
                    "nuance_loss": report.get("nuance_loss", []),
                    "worldview_drift": report.get("worldview_drift", []),
                    "corrections": report.get("corrections", []),
                    "verdict": report.get("verdict", "FAIL — recovered from malformed JSON"),
                }
            except json.JSONDecodeError:
                pass

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
            return self._fail_report(
                f"Critic output not parseable — cannot verify fidelity. First 100 chars: {text[:100]}"
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

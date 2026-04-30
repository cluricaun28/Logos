"""Multi-Pass Synthesis Engine — Phase 4 of Deep Research & Continuity Engine.

Takes vetted facts from Phases 1-3 and performs multi-pass synthesis via local
LM Studio inference, producing a compact hidden context block (~4-8KB) injected
during reasoning. Also detects when Reference Library pages need updating based
on new research findings.

Architecture:
  - SynthesisEngine: Multi-pass refinement orchestrator (local LM Studio only)
  - ContextBlockFormatter: Formats synthesis into injection-ready blocks with budget caps
  - RLUpdateDetector: Detects stale/contradictory RL pages needing updates

All operations degrade gracefully — returns original draft if LM Studio unavailable.
NO data leaves the local system. All inference runs on your machine via LM Studio.
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants — LOCAL INFERENCE ONLY
# ---------------------------------------------------------------------------
SYNTHESIS_PASS_TIMEOUT = 30        # Seconds per LM Studio inference pass
CONTEXT_BUDGET_KB_DEFAULT = 6      # Default KB budget for context block (4-8 recommended)
MAX_SYNTHESIS_PASSES = 4           # Hard cap on refinement passes
LM_STUDIO_URL_DEFAULT = "http://127.0.0.1:1234/v1"
SYNTHESIS_MODEL_DEFAULT = "qwen3.6-27b"

# Pre-compiled regex for budget enforcement
_SECTION_PATTERN = _re.compile(r'^##\s+(.+)$', _re.MULTILINE)


class SynthesisEngine:
    """Multi-pass synthesis engine using local LM Studio inference.

    Passes:
    1. Draft compilation — assemble raw facts into structured draft
    2. Refinement/cross-reference — verify internal consistency, resolve contradictions
    3. Polish/worldview filter — apply final formatting and worldview alignment notes
    (Optional) 4. Review pass — for high-sensitivity topics

    All inference runs locally via LM Studio. No data leaves the machine.
    Falls back to draft-only mode if LM Studio is unavailable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self._lm_studio_url: str = cfg.get("lm_studio_url", LM_STUDIO_URL_DEFAULT)
        self._model: str = cfg.get("synthesis_model", SYNTHESIS_MODEL_DEFAULT)
        self._max_passes: int = min(cfg.get("synthesis_passes", 3), MAX_SYNTHESIS_PASSES)
        self._context_budget_kb: int = cfg.get("context_budget_kb", CONTEXT_BUDGET_KB_DEFAULT)

    def synthesize(self, facts: List[Dict[str, Any]], query: str,
                   sensitivity: str = "low",
                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform multi-pass synthesis of vetted facts.

        Returns dict with:
        - 'context_block': Formatted text block (~4-8KB) ready for injection
        - 'pass_count': Number of passes performed
        - 'sources_used': List of source attributions
        - 'warnings': Any concerns from synthesis process
        """
        if not facts:
            return {
                "context_block": "",
                "pass_count": 0,
                "sources_used": [],
                "warnings": ["No facts provided for synthesis"],
            }

        start_time = time.time()
        warnings: List[str] = []
        sources_used: List[str] = []

        # Collect unique sources
        for fact in facts:
            src = fact.get("source") or fact.get("_source_stance", "unknown")
            url = fact.get("url", "")
            source_id = f"{src} ({url[:40]})" if url else str(src)
            if source_id not in sources_used:
                sources_used.append(source_id)

        # Pass 1: Compile draft (always runs, no LM Studio needed)
        draft = self._compile_draft(facts, query)
        current_text = draft

        if progress_callback:
            progress_callback(pass_num=1, total_passes=self._max_passes, status="Draft compiled")

        # Determine pass count based on sensitivity
        passes_to_run = 3 if sensitivity == "high" else self._max_passes

        # Passes 2+: Refinement via LM Studio (with graceful fallback)
        for pass_num in range(2, min(passes_to_run + 1, MAX_SYNTHESIS_PASSES + 1)):
            try:
                refined = self._refine_via_lm_studio(current_text, query, pass_num)
                if refined and len(refined.strip()) > 0:
                    current_text = refined
                    logger.debug("Pass %d refinement complete (%d chars)", pass_num, len(current_text))
                else:
                    warnings.append(f"LM Studio returned empty response on pass {pass_num} — using previous draft")
            except Exception as e:
                warnings.append(f"LM Studio unavailable for pass {pass_num}: {e}")
                logger.warning("Synthesis pass %d failed, falling back to current draft: %s", pass_num, e)

            if progress_callback:
                progress_callback(pass_num=pass_num, total_passes=self._max_passes, status=f"Pass {pass_num} complete")

        # Format final context block with budget enforcement
        formatter = ContextBlockFormatter(budget_kb=self._context_budget_kb)
        metadata = {
            "sensitivity": sensitivity,
            "pass_count": self._max_passes if sensitivity == "high" else 1,
            "query": query[:100],
            "timestamp": datetime.now().isoformat(),
        }

        context_block = formatter.format(current_text, facts, metadata)
        elapsed = time.time() - start_time

        return {
            "context_block": context_block,
            "pass_count": self._max_passes if sensitivity == "high" else 1,
            "sources_used": sources_used,
            "warnings": warnings,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _compile_draft(self, facts: List[Dict[str, Any]], query: str) -> str:
        """Pass 1: Compile raw facts into structured draft.

        Groups facts by topic, orders by relevance, extracts key claims.
        Returns plain text draft (~2-4KB). No LM Studio needed.
        """
        lines = [f"# Research Synthesis — {query}", ""]

        # Group facts by source for organization
        by_source: Dict[str, List[Dict]] = {}
        for fact in facts:
            src = fact.get("source", "unknown")
            if src not in by_source:
                by_source[src] = []
            by_source[src].append(fact)

        # Extract key claims from each fact
        lines.append("## Key Findings")
        lines.append("")

        for i, fact in enumerate(facts, 1):
            title = fact.get("title", "")[:80]
            snippet = (fact.get("snippet") or fact.get("content", ""))[:200]
            confidence = fact.get("_confidence", "N/A")
            source = fact.get("source", "unknown")

            lines.append(f"{i}. **{title}** (Source: {source}, Confidence: {confidence})")
            if snippet:
                # Clean up whitespace for compact display
                clean_snippet = " ".join(snippet.split())[:150]
                lines.append(f"   {clean_snippet}")
            lines.append("")

        # Add source analysis section
        lines.append("## Source Analysis")
        lines.append("")
        for src, src_facts in by_source.items():
            stance = src_facts[0].get("_source_stance", "unknown") if src_facts else "unknown"
            bias_notes = src_facts[0].get("_bias_notes", []) if src_facts else []
            lines.append(f"- **{src}** ({len(src_facts)} results, stance: {stance})")
            if bias_notes:
                for note in bias_notes[:2]:  # Limit to first 2 notes per source
                    lines.append(f"  - Bias note: {note}")
        lines.append("")

        return "\n".join(lines)

    def _refine_via_lm_studio(self, draft: str, query: str, pass_number: int) -> Optional[str]:
        """Send draft to LM Studio for refinement via local inference.

        Constructs a system prompt tailored to the current pass's objective.
        Returns refined text or None if LM Studio unavailable.
        Falls back to returning original draft on failure — never breaks synthesis.
        """
        try:
            import httpx as _httpx  # noqa: F811
        except ImportError:
            logger.debug("httpx not available for LM Studio inference")
            return None

        # Build system prompt based on pass number
        system_prompt = self._build_system_prompt(pass_number)

        user_message = f"Query: {query}\n\nCurrent draft:\n{draft}"

        try:
            with _httpx.Client(timeout=SYNTHESIS_PASS_TIMEOUT) as client:
                resp = client.post(
                    f"{self._lm_studio_url}/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0.3,  # Low temperature for consistency
                        "max_tokens": 2048,   # Keep responses compact
                    },
                )

            if resp.status_code != 200:
                logger.warning("LM Studio returned status %d during pass %d", resp.status_code, pass_number)
                return None

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content or len(content.strip()) == 0:
                logger.debug("LM Studio returned empty response on pass %d", pass_number)
                return None

            return content.strip()

        except Exception as e:
            logger.warning("LM Studio inference failed on pass %d: %s — using current draft", pass_number, e)
            return None  # Caller will use original draft

    def _build_system_prompt(self, pass_number: int) -> str:
        """Build system prompt tailored to the synthesis pass objective."""

        prompts = {
            2: (
                "You are a research synthesizer. Review this synthesized research draft. "
                "Check for internal consistency between sources, resolve any contradictions, "
                "and ensure all claims have proper attribution. Improve clarity and organization "
                "without adding new information that isn't in the draft. Keep source attributions intact."
            ),
            3: (
                "You are a research editor performing a final polish pass. Ensure the synthesis is "
                "well-formatted, properly attributed, and includes confidence notes where appropriate. "
                "Flag any remaining uncertainties with [UNCERTAIN] markers. Keep it concise — remove "
                "redundant information while preserving all key findings."
            ),
            4: (
                "You are a research reviewer performing a final quality check. Verify that: "
                "(1) All claims have source attribution, (2) No contradictions remain between sources, "
                "(3) Confidence levels are appropriate, (4) The synthesis is well-organized and readable. "
                "Make only essential corrections — do not rewrite the entire document."
            ),
        }

        return prompts.get(pass_number, prompts[2])


class ContextBlockFormatter:
    """Formats synthesized facts into hidden context blocks for injection."""

    def __init__(self, budget_kb: int = CONTEXT_BUDGET_KB_DEFAULT) -> None:
        self._budget_bytes = budget_kb * 1024

    def format(self, synthesized_text: str, sources: List[Dict[str, Any]],
               metadata: Dict[str, Any]) -> str:
        """Format synthesis output into a structured context block.

        Enforces hard cap at context_budget_kb. Truncates from bottom if needed.
        Never exceeds the budget — adds truncation note if necessary.
        """
        sensitivity = metadata.get("sensitivity", "low")
        pass_count = metadata.get("pass_count", 1)
        timestamp = metadata.get("timestamp", datetime.now().isoformat())

        # Build header
        source_count = len(set(s.get("source", "unknown") for s in sources)) if sources else 0
        header = (
            f"[Synthesized Context — {source_count} sources, sensitivity: {sensitivity}, "
            f"passes: {pass_count}, generated: {timestamp[:16]}]\n\n"
        )

        # Combine header + content
        full_block = header + synthesized_text

        # Enforce budget cap
        if len(full_block.encode("utf-8")) > self._budget_bytes:
            full_block = self._enforce_budget(full_block, header)

        # Add footer with stats
        kb_used = round(len(full_block.encode("utf-8")) / 1024, 1)
        footer = f"\n[Synthesis complete — {pass_count} passes, budget: {kb_used}/{self._budget_bytes // 1024}KB]\n"
        full_block = full_block.rstrip() + footer

        return full_block

    def _enforce_budget(self, text: str, header: str) -> str:
        """Enforce hard budget cap by removing low-confidence facts first, then truncating."""
        # Strategy 1: Remove sections marked as uncertain or low confidence
        lines = text.split("\n")
        filtered_lines = []
        skip_section = False

        for line in lines:
            if "[UNCERTAIN]" in line or "confidence: low" in line.lower():
                continue
            filtered_lines.append(line)

        reduced = "\n".join(filtered_lines)

        # Strategy 2: If still over budget, truncate from bottom (preserve header + key findings)
        full = header + reduced
        if len(full.encode("utf-8")) > self._budget_bytes:
            # Find the end of "## Key Findings" section to preserve it
            available = self._budget_bytes - len(header.encode("utf-8")) - 200  # Reserve for footer
            truncated_content = reduced[:available]

            # Cut at a reasonable boundary (end of paragraph)
            last_newline = truncated_content.rfind("\n")
            if last_newline > available * 0.5:  # Only cut if we have enough content
                truncated_content = truncated_content[:last_newline]

            full = header + truncated_content + "\n\n[...truncated to fit budget...]\n"

        return full


class RLUpdateDetector:
    """Detects stale or contradictory RL pages that should be updated."""

    def __init__(self, rl_dir: str = "~/.hermes/reference-library/") -> None:
        self._rl_dir = Path(os.path.expanduser(rl_dir))
        # Cache file contents to avoid re-reading on every comparison
        self._file_cache: Dict[str, Tuple[str, float]] = {}  # path -> (content, mtime)
        self._cache_ttl = 60  # Seconds to cache file contents

    def _read_file_cached(self, md_file: Path, max_bytes: int = 5000) -> Optional[str]:
        """Read RL file with caching based on modification time."""
        try:
            mtime = md_file.stat().st_mtime
            if md_file in self._file_cache:
                cached_content, cached_mtime = self._file_cache[md_file]
                if abs(mtime - cached_mtime) < self._cache_ttl:
                    return cached_content[:max_bytes]

            content = md_file.read_text(encoding="utf-8")[:max_bytes]
            self._file_cache[md_file] = (content, mtime)
            return content
        except Exception as e:
            logger.debug("Failed to read RL page %s: %s", md_file.name, e)
            return None

    def check_for_updates(self, new_facts: List[Dict[str, Any]],
                          rl_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compare new facts against existing RL pages.

        For each fact, checks if any RL page covers the same topic with outdated info.
        Returns list of update recommendations.
        """
        target_dir = Path(os.path.expanduser(rl_dir)) if rl_dir else self._rl_dir
        recommendations: List[Dict[str, Any]] = []

        if not target_dir.exists():
            logger.debug("RL directory not found: %s", target_dir)
            return recommendations

        # Get all markdown files in RL
        md_files = list(target_dir.rglob("*.md"))

        for fact in new_facts:
            try:
                title = (fact.get("title") or "").lower()
                snippet = (fact.get("snippet") or fact.get("content", "")).lower()
                source_url = fact.get("url", "")

                if not title and not snippet:
                    continue

                # Extract key terms from the fact
                fact_terms = set(title.split()) | set(snippet[:200].split())

                for md_file in md_files:
                    try:
                        content = self._read_file_cached(md_file)
                        if not content:
                            continue

                        content_lower = content.lower()

                        # Check for topic overlap
                        content_terms = set(content_lower.split())
                        overlap = fact_terms & content_terms

                        if len(overlap) >= RL_CONTRADICTION_THRESHOLD:
                            # Potential match — check for contradictions or outdated info
                            reason = self._determine_update_reason(fact, content, md_file)
                            if reason:
                                recommendations.append({
                                    "page": str(md_file),
                                    "reason": reason,
                                    "new_content_summary": (fact.get("snippet") or "")[:200],
                                    "source_url": source_url,
                                    "overlap_terms": list(overlap)[:5],
                                })
                    except Exception as e:
                        logger.debug("Failed to check RL page %s: %s", md_file.name, e)

            except Exception as e:
                logger.debug("Fact comparison failed: %s", e)

        # Deduplicate recommendations by page
        seen_pages = set()
        unique_recs = []
        for rec in recommendations:
            if rec["page"] not in seen_pages:
                seen_pages.add(rec["page"])
                unique_recs.append(rec)

        return unique_recs

    def _determine_update_reason(self, fact: Dict[str, Any], rl_content: str,
                                  md_file: Path) -> Optional[str]:
        """Determine why an RL page might need updating."""
        snippet = (fact.get("snippet") or "").lower()
        title = (fact.get("title") or "").lower()

        # Check for temporal indicators that suggest outdated info
        time_indicators = ["2024", "2025", "last year", "previous version"]
        rl_lower = rl_content.lower()

        for indicator in time_indicators:
            if indicator in snippet and indicator not in rl_lower:
                return f"New information from {indicator} may update existing content"

        # Check for contradictory claims (simple heuristic)
        if "not" in title or "no longer" in title or "changed" in title:
            if any(word in rl_lower for word in title.split()[:5]):
                return "Fact suggests change to previously documented information"

        # Check for significant new data not covered by existing page
        snippet_words = set(snippet.split())
        rl_words = set(rl_lower.split())
        unique_to_fact = snippet_words - rl_words

        if len(unique_to_fact) > 20:  # Significant new content
            return "Substantial new information not covered in existing page"

        return None


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def synthesize_research(facts: List[Dict[str, Any]], query: str,
                        sensitivity: str = "low", config: Optional[Dict] = None) -> Dict[str, Any]:
    """Convenience wrapper for the full synthesis pipeline.

    Usage:
        result = synthesize_research(vetted_facts, "RTX 5090 pricing")
        context_block = result["context_block"]  # Ready for injection
    """
    engine = SynthesisEngine(config)
    return engine.synthesize(facts, query, sensitivity=sensitivity)


# Import guard for constants used by other modules
RL_CONTRADICTION_THRESHOLD = 3

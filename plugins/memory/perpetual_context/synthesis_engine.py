"""Multi-Pass Synthesis Engine — Phase 4 of Deep Research & Continuity Engine.

Takes vetted facts from Phases 1-3 and performs multi-pass synthesis via the
active local inference provider (vLLM at localhost:8000), producing a compact
hidden context block (~4-8KB) injected during reasoning. Also detects when
Reference Library pages need updating based on new research findings.

Architecture:
  - SynthesisEngine: Multi-pass refinement orchestrator (uses configured provider)
  - ContextBlockFormatter: Formats synthesis into injection-ready blocks with budget caps
  - RLUpdateDetector: Detects stale/contradictory RL pages needing updates

All operations degrade gracefully — returns original draft if provider unavailable.
NO data leaves the local system. All inference runs on Patrick's machine.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants — LOCAL INFERENCE ONLY
# Reads active provider from Hermes config; defaults to vLLM on localhost:8000
# ---------------------------------------------------------------------------
SYNTHESIS_PASS_TIMEOUT = 30  # Seconds per inference pass
CONTEXT_BUDGET_KB_DEFAULT = 6  # Default KB budget for context block (4-8 recommended)
MAX_SYNTHESIS_PASSES = 4  # Hard cap on refinement passes
INFERENCE_URL_DEFAULT = "http://localhost:8000/v1"  # vLLM (not LM Studio)
SYNTHESIS_MODEL_DEFAULT = ""  # Empty = use whatever model the provider has loaded
RL_CONTRADICTION_THRESHOLD = 3  # Min term overlap to flag potential contradiction


def get_active_model() -> dict[str, Any]:
    """Read the active model from Hermes config.

    All auxiliary tools (synthesis, distillation, archiving, web_extract)
    use this single source of truth. No hardcoded model names anywhere.

    Returns dict with 'model', 'provider', 'base_url'.
    """
    import yaml as _yaml

    config_path = Path(os.path.expanduser("~/.hermes/config.yaml"))
    try:
        with open(config_path, encoding='utf-8') as f:            config = _yaml.safe_load(f) or {}
        model_cfg = config.get("model", {})
        return {
            "model": model_cfg.get("default", SYNTHESIS_MODEL_DEFAULT),
            "provider": model_cfg.get("provider", "custom"),
            "base_url": model_cfg.get("base_url", INFERENCE_URL_DEFAULT),
        }
    except (OSError, _yaml.YAMLError) as exc:
        logger.debug("Could not read config for active model, using defaults: %s", exc)
        return {
            "model": SYNTHESIS_MODEL_DEFAULT,
            "provider": "custom",
            "base_url": INFERENCE_URL_DEFAULT,
        }


class SynthesisEngine:
    """Multi-pass synthesis engine using local inference (vLLM).

    Passes:
    1. Draft compilation — assemble raw facts into structured draft
    2. Refinement/cross-reference — verify internal consistency, resolve contradictions
    3. Polish/worldview filter — apply final formatting and worldview alignment notes
    (Optional) 4. Review pass — for high-sensitivity topics

    All inference runs locally. No data leaves the machine.
    Falls back to draft-only mode if the provider is unavailable.
    """

    def __init__(self, config: dict[str, Any | None] = None) -> None:
        cfg = config or {}
        active = get_active_model()
        # Support both old key (lm_studio_url) and new key for backward compat
        self._inference_url: str = cfg.get("lm_studio_url", cfg.get("inference_url", active["base_url"]))
        self._model: str = cfg.get("synthesis_model", active["model"])
        self._max_passes: int = min(cfg.get("synthesis_passes", 3), MAX_SYNTHESIS_PASSES)
        self._context_budget_kb: int = cfg.get("context_budget_kb", CONTEXT_BUDGET_KB_DEFAULT)

    def synthesize(
        self, facts: list[dict[str, Any]], query: str, sensitivity: str = "low", progress_callback: Callable | None = None
    ) -> dict[str, Any]:
        """Perform multi-pass synthesis of vetted facts.

        Returns dict with:
        - 'context_block': Formatted text block (~4-8KB) ready for injection
        - 'pass_count': Number of passes performed
        - 'sources_used': List of source attributions
        - 'warnings': Any concerns from synthesis process
        - 'elapsed_seconds': Time taken
        - 'rl_update_flags': List of RL pages flagged for potential updates
        """
        if not facts:
            return {
                "context_block": "",
                "pass_count": 0,
                "sources_used": [],
                "warnings": ["No facts provided for synthesis"],
                "elapsed_seconds": 0,
                "rl_update_flags": [],
            }

        start_time = time.time()
        warnings: list[str] = []
        sources_used: list[str] = []

        # Collect unique sources
        for fact in facts:
            src = fact.get("source") or fact.get("_source_stance", "unknown")
            url = fact.get("url", "")
            source_id = f"{src} ({url[:40]})" if url else str(src)
            if source_id not in sources_used:
                sources_used.append(source_id)

        # Pass 1: Compile draft (always runs, no inference needed)
        draft = self._compile_draft(facts, query)
        current_text = draft

        if progress_callback:
            progress_callback(pass_num=1, total_passes=self._max_passes, status="Draft compiled")

        # Determine pass count based on sensitivity
        passes_to_run = self._max_passes if sensitivity == "high" else 1
        actual_passes = 1  # Always includes the draft pass

        # Passes 2+: Refinement via local inference (with graceful fallback)
        for pass_num in range(2, min(passes_to_run + 1, MAX_SYNTHESIS_PASSES + 1)):
            try:
                refined = self._refine_via_inference(current_text, query, pass_num)
                if refined and len(refined.strip()) > 0:
                    current_text = refined
                    actual_passes = pass_num
                    logger.debug("Pass %d refinement complete (%d chars)", pass_num, len(current_text))
                else:
                    warnings.append(f"Provider returned empty response on pass {pass_num} — using previous draft")
            except (AttributeError, TypeError, ConnectionError, TimeoutError) as e:
                warnings.append(f"Provider unavailable for pass {pass_num}: {e}")
                logger.warning("Synthesis pass %d failed, falling back to current draft: %s", pass_num, e)

            if progress_callback:
                progress_callback(pass_num=pass_num, total_passes=passes_to_run, status=f"Pass {pass_num} complete")

        # Format final context block with budget enforcement
        formatter = ContextBlockFormatter(budget_kb=self._context_budget_kb)
        metadata = {
            "sensitivity": sensitivity,
            "pass_count": actual_passes,
            "query": query[:100],
            "timestamp": datetime.now().isoformat(),
        }

        context_block = formatter.format(current_text, facts, metadata)
        elapsed = time.time() - start_time

        # Run RLUpdateDetector to flag stale RL pages — read-only, no auto-writes
        rl_update_flags: list[dict[str, Any]] = []
        try:
            detector = RLUpdateDetector()
            rl_update_flags = detector.check_for_updates(facts)
            if rl_update_flags:
                logger.info(
                    "RLUpdateDetector flagged %d page(s) for potential update",
                    len(rl_update_flags),
                )
        except (OSError, KeyError, TypeError, AttributeError) as e:
            logger.debug("RLUpdateDetector failed (non-fatal): %s", e)

        return {
            "context_block": context_block,
            "pass_count": actual_passes,
            "sources_used": sources_used,
            "warnings": warnings,
            "elapsed_seconds": round(elapsed, 2),
            "rl_update_flags": rl_update_flags,
        }

    def _compile_draft(self, facts: list[dict[str, Any]], query: str) -> str:
        """Pass 1: Compile raw facts into structured draft.

        Groups facts by topic, orders by relevance, extracts key claims.
        Returns plain text draft (~2-4KB). No LM Studio needed.
        """
        lines = [f"# Research Synthesis — {query}", ""]

        # Group facts by source for organization
        by_source: dict[str, list[dict]] = {}
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

            # Surface source intelligence if available
            profile = fact.get("_source_profile")
            if profile:
                alignment = profile.get("alignment", "")
                if alignment and alignment != "unknown":
                    lines.append(f"   [Source: {alignment}]")
                truthful = profile.get("truthful_on", [])
                omits = profile.get("omits", [])
                if truthful:
                    lines.append(f"   [Reliable on: {', '.join(truthful[:2])}]")
                if omits:
                    lines.append(f"   [Consistently omits: {', '.join(omits[:2])}]")

            # Surface narrative deviation (strong signal)
            narrative = fact.get("_narrative_signal", {})
            deviation = narrative.get("deviation")
            if deviation:
                lines.append(f"   [⚠ {deviation}]")

            lines.append("")

        # Add source analysis section
        lines.append("## Source Analysis")
        lines.append("")
        for src, src_facts in by_source.items():
            stance = src_facts[0].get("_source_stance", "unknown") if src_facts else "unknown"
            bias_notes = src_facts[0].get("_bias_notes", []) if src_facts else []
            lines.append(f"- **{src}** ({len(src_facts)} results, stance: {stance})")

            # Pull source profile from SourceAnalyzer if available
            profile = src_facts[0].get("_source_profile", {})
            if profile:
                cluster = profile.get("cluster", "")
                if cluster and cluster != "unknown":
                    lines.append(f"  - Cluster: {cluster}")
                omits = profile.get("omits", [])
                if omits:
                    lines.append(f"  - Known omissions: {', '.join(omits[:3])}")

            # Bias analysis markers
            bias_analysis = src_facts[0].get("_bias_analysis", {})
            if bias_analysis.get("markers"):
                marker_str = ", ".join(bias_analysis["markers"][:3])
                score = bias_analysis.get("score", 0)
                lines.append(f"  - Bias markers ({score:.2f}): {marker_str}")

            if bias_notes:
                for note in bias_notes[:2]:  # Limit to first 2 notes per source
                    lines.append(f"  - Bias note: {note}")

            # Flag narrative deviations across all facts for this source
            deviations = [f.get("_narrative_flag") for f in src_facts if f.get("_narrative_flag")]
            if deviations:
                lines.append(f"  - ⚠ {deviations[0]}")

        lines.append("")

        return "\n".join(lines)

    def _refine_via_inference(self, draft: str, query: str, pass_number: int) -> str | None:
        """Send draft to active provider for refinement.

        Constructs a system prompt tailored to the current pass's objective.
        Returns refined text or None if provider unavailable.
        Falls back to returning original draft on failure — never breaks synthesis.
        """
        try:
            import httpx as _httpx  # noqa: F811
        except ImportError:
            logger.debug("httpx not available for inference")
            return None

        # Build system prompt based on pass number
        system_prompt = self._build_system_prompt(pass_number)

        user_message = f"Query: {query}\n\nCurrent draft:\n{draft}"

        try:
            with _httpx.Client(timeout=SYNTHESIS_PASS_TIMEOUT) as client:
                resp = client.post(
                    f"{self._inference_url}/chat/completions",
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                        "temperature": 0.3,  # Low temperature for consistency
                        "max_tokens": 2048,  # Keep responses compact
                    },
                )

            if resp.status_code != 200:
                logger.warning("Provider returned status %d during pass %d", resp.status_code, pass_number)
                return None

            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content or len(content.strip()) == 0:
                logger.debug("Provider returned empty response on pass %d", pass_number)
                return None

            return content.strip()

        except (ConnectionError, TimeoutError, KeyError, AttributeError) as e:
            logger.warning("Provider inference failed on pass %d: %s — using current draft", pass_number, e)
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

    def format(self, synthesized_text: str, sources: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
        """Format synthesis output into a structured context block.

        Enforces hard cap at context_budget_kb. Truncates from bottom if needed.
        Never exceeds the budget — adds truncation note if necessary.
        """
        sensitivity = metadata.get("sensitivity", "low")
        pass_count = metadata.get("pass_count", 1)
        timestamp = metadata.get("timestamp", datetime.now().isoformat())

        # Build header
        source_count = len(set(s.get("source", "unknown") for s in sources)) if sources else 0
        header = f"[Synthesized Context — {source_count} sources, sensitivity: {sensitivity}, passes: {pass_count}, generated: {timestamp[:16]}]\n\n"

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

    def check_for_updates(self, new_facts: list[dict[str, Any]], rl_dir: str | None = None) -> list[dict[str, Any]]:
        """Compare new facts against existing RL pages.

        For each fact, checks if any RL page covers the same topic with outdated info.
        Returns list of update recommendations.

        Scans only topics/ and entities/ directories — skips bulk data like
        Britannica entries (32K+ files would make this O(n) unworkable).
        """
        target_dir = Path(os.path.expanduser(rl_dir)) if rl_dir else self._rl_dir
        recommendations: list[dict[str, Any]] = []

        if not target_dir.exists():
            logger.debug("RL directory not found: %s", target_dir)
            return recommendations

        # Only scan topics/ and entities/ — skip bulk data directories
        # that would make this prohibitively expensive (e.g. Britannica 32K entries)
        scan_dirs = [target_dir / "topics", target_dir / "entities"]
        md_files = []
        for scan_dir in scan_dirs:
            if scan_dir.exists():
                md_files.extend(scan_dir.glob("*.md"))

        # Cache: read each file ONCE before comparing against all facts
        file_cache: dict[Path, str] = {}
        for md_file in md_files:
            try:
                file_cache[md_file] = md_file.read_text(encoding="utf-8")[:5000]
            except (OSError, PermissionError) as e:
                logger.debug("Failed to read RL page %s: %s", md_file.name, e)

        for fact in new_facts:
            try:
                title = (fact.get("title") or "").lower()
                snippet = (fact.get("snippet") or fact.get("content", "")).lower()
                source_url = fact.get("url", "")

                if not title and not snippet:
                    continue

                # Extract key terms from the fact
                fact_terms = set(title.split()) | set(snippet[:200].split())

                for md_file, content in file_cache.items():
                    try:
                        content_lower = content.lower()

                        # Check for topic overlap
                        content_terms = set(content_lower.split())
                        overlap = fact_terms & content_terms

                        if len(overlap) >= RL_CONTRADICTION_THRESHOLD:
                            # Potential match — check for contradictions or outdated info
                            reason = self._determine_update_reason(fact, content_lower, md_file)
                            if reason:
                                recommendations.append(
                                    {
                                        "page": str(md_file),
                                        "reason": reason,
                                        "new_content_summary": (fact.get("snippet") or "")[:200],
                                        "source_url": source_url,
                                        "overlap_terms": list(overlap)[:5],
                                    }
                                )
                    except (KeyError, TypeError, AttributeError) as e:
                        logger.debug("Failed to check RL page %s: %s", md_file.name, e)

            except (KeyError, TypeError, AttributeError) as e:
                logger.debug("Fact comparison failed: %s", e)

        # Deduplicate recommendations by page
        seen_pages = set()
        unique_recs = []
        for rec in recommendations:
            if rec["page"] not in seen_pages:
                seen_pages.add(rec["page"])
                unique_recs.append(rec)

        return unique_recs

    def _determine_update_reason(self, fact: dict[str, Any], rl_content: str, md_file: Path) -> str | None:
        """Determine why an RL page might need updating."""
        snippet = (fact.get("snippet") or "").lower()
        title = (fact.get("title") or "").lower()

        # Check for temporal indicators that suggest outdated info
        time_indicators = ["2024", "2025", "last year", "previous version"]
        rl_lower = rl_content  # Already lowercased by caller

        for indicator in time_indicators:
            if indicator in snippet and indicator not in rl_lower:
                return f"New information from {indicator} may update existing content"

        # Check for contradictory claims (simple heuristic)
        if ("not" in title or "no longer" in title or "changed" in title) and any(word in rl_lower for word in title.split()[:5]):
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


def synthesize_research(facts: list[dict[str, Any]], query: str, sensitivity: str = "low", config: dict | None = None) -> dict[str, Any]:
    """Convenience wrapper for the full synthesis pipeline.

    Usage:
        result = synthesize_research(vetted_facts, "RTX 5090 pricing")
        context_block = result["context_block"]  # Ready for injection
    """
    engine = SynthesisEngine(config)
    return engine.synthesize(facts, query, sensitivity=sensitivity)

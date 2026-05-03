"""LogosOrchestrator: Tie Synthesis → Audit → Commit into atomic pipeline.

Stage 3 of the Logos Engine distillation pipeline. Coordinates the full workflow
from high-signal cluster detection through RL commit, ensuring epistemic integrity
at each step.

Design principles:
  - Atomic pipeline: if any stage fails, the whole operation aborts cleanly
  - Provenance tracking: every committed page links back to source PM turn IDs
  - SignalRegistry integration: marks clusters as distilled upon successful commit
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LogosOrchestrator:
    """Coordinate Synthesis → Audit → Commit pipeline.

    Takes high-signal clusters from SignalRegistry and distills them into
    authoritative Reference Library entries with full provenance tracking.
    """

    def __init__(
        self,
        registry=None,
        synthesis=None,
        audit=None,
        rl_dir: Optional[Path] = None,
        staging_dir: Optional[Path] = None,
        db_path: Optional[Path] = None,
    ):
        from agent.signal_registry import SignalRegistry
        from agent.synthesis_service import SynthesisService
        from agent.audit_service import AuditService

        self.rl_dir = rl_dir or Path.home() / ".hermes" / "reference-library"
        self.staging_dir = staging_dir or Path.home() / ".hermes" / "staging"
        self.db_path = db_path or Path.home() / ".hermes" / "perpetual_context.db"

        # Accept pre-built services (plugin pattern) or build our own
        if registry is None:
            self.registry = SignalRegistry(db_path=self.db_path)
        else:
            self.registry = registry

        if synthesis is None:
            self.synthesis = SynthesisService(db_path=self.db_path, staging_dir=self.staging_dir)
        else:
            self.synthesis = synthesis

        if audit is None:
            self.audit = AuditService(db_path=self.db_path)
        else:
            self.audit = audit

    def distill_cluster(
        self,
        cluster_id: int,
        turn_ids: List[int],
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run full distillation pipeline for a single cluster.

        Args:
            cluster_id: Unique identifier from SignalRegistry.
            turn_ids: PM message IDs in this cluster.
            main_runtime: LLM runtime config for synthesis/audit calls.

        Returns:
            Pipeline result dict with:
                - success: bool
                - stage: which stage completed ("synthesis", "audit", "commit")
                - draft_path: path to synthesized draft (if reached)
                - audit_report: audit results (if reached)
                - rl_path: final RL page path (if committed)
                - error: error message if failed
        """
        result = {
            "success": False,
            "stage": "init",
            "cluster_id": cluster_id,
            "turn_ids": turn_ids,
            "draft_path": None,
            "audit_report": None,
            "rl_path": None,
            "error": None,
        }

        try:
            # STAGE 1: Synthesis
            logger.info(f"Distillation pipeline: cluster {cluster_id} → SYNTHESIS")
            draft_path, metadata = self.synthesis.synthesize_draft(
                cluster_id=cluster_id,
                turn_ids=turn_ids,
                main_runtime=main_runtime,
            )
            result["stage"] = "synthesis"
            result["draft_path"] = str(draft_path)

            # STAGE 2: Audit
            logger.info(f"Distillation pipeline: cluster {cluster_id} → AUDIT")
            audit_report = self.audit.audit_draft(
                draft_path=draft_path,
                turn_ids=turn_ids,
                main_runtime=main_runtime,
            )
            result["stage"] = "audit"
            result["audit_report"] = audit_report

            if not audit_report.get("passed", False):
                result["error"] = f"Audit failed: {audit_report['verdict']}"
                logger.warning(f"Distillation aborted: cluster {cluster_id} — {result['error']}")
                return result

            # STAGE 3: Commit to RL
            logger.info(f"Distillation pipeline: cluster {cluster_id} → COMMIT")
            rl_path = self._commit_to_rl(draft_path, metadata, turn_ids)
            result["stage"] = "commit"
            result["rl_path"] = str(rl_path)

            # Mark as distilled in SignalRegistry
            self.registry.mark_distilled(cluster_id)
            result["success"] = True

            logger.info(f"Distillation complete: cluster {cluster_id} → {rl_path}")
            return result

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Distillation pipeline failed at stage '{result['stage']}': {e}")
            return result

    def distill_hotspots(
        self,
        min_size: int = 3,
        max_clusters: int = 5,
        main_runtime: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Distill all undistilled hotspots from SignalRegistry.

        Args:
            min_size: Minimum cluster size to consider.
            max_clusters: Maximum clusters to process in one run.
            main_runtime: LLM runtime config.

        Returns:
            List of pipeline result dicts for each processed cluster.
        """
        hotspots = self.registry.get_hotspots(min_size=min_size)
        if not hotspots:
            logger.info("No undistilled hotspots found")
            return []

        # Limit to max_clusters
        hotspots = hotspots[:max_clusters]
        logger.info(f"Distilling {len(hotspots)} hotspot(s)")

        results = []
        for hotspot in hotspots:
            result = self.distill_cluster(
                cluster_id=hotspot["cluster_id"],
                turn_ids=hotspot["ids"],
                main_runtime=main_runtime,
            )
            results.append(result)

        # Summary
        success_count = sum(1 for r in results if r["success"])
        logger.info(f"Distillation batch complete: {success_count}/{len(results)} committed")
        return results

    def _commit_to_rl(
        self,
        draft_path: Path,
        metadata: Dict[str, Any],
        turn_ids: List[int],
    ) -> Path:
        """Commit approved draft to Reference Library with atomic write.

        Args:
            draft_path: Path to the audited draft Markdown.
            metadata: Synthesis metadata (cluster_id, title, etc.)
            turn_ids: Source PM turn IDs for provenance.

        Returns:
            Path to the committed RL page.
        """
        # Read draft content
        draft_content = draft_path.read_text()

        # Guard: refuse to commit empty or minimal drafts
        if len(draft_content.strip()) < 100:
            logger.warning(f"Refusing to commit empty/minimal draft ({len(draft_content)} chars)")
            raise ValueError("Draft too short — synthesis likely timed out")

        # Generate RL filename from title or cluster ID
        title = metadata.get("draft_title", f"cluster_{metadata['cluster_id']}")
        safe_filename = self._slugify(title) + ".md"

        # Determine subdirectory (topics/ for general, system/ for technical)
        if "system" in draft_content.lower() or "config" in draft_content.lower():
            subdir = self.rl_dir / "system"
        else:
            subdir = self.rl_dir / "topics"
        subdir.mkdir(parents=True, exist_ok=True)

        rl_path = subdir / safe_filename

        # Add provenance footer if not present
        if "## Provenance" not in draft_content:
            draft_content += f"""

## Provenance
Source: Perpetual Memory turns {turn_ids}
Cluster ID: {metadata['cluster_id']}
Distilled: {datetime.now(timezone.utc).isoformat()}
"""

        # Atomic write: temp file + os.replace
        fd, tmp_path = tempfile.mkstemp(dir=str(subdir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(draft_content)
            os.replace(tmp_path, rl_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # Update index.md (append link if not present)
        self._update_index(rl_path, title)

        return rl_path

    def _update_index(self, rl_path: Path, title: str) -> None:
        """Append new page to reference-library/index.md."""
        index_path = self.rl_dir / "index.md"

        # Read existing index
        if index_path.exists():
            content = index_path.read_text()
        else:
            content = "# Reference Library Index\n\n"

        # Check if already indexed
        relative_path = rl_path.name
        if relative_path in content:
            return  # Already indexed

        # Append new entry
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        content += f"- [{title}]({relative_path}) — *{timestamp}*\n"

        # Atomic write
        fd, tmp_path = tempfile.mkstemp(dir=str(self.rl_dir), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            os.replace(tmp_path, index_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _slugify(text: str) -> str:
        """Convert title to URL-safe filename slug."""
        import re
        slug = text.lower()
        slug = re.sub(r"[^a-z0-9\s-]", "", slug)
        slug = re.sub(r"[\s]+", "-", slug.strip())
        return slug[:80] or "untitled"  # Max 80 chars

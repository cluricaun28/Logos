"""Logos Engine — Sovereign knowledge distillation pipeline.

Synthesis → Audit → Commit pipeline that transforms high-signal perpetual
memory into authoritative reference material. SQLite-backed with multi-
dimensional scoring and temporal decay.

Services:
  • SignalRegistry      — Cluster management, hotspot detection, scoring, pinning
  • SynthesisService    — Draft generation from pinned turns and hotspots
  • AuditService        — Multi-dimensional quality scoring (truth, completeness, bias)
  • LogosOrchestrator   — Full distillation pipeline orchestration
  • SovereignSieve      — Source credibility verification with local-first fallback

Config in ~/.hermes/config.yaml:
  memory:
    logos_engine:
      enabled: true
      db_path: ~/.hermes/perpetual_context.db
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_logos_services(db_path: str = None) -> Dict[str, Any]:
    """Instantiate and return all Logos Engine services.

    Args:
        db_path: Path to perpetual context database. Defaults to ~/.hermes/perpetual_context.db

    Returns dict with keys: signal_registry, synthesis_service, audit_service, orchestrator, sovereign_sieve
    """
    if db_path is None:
        from hermes_constants import get_hermes_home
        db_path = str(get_hermes_home() / "perpetual_context.db")

    # Lazy imports to avoid hard dependency on torch/sentence-transformers at module load
    from agent.signal_registry import SignalRegistry
    from agent.synthesis_service import SynthesisService
    from agent.audit_service import AuditService
    from agent.logos_orchestrator import LogosOrchestrator
    from agent.sovereign_sieve import SovereignSieve

    db_path_obj = Path(db_path) if not isinstance(db_path, Path) else db_path
    registry = SignalRegistry(db_path=db_path_obj)
    synthesis = SynthesisService()
    audit = AuditService()
    orchestrator = LogosOrchestrator(registry=registry, synthesis=synthesis, audit=audit)
    sieve = SovereignSieve()

    return {
        "signal_registry": registry,
        "synthesis_service": synthesis,
        "audit_service": audit,
        "orchestrator": orchestrator,
        "sovereign_sieve": sieve,
    }


def register(ctx):
    """Register Logos Engine as a memory provider extension."""

    class LogosEngineProvider:
        """Memory provider wrapper for Logos Engine services."""

        name = "logos_engine"
        description = "Sovereign knowledge distillation pipeline (Synthesis → Audit → Commit)"

        def __init__(self, config: Dict[str, Any] = None):
            self.config = config or {}
            db_path = self.config.get("db_path")
            self.services = get_logos_services(db_path)

        @classmethod
        def is_available(cls) -> bool:
            return True

        @property
        def signal_registry(self):
            return self.services["signal_registry"]

        @property
        def synthesis_service(self):
            return self.services["synthesis_service"]

        @property
        def audit_service(self):
            return self.services["audit_service"]

        @property
        def orchestrator(self):
            return self.services["orchestrator"]

        @property
        def sovereign_sieve(self):
            return self.services["sovereign_sieve"]

    ctx.register_memory_provider(LogosEngineProvider)

"""SignalRegistry: SQLite-backed continuous signal tracking for Perpetual Memory.

Frontier-grade scoring model replacing flat binary flags with multi-dimensional
heat maps, temporal decay, and automatic re-queueing of stale distilled content.

Architecture:
  - Persistent storage: SQLite tables in perpetual_context.db (signal_clusters, turn_signals)
  - In-memory cache: LRU dict for O(1) lookups during context compression
  - Scoring dimensions: recency, frequency, depth, centrality with configurable weights
  - Decay model: Exponential decay on recency and post-distillation heat

Design principles:
  - Zero blocking I/O in critical path (cache serves compression queries)
  - WAL mode for concurrent reads from other sessions
  - Append-only safety — signal data is an index layer over immutable PM messages
"""
from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from logos_constants import logos_env

logger = logging.getLogger(__name__)


# Default scoring weights (sum to 1.0)
DEFAULT_WEIGHTS = {
    "recency": 0.30,
    "frequency": 0.25,
    "depth": 0.20,
    "centrality": 0.15,
    "distillation_freshness": 0.10,
}

# Decay constants
RECENCY_HALF_LIFE_DAYS = 7.0  # Recency score halves every 7 days
DISTILLATION_DECAY_DAYS = 30.0  # Distilled content re-queues after ~30 days


class SignalRegistry:
    """SQLite-backed continuous signal tracker for PM hotspots.

    Maintains multi-dimensional heat scores per turn with temporal decay,
    persisted to the Perpetual Memory database and cached in-memory for
    O(1) lookups during context compression.
    """

    _instance: Optional["SignalRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls, db_path: Optional[Path] = None) -> "SignalRegistry":
        with cls._lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
            return cls._instance

    def __init__(
        self,
        db_path: Optional[Path] = None,
        weights: Optional[Dict[str, float]] = None,
        cache_size: int = 1024,
    ):
        if self._initialized:
            return

        self.db_path = db_path or Path(logos_env("HOME") or str(Path.home() / ".hermes")) / "perpetual_context.db"
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.cache_size = cache_size

        # In-memory LRU cache: turn_id -> {cluster_ids, composite_score, pinned}
        self._cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.RLock()

        # Cluster metadata cache: cluster_id -> cluster dict
        self._clusters_cache: Dict[int, Dict[str, Any]] = {}

        self._initialized = True
        logger.info(f"SignalRegistry initialized (db={self.db_path}, cache_size={cache_size})")

    @classmethod
    def reset(cls) -> None:
        """Reset singleton state (for testing or session resets)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._cache.clear()
                cls._instance._clusters_cache.clear()
                cls._instance = None

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Get a WAL-mode connection to the PM database."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s for locks
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # Scoring model
    # ------------------------------------------------------------------

    def compute_composite_score(
        self,
        recency: float = 0.5,
        frequency: int = 1,
        depth: float = 0.5,
        centrality: float = 0.5,
        distillation_age_days: float = 999.0,
    ) -> float:
        """Compute composite signal score from dimensions.

        Args:
            recency: Normalized recency score (0-1), higher = more recent.
            frequency: Raw cross-session mention count (will be normalized).
            depth: Content volume/length score (0-1).
            centrality: Cross-reference density (0-1).
            distillation_age_days: Days since last distillation (higher = stale).

        Returns:
            Composite score in [0.0, 1.0].
        """
        w = self.weights

        # Normalize frequency to 0-1 using sigmoid-like curve (saturates at ~20 mentions)
        freq_norm = min(1.0, frequency / 20.0)

        # Distillation freshness: decays from 1.0 to 0.0 over DISTILLATION_DECAY_DAYS
        if distillation_age_days < 0.01:
            freshness = 1.0  # Just distilled
        else:
            freshness = max(0.0, math.exp(-distillation_age_days / DISTILLATION_DECAY_DAYS))

        composite = (
            w["recency"] * recency
            + w["frequency"] * freq_norm
            + w["depth"] * depth
            + w["centrality"] * centrality
            + w["distillation_freshness"] * freshness
        )

        return round(min(1.0, max(0.0, composite)), 4)

    def decay_recency(self, score: float, days_elapsed: float) -> float:
        """Apply exponential decay to a recency score.

        Uses half-life model: score *= exp(-ln(2) * days / half_life).
        """
        if days_elapsed <= 0:
            return score
        decay = math.exp(-math.log(2) * days_elapsed / RECENCY_HALF_LIFE_DAYS)
        return round(max(0.0, min(1.0, score * decay)), 4)

    # ------------------------------------------------------------------
    # Cluster management (persisted to SQLite)
    # ------------------------------------------------------------------

    def create_cluster(
        self,
        topic_summary: str,
        turn_ids: List[int],
        initial_scores: Optional[Dict[int, Dict[str, float]]] = None,
    ) -> int:
        """Create a new signal cluster and seed it with turns.

        Args:
            topic_summary: Human-readable summary of what this cluster is about.
            turn_ids: PM message IDs belonging to this cluster.
            initial_scores: Optional per-turn score overrides.

        Returns:
            The new cluster_id.
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            now = time.time()

            # Insert cluster
            cursor.execute(
                "INSERT INTO signal_clusters (topic_summary, created_at, updated_at, distilled_status) VALUES (?, ?, ?, 'undistilled')",
                (topic_summary, now, now),
            )
            cluster_id = cursor.lastrowid

            # Seed turn signals
            for tid in turn_ids:
                scores = initial_scores.get(tid, {}) if initial_scores else {}
                recency = scores.get("recency_score", 0.7)  # New clusters start hot
                depth = scores.get("depth_score", 0.5)
                centrality = scores.get("centrality_score", 0.3)
                freq = scores.get("frequency_count", 1)

                composite = self.compute_composite_score(
                    recency=recency,
                    frequency=freq,
                    depth=depth,
                    centrality=centrality,
                    distillation_age_days=999.0,  # Not yet distilled
                )

                cursor.execute(
                    """INSERT OR IGNORE INTO turn_signals
                       (turn_id, cluster_id, heat_score, recency_score, frequency_count,
                        depth_score, centrality_score, composite_score, last_updated)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (tid, cluster_id, composite, recency, freq, depth, centrality, composite, now),
                )

            conn.commit()

            # Invalidate cache for affected turns
            with self._cache_lock:
                for tid in turn_ids:
                    self._cache.pop(tid, None)

            logger.info(f"Created cluster {cluster_id}: '{topic_summary[:50]}...' ({len(turn_ids)} turns)")
            return cluster_id

        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Failed to create cluster: {e}")
            raise
        finally:
            conn.close()

    def get_cluster(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific cluster."""
        if cluster_id in self._clusters_cache:
            return self._clusters_cache[cluster_id]

        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT cluster_id, topic_summary, created_at, updated_at, distilled_status, "
                "rl_page_path, distillation_timestamp FROM signal_clusters WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()

            if not row:
                return None

            cluster = {
                "cluster_id": row[0],
                "topic_summary": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "distilled_status": row[4],
                "rl_page_path": row[5],
                "distillation_timestamp": row[6],
            }

            # Get turn count and average score
            stats = conn.execute(
                "SELECT COUNT(*), AVG(composite_score) FROM turn_signals WHERE cluster_id=?",
                (cluster_id,),
            ).fetchone()

            cluster["turn_count"] = stats[0] or 0
            cluster["avg_score"] = round(stats[1], 4) if stats[1] else 0.0

            self._clusters_cache[cluster_id] = cluster
            return cluster

        finally:
            conn.close()

    def get_all_clusters(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all clusters, optionally filtered by distillation status."""
        conn = self._get_conn()
        try:
            if status:
                rows = conn.execute(
                    "SELECT cluster_id, topic_summary, distilled_status FROM signal_clusters WHERE distilled_status=?",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT cluster_id, topic_summary, distilled_status FROM signal_clusters"
                ).fetchall()

            return [
                {
                    "cluster_id": r[0],
                    "topic_summary": r[1],
                    "distilled_status": r[2],
                }
                for r in rows
            ]
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Turn signal queries (compression path — O(1) from cache)
    # ------------------------------------------------------------------

    def get_pinned_turns(self, turn_ids: List[int]) -> Set[int]:
        """Return which turns are high-signal and should be pinned during compression.

        A turn is 'pinned' if it belongs to an undistilled cluster with composite_score > 0.3.
        Uses in-memory cache for O(1) lookups.

        Args:
            turn_ids: PM message IDs being considered for compression.

        Returns:
            Set of turn IDs that should be protected from lossy summarization.
        """
        pinned = set()
        now = time.time()

        with self._cache_lock:
            for tid in turn_ids:
                cached = self._cache.get(tid)

                if not cached:
                    # Cache miss — query SQLite and populate
                    info = self._load_turn_info(tid)
                    if info:
                        self._cache[tid] = info
                        if len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)  # Evict oldest
                        cached = info

                if cached and cached.get("pinned"):
                    pinned.add(tid)

        return pinned

    def _load_turn_info(self, turn_id: int) -> Optional[Dict[str, Any]]:
        """Load signal info for a single turn from SQLite."""
        conn = self._get_conn()
        try:
            # Get all cluster memberships for this turn with their scores
            rows = conn.execute(
                """SELECT ts.cluster_id, ts.composite_score, sc.distilled_status
                   FROM turn_signals ts
                   JOIN signal_clusters sc ON ts.cluster_id = sc.cluster_id
                   WHERE ts.turn_id = ?""",
                (turn_id,),
            ).fetchall()

            if not rows:
                return None

            cluster_ids = [r[0] for r in rows]
            max_score = max(r[1] for r in rows)
            has_undistilled = any(r[2] == "undistilled" for r in rows)

            # Apply temporal decay to composite score
            now = time.time()
            latest_row = conn.execute(
                "SELECT MAX(last_updated) FROM turn_signals WHERE turn_id=?", (turn_id,)
            ).fetchone()
            latest_updated = latest_row[0] if latest_row and latest_row[0] else now
            days_elapsed = max(0, (now - latest_updated) / 86400.0)
            decayed_score = self.decay_recency(max_score, days_elapsed)

            return {
                "cluster_ids": set(cluster_ids),
                "composite_score": round(decayed_score, 4),
                "pinned": has_undistilled and decayed_score > 0.3,
            }

        finally:
            conn.close()

    def is_high_signal(self, turn_id: int) -> bool:
        """Check if a single turn ID belongs to an undistilled high-signal cluster."""
        pinned = self.get_pinned_turns([turn_id])
        return turn_id in pinned

    # ------------------------------------------------------------------
    # Hotspot discovery (for distillation pipeline)
    # ------------------------------------------------------------------

    def get_hotspots(self, min_score: float = 0.3, limit: int = 10) -> List[Dict[str, Any]]:
        """Return undistilled hotspots suitable for distillation.

        Args:
            min_score: Minimum composite score threshold.
            limit: Maximum number of clusters to return.

        Returns:
            List of cluster dicts with turn IDs, sorted by avg composite score descending.
        """
        conn = self._get_conn()
        try:
            # Find undistilled clusters with high average scores
            rows = conn.execute(
                """SELECT sc.cluster_id, sc.topic_summary, COUNT(*) as turn_count,
                          AVG(ts.composite_score) as avg_score
                   FROM signal_clusters sc
                   JOIN turn_signals ts ON sc.cluster_id = ts.cluster_id
                   WHERE sc.distilled_status = 'undistilled'
                   GROUP BY sc.cluster_id
                   HAVING avg_score >= ?
                   ORDER BY avg_score DESC, turn_count DESC
                   LIMIT ?""",
                (min_score, limit),
            ).fetchall()

            hotspots = []
            for row in rows:
                cid = row[0]
                # Get the actual turn IDs
                turn_ids = [
                    r[0]
                    for r in conn.execute(
                        "SELECT turn_id FROM turn_signals WHERE cluster_id=? ORDER BY composite_score DESC",
                        (cid,),
                    ).fetchall()
                ]

                hotspots.append({
                    "cluster_id": cid,
                    "topic_summary": row[1],
                    "turn_count": row[2],
                    "avg_score": round(row[3], 4),
                    "turn_ids": turn_ids,
                })

            return hotspots

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Distillation lifecycle
    # ------------------------------------------------------------------

    def mark_distilling(self, cluster_id: int) -> None:
        """Mark a cluster as currently being distilled."""
        self._update_cluster_status(cluster_id, "distilling")

    def mark_distilled(
        self, cluster_id: int, rl_page_path: Optional[str] = None
    ) -> None:
        """Mark a cluster as distilled into RL.

        Once distilled, its turns no longer need to be pinned during compression
        because the truth is now in the Reference Library. Heat scores decay but
        don't reset — if the RL page becomes stale, re-queueing happens automatically.

        Args:
            cluster_id: The cluster that was just distilled.
            rl_page_path: Path to the committed RL page (for provenance tracking).
        """
        conn = self._get_conn()
        try:
            now = time.time()
            conn.execute(
                "UPDATE signal_clusters SET distilled_status='distilled', "
                "rl_page_path=?, distillation_timestamp=? WHERE cluster_id=?",
                (rl_page_path, now, cluster_id),
            )

            # Reset composite scores to reflect post-distillation state
            # (freshness dimension drops, but recency/frequency remain for re-queueing)
            conn.execute(
                "UPDATE turn_signals SET composite_score=0.1 WHERE cluster_id=?",
                (cluster_id,),
            )

            conn.commit()

            # Invalidate cache
            rows = conn.execute(
                "SELECT turn_id FROM turn_signals WHERE cluster_id=?", (cluster_id,)
            ).fetchall()
            with self._cache_lock:
                for (tid,) in rows:
                    self._cache.pop(tid, None)

            # Clear cluster metadata cache
            self._clusters_cache.pop(cluster_id, None)

            logger.info(f"Cluster {cluster_id} marked as distilled → {rl_page_path}")

        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Failed to mark cluster {cluster_id} as distilled: {e}")
            raise
        finally:
            conn.close()

    def get_stale_clusters(self, max_age_days: float = 30.0) -> List[Dict[str, Any]]:
        """Find distilled clusters whose RL pages have become stale.

        Stale content should be re-distilled to capture new context or corrections.

        Args:
            max_age_days: Days since distillation before considering a cluster stale.

        Returns:
            List of stale cluster dicts ready for refresh distillation.
        """
        conn = self._get_conn()
        try:
            cutoff = time.time() - (max_age_days * 86400.0)

            rows = conn.execute(
                """SELECT sc.cluster_id, sc.topic_summary, sc.distillation_timestamp,
                          COUNT(*) as turn_count, AVG(ts.composite_score) as avg_score
                   FROM signal_clusters sc
                   LEFT JOIN turn_signals ts ON sc.cluster_id = ts.cluster_id
                   WHERE sc.distilled_status = 'distilled'
                     AND (sc.distillation_timestamp < ? OR sc.distillation_timestamp IS NULL)
                   GROUP BY sc.cluster_id
                   ORDER BY sc.distillation_timestamp ASC""",
                (cutoff,),
            ).fetchall()

            return [
                {
                    "cluster_id": r[0],
                    "topic_summary": r[1],
                    "distilled_at": r[2],
                    "turn_count": r[3] or 0,
                    "avg_score": round(r[4], 4) if r[4] else 0.0,
                }
                for r in rows
            ]

        finally:
            conn.close()

    def _update_cluster_status(self, cluster_id: int, status: str) -> None:
        """Update distillation status of a cluster."""
        conn = self._get_conn()
        try:
            now = time.time()
            conn.execute(
                "UPDATE signal_clusters SET distilled_status=?, updated_at=? WHERE cluster_id=?",
                (status, now, cluster_id),
            )
            conn.commit()
            self._clusters_cache.pop(cluster_id, None)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Legacy compatibility (for existing distillation pipeline)
    # ------------------------------------------------------------------

    def update_signals(self, cluster_data: List[Dict[str, Any]]) -> int:
        """Legacy interface: bulk import from pm_scanner.py format.

        Args:
            cluster_data: [{cluster_id, size, ids, samples, signal_score}, ...]

        Returns:
            Number of clusters created/updated.
        """
        created = 0
        for cluster in cluster_data:
            cid = cluster.get("cluster_id")
            turn_ids = cluster.get("ids", [])
            if cid is None or not turn_ids:
                continue

            # Check if cluster already exists
            existing = self.get_cluster(cid)
            if not existing:
                topic = f"Cluster {cid} — auto-generated from scanner"
                self.create_cluster(topic, turn_ids)
                created += 1

        return created

    def load_from_file(self, report_path: Path) -> int:
        """Legacy interface: load signals from distillation_signal_report.json."""
        import json

        if not report_path.exists():
            logger.debug(f"No signal report found at {report_path}")
            return 0

        try:
            with open(report_path, "r", encoding='utf-8') as f:                data = json.load(f)
            return self.update_signals(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load signal report from {report_path}: {e}")
            return 0

    # ------------------------------------------------------------------
    # Stats & diagnostics
    # ------------------------------------------------------------------

    @property
    def total_clusters(self) -> int:
        conn = self._get_conn()
        try:
            return conn.execute("SELECT COUNT(*) FROM signal_clusters").fetchone()[0]
        finally:
            conn.close()

    @property
    def undistilled_count(self) -> int:
        conn = self._get_conn()
        try:
            return conn.execute(
                "SELECT COUNT(*) FROM signal_clusters WHERE distilled_status='undistilled'"
            ).fetchone()[0]
        finally:
            conn.close()

    @property
    def tracked_turns(self) -> int:
        conn = self._get_conn()
        try:
            return conn.execute("SELECT COUNT(DISTINCT turn_id) FROM turn_signals").fetchone()[0]
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get full registry statistics."""
        return {
            "total_clusters": self.total_clusters,
            "undistilled_count": self.undistilled_count,
            "tracked_turns": self.tracked_turns,
            "cache_size": len(self._cache),
            "weights": self.weights.copy(),
        }


# Module-level singleton accessor for convenience
def get_registry(db_path: Optional[Path] = None) -> SignalRegistry:
    """Get the global SignalRegistry singleton."""
    return SignalRegistry(db_path=db_path)

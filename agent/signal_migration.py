"""Migration: Add signal tracking tables to Perpetual Memory database.

Creates two new tables with zero impact on existing schema:
  - signal_clusters: Semantic cluster metadata (topic summary, distillation status)
  - turn_signals: Multi-dimensional heat scores per turn linked to clusters

Idempotent — safe to run multiple times. Uses CREATE TABLE IF NOT EXISTS.
"""

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def migrate(db_path: Path, dry_run: bool = False) -> dict:
    """Run the signal tracking migration on the given database.

    Args:
        db_path: Path to the Perpetual Memory SQLite database.
        dry_run: If True, only validate schema without writing.

    Returns:
        Dict with migration status and table counts.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    cursor = conn.cursor()

    try:
        # --- signal_clusters table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_clusters (
                cluster_id INTEGER PRIMARY KEY,
                topic_summary TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                distilled_status TEXT NOT NULL DEFAULT 'undistilled'
                    CHECK(distilled_status IN ('undistilled', 'distilling', 'distilled', 'stale')),
                rl_page_path TEXT DEFAULT NULL,
                distillation_timestamp REAL DEFAULT NULL,
                last_refresh REAL DEFAULT NULL
            )
        """)

        # Index for fast status lookups (e.g., "give me all undistilled clusters")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_clusters_status
            ON signal_clusters(distilled_status)
        """)

        # --- turn_signals table ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turn_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id INTEGER NOT NULL REFERENCES messages(id),
                cluster_id INTEGER NOT NULL REFERENCES signal_clusters(cluster_id),
                heat_score REAL NOT NULL DEFAULT 0.5 CHECK(heat_score >= 0.0 AND heat_score <= 1.0),
                recency_score REAL NOT NULL DEFAULT 0.5 CHECK(recency_score >= 0.0 AND recency_score <= 1.0),
                frequency_count INTEGER NOT NULL DEFAULT 1,
                depth_score REAL NOT NULL DEFAULT 0.5 CHECK(depth_score >= 0.0 AND depth_score <= 1.0),
                centrality_score REAL NOT NULL DEFAULT 0.5 CHECK(centrality_score >= 0.0 AND centrality_score <= 1.0),
                composite_score REAL NOT NULL DEFAULT 0.5,
                last_updated REAL NOT NULL,
                UNIQUE(turn_id, cluster_id)
            )
        """)

        # Index for fast turn lookups (compression path: "is this turn high-signal?")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_turn_signals_turn
            ON turn_signals(turn_id)
        """)

        # Index for cluster membership queries ("what turns are in cluster N?")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_turn_signals_cluster
            ON turn_signals(cluster_id)
        """)

        if not dry_run:
            conn.commit()

        # Verify tables exist and count rows
        tables = {
            "signal_clusters": cursor.execute(
                "SELECT COUNT(*) FROM signal_clusters"
            ).fetchone()[0],
            "turn_signals": cursor.execute(
                "SELECT COUNT(*) FROM turn_signals"
            ).fetchone()[0],
        }

        logger.info(f"Signal migration complete: {tables}")
        return tables

    except sqlite3.Error as e:
        if not dry_run:
            conn.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    db = Path.home() / ".hermes" / "perpetual_context.db"
    dry = "--dry-run" in sys.argv

    if dry:
        print("Dry run mode — no writes")

    result = migrate(db, dry_run=dry)
    print(f"Migration result: {result}")

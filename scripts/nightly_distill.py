#!/usr/bin/env python3
"""
Nightly distillation pipeline runner.

1. Run signal scanner (new signals since last run)
2. Process undistilled clusters from queue
3. Log results for morning delivery

Usage:
    python3 nightly_distill.py [--clusters N]  # Process up to N clusters per night
"""
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("nightly_distill")

QUEUE_PATH = Path("/home/pmdal/.hermes/staging/distillation_queue.json")
SCANNER_SCRIPT = Path("/home/pmdal/.hermes/scripts/phase3_signal_scanner.py")
PROJECT_DIR = Path("/home/pmdal/.hermes/hermes-agent")


def run_signal_scan():
    """Run signal scanner to detect new topics since last run."""
    logger.info("Running signal scanner...")
    try:
        result = subprocess.run(
            ["python3", str(SCANNER_SCRIPT), "--full-pipeline"],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            logger.info(f"Signal scan complete:\n{result.stdout[-500:]}")
        else:
            logger.warning(f"Signal scan failed (rc={result.returncode}): {result.stderr[:300]}")
    except subprocess.TimeoutExpired:
        logger.error("Signal scanner timed out after 120s")
    except Exception as e:
        logger.error(f"Signal scanner error: {e}")


def load_queue() -> list:
    """Load distillation queue."""
    if QUEUE_PATH.exists():
        with open(QUEUE_PATH) as f:
            return json.load(f)
    return []


def get_undistilled(queue: list, limit: int = None) -> list:
    """Get undistilled signals from queue, sorted by score desc."""
    undistilled = [s for s in queue if not s.get("distilled", False)]
    undistilled.sort(key=lambda x: x.get("score", 0), reverse=True)
    if limit:
        return undistilled[:limit]
    return undistilled


def distill_cluster(cluster_id_str: str, turn_ids: list, topic: str) -> dict:
    """Run full distillation pipeline on a single cluster."""
    import sys
    sys.path.insert(0, str(PROJECT_DIR))

    from agent.logos_orchestrator import LogosOrchestrator
    orchestrator = LogosOrchestrator()

    # Parse cluster_id string "cluster_N" → int N
    cid_num = int(cluster_id_str.split('_')[1])

    logger.info(f"Distilling {cluster_id_str} (topic={topic}, turns={len(turn_ids)})")
    start = datetime.now()

    try:
        result = orchestrator.distill_cluster(cid_num, turn_ids)
        elapsed = (datetime.now() - start).total_seconds()
        success = result.get("success", False)
        rl_path = result.get("rl_path", "") or ""
        error = result.get("error", "")

        status_str = "COMMITTED" if success else f"FAILED: {error}"

        # Mark as distilled in queue (even on failure, don't retry indefinitely)
        for s in load_queue():
            if s["cluster_id"] == cluster_id_str:
                s["distilled"] = True  # Mark done regardless of outcome
                s["distilled_at"] = datetime.now().isoformat()
                break
        save_queue(load_queue())

        return {
            "cluster_id": cluster_id_str,
            "topic": topic,
            "status": status_str,
            "rl_path": rl_path,
            "time_seconds": elapsed,
        }
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        logger.error(f"Distillation failed for {cluster_id_str}: {e}")
        # Still mark as distilled to avoid infinite retry loops
        for s in load_queue():
            if s["cluster_id"] == cluster_id_str:
                s["distilled"] = True
                s["distilled_at"] = datetime.now().isoformat()
                break
        save_queue(load_queue())
        return {
            "cluster_id": cluster_id_str,
            "topic": topic,
            "status": f"ERROR: {e}",
            "rl_path": "",
            "time_seconds": elapsed,
        }


def save_queue(queue: list):
    """Save distillation queue."""
    with open(QUEUE_PATH, 'w') as f:
        json.dump(queue, f, indent=2)


def main():
    max_clusters = 3  # Default: process 3 per night

    for arg in sys.argv[1:]:
        if arg == "--clusters" and len(sys.argv) > sys.argv.index(arg) + 1:
            max_clusters = int(sys.argv[sys.argv.index(arg) + 1])

    logger.info(f"=== Nightly Distillation Pipeline ===")
    logger.info(f"Max clusters per run: {max_clusters}")

    # Step 1: Signal scan
    run_signal_scan()

    # Step 2: Process undistilled clusters
    queue = load_queue()
    undistilled = get_undistilled(queue, limit=max_clusters)

    if not undistilled:
        logger.info("No undistilled clusters in queue — nothing to do")
        return

    results = []
    for signal in undistilled:
        cid = signal["cluster_id"]
        result = distill_cluster(cid, signal["turn_ids"], signal.get("topic", "unknown"))
        results.append(result)
        logger.info(f"  [{cid}] topic={signal['topic']} → {result['status']} ({result['time_seconds']:.0f}s)")

    # Step 3: Summary
    committed = sum(1 for r in results if "COMMIT" in r["status"].upper())
    failed = len(results) - committed
    total_time = sum(r["time_seconds"] for r in results)

    logger.info(f"=== Nightly Distillation Complete ===")
    logger.info(f"Processed: {len(results)} | Committed: {committed} | Failed: {failed}")
    logger.info(f"Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Log committed paths for morning delivery
    for r in results:
        if r["rl_path"]:
            logger.info(f"  New RL page: {r['rl_path']}")


if __name__ == "__main__":
    main()

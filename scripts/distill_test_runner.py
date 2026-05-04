#!/usr/bin/env python3
"""Distillation Pipeline Test Runner

Processes queued signals from ~/.hermes/staging/distillation_queue.json through the full
Logos Engine pipeline (Synthesis → Audit → Commit) and reports results.

Usage:
    python3 distill_test_runner.py [--clusters N]  # Process top N clusters (default: 2)
    python3 distill_test_runner.py --all           # Process all queued signals
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add hermes-agent to path
sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))

from agent.logos_orchestrator import LogosOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("distill_test")


def load_queue(queue_path: Path) -> list:
    """Load distillation queue from staging."""
    if not queue_path.exists():
        logger.error(f"Queue not found at {queue_path}")
        return []

    with open(queue_path, "r") as f:
        queue = json.load(f)

    if not isinstance(queue, list):
        logger.error("Invalid queue format (expected list)")
        return []

    # Filter out already-distilled signals
    undistilled = [s for s in queue if not s.get("distilled", False)]
    logger.info(f"Queue: {len(queue)} total, {len(undistilled)} undistilled")
    return undistilled


def mark_distilled(queue_path: Path, cluster_id: str):
    """Mark a signal as distilled in the queue."""
    if not queue_path.exists():
        return

    with open(queue_path, "r") as f:
        queue = json.load(f)

    for signal in queue:
        if signal.get("cluster_id") == cluster_id:
            signal["distilled"] = True
            break

    with open(queue_path, "w") as f:
        json.dump(queue, f, indent=2)


def run_distillation(undistilled_queue: list, max_clusters: int = None):
    """Run distillation pipeline on queued signals."""
    if not undistilled_queue:
        logger.info("No undistilled signals to process")
        return []

    # Limit clusters if specified
    queue_to_process = undistilled_queue[:max_clusters] if max_clusters else undistilled_queue

    orchestrator = LogosOrchestrator()
    results = []

    for i, signal in enumerate(queue_to_process, 1):
        cluster_id = signal.get("cluster_id", f"unknown_{i}")
        topic = signal.get("topic", "unknown")
        turn_ids = signal.get("turn_ids", [])
        score = signal.get("score", 0)

        logger.info(f"\n{'='*60}")
        logger.info(f"[{i}/{len(queue_to_process)}] Distilling: {cluster_id} (topic: '{topic}', score: {score:.3f})")
        logger.info(f"Turns: {len(turn_ids)}, Cross-session: {signal.get('cross_session', False)}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        try:
            # Run full pipeline: Synthesis → Audit → Commit
            result = orchestrator.distill_cluster(
                cluster_id=hash(cluster_id) % 10000,  # Convert string ID to int for registry
                turn_ids=turn_ids[:20],  # Limit to first 20 turns per cluster
            )

            elapsed = time.time() - start_time
            result["topic"] = topic
            result["cluster_id_str"] = cluster_id
            result["elapsed_seconds"] = round(elapsed, 1)

            status = "✅ COMMITTED" if result["success"] else f"❌ FAILED at {result['stage']}"
            logger.info(f"Result: {status} ({elapsed:.1f}s)")

            if result.get("rl_path"):
                logger.info(f"RL Path: {result['rl_path']}")

            if result.get("error"):
                logger.warning(f"Error: {result['error']}")

            # Mark as distilled in queue (even on failure, to avoid infinite retries)
            mark_distilled(Path.home() / ".hermes" / "staging" / "distillation_queue.json", cluster_id)

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Unexpected error processing {cluster_id}: {e}")
            result = {
                "success": False,
                "stage": "error",
                "cluster_id_str": cluster_id,
                "topic": topic,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
            }

        results.append(result)

    return results


def print_summary(results: list):
    """Print distillation summary report."""
    if not results:
        print("No results to summarize.")
        return

    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success
    total_time = sum(r.get("elapsed_seconds", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"Distillation Test Summary")
    print(f"{'='*60}")
    print(f"Total clusters processed: {total}")
    print(f"Successfully committed:   {success}")
    print(f"Failed/aborted:           {failed}")
    print(f"Total runtime:            {total_time:.1f}s ({total_time/60:.1f} min)")
    print()

    for r in results:
        status = "✅" if r["success"] else "❌"
        topic = r.get("topic", "unknown")
        stage = r.get("stage", "?")
        elapsed = r.get("elapsed_seconds", 0)
        error = f" — {r['error'][:60]}" if r.get("error") else ""

        print(f"  {status} [{r.get('cluster_id_str', '?')}] topic='{topic}' "
              f"stage={stage} time={elapsed:.1f}s{error}")

    if success > 0:
        print(f"\nCommitted pages are in ~/.hermes/reference-library/topics/")


def main():
    parser = argparse.ArgumentParser(description="Distillation Pipeline Test Runner")
    parser.add_argument("--clusters", type=int, default=2, help="Number of clusters to process (default: 2)")
    parser.add_argument("--all", action="store_true", help="Process all queued signals")
    args = parser.parse_args()

    queue_path = Path.home() / ".hermes" / "staging" / "distillation_queue.json"
    undistilled = load_queue(queue_path)

    if not undistilled:
        print("No undistilled signals in queue. Run signal scanner first:")
        print("  python3 ~/.hermes/scripts/phase3_signal_scanner.py --full-pipeline")
        return

    max_clusters = None if args.all else args.clusters
    results = run_distillation(undistilled, max_clusters=max_clusters)
    print_summary(results)


if __name__ == "__main__":
    main()

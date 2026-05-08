#!/usr/bin/env python3
"""Nightly FAISS reindex — embed unembedded messages and rebuild the vector index.

Wired as a cron job that runs at 4:00 AM. Skips entirely if message count
is below 15,000 (not worth the compute for small databases).

Usage:
    python -m hermes.scripts.reindex_embeddings
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure the hermes-agent package is importable
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


def main() -> dict:
    """Run the nightly reindex."""
    from agent.perpetual_context_db import PerpetualContextDB

    db = PerpetualContextDB()
    if not db.initialize():
        logger.error("Failed to initialize PerpetualContextDB")
        return {"status": "failed", "reason": "initialization failed"}

    try:
        result = db.reindex_embeddings()
        logger.info("Reindex result: %s", result)
        return result
    except Exception as e:
        logger.error("Reindex error: %s", e)
        return {"status": "failed", "reason": str(e)}
    finally:
        db.shutdown()


if __name__ == "__main__":
    result = main()
    if result.get("action_taken") == "skipped":
        print(f"Skipped: {result.get('reason', 'no action needed')}")
    elif result.get("action_taken") == "completed":
        print(
            f"Completed: embedded {result.get('messages_embedded', 0)} messages, "
            f"FAISS rebuilt: {result.get('faiss_rebuilt', False)}"
        )
    else:
        print(f"Failed: {result.get('reason', 'unknown error')}")
        sys.exit(1)

"""Standalone RL Index Build Script.

from __future__ import annotations

Usage:
    # From the hermes-agent repo root:
    python -m plugins.memory.perpetual_context.rl_build

    # Or directly:
    python /path/to/rl_build.py

Builds the full FTS5 + embedding index for the Reference Library.
Takes 2-5 minutes for ~32K files on a local 5090 GPU.

The resulting index lives at ~/.hermes/rl_index.db and is
automatically used by handle_reference_library_search on next startup.
"""

import logging
import sys
import time

# Ensure the hermes-agent repo is on the path
_HERE = __import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rl_build")


def main() -> None:
    from plugins.memory.perpetual_context.rl_index import RLIndex, RL_BASE_DIR

    logger.info("Building RL index from %s ...", RL_BASE_DIR)

    rl = RLIndex()
    if not rl.initialize():
        logger.error("Failed to initialize RLIndex. Aborting.")
        sys.exit(1)

    start = time.time()
    stats = rl.build_index()
    elapsed = time.time() - start

    logger.info("Build complete in %.1fs:", elapsed)
    logger.info("  Files processed : %d", stats.get("files_processed", "?"))
    logger.info("  Files indexed   : %d", stats.get("files_indexed", "?"))
    logger.info("  Files embedded  : %d", stats.get("files_embedded", "?"))
    logger.info("  Files failed    : %d", stats.get("files_failed", "?"))
    logger.info("  Categories      : %s", stats.get("categories", "?"))

    rl.shutdown()
    logger.info("Done. Index saved to %s", rl._db_path)


if __name__ == "__main__":
    main()

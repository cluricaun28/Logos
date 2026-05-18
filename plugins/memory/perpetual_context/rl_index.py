"""Reference Library Index — Pre-computed FTS5 + embedding search over RL files.

Thin orchestrator class. Delegates to sub-modules:
  - rl_schema  — table/trigger creation
  - rl_builder — build, update, remove, reindex_stale
  - rl_search  — FTS5, semantic, hybrid fusion, snippets, stats

Usage:
    from plugins.memory.perpetual_context.rl_index import RLIndex

    rl = RLIndex()
    rl.initialize()
    rl.build_index()
    results = rl.search("dog training methods", top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
import re as _re
import sqlite3
import struct as _struct
import threading
import time
from pathlib import Path
from typing import Any

import tqdm
tqdm.disable = True

logger = logging.getLogger(__name__)

RL_INDEX_DB_PATH = os.path.expanduser("~/.hermes/rl_index.db")
RL_BASE_DIR = os.path.expanduser("~/.hermes/reference-library")
RL_EMBED_DIM = 384
RL_EMBED_MAX_CONTENT_LEN = 8000
EMBED_BATCH_SIZE = 64
PROGRESS_LOG_INTERVAL = 512
RL_SEMANTIC_WEIGHT = 0.4
RL_FTS5_WEIGHT = 0.6
RL_COSINE_THRESHOLD = 0.05

YAML_FRONTMATTER_RE = _re.compile(r"^---\s*\n(.*?)\n---\s*\n", _re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers — module-level, reused by sub-modules
# ---------------------------------------------------------------------------

def _parse_yaml_frontmatter(text: str) -> dict:
    """Parse simple YAML frontmatter (flat key: value pairs only)."""
    result = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value:
                result[key] = value
    return result


def _extract_file_info(file_path: Path) -> dict | None:
    """Read a markdown file and extract title, frontmatter, body, category."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        logger.debug("Failed to read %s: %s", file_path.name, e)
        return None

    try:
        rel_path = str(file_path.relative_to(RL_BASE_DIR))
    except ValueError:
        rel_path = str(file_path)

    parts = rel_path.split("/")
    category = parts[0] if parts else "other"
    if len(parts) >= 2 and parts[0] == "entities":
        sub = parts[1].split("/")[0]
        if sub == "britannica":
            category = "britannica"
        elif sub == "aquinas-library":
            category = "aquinas"

    frontmatter = "{}"
    body = content
    title = ""
    match = YAML_FRONTMATTER_RE.search(content)
    if match:
        fm_data = _parse_yaml_frontmatter(match.group(1))
        frontmatter = json.dumps(fm_data)
        body = content[match.end():]
        title = fm_data.get("title") or fm_data.get("name") or fm_data.get("topic", "")
    else:
        title = ""

    if not title:
        h1_match = _re.search(r"^#\s+(.+)$", body, _re.MULTILINE)
        if h1_match:
            title = h1_match.group(1).strip()

    if not title:
        title = file_path.stem.replace("-", " ").title()

    body_clean = _re.sub(r"```.*?```", "", body, flags=_re.DOTALL)
    body_clean = _re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", body_clean)
    body_clean = _re.sub(r"[*_~]{1,2}", "", body_clean)
    body_clean = _re.sub(r"#{1,6}\s*", " ", body_clean)
    body_clean = _re.sub(r"---+", "", body_clean)
    body_clean = _re.sub(r"\[!\[.*?\]\(.*?\)\]", "", body_clean)
    body_clean = _re.sub(r"\|.*\|", "", body_clean)
    body_clean = _re.sub(r"\n{3,}", "\n\n", body_clean)
    body_clean = body_clean.strip()

    return {
        "file_path": rel_path,
        "title": title,
        "frontmatter": frontmatter,
        "body": body_clean[:RL_EMBED_MAX_CONTENT_LEN],
        "category": category,
        "mtime": file_path.stat().st_mtime,
        "size": len(content),
    }


def _embed_batch_fn(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float | None]]:
    """Embed a batch of texts using the cached embedding model.

    Called by rl_builder. Uses self._embedding_model.
    """
    # This is called via closure from RLIndex — see _make_embed_batch_fn
    raise NotImplementedError("Use instance method")


# ---------------------------------------------------------------------------
# RLIndex — Thin Orchestrator (~200 lines target)
# ---------------------------------------------------------------------------

class RLIndex:
    """Search index for the Reference Library."""

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or RL_INDEX_DB_PATH
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._initialized = False
        self._embedding_engine: Any | None = None
        self._embedding_model: Any | None = None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # -------------------------------------------------------------------
    # Embedding cache
    # -------------------------------------------------------------------

    def _get_embedding_engine(self) -> Any:
        if self._embedding_engine is None:
            from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415
            self._embedding_engine = EmbeddingEngine.get()
        return self._embedding_engine

    def _get_embedding_model(self) -> Any:
        if self._embedding_model is None:
            self._embedding_model = self._get_embedding_engine()._load_model()
        return self._embedding_model

    def _make_embed_batch_fn(self):
        """Create a closure that uses the cached model."""
        def fn(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float | None]]:
            model = self._get_embedding_model()
            if model is None:
                return [None] * len(texts)
            results: list[list[float | None]] = []
            for i in range(0, len(texts), batch_size):
                batch = [t[:RL_EMBED_MAX_CONTENT_LEN] for t in texts[i: i + batch_size]]
                try:
                    vectors = model.encode(
                        batch, convert_to_numpy=True, batch_size=batch_size,
                    )
                    if hasattr(vectors, "__len__") and not isinstance(vectors, str):
                        results.extend(v.tolist() for v in vectors)
                    else:
                        results.append(vectors.tolist())
                except (AttributeError, TypeError, ValueError) as e:
                    logger.debug("Batch embedding failed: %s", e)
                    results.extend([None] * len(batch))
            return results
        return fn

    def _make_embed_fn(self):
        """Create a closure for single-file embedding."""
        def fn(text: str) -> list[float] | None:
            return self._get_embedding_engine().embed(text)
        return fn

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def initialize(self) -> bool:
        try:
            with self._lock:
                self._conn = sqlite3.connect(
                    self._db_path, timeout=30, check_same_thread=False
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._conn.execute("PRAGMA busy_timeout=5000")
                from . import rl_schema  # noqa: PLC0415
                rl_schema.create_tables(self._conn)
                self._initialized = True
                return True
        except (sqlite3.Error, AttributeError) as e:
            logger.error("Failed to initialize RLIndex: %s", e)
            self._conn = None
            return False

    def shutdown(self) -> None:
        with self._lock:
            if self._conn:
                try:
                    self._conn.commit()
                except (sqlite3.Error, AttributeError) as e:
                    logger.debug("RLIndex shutdown commit error: %s", e)
                try:
                    self._conn.close()
                except (sqlite3.Error, AttributeError) as e:
                    logger.debug("RLIndex close error: %s", e)
                self._conn = None
            self._initialized = False

    # -------------------------------------------------------------------
    # Delegation to sub-modules
    # -------------------------------------------------------------------

    def build_index(self, rl_base: str | None = None) -> dict:
        from . import rl_builder  # noqa: PLC0415
        return rl_builder.build_index(
            conn=self._conn, lock=self._lock,
            rl_base=rl_base or RL_BASE_DIR,
            file_info_extractor=_extract_file_info,
            embed_batch_fn=self._make_embed_batch_fn(),
        )

    def update_file(self, file_path_str: str) -> bool:
        from . import rl_builder  # noqa: PLC0415
        return rl_builder.update_file(
            conn=self._conn, lock=self._lock,
            file_path_str=file_path_str,
            file_info_extractor=_extract_file_info,
            embed_fn=self._make_embed_fn(),
            embed_dim=RL_EMBED_DIM,
        )

    def remove_file(self, file_path_str: str) -> bool:
        from . import rl_builder  # noqa: PLC0415
        return rl_builder.remove_file(
            conn=self._conn, lock=self._lock,
            file_path_str=file_path_str,
            rl_base=RL_BASE_DIR,
        )

    def reindex_stale(self) -> int:
        from . import rl_builder  # noqa: PLC0415
        return rl_builder.reindex_stale(
            conn=self._conn, lock=self._lock,
            file_info_extractor=_extract_file_info,
            embed_batch_fn=self._make_embed_batch_fn(),
            update_file_fn=self.update_file,
            rl_base=RL_BASE_DIR,
            embed_dim=RL_EMBED_DIM,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
        semantic_weight: float = RL_SEMANTIC_WEIGHT,
        fts5_weight: float = RL_FTS5_WEIGHT,
    ) -> list[dict[str, Any]]:
        if not self._initialized or not self._conn:
            return []
        from . import rl_search  # noqa: PLC0415
        return rl_search.search(
            conn=self._conn,
            query=query,
            top_k=top_k,
            category=category,
            semantic_weight=semantic_weight,
            fts5_weight=fts5_weight,
            embedding_engine=self._embedding_engine or self._get_embedding_engine(),
        )

    def get_stats(self) -> dict:
        if not self._conn:
            return {"error": "not initialized"}
        from . import rl_search  # noqa: PLC0415
        return rl_search.get_stats(conn=self._conn)

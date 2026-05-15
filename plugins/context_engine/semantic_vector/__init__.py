"""Semantic Vector Context Engine plugin.

Tracks conversation topics using local embeddings and prunes dormant chatter
instead of applying lossy summarization. Preserves raw signal fidelity while
adapting aggressiveness based on context window pressure.

CPU-only embedding — uses its own independent SentenceTransformer instance
loaded on device="cpu" to avoid GPU contention with vLLM.

The embedding model is cached in a module-level singleton so that new engine
instances (created after session splits) reuse the already-loaded model
instead of reloading from disk on every archive.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
# AIAgent creates a fresh SemanticVectorContextEngine on every session split.
# Without this cache, the SentenceTransformer reloads from disk each time
# (~100 ms each).  With it, the model loads once per process lifetime.

_model_cache: Any = None
_model_path_cache: str = ""


def _get_model_cache() -> tuple[Any, str]:
    """Return the cached (model, path) tuple."""
    return _model_cache, _model_path_cache


def _set_model_cache(model: Any, path: str) -> None:
    """Store model in module-level cache for reuse across engine instances."""
    global _model_cache, _model_path_cache
    _model_cache = model
    _model_path_cache = path


# -- Data model ---------------------------------------------------------------


class ConversationVector:
    """Tracks a semantic topic vector in the conversation."""

    __slots__ = [
        "vector_id",
        "name",
        "centroid",
        "turn_indices",
        "turn_ids",
        "last_seen_turn",
        "last_active_turn",
        "status",
        "dim",
    ]

    def __init__(
        self,
        vector_id: int,
        name: str,
        centroid: List[float],
        turn_indices: List[int] = None,
        last_seen_turn: int = 0,
        status: str = "Active",
    ):
        self.vector_id = vector_id
        self.name = name
        self.centroid = centroid
        self.turn_indices = turn_indices or []
        # Aliases for compatibility with different test suites
        self.turn_ids = self.turn_indices
        self.last_seen_turn = last_seen_turn
        self.last_active_turn = last_seen_turn
        self.status = status
        self.dim = len(centroid)

    def __repr__(self):
        return (
            f"ConversationVector(id={self.vector_id}, name={self.name!r}, "
            f"turns={len(self.turn_indices)}, status={self.status!r})"
        )


# -- Engine implementation ----------------------------------------------------


class SemanticVectorContextEngine(ContextEngine):
    """Context engine that uses semantic vectors to track conversation topics.

    Instead of summarizing, it identifies when topics have gone dormant or
    resolved and prunes only those turns. Active topics are preserved in full.

    Falls back to positional tail-off pruning if embedding is unavailable.
    """

    name: str = "semantic_vector"

    # Configurable parameters (can be overridden via config.yaml)
    similarity_threshold: float = 0.75
    dormancy_decay: int = 10
    resolution_decay: int = 40
    state_map_max_chars: int = 800
    max_dormant_vectors: int = 10

    # Embedding state
    model: Any = None  # The SentenceTransformer model (also aliased as _model)
    _embedding_engine: Any = None  # For test mocking
    _embed_cache: Dict[str, List[float]] = None
    _vectors: List[ConversationVector] = None
    _next_vector_id: int = 0
    _current_turn: int = 0

    # Local model path
    _model_path: str = ""

    def __init__(self, **kwargs):
        super().__init__()
        if self._embed_cache is None:
            self._embed_cache = {}
        if self._vectors is None:
            self._vectors = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if not self._model_path:
            self._model_path = os.path.expanduser(
                "~/.hermes/models/embeddings/all-MiniLM-L6-v2"
            )

    # -- Abstract methods -----------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        self.last_prompt_tokens = usage.get("prompt_tokens", self.last_prompt_tokens)
        self.last_completion_tokens = usage.get("completion_tokens", self.last_completion_tokens)
        self.last_total_tokens = usage.get("total_tokens", self.last_total_tokens)

    def should_archive(self, prompt_tokens: int = None) -> bool:
        """Return True if archiving should fire this turn."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if not self.threshold_tokens:
            return False
        return tokens >= self.threshold_tokens

    def archive(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Archive old messages and return the new message list."""
        if not messages:
            return []

        if len(messages) <= self.protect_first_n + self.protect_last_n:
            logger.info(
                "SemanticVector archive skipped: %d messages <= %d protected",
                len(messages),
                self.protect_first_n + self.protect_last_n,
            )
            return messages

        # Try semantic pruning first
        engine = self._get_embedding_model()
        if engine is not None:
            try:
                return self._semantic_archive(
                    messages, focus_topic, current_tokens
                )
            except Exception as e:
                logger.warning(
                    "Semantic archive failed, falling back to tail-off: %s", e
                )

        # Fallback: tail-off positional pruning
        return self._tailoff_archive(messages)

    # -- Semantic archive -----------------------------------------------------

    def _semantic_archive(
        self,
        messages: List[Dict[str, Any]],
        focus_topic: str = None,
        current_tokens: int = None,
    ) -> List[Dict[str, Any]]:
        """Full semantic archive: embed all turns, cluster into vectors,
        prune dormant/resolved ones, inject state map."""

        # Reset per-call state
        self._vectors = []
        self._next_vector_id = 0
        self._current_turn = 0

        # Embed each non-system message and assign to vectors
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                continue
            content = msg.get("content", "")
            if not content:
                continue

            embedding = self._get_embedding(content)
            if embedding is None:
                continue

            self._assign_turn_to_vector(self._current_turn, content, embedding)
            self._current_turn += 1

        # Update vector states based on current turn
        current = self._current_turn - 1
        self._update_vector_states(current)

        # Log vector summary
        active_v = sum(1 for v in self._vectors if v.status == "Active")
        dormant_v = sum(1 for v in self._vectors if v.status == "Dormant")
        resolved_v = sum(1 for v in self._vectors if v.status == "Resolved")
        logger.info(
            "SemanticVector vectors: %d total "
            "(%d Active, %d Dormant, %d Resolved), threshold=%.2f",
            len(self._vectors),
            active_v,
            dormant_v,
            resolved_v,
            self.similarity_threshold,
        )

        # Determine which turns to keep
        keep_indices = set()

        # Always protect first N (system prompt area)
        for i in range(self.protect_first_n):
            if i < len(messages):
                keep_indices.add(i)

        # Always protect last N
        for i in range(
            max(0, len(messages) - self.protect_last_n), len(messages)
        ):
            keep_indices.add(i)

        # Keep all turns belonging to Active vectors
        for vec in self._vectors:
            if vec.status == "Active":
                for ti in vec.turn_indices:
                    # Map turn index back to message index (skip system messages)
                    msg_idx = self._turn_to_msg_index(ti, messages)
                    if msg_idx is not None:
                        keep_indices.add(msg_idx)

        # Estimate tokens from actual message content if not provided
        estimated_tokens = sum(len(m.get("content", "")) for m in messages) // 4
        effective_tokens = (
            current_tokens if current_tokens is not None else estimated_tokens
        )

        # If we're not over threshold, just return original with state map
        if not self.should_archive(effective_tokens):
            result = list(messages)
            state_map = self._build_state_map()
            if state_map and result:
                self._inject_state_map(result, state_map)
                logger.info(
                    "SemanticVector archive: below threshold (%d < %d tokens), "
                    "injected state map (%d chars), returned %d messages",
                    effective_tokens,
                    self.threshold_tokens,
                    len(state_map),
                    len(result),
                )
            return result

        # Build result from kept indices
        result = [
            messages[i] for i in sorted(keep_indices) if i < len(messages)
        ]

        # Safety check: if result is still too large after semantic pruning,
        # fall back to tail-off
        result_chars = sum(len(m.get("content", "")) for m in result)
        result_tokens = result_chars // 4
        if result_tokens > self.threshold_tokens and self.threshold_tokens > 0:
            logger.info(
                "SemanticVector pruning insufficient (%d > %d tokens), "
                "applying tail-off fallback",
                result_tokens,
                self.threshold_tokens,
            )
            return self._tailoff_archive(result)

        # Inject state map
        state_map = self._build_state_map()
        if state_map and result:
            self._inject_state_map(result, state_map)
            logger.info(
                "SemanticVector archive: %d -> %d messages, "
                "pruned %d turns, injected state map (%d chars)",
                len(messages),
                len(result),
                len(messages) - len(result),
                len(state_map),
            )
        else:
            logger.info(
                "SemanticVector archive: %d -> %d messages, "
                "pruned %d turns (no state map)",
                len(messages),
                len(result),
                len(messages) - len(result),
            )

        self.archive_count += 1
        return result

    # -- Tail-off fallback ----------------------------------------------------

    def _tailoff_archive(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Fallback: keep first N and last N, discard middle."""
        # Protect system messages
        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if len(non_system) <= self.protect_last_n * 2:
            logger.info(
                "SemanticVector tail-off: too few messages (%d), returning as-is",
                len(messages),
            )
            return messages

        # Keep last protect_last_n messages
        keep = system + non_system[-self.protect_last_n :]
        pruned = len(messages) - len(keep)
        logger.info(
            "SemanticVector tail-off fallback: %d -> %d messages, "
            "pruned %d (kept system + last %d)",
            len(messages),
            len(keep),
            pruned,
            self.protect_last_n,
        )
        self.archive_count += 1
        return keep

    # -- Embedding helpers ----------------------------------------------------

    def _get_embedding_model(self) -> Any:
        """Get embedding engine, using module-level cache for reuse across instances.

        Checks in order:
        1. Instance-level _embedding_engine (for test mocking)
        2. Instance-level model
        3. Module-level model cache (shared across engine instances)
        4. Load from disk and cache

        Returns the model instance or None.
        """
        if self._embedding_engine is not None:
            return self._embedding_engine
        if self.model is not None:
            return self.model

        # Check module-level cache
        cached_model, cached_path = _get_model_cache()
        if cached_model is not None:
            # If the path matches, use the cached model directly
            if cached_path == self._model_path:
                self.model = cached_model
                logger.debug(
                    "SemanticVectorContextEngine: reused cached model from "
                    "module cache (path=%s)",
                    self._model_path,
                )
                return cached_model
            else:
                # Path differs — load new model but keep cache for future
                logger.debug(
                    "SemanticVectorContextEngine: cached path %s != requested %s, "
                    "loading new model",
                    cached_path,
                    self._model_path,
                )

        # Load fresh
        loaded = self._load_model()
        return self.model  # may be None if _load_model failed

    def _get_embedding(self, text: str) -> List[float] | None:
        """Get embedding for text, using cache."""
        cache_key = text[:200]  # Cache by prefix to avoid huge keys
        if cache_key in self._embed_cache:
            return self._embed_cache[cache_key]

        engine = self._embedding_engine or self.model
        if engine is None:
            return None

        try:
            if hasattr(engine, "embed"):
                # Test mock or direct embed method
                result = engine.embed(text)
            elif hasattr(engine, "encode"):
                # SentenceTransformer interface
                result = engine.encode(text)
                result = (
                    result.tolist() if hasattr(result, "tolist") else list(result)
                )
            else:
                return None
            self._embed_cache[cache_key] = result
            return result
        except Exception as e:
            logger.debug("Embedding failed for text: %s", e)
            return None

    def _assign_turn_to_vector(
        self, turn_idx: int, text: str, embedding: List[float] = None
    ):
        """Assign a turn to an existing vector or create a new one."""
        if embedding is None:
            embedding = self._get_embedding(text)
        if embedding is None:
            return

        # Find best matching vector
        best_vec = None
        best_sim = -1.0

        for vec in self._vectors:
            sim = self._cosine_similarity(embedding, vec.centroid)
            if sim > best_sim:
                best_sim = sim
                best_vec = vec

        if best_vec is not None and best_sim >= self.similarity_threshold:
            # Join existing vector
            best_vec.turn_indices.append(turn_idx)
            best_vec.last_seen_turn = turn_idx
            best_vec.last_active_turn = turn_idx
            # Update centroid (moving average)
            n = len(best_vec.turn_indices)
            for j in range(len(best_vec.centroid)):
                best_vec.centroid[j] = (
                    best_vec.centroid[j] * (n - 1) + embedding[j]
                ) / n
            # Update name from text
            words = text[:80].split()[:5]
            best_vec.name = (
                "-".join(words) if words else f"topic-{best_vec.vector_id}"
            )
        else:
            # Create new vector
            words = text[:80].split()[:5]
            name = (
                "-".join(words) if words else f"topic-{self._next_vector_id}"
            )
            vec = ConversationVector(
                vector_id=self._next_vector_id,
                name=name,
                centroid=list(embedding),
                turn_indices=[turn_idx],
                last_seen_turn=turn_idx,
                status="Active",
            )
            self._vectors.append(vec)
            self._next_vector_id += 1

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors (pure Python)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _update_vector_states(self, current_turn: int):
        """Update vector states based on dormancy and resolution decay."""
        for vec in self._vectors:
            gap = current_turn - vec.last_seen_turn
            if vec.status == "Active" and gap >= self.dormancy_decay:
                vec.status = "Dormant"
            elif vec.status in ("Active", "Dormant") and gap >= self.resolution_decay:
                vec.status = "Resolved"

    # -- State map -------------------------------------------------------------

    def _build_state_map(self) -> str:
        """Build a conversation state map string for injection."""
        if not self._vectors:
            return ""

        lines = ["[Conversation State]"]
        for vec in sorted(
            self._vectors, key=lambda v: v.last_seen_turn, reverse=True
        ):
            lines.append(
                f"  #{vec.vector_id} {vec.name}: {vec.status} "
                f"(turns {vec.turn_indices}, last seen turn {vec.last_seen_turn})"
            )

        result = "\n".join(lines)
        if len(result) > self.state_map_max_chars:
            result = result[: self.state_map_max_chars] + "\n  ... (truncated)"
        return result

    def _inject_state_map(
        self, messages: List[Dict[str, Any]], state_map: str
    ):
        """Inject state map into message list."""
        if not messages or not state_map:
            return

        # Find last assistant message to prepend state map to
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                content = messages[i].get("content", "")
                messages[i]["content"] = state_map + "\n\n" + content
                return

        # If no assistant message, prepend as a system-like note
        if messages:
            first = messages[0]
            if first.get("role") == "user":
                first["content"] = state_map + "\n\n" + first.get("content", "")

    def _turn_to_msg_index(
        self, turn_idx: int, messages: List[Dict[str, Any]]
    ) -> int | None:
        """Map a turn index (which skips system messages) back to message index."""
        non_system_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                continue
            if non_system_idx == turn_idx:
                return i
            non_system_idx += 1
        return None

    # -- Model loading --------------------------------------------------------

    def _load_model(self) -> bool:
        """Load the embedding model on CPU. Returns True if successful.

        After loading, stores the model in the module-level cache so that
        subsequent engine instances reuse it instead of reloading.
        """
        if self.model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer

            if os.path.isdir(self._model_path):
                self.model = SentenceTransformer(
                    self._model_path, device="cpu"
                )
                logger.info(
                    "SemanticVectorContextEngine: loaded local model from %s "
                    "on CPU",
                    self._model_path,
                )
                # Store in module-level cache for reuse
                _set_model_cache(self.model, self._model_path)
                return True
            else:
                logger.warning(
                    "SemanticVectorContextEngine: local model not found at %s",
                    self._model_path,
                )
                return False
        except ImportError:
            logger.warning(
                "SemanticVectorContextEngine: sentence-transformers not installed"
            )
            return False
        except Exception as e:
            logger.warning(
                "SemanticVectorContextEngine: failed to load model: %s", e
            )
            return False

    # -- Legacy method aliases (for compatibility with test suites) ------------

    def should_compress(
        self, messages: List[Dict[str, Any]]
    ) -> bool:
        """Legacy alias for should_archive — checks token threshold against messages."""
        total_chars = sum(len(m.get("content", "")) for m in messages)
        estimated_tokens = total_chars // 4
        return self.should_archive(estimated_tokens)

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Legacy alias for archive."""
        return self.archive(messages, current_tokens, focus_topic)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_post_compress(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_pre_archive(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def on_post_archive(self, messages: List[Dict[str, Any]]) -> None:
        pass

    # -- Tools (none by default) ----------------------------------------------

    def get_tool_schemas(self) -> list:
        """Return tool schemas this engine provides. Default: none."""
        return []

    # -- Session lifecycle ----------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins.

        Tries to get the embedding model (from cache or by loading).
        Does NOT nuke the model — the module-level cache persists.
        """
        try:
            self._get_embedding_model()
        except Exception as e:
            logger.debug(
                "Failed to get embedding model on session start: %s", e
            )

    def on_session_end(
        self, session_id: str, messages: List[Dict[str, Any]]
    ) -> None:
        """Called at real session boundaries."""
        pass

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Reset all per-session state.

        Resets per-session vectors, embed cache, and turn counters.
        Does NOT nuke the model — the module-level cache survives across
        session resets so the model isn't reloaded from disk.
        """
        super().on_session_reset()
        self._vectors = []
        self._embed_cache = {}
        self._next_vector_id = 0
        self._current_turn = 0
        self._embedding_engine = None
        # Note: we deliberately do NOT set self.model = None here.
        # If the model was loaded (from cache or fresh load), it stays.
        # The module-level cache is the ultimate safety net.

    # -- Status ---------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging."""
        status = super().get_status()
        status["vectors"] = len(self._vectors)
        status["active_vectors"] = sum(
            1 for v in self._vectors if v.status == "Active"
        )
        status["model_cached"] = _get_model_cache()[0] is not None
        return status

    def update_model(
        self,
        model: str = "",
        context_length: int = 0,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        """Called when the user switches models."""
        if context_length:
            self.context_length = context_length
            self.threshold_tokens = int(
                context_length * self.threshold_percent
            )
        if model:
            super().update_model(
                model, context_length, base_url, api_key, provider
            )


# -- Plugin registration (required for discovery) ----------------------------


def register(collector, config=None):
    """Register this engine with the Hermes plugin system."""
    engine = SemanticVectorContextEngine(**(config or {}))
    collector.register_context_engine(engine)

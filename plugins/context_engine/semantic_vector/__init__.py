"""Semantic Vector Context Engine plugin.

Tracks conversation topics using local embeddings and prunes dormant chatter
instead of applying lossy summarization. Preserves raw signal fidelity while
adapting aggressiveness based on context window pressure.

Device-aware embedding (c4-backbone, 2026-08-19) — its own independent
SentenceTransformer instance, loaded on the best available device
(explicit `device` config or HERMES_EMBED_DEVICE > free-GPU ranked > CPU
fallback). On a 32 GB GPU with vLLM, MiniLM (~90 MB) adds negligible
contention; CPU remains the fallback when no GPU has headroom.

The embedding model is cached in a module-level singleton so that new engine
instances (created after session splits) reuse the already-loaded model
instead of reloading from disk on every archive.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, List

import tqdm
tqdm.disable = True

from agent.context_engine import (
    ContextEngine,
    context_engine_log,
    estimate_content_tokens,
)
from logos_constants import logos_env

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache
# ---------------------------------------------------------------------------
# AIAgent creates a fresh SemanticVectorContextEngine on every session split.
# Without this cache, the SentenceTransformer reloads from disk each time
# (~100 ms each).  With it, the model loads once per process lifetime.

_model_cache: Any = None
_model_path_cache: str = ""
_model_device_cache: str = ""


def _get_model_cache() -> tuple[Any, str, str]:
    """Return the cached (model, path, device) tuple."""
    return _model_cache, _model_path_cache, _model_device_cache


def _set_model_cache(model: Any, path: str, device: str = "") -> None:
    """Store model in module-level cache for reuse across engine instances.

    c4: keyed by (path, device) — a CPU-loaded cache must not be served to
    an instance that explicitly wants CUDA (or vice versa).
    """
    global _model_cache, _model_path_cache, _model_device_cache
    _model_cache = model
    _model_path_cache = path
    _model_device_cache = device


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
    # c5 (2026-08-19): per-topic rolling tail. 0 = legacy (keep ALL turns of
    # Active topics). K > 0 = keep only the last K turns of each Active topic;
    # older turns roll off into Perpetual Memory. This is what makes the
    # window behave like a rolling window: per-topic in-context footprint is
    # bounded regardless of session length or topic re-mentions.
    active_tail_turns: int = 0

    # Rolling-window fallback knobs (C9-A, 2026-08-17): wired from the
    # context.rolling_window section of config.yaml. Defaults preserve the
    # pre-C9-A hardcoded behavior so other users see no change; a user's
    # config section now actually controls the fallback.
    #   window_size        — minimum non-system messages the fallback keeps
    #                        (0 = legacy floor of protect_last_n * 2)
    #   archive_target     — fraction of context_length to prune down to
    #                        (0 = legacy target of threshold_tokens)
    #   hard_ceiling_percent — absolute OOM guard (was hardcoded 0.85)
    #   danger_zone_percent  — fallback trigger ceiling (was hardcoded 0.90)
    window_size: int = 0
    archive_target: float = 0.0
    hard_ceiling_percent: float = 0.85
    danger_zone_percent: float = 0.90
    # Phase C (2026-08-22): task-aware smart pre-prune. Wired from
    # context.rolling_window.task_aware (the fleet already sets it true;
    # the kwargs loop in __init__ picks it up because it's a class attr).
    # False when unset so homes without the key see unchanged behavior.
    # Under this engine the knob runs TaskAwarePruner as a SELECTION pass
    # before the deterministic emergency brake: it chooses which semantic
    # survivors fit the token budget by importance (active-task turns,
    # task markers, user queries, recency), then the deterministic
    # strip/truncate/drop-oldest path only engages if the smart selection
    # still exceeds the target/ceiling. See _task_aware_preprune().
    task_aware: bool = False

    # F13 fix: class attr so the config kwarg 'model_path' lands here
    # (the kwargs loop only sets existing attributes; the class attr was
    # previously '_model_path', so config values were silently dropped).
    model_path: str = ""

    # c4-backbone: embedding device. Empty = auto-select (free-GPU ranked,
    # CPU fallback) via _device_candidates(). Config or HERMES_EMBED_DEVICE
    # override force a specific device. Mirrors the EmbeddingEngine pattern.
    device: str = ""

    # Embedding state
    model: Any = None  # The SentenceTransformer model (also aliased as _model)
    _embedding_engine: Any = None  # For test mocking
    _embed_cache: Dict[str, List[float]] = None
    _vectors: List[ConversationVector] = None
    _next_vector_id: int = 0
    _current_turn: int = 0

    # C-E (2026-08-22): estimate of the static prompt payload (system
    # prompt + tool schemas + injections, ~50-55K tokens) that the chars//4
    # message-content estimate omits. Calibrated in update_from_response
    # from the c2 pair (actual prompt_tokens - message-only estimate);
    # reset in on_session_reset. Added back ONLY at full-context threshold
    # comparisons — _last_archive_post_est and the calibration JSONL keep
    # the message-only number so the c2 delta stays the overhead signal.
    _overhead_est: int = 0

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
        if self.model_path:
            self._model_path = os.path.expanduser(self.model_path)
        if not self._model_path:
            self._model_path = os.path.expanduser(
                "~/.hermes/models/embeddings/all-MiniLM-L6-v2"
            )
        # Phase C (2026-08-22): lazily constructed TaskAwarePruner (see
        # _get_pruner) and the drop count from the last
        # _rolling_window_fallback run (ceiling_override calibration stat).
        self._pruner = None
        self._last_fallback_dropped = 0
        # Set on every archive pass ("semantic" / "semantic_below_threshold"
        # / "rw_fallback" / "tailoff"). Initialized here so readers (e.g.
        # _log_task_aware) are safe on the very first archive call.
        self._last_archive_path = "unknown"

    # -- Abstract methods -----------------------------------------------------

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response."""
        # c2 calibration: if we archived since the previous response, pair
        # the ACTUAL prompt_tokens against our chars//4 estimate of the
        # post-archive context. This is the data source for threshold tuning.
        est = getattr(self, "_last_archive_post_est", 0)
        actual = int(usage.get("prompt_tokens", 0) or 0)
        if est > 0 and actual > 0:
            # C-E (2026-08-22): actual - est isolates the static payload
            # the content estimate omits. Clamp at 0 so an actual that
            # comes in under the estimate (mid-session drift) can't drive
            # the overhead negative.
            overhead = max(0, actual - est)
            self._overhead_est = overhead
            context_engine_log({
                "type": "calibration",
                "engine": self.name,
                "path": getattr(self, "_last_archive_path", "unknown"),
                "estimated": est,
                "actual": actual,
                "ratio": round(actual / est, 3),
                "overhead": overhead,
            })
        self._last_archive_post_est = 0
        self.last_prompt_tokens = usage.get("prompt_tokens", self.last_prompt_tokens)
        self.last_completion_tokens = usage.get("completion_tokens", self.last_completion_tokens)
        self.last_total_tokens = usage.get("total_tokens", self.last_total_tokens)

    def _estimate_total(self, messages: List[Dict[str, Any]]) -> int:
        """Message-content estimate + calibrated static overhead.

        C-E (2026-08-22): estimate_content_tokens() deliberately counts
        message content only — it is the c2 calibration left-hand side,
        where the delta vs actual prompt_tokens IS the overhead signal.
        Every comparison of the engine's estimate against a FULL-context
        threshold (threshold_tokens, danger zone, archive target) must
        use this instead, or the ~50-55K static payload goes uncounted
        and the estimate undercounts the real prompt 2.7-3.1x.
        """
        return estimate_content_tokens(messages) + self._overhead_est

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

        # Keep turns belonging to Active vectors.
        # c5 (2026-08-19): per-topic rolling tail. When active_tail_turns > 0,
        # keep only the LAST K turns of each active topic — older turns of a
        # still-active topic roll off (they live in Perpetual Memory). This
        # bounds per-topic in-context footprint so long multi-topic sessions
        # behave like a rolling window instead of accumulating the full
        # history of every re-mentioned topic. 0 = legacy (keep all).
        tail = self.active_tail_turns
        for vec in self._vectors:
            if vec.status == "Active":
                indices = vec.turn_indices
                if tail > 0:
                    indices = indices[-tail:]
                for ti in indices:
                    # Map turn index back to message index (skip system messages)
                    msg_idx = self._turn_to_msg_index(ti, messages)
                    if msg_idx is not None:
                        keep_indices.add(msg_idx)

        # Estimate tokens from actual message content if not provided.
        # C-E (2026-08-22): use _estimate_total so the calibrated static
        # overhead is counted against the full-context threshold. Without
        # it, a below-threshold short-circuit fires on the message-only
        # number and returns the ORIGINAL list — the c5 per-topic tail
        # (keep_indices computed above) is never applied.
        estimated_tokens = self._estimate_total(messages)
        effective_tokens = (
            current_tokens if current_tokens is not None else estimated_tokens
        )

        # If we're not over threshold, just return original with state map
        if not self.should_archive(effective_tokens):
            result = list(messages)
            self._last_archive_post_est = estimate_content_tokens(result)
            self._last_archive_path = "semantic_below_threshold"
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
        # Mark this path BEFORE the smart pass so _log_task_aware records
        # the right provenance (not the "unknown" init value).
        self._last_archive_path = "semantic"

        # Phase C (2026-08-22): task-aware smart pre-prune. When
        # context.rolling_window.task_aware is set and the pruner finds an
        # active (unclosed) task, TaskAwarePruner selects which semantic
        # survivors fit the prune target by importance — BEFORE the
        # deterministic emergency brake. Selection only: it can drop
        # survivors but never re-keeps a turn the semantic pass archived.
        # No-op (stats None) when there is no task signal, in which case
        # the deterministic path below is exactly the pre-Phase C behavior.
        ta_stats = None
        if self.task_aware:
            result, ta_stats = self._task_aware_preprune(
                result, self._prune_target_tokens()
            )

        # Safety check: if result is still too large after semantic pruning,
        # fall back to rolling window for aggressive pruning.
        # Two triggers:
        #   1. Pruning insufficient — result still over threshold_tokens (75%)
        #   2. Danger zone — current_tokens > 90% of context_length, regardless
        #      of semantic pruning outcome. This is the emergency brake to
        #      prevent OOM/crash.
        result_chars = sum(len(m.get("content", "")) for m in result)
        result_tokens = result_chars // 4
        # C-E (2026-08-22): these thresholds are full-context values, so
        # compare the calibrated total (content + static overhead), not
        # the message-only estimate — otherwise the emergency brake
        # engages up to ~3x too late.
        total_result_tokens = result_tokens + self._overhead_est
        effective_tokens = (
            current_tokens if current_tokens is not None else total_result_tokens
        )
        danger_zone = int(self.context_length * self.danger_zone_percent) if self.context_length else 0
        needs_fallback = False
        if self.threshold_tokens > 0 and total_result_tokens > self.threshold_tokens:
            logger.info(
                "SemanticVector pruning insufficient (%d > %d tokens), "
                "engaging rolling_window fallback",
                total_result_tokens,
                self.threshold_tokens,
            )
            needs_fallback = True
        elif danger_zone > 0 and effective_tokens > danger_zone:
            logger.info(
                "SemanticVector danger zone: %d > %d tokens (90%% ceiling), "
                "engaging rolling_window emergency fallback",
                effective_tokens,
                danger_zone,
            )
            needs_fallback = True
        fallback_input = None
        if needs_fallback:
            # Last resort: deterministic strip/truncate + drop-oldest on
            # what the smart pass (or plain semantic pass) left.
            fallback_input = result
            result = self._rolling_window_fallback(result)
        # Phase C (2026-08-22): record the smart pass whenever it did work —
        # whether or not the last-resort brake had to follow. This data
        # tunes the feature (protected-vs-deterministic, ceiling overrides).
        if ta_stats is not None:
            self._log_task_aware(ta_stats, fallback_input, result)
        if needs_fallback:
            return result

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
        # c2: record post-archive estimate for calibration pairing in
        # update_from_response (actual prompt_tokens arrive with the next
        # API response).
        self._last_archive_post_est = estimate_content_tokens(result)
        self._last_archive_path = "semantic"
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
        self._last_archive_post_est = estimate_content_tokens(keep)
        self._last_archive_path = "tailoff"
        return keep

    # -- Phase C: task-aware smart pre-prune ----------------------------------

    def _prune_target_tokens(self) -> int:
        """Full-context token target for the prune path (Phase C / C9-A).

        The same target the rolling-window fallback drops toward:
        context.rolling_window.archive_target fraction of context_length
        when set, else threshold_tokens, clamped to the hard ceiling.
        Full-context values are compared against content + calibrated
        static overhead (C-E), so the smart pass and the deterministic
        brake aim at the same number.
        """
        if self.context_length and self.archive_target > 0:
            target = int(self.context_length * self.archive_target)
        else:
            target = (
                self.threshold_tokens
                if self.threshold_tokens > 0
                else int(self.context_length * 0.75)
            )
        hard_ceiling = (
            int(self.context_length * self.hard_ceiling_percent)
            if self.context_length
            else 0
        )
        if hard_ceiling > 0 and target > hard_ceiling:
            target = hard_ceiling
        return target

    def _get_pruner(self):
        """Lazily construct the TaskAwarePruner (import pattern mirrors
        plugins/context_engine/rolling_window/__init__.py). Returns None
        if unavailable — the smart pass degrades to the deterministic
        path (task-aware is an enhancement, not a dependency)."""
        if self._pruner is not None:
            return self._pruner
        TaskAwarePruner = None
        try:
            from ..rolling_window.task_aware_pruner import TaskAwarePruner
        except ImportError:
            try:
                from plugins.context_engine.rolling_window.task_aware_pruner import (
                    TaskAwarePruner,
                )
            except ImportError:
                try:
                    import importlib.util
                    import sys

                    rw_dir = os.path.join(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "rolling_window",
                    )
                    # The pruner's own sibling import needs the dir on path.
                    if rw_dir not in sys.path:
                        sys.path.insert(0, rw_dir)
                    spec = importlib.util.spec_from_file_location(
                        "logos_task_aware_pruner",
                        os.path.join(rw_dir, "task_aware_pruner.py"),
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    TaskAwarePruner = mod.TaskAwarePruner
                except Exception as e:
                    logger.debug("TaskAwarePruner unavailable: %s", e)
        if TaskAwarePruner is None:
            return None
        try:
            self._pruner = TaskAwarePruner(
                window_size=self.window_size or 20,
                protect_first_n=self.protect_first_n,
                protect_last_n=self.protect_last_n,
            )
        except Exception as e:
            logger.debug("TaskAwarePruner init failed: %s", e)
            return None
        return self._pruner

    def _deterministic_shrink(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Phase C extraction of the fallback's size-reduction steps
        (1-2): strip assistant tool_calls, truncate >6-line tool results.
        Position-preserving (1:1 with the input) and non-mutating —
        selection is NOT done here; that is either the smart pass or the
        drop-oldest last resort."""
        out = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg = {**msg, "tool_calls": None}
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.split("\n")) > 6:
                    lines = content.split("\n")
                    msg = {
                        **msg,
                        "content": (
                            "\n".join(lines[:3])
                            + "\n...[truncated]...\n"
                            + "\n".join(lines[-3:])
                        ),
                    }
            out.append(msg)
        return out

    def _deterministic_keep_set(
        self, messages: List[Dict[str, Any]], target_tokens: int
    ) -> set:
        """Counterfactual (Phase C calibration): the input indices the
        pre-Phase-C deterministic path — strip + truncate + drop-oldest
        to the same target — would have kept. Mirrors the fallback's
        accounting exactly (drops only ever come from the oldest
        non-system prefix, so positions stay aligned)."""
        shrunken = self._deterministic_shrink(messages)
        hard_ceiling = (
            int(self.context_length * self.hard_ceiling_percent)
            if self.context_length
            else 0
        )
        window_floor = max(self.protect_last_n * 2, self.window_size)
        current = (
            sum(len(m.get("content", "") or "") for m in shrunken) // 4
            + self._overhead_est
        )
        non_system_pos = [
            i for i, m in enumerate(shrunken) if m.get("role") != "system"
        ]
        dropped = 0
        while current > target_tokens:
            if dropped >= len(non_system_pos):
                break
            if (
                len(non_system_pos) - dropped <= window_floor
                and (hard_ceiling == 0 or current <= hard_ceiling)
            ):
                break
            pos = non_system_pos[dropped]
            current -= len(shrunken[pos].get("content", "") or "") // 4
            dropped += 1
        kept = set(range(len(messages)))
        for j in range(dropped):
            kept.discard(non_system_pos[j])
        return kept

    def _task_aware_preprune(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any] | None]:
        """Phase C smart pre-prune: TaskAwarePruner picks which of the
        semantic survivors fit under target_tokens, ranked by importance
        (its existing heuristics: active-task turns, task markers, user
        queries, status updates, recency).

        Selection only — it runs on the semantic pass's OUTPUT, so it can
        drop survivors but never re-keeps a turn _semantic_archive
        already archived (the archive's summary wins). Returns
        (selected, stats); stats is None when the pass is a no-op (no
        pruner, no active/unclosed task, or already under target), in
        which case the deterministic path below runs exactly as it did
        pre-Phase C (regression guard).
        """
        if not self.task_aware or not messages:
            return messages, None
        if self._estimate_total(messages) <= target_tokens:
            return messages, None
        pruner = self._get_pruner()
        if pruner is None:
            return messages, None
        try:
            scores = pruner.score_turns(messages)
        except Exception as e:
            logger.warning(
                "Task-aware scoring failed, deterministic path: %s", e
            )
            return messages, None
        if not any(s.task_bonus > 0 for s in scores):
            # No active (unclosed) task in this context — the pruner has
            # no task signal, so skip the smart pass entirely.
            return messages, None

        est_tokens = [len(m.get("content", "") or "") // 4 for m in messages]
        total = len(messages)

        # Protected set (mirrors the pruner's own convention): system
        # messages + the newest protect_last_n messages.
        protected = set()
        for i, m in enumerate(messages):
            if m.get("role") == "system":
                protected.add(i)
        for i in range(max(0, total - self.protect_last_n), total):
            protected.add(i)

        # Token budget for score-ranked fills: target is a full-context
        # value, so subtract the calibrated static overhead and the
        # protected set's cost first.
        budget = (
            target_tokens
            - self._overhead_est
            - sum(est_tokens[i] for i in protected)
        )

        keep = set(protected)
        rest = sorted(
            (s for s in scores if s.turn_index not in protected),
            key=lambda s: s.total_score,
            reverse=True,
        )
        for s in rest:
            if est_tokens[s.turn_index] <= budget:
                keep.add(s.turn_index)
                budget -= est_tokens[s.turn_index]

        sorted_keep = sorted(keep)
        selected = [messages[i] for i in sorted_keep]

        # Calibration counterfactual: what the pre-Phase-C deterministic
        # path would have kept from the SAME input at the SAME target.
        det_kept = self._deterministic_keep_set(messages, target_tokens)
        task_idx = {s.turn_index for s in scores if s.task_bonus > 0}
        pos_of_idx = {idx: pos for pos, idx in enumerate(sorted_keep)}
        stats = {
            "messages_in": total,
            "kept": len(selected),
            "protected_turns": len(keep - det_kept),
            "deterministic_would_keep": len(det_kept),
            "task_protected_kept": len(task_idx & keep),
            # Internal: positions of task-protected turns inside the
            # selected list (the fallback input) — used to compute
            # ceiling_override after the last-resort drop pass.
            "_protected_positions": {
                pos_of_idx[i] for i in (task_idx & keep)
            },
        }
        return selected, stats

    def _log_task_aware(
        self,
        ta_stats: Dict[str, Any],
        fallback_input: List[Dict[str, Any]],
        final_result: List[Dict[str, Any]],
    ) -> None:
        """Phase C calibration event: record when the smart pass ran,
        how many turns it protected vs the deterministic path, and
        whether the last-resort pass overrode the pruner. This data
        tunes the feature (c2 JSONL sibling: task_aware_prune)."""
        protected_positions = ta_stats.pop("_protected_positions", set())
        if fallback_input is None or fallback_input is final_result:
            # No last-resort drop pass ran — nothing was overridden.
            ceiling_override = False
        else:
            # The fallback drops only from the oldest non-system prefix
            # (positions 1:1 with fallback_input), so a protected turn
            # was overridden iff one sits in that dropped prefix.
            ns_pos = [
                i
                for i, m in enumerate(fallback_input)
                if m.get("role") != "system"
            ]
            dropped = set(ns_pos[: self._last_fallback_dropped])
            ceiling_override = bool(protected_positions & dropped)
        ta_stats["ceiling_override"] = ceiling_override
        context_engine_log({
            "type": "task_aware_prune",
            "engine": self.name,
            "path": self._last_archive_path,
            "messages_in": ta_stats["messages_in"],
            "kept": ta_stats["kept"],
            "protected_turns": ta_stats["protected_turns"],
            "deterministic_would_keep": ta_stats["deterministic_would_keep"],
            "task_protected_kept": ta_stats["task_protected_kept"],
            "ceiling_override": ceiling_override,
        })

    # -- Rolling window fallback (hard prune when nearing OOM) -----------------

    def _rolling_window_fallback(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Hard prune using rolling window logic: strip tool calls,
        truncate verbose results, then drop oldest until under ceiling.

        This is the emergency brake when semantic pruning hasn't trimmed
        enough and we're approaching the hard ceiling."""
        if not messages:
            return messages

        # Step 1: Strip raw assistant tool calls (verbose JSON bloat)
        stripped = []
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                stripped.append({**msg, "tool_calls": None})
            else:
                stripped.append(msg)

        # Step 2: Truncate verbose tool results to first/last 3 lines
        truncated = []
        for msg in stripped:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.split("\n")) > 6:
                    lines = content.split("\n")
                    truncated_content = "\n".join(lines[:3]) + "\n...[truncated]...\n" + "\n".join(lines[-3:])
                    msg = {**msg, "content": truncated_content}
            truncated.append(msg)

        # Step 3: Drop oldest unprotected messages until under target.
        # C9-A (2026-08-17): targets and floors now come from config
        # (context.rolling_window) via class attributes. Legacy defaults
        # (window_size=0, archive_target=0) reproduce the old hardcoded
        # behavior exactly: target = threshold_tokens, floor = protect_last_n*2.
        if self.context_length and self.archive_target > 0:
            target_tokens = int(self.context_length * self.archive_target)
        else:
            target_tokens = self.threshold_tokens if self.threshold_tokens > 0 else int(self.context_length * 0.75)
        hard_ceiling_tokens = int(self.context_length * self.hard_ceiling_percent) if self.context_length else 0

        # Safety: never let target exceed the hard ceiling
        if hard_ceiling_tokens > 0 and target_tokens > hard_ceiling_tokens:
            target_tokens = hard_ceiling_tokens

        # Working-window floor: never shrink below window_size messages
        # (or the legacy protect_last_n*2 floor when window_size is unset).
        window_floor = max(self.protect_last_n * 2, self.window_size)

        # Estimate current tokens.
        # C-E (2026-08-22): bake the calibrated static overhead into the
        # counter — target_tokens and hard_ceiling_tokens are full-context
        # values, so the drop-oldest loop and the floor/ceiling stop
        # conditions compare apples to apples. Message drops subtract the
        # same per-message amount as before; the overhead is constant.
        current_chars = sum(len(m.get("content", "")) for m in truncated)
        current_tokens = current_chars // 4 + self._overhead_est

        # Protect system messages and last N messages
        system_msgs = [m for m in truncated if m.get("role") == "system"]
        non_system_msgs = [m for m in truncated if m.get("role") != "system"]

        # Drop oldest non-system messages until under target. Stop when the
        # token target is met, or when the working-window floor is reached
        # AND we're under the OOM guard — the guard wins over the floor.
        while current_tokens > target_tokens:
            if not non_system_msgs:
                break
            if (
                len(non_system_msgs) <= window_floor
                and (hard_ceiling_tokens == 0 or current_tokens <= hard_ceiling_tokens)
            ):
                logger.warning(
                    "Rolling window fallback: at working-window floor (%d msgs) "
                    "still at %d tokens (target %d, ceiling %d) — stopping",
                    len(non_system_msgs), current_tokens, target_tokens, hard_ceiling_tokens,
                )
                break
            # Drop one message from the oldest unprotected
            dropped = non_system_msgs.pop(0)
            dropped_chars = len(dropped.get("content", ""))
            current_tokens -= dropped_chars // 4
            logger.debug(
                "Rolling window fallback: dropped oldest message, "
                "tokens %d -> %d (target %d)",
                current_tokens + dropped_chars // 4,
                current_tokens,
                target_tokens,
            )

        result = system_msgs + non_system_msgs
        pruned_count = len(messages) - len(result)

        # Phase C (2026-08-22): record how many non-system messages the
        # last-resort drop pass removed, so _log_task_aware can compute
        # ceiling_override (did the drop pass eat a task-protected turn?).
        self._last_fallback_dropped = pruned_count

        logger.info(
            "Rolling window fallback: %d -> %d messages, "
            "pruned %d (tokens %d, target %d, hard_ceiling %d)",
            len(messages),
            len(result),
            pruned_count,
            current_tokens,
            target_tokens,
            hard_ceiling_tokens,
        )
        self.archive_count += 1
        self._last_archive_post_est = estimate_content_tokens(result)
        self._last_archive_path = "rw_fallback"
        return result

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
        cached_model, cached_path, cached_device = _get_model_cache()
        if cached_model is not None:
            # c4: reuse only if the path matches AND the device is compatible.
            # An explicit `device` config must be honored (a CPU cache is not
            # served to a CUDA request); auto-select reuses whatever device
            # was previously chosen.
            want = (self.device or "").strip()
            device_ok = (not want) or (cached_device == want)
            # If the path matches and the device is compatible, use the cached
            # model directly.
            if cached_path == self._model_path and device_ok:
                self.model = cached_model
                logger.debug(
                    "SemanticVectorContextEngine: reused cached model from "
                    "module cache (path=%s device=%s)",
                    self._model_path,
                    cached_device,
                )
                return cached_model
            else:
                # Path or device differs — load new model but keep cache for
                # future instances that match.
                logger.debug(
                    "SemanticVectorContextEngine: cached (path=%s device=%s) "
                    "!= requested (path=%s device=%s), loading new model",
                    cached_path,
                    cached_device,
                    self._model_path,
                    want or "auto",
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
                # SentenceTransformer interface. show_progress_bar=False is
                # required: ST passes an explicit disable= to tqdm, which
                # overrides the class-level tqdm.disable=True set above.
                result = engine.encode(text, show_progress_bar=False)
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
                # Replace (not stack) any previously-injected state map —
                # legacy behavior left one map per archive event in the
                # stored transcript.
                from agent.context_scaffolding import strip_state_map

                content = strip_state_map(content)
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

    def _device_candidates(self) -> List[str]:
        """Ordered embedding-device candidates (c4-backbone, 2026-08-19).

        Mirrors ``agent.perpetual_context_db.EmbeddingEngine._select_device_candidates``:

        1. Explicit ``device`` config or ``HERMES_EMBED_DEVICE`` env — tried
           first and *only* (one knob controls all local embedders).
        2. Otherwise every CUDA device ranked by free memory (most-free
           first). Devices with < 2 GB free (e.g. a vLLM-packed GPU) are
           tried last — a load there can OOM, but it can also work.
        3. ``cpu`` as the final fallback.
        """
        forced = (self.device or logos_env("EMBED_DEVICE", "")).strip()
        if forced:
            return [forced]
        ordered: List[tuple] = []
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        free, _total = torch.cuda.mem_get_info(i)
                    except Exception:
                        free = 0
                    ordered.append((free, i))
                ordered.sort(key=lambda t: t[0], reverse=True)
        except Exception:
            ordered = []
        min_free = 2 * 1024 ** 3
        free_gpus = [f"cuda:{i}" for free, i in ordered if free > min_free]
        packed_gpus = [f"cuda:{i}" for free, i in ordered if free <= min_free]
        return free_gpus + packed_gpus + ["cpu"]

    def _load_model(self) -> bool:
        """Load the embedding model on the best available device.

        c4-backbone: walks ``_device_candidates()`` in order; the first
        successful load wins and is stored in the module-level cache keyed
        by (path, device) so subsequent engine instances reuse it instead of
        reloading. Any device failure degrades to the next candidate — a GPU
        hiccup must never take the engine down to tail-off.

        Returns True if a model is loaded (on any device).
        """
        if self.model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "SemanticVectorContextEngine: sentence-transformers not installed"
            )
            return False

        if not os.path.isdir(self._model_path):
            logger.warning(
                "SemanticVectorContextEngine: local model not found at %s",
                self._model_path,
            )
            return False

        # Suppress tqdm progress bars during load and all future encode() calls
        import tqdm
        tqdm.disable = True

        for device in self._device_candidates():
            try:
                self.model = SentenceTransformer(self._model_path, device=device)
                # Store in module-level cache for reuse
                _set_model_cache(self.model, self._model_path, device)
                logger.info(
                    "SemanticVectorContextEngine: loaded local model from %s "
                    "on %s (c4)",
                    self._model_path,
                    device,
                )
                return True
            except Exception as e:
                logger.warning(
                    "SemanticVectorContextEngine: load on %s failed: %s — "
                    "trying next candidate",
                    device,
                    e,
                )
        logger.warning(
            "SemanticVectorContextEngine: all device candidates failed for %s",
            self._model_path,
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
        # C-E: overhead is a per-session measurement (tool schemas /
        # injections differ per session); don't leak it across resets.
        self._overhead_est = 0
        # Phase C: last-resort drop count is per-run state.
        self._last_fallback_dropped = 0
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
    """Register this engine with the Logos plugin system."""
    engine = SemanticVectorContextEngine(**(config or {}))
    collector.register_context_engine(engine)

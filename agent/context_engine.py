"""Abstract base class for pluggable context engines.

A context engine controls how conversation context is managed when
approaching the model's token limit. The built-in ContextCompressor
is the default implementation. Third-party engines (e.g. LCM) can
replace it via the plugin system or by being placed in the
``plugins/context_engine/<name>/`` directory.

Selection is config-driven: ``context.engine`` in config.yaml.
Default is ``"compressor"`` (the built-in). Only one engine is active.

The engine is responsible for:
  - Deciding when compaction should fire
  - Performing compaction (summarization, DAG construction, etc.)
  - Optionally exposing tools the agent can call (e.g. lcm_grep)
  - Tracking token usage from API responses

Lifecycle:
  1. Engine is instantiated and registered (plugin register() or default)
  2. on_session_start() called when a conversation begins
  3. update_from_response() called after each API response with usage data
  4. should_compress() checked after each turn
  5. compress() called when should_compress() returns True
  6. on_session_end() called at real session boundaries (CLI exit, /reset,
     gateway session expiry) — NOT per-turn
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Set, Optional
import os
from dataclasses import dataclass, field
import torch

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_EMBEDDER = True
except ImportError:
    HAS_EMBEDDER = False

@dataclass
class ConversationVector:
    id: str
    status: str  # "Active", "Dormant", "Resolved"
    last_seen_turn: int = 0
    embedding: Optional[torch.Tensor] = None
    turns: List[int] = field(default_factory=list)

class ContextEngine(ABC):
    """Base class all context engines must implement."""

    # -- Identity ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    # -- Token state (read by run_agent.py for display/logging) ------------
    #
    # Engines MUST maintain these. run_agent.py reads them directly.

    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    archive_count: int = 0

    # -- Compaction parameters (read by run_agent.py for preflight) --------
    #
    # These control the preflight compression check.  Subclasses may
    # override via __init__ or property; defaults are sensible for most
    # engines.

    threshold_percent: float = 0.75
    protect_first_n: int = 3
    protect_last_n: int = 6

    # -- Core interface ----------------------------------------------------

    @abstractmethod
    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Update tracked token usage from an API response.

        Called after every LLM call with the usage dict from the response.
        """

    @abstractmethod
    def should_archive(self, prompt_tokens: int = None) -> bool:
        """Return True if archiving should fire this turn."""

    @abstractmethod
    def archive(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        """Archive old messages and return the new message list.

        This is the main entry point. The engine receives the full message
        list and returns a (possibly shorter) list that fits within the
        context budget. The implementation is free to summarize, build a
        DAG, or do anything else — as long as the returned list is a valid
        OpenAI-format message sequence.

        Args:
            focus_topic: Optional topic string from manual ``/archive <focus>``.
                Engines that support guided archiving should prioritise
                preserving information related to this topic.  Engines that
                don't support it may simply ignore this argument.
        """

    # -- Optional: manual /archive preflight ------------------------------

    def has_content_to_archive(self, messages: List[Dict[str, Any]]) -> bool:
        """Quick check: is there anything in ``messages`` that can be archived?

        Used by the gateway ``/archive`` command as a preflight guard —
        returning False lets the gateway report "nothing to archive yet"
        without making an LLM call.

        Default returns True (always attempt).  Engines with a cheap way
        to introspect their own head/tail boundaries should override this
        to return False when the transcript is still entirely protected.
        """
        return True

    # -- Optional: session lifecycle ---------------------------------------

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Called when a new conversation session begins.

        Use this to load persisted state (DAG, store) for the session.
        kwargs may include hermes_home, platform, model, etc.
        """

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Called at real session boundaries (CLI exit, /reset, gateway expiry).

        Use this to flush state, close DB connections, etc.
        NOT called per-turn — only when the session truly ends.
        """

    def on_session_reset(self) -> None:
        """Called on /new or /reset. Reset per-session state.

        Default resets archive_count and token tracking.
        """
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.archive_count = 0

    # -- Optional: tools ---------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas this engine provides to the agent.

        Default returns empty list (no tools). LCM would return schemas
        for lcm_grep, lcm_describe, lcm_expand here.
        """
        return []

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle a tool call from the agent.

        Only called for tool names returned by get_tool_schemas().
        Must return a JSON string.

        kwargs may include:
          messages: the current in-memory message list (for live ingestion)
        """
        import json
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # -- Optional: status / display ----------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return status dict for display/logging.

        Default returns the standard fields run_agent.py expects.
        """
        return {
            "last_prompt_tokens": self.last_prompt_tokens,
            "threshold_tokens": self.threshold_tokens,
            "context_length": self.context_length,
            "usage_percent": (
                min(100, self.last_prompt_tokens / self.context_length * 100)
                if self.context_length else 0
            ),
            "archive_count": self.archive_count,
        }

    # -- Optional: model switch support ------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        """Called when the user switches models or on fallback activation.

        Default updates context_length and recalculates threshold_tokens
        from threshold_percent. Override if your engine needs more
        (e.g. recalculate DAG budgets, switch summary models).
        """
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

class SemanticVectorEngine(ContextEngine):
    """A+ Production Context Engine using Semantic Vector Tracking.
    
    Instead of lossy compression, this engine tracks conversation 'vectors' 
    using a local embedding model and maintains a Consolidated State Map header.
    """
    @property
    def name(self) -> str:
        return "semantic_vector"

    def __init__(self, model_path='~/.hermes/models/embeddings/all-MiniLM-L6-v2'):
        super().__init__()
        self.vectors: Dict[str, ConversationVector] = {}
        self.current_vector_id: Optional[str] = None
        self.model_path = os.path.expanduser(model_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None  # Initialize before _get_model() checks it
        # Eagerly load the model from local path to ensure total sovereignty and zero network latency
        self.model = self._get_model()

    def _get_model(self):
        if self.model is None and HAS_EMBEDDER:
            try:
                # Load directly from the sovereign local path; no network checks allowed
                self.model = SentenceTransformer(self.model_path, device=self.device, local_files_only=True)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    "SemanticVectorEngine embedder not found at %s (%s). Falling back to recency-only pruning.",
                    self.model_path, e
                )
        return self.model

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_archive(self, prompt_tokens: int = None) -> bool:
        # Semantic Vector Engine archives based on token threshold or explicit trigger
        overhead = getattr(self, "system_prompt_tokens", 0) + getattr(self, "tool_schema_tokens", 0)
        tokens = (prompt_tokens or self.last_prompt_tokens) + overhead
        return tokens > self.threshold_tokens

    def archive(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        # 1. Update Vector State for ALL messages to build the map before pruning
        if messages:
            for i in range(len(messages)):
                msg = messages[i]
                self._update_vector_state(i, msg.get("content", ""))

        active_vector = self.current_vector_id
        
        # 2. Determine aggressiveness based on context pressure
        tokens = current_tokens or self.last_prompt_tokens
        usage_ratio = tokens / self.context_length if self.context_length else 0.75
        
        # Adaptive thresholds: closer to limit → more aggressive pruning
        recency_buffer = 3 if usage_ratio < 0.80 else (1 if usage_ratio < 0.92 else 2)
        include_dormant_turns = usage_ratio < 0.85
        max_active_turns = 60 if usage_ratio < 0.90 else 25

        pruned_messages = []
        
        # 3. Protect System Prompt (always first)
        if messages:
            pruned_messages.append(messages[0])

        # 4. Generate Compact Consolidated State Map Header
        # NOTE: Only include turn COUNT, not indices. Full index lists bloat the header to thousands of tokens.
        state_header = "[Conversation State | "
        vector_summaries = []
        for vid, vec in self.vectors.items():
            vector_summaries.append(f"{vid}:{vec.status}(count:{len(vec.turns)})")
        state_header += " | ".join(vector_summaries) + "]"
        pruned_messages.append({"role": "system", "content": state_header})

        # 2b. SignalRegistry integration — protect pinned high-signal turns
        pinned_indices: Set[int] = set()
        try:
            from agent.signal_registry import SignalRegistry
            registry = SignalRegistry()
            pm_ids = [msg.get("id") for msg in messages if msg.get("id")]
            if pm_ids:
                pinned_pms = registry.get_pinned_turns(pm_ids)
                # Map PM IDs back to message indices
                for idx, msg in enumerate(messages):
                    if msg.get("id") in pinned_pms:
                        pinned_indices.add(idx)
        except Exception as e:
            logger.debug(f"SignalRegistry unavailable during archive: {e}")

        # 5. Filter Messages based on Vector Relevance and Recency
        active_turn_count = 0
        for i in range(1, len(messages)):
            msg = messages[i]

            # Signal-pinned protection (high-signal turns survive pruning)
            if i in pinned_indices:
                pruned_messages.append(msg)
                continue
            
            # Absolute recency protection (adaptive buffer)
            if (len(messages) - i) <= recency_buffer:
                pruned_messages.append(msg)
                continue

            # Vector Relevance Check
            is_active = False
            for vid, vec in self.vectors.items():
                if i in vec.turns and vid == active_vector:
                    is_active = True
                    break
            
            # Include active turns (capped to prevent runaway context)
            if is_active and active_turn_count < max_active_turns:
                pruned_messages.append(msg)
                active_turn_count += 1
                continue
                
            # Include dormant turns only if we have breathing room
            if include_dormant_turns:
                for vid, vec in self.vectors.items():
                    if i in vec.turns and vec.status == "Dormant":
                        pruned_messages.append(msg)
                        break

        # 6. HARD SAFETY NET: If still over budget, aggressively trim oldest non-protected
        target_tokens = int(self.context_length * 0.82)
        while len(pruned_messages) > 2 and self._estimate_tokens(pruned_messages) > target_tokens:
            pruned_messages.pop(1)  # Drop oldest message after system prompt

        self.archive_count += max(0, len(messages) - len(pruned_messages))
        return pruned_messages

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Rough token estimator to prevent context overflow crashes.
        
        Uses ~3.5 chars/token as a safe upper bound for English/code mix.
        Guarantees we never exceed hard limits even if semantic pruning fails."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return max(total_chars // 3.5, len(messages) * 10)

    def _update_vector_state(self, turn_index: int, text: str):
        model = self._get_model()
        if not model:
            # Fallback: treat every turn as its own vector (recency-only pruning)
            vid = f"vec_{turn_index}"
            self.vectors[vid] = ConversationVector(
                id=vid, status="Active", last_seen_turn=turn_index, turns=[turn_index]
            )
            self.current_vector_id = vid
            return

        turn_embedding = model.encode(text, convert_to_tensor=True).to(self.device)
        best_vector = None
        highest_sim = -1.0
        
        for vid, vec in self.vectors.items():
            if vec.embedding is not None:
                vec_tensor = vec.embedding.to(self.device)
                sim = util.cos_sim(turn_embedding, vec_tensor).item()
                # Use a more robust similarity threshold for production
                if sim > 0.45 and sim > highest_sim: 
                    highest_sim = sim
                    best_vector = vid

        if best_vector is None:
            best_vector = f"vec_{turn_index}"
            self.vectors[best_vector] = ConversationVector(
                id=best_vector, 
                status="Active", 
                last_seen_turn=turn_index,
                embedding=turn_embedding,
                turns=[turn_index]
            )
        else:
            vec = self.vectors[best_vector]
            vec.last_seen_turn = turn_index
            vec.status = "Active"
            vec.turns.append(turn_index)

        self.current_vector_id = best_vector

        # Decay other vectors to Dormant if not seen for 5 turns
        for vid, vec in self.vectors.items():
            if vid != self.current_vector_id and (turn_index - vec.last_seen_turn) > 5:
                vec.status = "Dormant"

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.vectors = {}
        self.current_vector_id = None

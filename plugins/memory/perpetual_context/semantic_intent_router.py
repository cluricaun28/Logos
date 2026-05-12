"""Semantic Intent Router — centroid-based classification using local embeddings.

Replaces keyword matching with embedding-based intent classification. Embeds the
user's query once, then computes cosine similarity to pre-computed centroids for
each intent category. The single embedding call is shared with the prefetch
pipeline's Phase 1b hybrid search — no extra model load, no extra GPU memory.

Architecture:
- Centroids are pre-embedded representative queries for each intent
- Runtime: embed query once (reuses EmbeddingEngine singleton), cosine to each centroid
- Threshold-based: only fire intents that exceed similarity threshold
- Graceful degradation: if embedding model unavailable, falls back to keyword router

Performance: one additional embedding call per turn (~50-100ms), but this is the
*same* call the prefetch pipeline already makes in Phase 1b. By sharing the
embedding, total cost is one embed, not two.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent definitions — each has a centroid computed from representative queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntentDef:
    """Definition of an injection intent with routing flags and training queries."""

    name: str
    # Routing flags — what to fire when this intent is detected
    fire_prefetch: bool
    fire_recall: bool
    fire_web: bool
    needs_recent_context: bool
    needs_clarification: bool
    confidence: str  # "high" | "medium" | "low"
    # Representative training queries for centroid computation
    training_queries: tuple[str, ...]


# Each intent has hand-crafted representative queries that define its semantic
# center. At module load time, these are embedded to create centroid vectors.
_INTENTS: list[IntentDef] = [
    IntentDef(
        name="past_work",
        fire_prefetch=False,
        fire_recall=True,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "did we discuss this before",
            "what did we do last time about X",
            "remember when we fixed",
            "check our past conversations about",
            "continue where we left off",
            "what happened with the",
            "back up our progress on",
        ),
    ),
    IntentDef(
        name="recent",
        fire_prefetch=False,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=True,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "what just happened in this session",
            "in the last few turns",
            "right before that",
            "what were we just talking about",
            "going back to what I said earlier",
        ),
    ),
    IntentDef(
        name="factual",
        fire_prefetch=True,
        fire_recall=False,
        fire_web=True,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "what is X and how does it work",
            "explain the concept of",
            "tell me about",
            "define and describe",
            "what are the details of",
            "how does this work",
        ),
    ),
    IntentDef(
        name="research",
        fire_prefetch=True,
        fire_recall=False,
        fire_web=True,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "search for current information about",
            "find out the latest on",
            "what's happening with right now",
            "research and summarize",
            "look up the current status of",
        ),
    ),
    IntentDef(
        name="reference",
        fire_prefetch=True,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "read the docs about",
            "check the reference library for",
            "what's in our knowledge base about",
            "do we have a page on",
            "look up in the reference library",
        ),
    ),
    IntentDef(
        name="code",
        fire_prefetch=False,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "fix this bug in the code",
            "implement the feature",
            "refactor this module",
            "build and test",
            "deploy the changes",
            "fix this bug",
            "implement the new api endpoint",
            "audit the code for issues",
            "push the changes to the repo",
            "restart the service",
        ),
    ),
    IntentDef(
        name="opinion",
        fire_prefetch=False,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "what do you think about",
            "should I do this",
            "is it better to",
            "in your opinion",
            "what's your take on",
            "would you recommend",
        ),
    ),
    IntentDef(
        name="comparison",
        fire_prefetch=True,
        fire_recall=False,
        fire_web=True,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "compare these two options",
            "what's the difference between",
            "how does this compare to",
            "versus",
            "which is better",
        ),
    ),
    IntentDef(
        name="system",
        fire_prefetch=True,
        fire_recall=True,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="medium",
        training_queries=(
            "are you using the prefetch pipeline",
            "is the system working correctly",
            "how are the injections configured",
            "check the system status",
            "what modules are active",
            "verify the setup",
        ),
    ),
    IntentDef(
        name="status",
        fire_prefetch=False,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=True,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "is it done yet",
            "still broken",
            "any updates",
            "is it working now",
            "has that been fixed",
        ),
    ),
    IntentDef(
        name="conversation",
        fire_prefetch=False,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "yes",
            "no",
            "ok",
            "thanks",
            "hello",
            "agreed",
        ),
    ),
    IntentDef(
        name="topic",
        fire_prefetch=True,
        fire_recall=False,
        fire_web=False,
        needs_recent_context=False,
        needs_clarification=False,
        confidence="high",
        training_queries=(
            "let's talk about",
            "on the topic of",
            "speaking of",
            "regarding",
            "concerning",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Centroid cache — pre-computed at module load from training queries
# ---------------------------------------------------------------------------

_centroids: dict[str, list[float]] = {}
_centroid_loaded: bool = False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_centroids() -> None:
    """Pre-compute centroid vectors for all intents from training queries.

    Each centroid is the mean embedding of its training queries.
    Falls back gracefully if the embedding model is unavailable.
    """
    global _centroids, _centroid_loaded

    try:
        from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415

        engine = EmbeddingEngine.get()
    except ImportError:
        logger.warning(
            "SemanticIntentRouter: EmbeddingEngine not available — "
            "falling back to keyword router"
        )
        _centroid_loaded = False
        return

    for intent in _INTENTS:
        vectors: list[list[float]] = []
        for query in intent.training_queries:
            vec = engine.embed(query)
            if vec is not None:
                vectors.append(vec)

        if vectors:
            # Mean vector as centroid
            dim = len(vectors[0])
            centroid = [0.0] * dim
            for v in vectors:
                for d in range(dim):
                    centroid[d] += v[d]
            for d in range(dim):
                centroid[d] /= len(vectors)
            _centroids[intent.name] = centroid
        else:
            logger.warning(
                "SemanticIntentRouter: no embeddings for intent '%s' — skipping",
                intent.name,
            )

    _centroid_loaded = len(_centroids) > 0
    if _centroid_loaded:
        logger.info(
            "SemanticIntentRouter: computed %d/%d centroids from %d training queries",
            len(_centroids),
            len(_INTENTS),
            sum(len(i.training_queries) for i in _INTENTS),
        )
    else:
        logger.warning(
            "SemanticIntentRouter: failed to compute any centroids — "
            "embedding model may not be loaded yet"
        )


def _ensure_centroids() -> bool:
    """Ensure centroids are computed. Returns True if available."""
    if not _centroid_loaded:
        _compute_centroids()
    return _centroid_loaded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.65  # Minimum cosine similarity to fire an intent — high bar to prevent noise injections


def classify_injection_intent(query: str) -> dict[str, Any]:
    """Classify the user's message using semantic embedding matching.

    Embeds the query once, computes cosine similarity to each intent centroid,
    and returns the routing dict for the best-matching intent.

    Falls back to keyword-based router if embedding model is unavailable.

    Args:
        query: The user's message text.

    Returns:
        A routing dict with keys: fire_prefetch, fire_recall, fire_web,
        needs_recent_context, needs_clarification, confidence, intent.
    """
    if not query or not query.strip():
        return _fallback_empty()

    # Try semantic classification first
    if _ensure_centroids():
        try:
            from agent.perpetual_context_db import EmbeddingEngine  # noqa: PLC0415

            engine = EmbeddingEngine.get()
            query_vector = engine.embed(query)

            if query_vector is not None:
                result = _classify_semantic(query_vector, query)
                if result is not None:
                    return result
                # Embedding succeeded but no intent exceeded threshold —
                # fall through to keyword router
        except (AttributeError, TypeError) as e:
            logger.debug(
                "SemanticIntentRouter: embedding failed, falling back to keyword: %s", e
            )

    # Fall back to keyword router
    return _keyword_classify(query)


def _classify_semantic(
    query_vector: list[float], query: str
) -> dict[str, Any] | None:
    """Find the best-matching intent by cosine similarity to centroids."""
    best_score = 0.0
    best_intent: IntentDef | None = None

    for intent in _INTENTS:
        centroid = _centroids.get(intent.name)
        if centroid is None:
            continue
        score = _cosine_similarity(query_vector, centroid)
        if score > best_score:
            best_score = score
            best_intent = intent

    if best_intent is not None and best_score >= _SIMILARITY_THRESHOLD:
        return _intent_to_routing(best_intent, best_score)

    # No intent exceeded threshold — don't inject. Better to inject nothing than noise.
    return None  # Fall back to keyword router which is more conservative


def _intent_to_routing(intent: IntentDef, score: float) -> dict[str, Any]:
    """Convert an IntentDef to the routing dict format expected by the prefetch pipeline."""
    return {
        "fire_prefetch": intent.fire_prefetch,
        "fire_recall": intent.fire_recall,
        "fire_web": intent.fire_web,
        "needs_recent_context": intent.needs_recent_context,
        "needs_clarification": intent.needs_clarification,
        "confidence": intent.confidence,
        "intent": intent.name,
        "semantic_score": round(score, 3),
    }


# ---------------------------------------------------------------------------
# Keyword fallback — same logic as the old injection_router.py
# ---------------------------------------------------------------------------


def _phrase_match(triggers: set[str], lower: str) -> bool:
    return any(phrase in lower for phrase in triggers)


def _word_match(triggers: set[str], words: set) -> bool:
    return bool(words & triggers)


_KEYWORD_TRIGGERS: dict[str, set[str]] = {
    "past_work": {
        "did we", "have we", "were you able", "did you finish",
        "last time", "remember", "check pm", "from our conversation",
        "continue", "what happened", "what did we do",
    },
    "recent": {
        "recent", "recently", "recent turns", "what just",
        "in the last", "this session", "just now", "right before",
    },
    "factual": {
        "what is", "what are", "who is", "who are",
        "tell me about", "explain", "define", "describe",
        "how does it work", "tell me more",
    },
    "factual_words": {
        "what", "who", "where", "when", "why", "how",
        "explain", "define", "describe",
    },
    "opinion": {
        "what do you think", "should i", "is it better",
        "would you recommend", "do you agree",
        "what's your take", "in your opinion",
    },
    "short": {
        "yes", "no", "ok", "okay", "thanks", "hello", "hi",
        "good", "right", "correct", "wrong", "agreed",
    },
    "status": {
        "still broken", "is it done", "any updates", "working",
        "still no", "still happening", "fixed",
    },
    "code": {
        "fix", "audit", "refactor", "implement", "build",
        "deploy", "push to", "backup", "restart",
    },
    "reference": {
        "how does", "what's the config", "read the docs",
        "is there an rl page", "rl page", "reference library",
        "read file", "take a look",
    },
    "research": {
        "research", "search for", "find out about",
        "what's happening", "latest news", "current status of",
    },
    "comparison": {
        "compare", "compared to", "difference between",
        "versus", "vs ", "vs.", "better than",
    },
}


def _keyword_classify(query: str) -> dict[str, Any]:
    """Fallback keyword-based classification from the old injection_router."""
    lower = query.lower()
    raw_words = lower.split()
    words = {w.strip(".,!?;:\"'()[]{}") for w in raw_words}
    words = {w for w in words if w}

    # Short chit-chat
    if len(words) <= 3 and _word_match(_KEYWORD_TRIGGERS["short"], words):
        return _fallback_conversation()

    # Status
    if _phrase_match(_KEYWORD_TRIGGERS["status"], lower):
        return _fallback_status()

    # Code
    if _word_match(_KEYWORD_TRIGGERS["code"], words):
        if _phrase_match(_KEYWORD_TRIGGERS["recent"], lower):
            return _fallback_recent()
        if _phrase_match(_KEYWORD_TRIGGERS["past_work"], lower):
            return _fallback_past_work()
        return _fallback_code()

    # Recent
    if _phrase_match(_KEYWORD_TRIGGERS["recent"], lower):
        return _fallback_recent()

    # Past work
    if _phrase_match(_KEYWORD_TRIGGERS["past_work"], lower):
        return _fallback_past_work()

    # Reference
    if _word_match(_KEYWORD_TRIGGERS["reference"], words) or _phrase_match(_KEYWORD_TRIGGERS["reference"], lower):
        return _fallback_reference()

    # Research
    if _word_match(_KEYWORD_TRIGGERS["research"], words) or _phrase_match(_KEYWORD_TRIGGERS["research"], lower):
        return _fallback_research()

    # Factual
    if _phrase_match(_KEYWORD_TRIGGERS["factual"], lower):
        return _fallback_factual()

    # Comparison
    if _word_match(_KEYWORD_TRIGGERS["comparison"], words) or _phrase_match(_KEYWORD_TRIGGERS["comparison"], lower):
        return _fallback_comparison()

    # Opinion
    if _phrase_match(_KEYWORD_TRIGGERS["opinion"], lower):
        return _fallback_opinion()

    # Factual fallback for short wh- queries — but only if the query is short
    # AND contains multiple wh- words (genuine question, not just a statement with "what")
    if len(words) <= 4 and _word_match(_KEYWORD_TRIGGERS["factual_words"], words):
        return _fallback_factual_fallback()

    # Ambiguous — don't inject. Better to inject nothing than noise.
    return _fallback_ambiguous()


# -- Fallback intent configs --


def _fallback_empty() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "low", "intent": "empty",
    }


def _fallback_conversation() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "conversation",
    }


def _fallback_status() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": True, "needs_clarification": False,
        "confidence": "high", "intent": "status",
    }


def _fallback_code() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "code",
    }


def _fallback_recent() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": True, "needs_clarification": False,
        "confidence": "high", "intent": "recent",
    }


def _fallback_past_work() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": True, "fire_web": False,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "past_work",
    }


def _fallback_reference() -> dict[str, Any]:
    return {
        "fire_prefetch": True, "fire_recall": False, "fire_web": False,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "reference",
    }


def _fallback_research() -> dict[str, Any]:
    return {
        "fire_prefetch": True, "fire_recall": False, "fire_web": True,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "research",
    }


def _fallback_factual() -> dict[str, Any]:
    return {
        "fire_prefetch": True, "fire_recall": False, "fire_web": True,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "factual",
    }


def _fallback_factual_fallback() -> dict[str, Any]:
    return {
        "fire_prefetch": True, "fire_recall": False, "fire_web": True,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "medium", "intent": "factual_fallback",
    }


def _fallback_comparison() -> dict[str, Any]:
    return {
        "fire_prefetch": True, "fire_recall": False, "fire_web": True,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "comparison",
    }


def _fallback_opinion() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": False, "needs_clarification": False,
        "confidence": "high", "intent": "opinion",
    }


def _fallback_ambiguous() -> dict[str, Any]:
    return {
        "fire_prefetch": False, "fire_recall": False, "fire_web": False,
        "needs_recent_context": True, "needs_clarification": False,
        "confidence": "low", "intent": "ambiguous",
    }

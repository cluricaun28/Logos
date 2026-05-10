"""Injection Router — data-driven intent classification for perpetual context.

Replaces the copy-pasted return dicts in _classify_injection_intent with a
config-driven approach. Each intent maps to a routing dict (what to inject)
and trigger configs (how to match).

Design principle: only inject when it would improve the response.
When in doubt, inject nothing rather than noise.
"""

from __future__ import annotations

from typing import Any


def _phrase_match(triggers: set[str], lower: str) -> bool:
    """Check if any trigger phrase appears in the lowered query."""
    return any(phrase in lower for phrase in triggers)


def _word_match(triggers: set[str], words: set) -> bool:
    """Check if any trigger word intersects with the query words set."""
    return bool(words & triggers)


# ---------------------------------------------------------------------------
# Trigger phrase sets — extracted from class-level _INTENT_* attributes
# ---------------------------------------------------------------------------

_TRIGGERS: dict[str, set[str]] = {
    "past_work": {
        "did we", "have we", "were you able", "did you finish",
        "last time", "remember", "check pm", "from our conversation",
        "continue", "what happened", "what did we do",
        "back up our", "assess the errors", "review the",
    },
    "recent": {
        "recent", "recently", "recent turns", "what just",
        "in the last", "this session", "just now", "right before",
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
    "factual": {
        "what is", "what are", "who is", "who are",
        "tell me about", "explain", "define",
        "what's a", "what's an", "describe",
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
    "comparison": {
        "compare", "compared to", "difference between",
        "versus", "vs ", "vs.", "better than",
        "alternative to", "how does x compare",
    },
    "topic": {
        "let's talk about", "on the topic of", "speaking of",
        "about", "regarding", "concerning",
    },
    "code": {
        "fix", "audit", "refactor", "implement", "build",
        "deploy", "push to", "backup", "restart",
    },
    "command": {
        "add 10 new", "review the conversation above",
        "download is done", "let's see if that solves",
        "ok, built it",
    },
    "short": {
        "yes", "no", "ok", "okay", "thanks", "hello", "hi",
        "good", "right", "correct", "wrong", "agreed",
    },
    "status": {
        "still broken", "is it done", "any updates", "working",
        "still no", "still happening", "fixed",
    },
}


# ---------------------------------------------------------------------------
# Routing configs — one dict per intent outcome, no duplication
# ---------------------------------------------------------------------------

_INTENT_CONFIG: dict[str, dict[str, Any]] = {
    "conversation": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "conversation",
    },
    "status": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": True,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "status",
    },
    "command": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "command",
    },
    "recent": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": True,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "recent",
    },
    "past_work": {
        "fire_prefetch": False,
        "fire_recall": True,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "past_work",
    },
    "code": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "code",
    },
    "reference": {
        "fire_prefetch": True,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "reference",
    },
    "research": {
        "fire_prefetch": True,
        "fire_recall": False,
        "fire_web": True,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "research",
    },
    "factual": {
        "fire_prefetch": True,
        "fire_recall": False,
        "fire_web": True,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "factual",
    },
    "factual_fallback": {
        "fire_prefetch": True,
        "fire_recall": False,
        "fire_web": True,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "medium",
        "intent": "factual",
    },
    "comparison": {
        "fire_prefetch": True,
        "fire_recall": False,
        "fire_web": True,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "comparison",
    },
    "opinion": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "opinion",
    },
    "topic": {
        "fire_prefetch": True,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "high",
        "intent": "topic",
    },
    "ambiguous": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": True,
        "needs_clarification": False,
        "confidence": "low",
        "intent": "ambiguous",
    },
    "empty": {
        "fire_prefetch": False,
        "fire_recall": False,
        "fire_web": False,
        "needs_recent_context": False,
        "needs_clarification": False,
        "confidence": "low",
        "intent": "empty",
    },
}


# ---------------------------------------------------------------------------
# Core classifier — standalone function, no class dependency
# ---------------------------------------------------------------------------

def classify_injection_intent(query: str) -> dict[str, Any]:
    """Classify the user's message and decide which injections to fire.

    Returns a routing dict with boolean flags for each injection type,
    a confidence level, and flags for recent-context / clarification fallback.

    Args:
        query: The user's message text.

    Returns:
        A dict with keys: fire_prefetch, fire_recall, fire_web,
        needs_recent_context, needs_clarification, confidence, intent.
    """
    if not query:
        return _INTENT_CONFIG["empty"]

    lower = query.lower()
    # Tokenize and strip punctuation for word-level matching
    raw_words = lower.split()
    words = {w.strip(".,!?;:\"'()[]{}") for w in raw_words}
    words = {w for w in words if w}

    # --- Quick check: very short messages (chitchat) ---
    if len(words) <= 3 and _word_match(_TRIGGERS["short"], words):
        return _INTENT_CONFIG["conversation"]

    # --- Status check: "still X", "is it done", etc. ---
    if _phrase_match(_TRIGGERS["status"], lower):
        return _INTENT_CONFIG["status"]

    # --- Command/instruction: skip retrieval entirely ---
    if _phrase_match(_TRIGGERS["command"], lower):
        return _INTENT_CONFIG["command"]

    # --- Code task ---
    if _word_match(_TRIGGERS["code"], words):
        if _phrase_match(_TRIGGERS["recent"], lower):
            return _INTENT_CONFIG["recent"]
        if _phrase_match(_TRIGGERS["past_work"], lower):
            return _INTENT_CONFIG["past_work"]
        return _INTENT_CONFIG["code"]

    # --- Recent context (takes priority over past_work) ---
    if _phrase_match(_TRIGGERS["recent"], lower):
        return _INTENT_CONFIG["recent"]

    # --- Past work references ---
    if _phrase_match(_TRIGGERS["past_work"], lower):
        return _INTENT_CONFIG["past_work"]

    # --- Reference lookup ---
    if _word_match(_TRIGGERS["reference"], words) or _phrase_match(_TRIGGERS["reference"], lower):
        return _INTENT_CONFIG["reference"]

    # --- External research (explicit request) ---
    if _word_match(_TRIGGERS["research"], words) or _phrase_match(_TRIGGERS["research"], lower):
        return _INTENT_CONFIG["research"]

    # --- Factual questions: "what is X", "tell me about Y" ---
    if _phrase_match(_TRIGGERS["factual"], lower):
        return _INTENT_CONFIG["factual"]

    # --- Comparison: "how does X compare to Y" ---
    if _word_match(_TRIGGERS["comparison"], words) or _phrase_match(_TRIGGERS["comparison"], lower):
        return _INTENT_CONFIG["comparison"]

    # --- Opinion/advice: no injection, just reasoning ---
    if _phrase_match(_TRIGGERS["opinion"], lower):
        return _INTENT_CONFIG["opinion"]

    # --- Topic discussion: "let's talk about X" ---
    if _phrase_match(_TRIGGERS["topic"], lower):
        return _INTENT_CONFIG["topic"]

    # --- Fallback: short question-like phrases with wh- words ---
    # Detect simple factual questions that didn't match _INTENT_FACTUAL phrases
    if len(words) <= 8 and _word_match(_TRIGGERS["factual_words"], words):
        if not _phrase_match(_TRIGGERS["command"], lower):
            return _INTENT_CONFIG["factual_fallback"]

    # --- Ambiguous: no strong signal ---
    return _INTENT_CONFIG["ambiguous"]

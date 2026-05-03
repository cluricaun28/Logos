"""Semantic Vector Context Engine plugin.

Tracks conversation topics using local embeddings and prunes dormant chatter
instead of applying lossy summarization. Preserves raw signal fidelity while
adapting aggressiveness based on context window pressure.
"""

from agent.context_engine import SemanticVectorEngine  # noqa: F401

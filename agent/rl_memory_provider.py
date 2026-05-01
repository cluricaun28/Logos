
import logging
import os
from typing import Any, Dict, List, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from agent.memory_provider import MemoryProvider
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

class ReferenceLibraryMemoryProvider(MemoryProvider):
    """
    Hybrid Memory Provider that replaces flat-file snapshots with 
    on-demand retrieval from the Reference Library (RL).
    """
    def __init__(self, model_path="~/.hermes/models/embeddings/all-MiniLM-L6-v2"):
        self._model_path = os.path.expanduser(model_path)
        self.model = None 
        self.rl_dir = Path(get_hermes_home()) / "reference-library"

    @property
    def name(self) -> str:
        return "builtin"

    def is_available(self) -> bool:
        return self.rl_dir.exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            self.model = SentenceTransformer(self._model_path, device="cuda")
            logger.info("ReferenceLibraryMemoryProvider: Embedding model loaded on CUDA.")
        except Exception as e:
            logger.error(f"ReferenceLibraryMemoryProvider: Failed to load embedder: {e}")

    def system_prompt_block(self) -> str:
        return "Your personal memory is now managed via the Reference Library. Relevant facts are injected dynamically."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self.model: return ""

        docs = []
        user_profile = self.rl_dir / "entities/patrick-daley.md"
        if user_profile.exists():
            docs.append((user_profile.name, user_profile.read_text(encoding="utf-8")))

        topics_dir = self.rl_dir / "topics"
        if topics_dir.exists():
            for f in topics_dir.glob("*.md"):
                docs.append((f.name, f.read_text(encoding="utf-8")))

        if not docs: return ""

        texts = [d[1] for d in docs]
        names = [d[0] for d in docs]
        
        query_emb = self.model.encode([query])
        doc_embs = self.model.encode(texts)
        sims = cosine_similarity(query_emb, doc_embs)[0]

        keyword_scores = []
        query_words = set(query.lower().split())
        for text in texts:
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            keyword_scores.append(overlap)

        final_scores = []
        for i in range(len(texts)):
            score = (sims[i] * 0.7) + (min(keyword_scores[i]/10, 1.0) * 0.3)
            final_scores.append((score, names[i], texts[i]))

        final_scores.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, name, text in final_scores[:3]:
            if score > 0.3:
                snippet = text[:500] + "..." if len(text) > 500 else text
                results.append(f"[{name}]: {snippet}")

        return "\n\n".join(results)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return "Memory writes are now handled via Reference Library updates."

    def shutdown(self) -> None:
        self.model = None

from __future__ import annotations
from .chunking import Chunk

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  

        self._model = CrossEncoder(model_name)
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        pairs = [(query, chunk.text) for chunk, _ in candidates]
        scores = self._model.predict(pairs)
        reranked = sorted(zip((c for c, _ in candidates), scores), key=lambda x: x[1], reverse=True)
        return [(chunk, float(score)) for chunk, score in reranked[:top_k]]
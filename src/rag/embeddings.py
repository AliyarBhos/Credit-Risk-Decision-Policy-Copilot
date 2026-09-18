from __future__ import annotations
from typing import Protocol
import numpy as np


class Embedder(Protocol):
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray:
        ...

class FastEmbedEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  
        self._model = TextEmbedding(model_name)
        self.model_name = model_name
        self.dim = 384  

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        vectors = list(self._model.embed(texts))
        return np.array(vectors, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        prefixed = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed([prefixed])[0]
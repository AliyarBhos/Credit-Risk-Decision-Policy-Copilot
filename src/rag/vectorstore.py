from __future__ import annotations
import json
import os
import pickle
from dataclasses import asdict
import faiss
import numpy as np
from .chunking import Chunk


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if len(vectors) != len(chunks):
            raise ValueError("Number of vectors must match number of chunks")
        if len(vectors) == 0:
            return
        self.index.add(self._normalize(vectors))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        if self.index.ntotal == 0:
            return []
        k = min(k, self.index.ntotal)
        query = self._normalize(query_vector.reshape(1, -1))
        scores, indices = self.index.search(query, k)
        return [
            (self.chunks[idx], float(score))
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"dim": self.dim, "n_chunks": len(self.chunks)}, f)

    @classmethod
    def load(cls, directory: str) -> "FAISSVectorStore":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.pkl"), "rb") as f:
            store.chunks = pickle.load(f)
        return store
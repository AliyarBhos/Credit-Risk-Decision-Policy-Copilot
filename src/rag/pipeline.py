from __future__ import annotations
from .chunking import Chunk, chunk_document
from .embeddings import FastEmbedEmbedder
from .generation import generate_answer
from .ingestion import load_policy_documents
from .vectorstore import FAISSVectorStore

class RAGPipeline:
    def __init__(
        self,
        embedder: FastEmbedEmbedder | None = None,
        reranker=None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ):
        self.embedder = embedder or FastEmbedEmbedder()
        self.reranker = reranker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store: FAISSVectorStore | None = None

    def build_index(self, policies_dir: str) -> None:
        documents = load_policy_documents(policies_dir)

        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(
                chunk_document(
                    doc.doc_id, doc.source, doc.text,
                    chunk_size=self.chunk_size, overlap=self.chunk_overlap,
                )
            )

        texts = [c.text for c in all_chunks]
        vectors = self.embedder.embed(texts)

        self.store = FAISSVectorStore(dim=self.embedder.dim)
        self.store.add(vectors, all_chunks)

    def save_index(self, directory: str) -> None:
        if self.store is None:
            raise RuntimeError("No index built yet — call build_index() first")
        self.store.save(directory)

    def load_index(self, directory: str) -> None:
        self.store = FAISSVectorStore.load(directory)

    def retrieve(self, query: str, k: int = 5, rerank_k: int | None = None) -> list[tuple[Chunk, float]]:
        if self.store is None:
            raise RuntimeError("No index loaded — call build_index() or load_index() first")

        query_vector = self.embedder.embed_query(query)
        fetch_k = rerank_k or (k * 4 if self.reranker else k)
        candidates = self.store.search(query_vector, k=fetch_k)

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=k)
        return candidates[:k]

    def answer(
        self,
        question: str,
        pd_score: float,
        risk_drivers: dict[str, float],
        k: int = 5,
    ) -> dict:
        retrieved = self.retrieve(question, k=k)
        chunks = [chunk for chunk, _ in retrieved]

        answer_text = generate_answer(question, pd_score, risk_drivers, chunks)

        return {
            "answer": answer_text,
            "retrieved_sources": [
                {"source": c.source, "chunk_id": c.chunk_id, "score": score}
                for c, score in retrieved
            ],
        }
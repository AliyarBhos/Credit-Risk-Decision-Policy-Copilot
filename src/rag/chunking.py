from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def _split_into_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    paragraphs = _split_into_paragraphs(text)
    chunks: list[str] = []
    current = ""

    def flush(carry_overlap: bool = True):
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
            current = current[-overlap:] if carry_overlap else ""
        else:
            current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 > chunk_size:
                    flush()
                current = f"{current} {sent}".strip()
            continue

        if len(current) + len(para) + 2 > chunk_size:
            flush()
        current = f"{current}\n\n{para}".strip()

    flush(carry_overlap=False)
    return chunks


def chunk_document(doc_id: str, source: str, text: str, **chunk_kwargs) -> list[Chunk]:
    raw_chunks = chunk_text(text, **chunk_kwargs)
    return [
        Chunk(
            chunk_id=f"{doc_id}::chunk_{i}",
            doc_id=doc_id,
            source=source,
            text=chunk,
            metadata={"chunk_index": i, "n_chunks": len(raw_chunks)},
        )
        for i, chunk in enumerate(raw_chunks)
    ]
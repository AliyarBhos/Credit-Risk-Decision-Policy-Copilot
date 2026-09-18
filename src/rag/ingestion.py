from __future__ import annotations
import os
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class IngestedDocument:
    doc_id: str
    source: str
    text: str


def _extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_txt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


_LOADERS = {
    ".pdf": _extract_pdf_text,
    ".txt": _extract_txt_text,
    ".md": _extract_txt_text,
}


def load_policy_documents(policies_dir: str) -> list[IngestedDocument]:
    documents: list[IngestedDocument] = []

    if not os.path.isdir(policies_dir):
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    for filename in sorted(os.listdir(policies_dir)):
        ext = os.path.splitext(filename)[1].lower()
        loader = _LOADERS.get(ext)
        if loader is None:
            continue

        path = os.path.join(policies_dir, filename)
        text = loader(path)
        if not text.strip():
            continue

        doc_id = os.path.splitext(filename)[0]
        documents.append(IngestedDocument(doc_id=doc_id, source=filename, text=text))

    if not documents:
        raise ValueError(
            f"No supported policy documents (.pdf, .txt, .md) found in {policies_dir}"
        )

    return documents
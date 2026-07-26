"""Split extracted pages into embeddable chunks.

Chunks are built per page rather than over the concatenated document, so a chunk never
straddles a page boundary and its recorded page number is always exactly right. A
citation that points at the wrong page is worse than no citation — it looks authoritative
and sends the reader somewhere the claim isn't.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.rag.extraction import ExtractedPage


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str
    page: int | None
    label: str


def _splitter() -> RecursiveCharacterTextSplitter:
    settings = get_settings()
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        # Prefer breaking at paragraph, then line, then sentence, then word — the
        # defaults, made explicit because retrieval quality depends on them.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        keep_separator=False,
    )


def chunk_pages(pages: list[ExtractedPage]) -> list[Chunk]:
    """Flatten pages into chunks numbered consecutively across the whole document."""
    splitter = _splitter()
    chunks: list[Chunk] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            text = piece.strip()
            if not text:
                continue
            chunks.append(Chunk(index=len(chunks), content=text, page=page.page, label=page.label))
    return chunks


__all__ = ["Chunk", "chunk_pages"]

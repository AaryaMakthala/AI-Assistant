"""Chunking must preserve the locator, because citations are built from it."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.rag.chunking import chunk_pages
from app.rag.extraction import ExtractedPage

pytestmark = pytest.mark.usefixtures("valid_env")


def _page(number: int, text: str) -> ExtractedPage:
    return ExtractedPage(page=number, text=text, label=f"page {number}")


def test_chunk_indices_are_consecutive_across_the_whole_document() -> None:
    pages = [_page(1, "alpha " * 400), _page(2, "beta " * 400)]

    chunks = chunk_pages(pages)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))


def test_a_chunk_never_spans_two_pages() -> None:
    """Splitting per page is what keeps a chunk's recorded page number truthful."""
    pages = [_page(1, "alpha " * 300), _page(2, "beta " * 300)]

    chunks = chunk_pages(pages)

    for chunk in chunks:
        assert not ("alpha" in chunk.content and "beta" in chunk.content)
        expected_page = 1 if "alpha" in chunk.content else 2
        assert chunk.page == expected_page
        assert chunk.label == f"page {expected_page}"


def test_long_page_is_split_into_multiple_chunks() -> None:
    settings = get_settings()
    pages = [_page(1, "word " * 2000)]

    chunks = chunk_pages(pages)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= settings.chunk_size for chunk in chunks)


def test_short_page_stays_a_single_chunk() -> None:
    chunks = chunk_pages([_page(1, "A short refund policy.")])

    assert len(chunks) == 1
    assert chunks[0].content == "A short refund policy."


def test_unpaginated_page_keeps_a_null_page_number() -> None:
    pages = [ExtractedPage(page=None, text="Body text.", label="document")]

    chunks = chunk_pages(pages)

    assert chunks[0].page is None
    assert chunks[0].label == "document"


def test_whitespace_only_pages_produce_no_chunks() -> None:
    assert chunk_pages([_page(1, "   \n\n  ")]) == []


def test_no_pages_produce_no_chunks() -> None:
    assert chunk_pages([]) == []

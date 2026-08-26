"""Markdown joins the upload allowlist (Phase 10).

Markdown is treated as plain UTF-8 text: never rendered, never parsed for links, so it
carries the same risk profile as .txt. These tests hold that line — in particular that a
`.md` extension cannot smuggle non-text bytes past the sniffer, and that adding the type
did not widen the allowlist to anything else.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from app.config import get_settings
from app.rag.extraction import extract_pages
from app.security.uploads import (
    ALLOWED_TYPES,
    UploadRejected,
    resolve_type,
    stream_to_storage,
)


class FakeUpload:
    """Minimal stand-in for Starlette's UploadFile: an async chunked reader."""

    def __init__(self, data: bytes) -> None:
        self._stream = BytesIO(data)

    async def read(self, size: int) -> bytes:
        return self._stream.read(size)


@pytest.fixture
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_env: None):
    destination = tmp_path / "uploads"
    monkeypatch.setenv("UPLOAD_DIR", str(destination))
    get_settings.cache_clear()
    yield destination
    get_settings.cache_clear()


@pytest.mark.parametrize("filename", ["notes.md", "notes.markdown", "NOTES.MD"])
def test_both_markdown_spellings_are_accepted(filename: str) -> None:
    allowed = resolve_type(filename)

    assert allowed.mime_type == "text/markdown"
    # Both resolve to one canonical extension, so one extractor serves both.
    assert allowed.extension == "md"


def test_the_allowlist_gained_only_markdown() -> None:
    """A guard against the allowlist quietly growing — it is a security boundary."""
    assert set(ALLOWED_TYPES) == {"pdf", "docx", "xlsx", "csv", "txt", "md", "markdown"}


async def test_a_markdown_upload_is_stored(upload_dir: Path) -> None:
    body = b"# Refund policy\n\nRefunds are issued within 30 days.\n"

    stored = await stream_to_storage(FakeUpload(body), filename="policy.md")

    assert stored.mime_type == "text/markdown"
    assert stored.size_bytes == len(body)
    assert stored.path.read_bytes() == body


async def test_binary_content_wearing_a_markdown_name_is_rejected(upload_dir: Path) -> None:
    """The extension is a claim; the leading bytes decide."""
    with pytest.raises(UploadRejected) as caught:
        await stream_to_storage(FakeUpload(b"\x00\x01\x02binary"), filename="evil.md")

    assert caught.value.status_code == 415
    assert not list(upload_dir.glob("*")) if upload_dir.exists() else True


async def test_an_empty_markdown_file_is_rejected(upload_dir: Path) -> None:
    with pytest.raises(UploadRejected):
        await stream_to_storage(FakeUpload(b""), filename="empty.md")


def test_markdown_extraction_keeps_syntax_and_normalizes(tmp_path: Path, valid_env: None) -> None:
    """Syntax is retained as a structural hint; whitespace artefacts are not."""
    source = tmp_path / "notes.md"
    source.write_text(
        "# Refund policy\n\n\n\nRefunds are   issued within 30 days.\n",
        encoding="utf-8",
    )

    pages = extract_pages(source, extension="md")

    assert len(pages) == 1
    assert pages[0].page is None
    assert pages[0].label == "document"
    assert pages[0].text == "# Refund policy\n\nRefunds are issued within 30 days."


def test_markdown_links_are_not_resolved(tmp_path: Path, valid_env: None) -> None:
    """Nothing in a document is ever fetched — the link survives as literal text."""
    source = tmp_path / "notes.md"
    source.write_text("See [the policy](https://internal.example/secret).", encoding="utf-8")

    pages = extract_pages(source, extension="md")

    assert "https://internal.example/secret" in pages[0].text

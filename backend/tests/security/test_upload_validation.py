"""Upload validation is the boundary hostile files have to cross (CLAUDE.md 4.2).

Each test here corresponds to a way a real upload endpoint gets abused: wrong type, right
type but wrong bytes, too big, or a filename that tries to escape the storage directory.
"""

from __future__ import annotations

import uuid
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from app.config import get_settings
from app.security.uploads import (
    UploadRejected,
    safe_display_filename,
    storage_path_for,
    stream_to_storage,
)


class FakeUpload:
    """Minimal stand-in for Starlette's UploadFile: an async chunked reader."""

    def __init__(self, data: bytes) -> None:
        self._stream = BytesIO(data)

    async def read(self, size: int) -> bytes:
        return self._stream.read(size)


@pytest.fixture
def upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid_env: None) -> Path:
    destination = tmp_path / "uploads"
    monkeypatch.setenv("UPLOAD_DIR", str(destination))
    get_settings.cache_clear()
    yield destination
    get_settings.cache_clear()


def _pdf_bytes(body: bytes = b"stub") -> bytes:
    return b"%PDF-1.7\n" + body + b"\n%%EOF"


def _docx_bytes() -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", "<document/>")
    return buffer.getvalue()


def _stored_files(upload_dir: Path) -> list[Path]:
    return [p for p in upload_dir.glob("*")] if upload_dir.exists() else []


async def test_accepts_an_allowed_type(upload_dir: Path) -> None:
    stored = await stream_to_storage(FakeUpload(_pdf_bytes()), filename="policy.pdf")

    assert stored.mime_type == "application/pdf"
    assert stored.filename == "policy.pdf"
    assert stored.size_bytes == len(_pdf_bytes())
    assert stored.path.exists()
    assert stored.path.read_bytes() == _pdf_bytes()


async def test_rejects_a_type_not_on_the_allowlist(upload_dir: Path) -> None:
    with pytest.raises(UploadRejected) as exc:
        await stream_to_storage(FakeUpload(b"MZ\x90\x00"), filename="payload.exe")

    assert exc.value.status_code == 415
    assert _stored_files(upload_dir) == []


async def test_rejects_extensionless_file(upload_dir: Path) -> None:
    with pytest.raises(UploadRejected) as exc:
        await stream_to_storage(FakeUpload(b"hello"), filename="README")
    assert exc.value.status_code == 415


async def test_rejects_content_that_contradicts_the_extension(upload_dir: Path) -> None:
    """Renaming an executable to .pdf must not get it past the door."""
    with pytest.raises(UploadRejected) as exc:
        await stream_to_storage(FakeUpload(b"MZ\x90\x00executable"), filename="invoice.pdf")

    assert exc.value.status_code == 415
    assert "does not look like" in exc.value.message
    assert _stored_files(upload_dir) == []


async def test_rejects_binary_disguised_as_text(upload_dir: Path) -> None:
    with pytest.raises(UploadRejected):
        await stream_to_storage(FakeUpload(b"text\x00\x01\x02binary"), filename="notes.txt")


async def test_rejects_empty_file(upload_dir: Path) -> None:
    with pytest.raises(UploadRejected) as exc:
        await stream_to_storage(FakeUpload(b""), filename="empty.txt")

    assert exc.value.status_code == 400
    assert _stored_files(upload_dir) == []


async def test_rejects_oversized_file_and_leaves_nothing_behind(upload_dir: Path) -> None:
    """The cap trips mid-stream; a partial file must not survive the rejection."""
    oversized = _pdf_bytes(b"A" * 5000)

    with pytest.raises(UploadRejected) as exc:
        await stream_to_storage(FakeUpload(oversized), filename="huge.pdf", max_bytes=1024)

    assert exc.value.status_code == 413
    assert _stored_files(upload_dir) == []


async def test_accepts_ooxml_container(upload_dir: Path) -> None:
    stored = await stream_to_storage(FakeUpload(_docx_bytes()), filename="handbook.docx")
    assert stored.mime_type.endswith("wordprocessingml.document")


async def test_stored_path_is_a_uuid_inside_the_upload_dir(upload_dir: Path) -> None:
    """The bytes are never stored under a user-supplied name."""
    stored = await stream_to_storage(FakeUpload(b"a,b\n1,2\n"), filename="report.csv")

    assert stored.path.name == str(stored.storage_key)
    uuid.UUID(stored.path.name)  # raises if the name is not a generated UUID
    assert stored.path.parent == upload_dir.resolve()
    assert "report" not in str(stored.path)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\config", "config"),
        ("/absolute/path/notes.txt", "notes.txt"),
        ("with\x00null.txt", "with_null.txt"),
        ("....", "unnamed"),
        ("", "unnamed"),
    ],
)
def test_display_filename_is_stripped_of_path_and_control_characters(
    raw: str, expected: str
) -> None:
    assert safe_display_filename(raw) == expected


async def test_traversing_filename_cannot_escape_the_upload_dir(upload_dir: Path) -> None:
    stored = await stream_to_storage(FakeUpload(b"hello"), filename="../../../../etc/passwd.txt")

    assert stored.path.parent == upload_dir.resolve()
    assert stored.filename == "passwd.txt"


def test_storage_path_is_confined_to_the_upload_dir(upload_dir: Path) -> None:
    key = uuid.uuid4()
    assert storage_path_for(key).parent == upload_dir.resolve()

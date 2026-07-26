"""Upload validation and safe storage (CLAUDE.md 4.2).

Everything here runs before a byte reaches an extraction library. The order matters:
the size cap is enforced *while* streaming, so an oversized upload is abandoned partway
rather than being written out and measured afterwards, and the content sniff runs on the
first chunk, so a file whose bytes contradict its name never gets stored at all.

A declared content type is a claim by the client, not evidence — the leading bytes decide.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.config import get_settings

#: Read size while streaming an upload. Small enough that the size cap trips promptly.
CHUNK_BYTES = 64 * 1024

_ZIP_LOCAL_HEADER = b"PK\x03\x04"
_PDF_HEADER = b"%PDF-"


class UploadRejected(Exception):
    """An upload failed validation. The message is safe to show the user."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class _Sniffer(Protocol):
    def __call__(self, head: bytes) -> bool: ...


def _is_pdf(head: bytes) -> bool:
    return head.startswith(_PDF_HEADER)


def _is_ooxml(head: bytes) -> bool:
    """DOCX and XLSX are both ZIP containers; which one is settled during extraction.

    Only a non-empty archive qualifies: `PK\\x05\\x06` is an empty central directory,
    which cannot hold the parts a real document needs.
    """
    return head.startswith(_ZIP_LOCAL_HEADER)


def _is_text(head: bytes) -> bool:
    """Text must be NUL-free and valid UTF-8.

    A NUL byte is the cheapest reliable signal of a binary file wearing a .txt name.
    The head may end mid-codepoint, so a truncation error at the tail is tolerated.
    """
    if b"\x00" in head:
        return False
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as exc:
        return exc.reason == "unexpected end of data" or exc.end >= len(head) - 3
    return True


@dataclass(frozen=True)
class AllowedType:
    extension: str
    mime_type: str
    sniff: _Sniffer


#: The allowlist from CLAUDE.md 4.2. Anything not named here is refused.
ALLOWED_TYPES: dict[str, AllowedType] = {
    "pdf": AllowedType("pdf", "application/pdf", _is_pdf),
    "docx": AllowedType(
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        _is_ooxml,
    ),
    "xlsx": AllowedType(
        "xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        _is_ooxml,
    ),
    "csv": AllowedType("csv", "text/csv", _is_text),
    "txt": AllowedType("txt", "text/plain", _is_text),
}

_ALLOWED_LIST = ", ".join(sorted(ALLOWED_TYPES))

_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


def safe_display_filename(raw: str) -> str:
    """Reduce a client-supplied name to something safe to store and render.

    This is for display only — the bytes are stored under a generated UUID, so this
    value never reaches the filesystem. It is still sanitized because it *is* echoed
    back into a UI, and because a name like `../../etc/passwd` in a log or a future
    download header is a problem waiting to happen.
    """
    # NFKC first: otherwise fullwidth solidus and friends survive the character filter.
    name = unicodedata.normalize("NFKC", raw).strip()
    # Windows and POSIX separators both, then any residual traversal segments.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("_", name).strip(". ")
    if not name:
        name = "unnamed"
    return name[:255]


def extension_of(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def resolve_type(filename: str) -> AllowedType:
    """Look up the allowlist entry for a filename, or reject it."""
    extension = extension_of(filename)
    allowed = ALLOWED_TYPES.get(extension)
    if allowed is None:
        raise UploadRejected(
            f"Files of type '{extension or 'unknown'}' are not accepted. "
            f"Allowed types: {_ALLOWED_LIST}.",
            status_code=415,
        )
    return allowed


def verify_content_matches(allowed: AllowedType, head: bytes) -> None:
    """Reject a file whose leading bytes contradict its extension."""
    if not head:
        raise UploadRejected("The uploaded file is empty.")
    if not allowed.sniff(head):
        raise UploadRejected(
            f"File content does not look like a valid .{allowed.extension} file.",
            status_code=415,
        )


class _ChunkSource(Protocol):
    async def read(self, size: int) -> bytes: ...


@dataclass(frozen=True)
class StoredUpload:
    storage_key: uuid.UUID
    path: Path
    filename: str
    mime_type: str
    size_bytes: int


def storage_path_for(storage_key: uuid.UUID) -> Path:
    """Resolve where a stored object lives, by generated key only.

    No user-controlled component enters this path, so traversal is impossible by
    construction rather than by filtering.
    """
    return (get_settings().upload_dir / str(storage_key)).resolve()


async def stream_to_storage(
    source: _ChunkSource,
    *,
    filename: str,
    max_bytes: int | None = None,
) -> StoredUpload:
    """Validate and persist an upload, aborting as soon as it breaks a rule.

    Raises :class:`UploadRejected` for a disallowed type, a content/extension mismatch,
    an empty file, or one that exceeds ``max_bytes``. Nothing is left on disk when an
    upload is rejected.
    """
    settings = get_settings()
    limit = settings.max_upload_bytes if max_bytes is None else max_bytes

    display_name = safe_display_filename(filename)
    allowed = resolve_type(display_name)

    storage_key = uuid.uuid4()
    destination = storage_path_for(storage_key)
    destination.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    sniffed = False
    # Written under a partial suffix and renamed only on success, so a crashed or
    # rejected upload can never be mistaken for a complete one.
    partial = destination.with_suffix(".partial")
    try:
        with partial.open("wb") as handle:
            while chunk := await source.read(CHUNK_BYTES):
                if not sniffed:
                    verify_content_matches(allowed, chunk)
                    sniffed = True
                total += len(chunk)
                if total > limit:
                    raise UploadRejected(
                        f"File exceeds the maximum size of {limit // (1024 * 1024)} MB.",
                        status_code=413,
                    )
                handle.write(chunk)
        if not sniffed:
            raise UploadRejected("The uploaded file is empty.")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    return StoredUpload(
        storage_key=storage_key,
        path=destination,
        filename=display_name,
        mime_type=allowed.mime_type,
        size_bytes=total,
    )


__all__ = [
    "ALLOWED_TYPES",
    "AllowedType",
    "StoredUpload",
    "UploadRejected",
    "extension_of",
    "resolve_type",
    "safe_display_filename",
    "storage_path_for",
    "stream_to_storage",
    "verify_content_matches",
]

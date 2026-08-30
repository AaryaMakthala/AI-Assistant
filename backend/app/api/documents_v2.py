"""Document upload, status and lifecycle endpoints (Phase 3).

The canonical Phase 3 architecture (CLAUDE.md section 9 — "Document Upload &
Synchronous Ingestion"):

* The upload handler validates the file, computes a SHA-256 checksum, and for an
  **OWNER** runs extraction → chunking → embedding inline before persisting the
  document **READY together with its chunks in one transaction**. The RLS policy
  ``document_chunks_write`` (migration 0008) only permits chunk writes for a READY
  document, evaluated with same-transaction visibility — so READY is set before the
  chunk inserts, never after.
* A **MEMBER** upload stores the document row as **PENDING with no chunks**: the
  RLS policies structurally forbid a member from writing chunks or flipping a
  document to READY.
* Phase 4 adds the owner-side approval lifecycle (CLAUDE.md section 9): an OWNER
  approves a PENDING document (ingest inline, then READY + chunks in one
  transaction) or rejects it (REJECTED, never ingested). The transition guard is
  the document's PENDING status — enforced server-side and re-checked inside the
  transaction (``WHERE status = 'PENDING'``) so a concurrent double-approve cannot
  duplicate chunks.
* Any ingestion failure leaves the document **FAILED** with a safe error message —
  never READY with zero chunks, never PENDING with chunks (section 7 invariant).

Raw bytes live in ``documents.file_data`` (BYTEA) — the database is the source of
truth (CLAUDE.md section 6). No filename ever becomes a filesystem path; the
display name is sanitized and the bytes are stored under the row's UUID.

Workspace isolation: every query filters on ``workspace_id`` explicitly on top of
RLS, which is also workspace-scoped (CLAUDE.md section 4 — the frontend is never
trusted with the boundary).
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, func, insert, select, update

from app.api.workspace_deps import assert_workspace_role
from app.config import get_settings
from app.db.models import Document, DocumentChunk
from app.ingestion.pipeline import IngestionError, generate_document_description, prepare_document
from app.retrieval.routing_cache import invalidate_workspace_cache
from app.retrieval.workspace_knowledge import invalidate_workspace_knowledge
from app.security.auth import CurrentPrincipal
from app.security.rate_limit import UPLOAD_RATE_LIMIT, limiter
from app.security.rls import tenant_session
from app.security.uploads import UploadRejected, resolve_type, verify_content_matches

router = APIRouter(prefix="/documents", tags=["documents"])

#: How many leading bytes are content-sniffed to confirm a file is what it claims.
_SNIFF_BYTES = 64 * 1024


class DocumentResponse(BaseModel):
    """Canonical document shape (CLAUDE.md section 7)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    workspace_id: uuid.UUID
    uploaded_by: uuid.UUID
    filename: str
    mime_type: str
    file_size: int
    checksum: str
    status: str
    error_message: str | None = None
    description: str | None = None
    approved_at: datetime | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class UploadAcceptedResponse(BaseModel):
    document: DocumentResponse
    #: Number of chunks indexed when the document was ingested inline (READY), or
    #: None for a PENDING member upload. Lets the client show the result without
    #: a follow-up poll.
    chunk_count: int | None = Field(default=None, description="Chunks stored for a READY document.")


def _read_upload(request: Request, file: UploadFile) -> bytes:
    """Read an upload's bytes, rejecting anything over the configured cap.

    The cap is checked from Content-Length *before* the body is read (CLAUDE.md
    section 6), then re-checked on the actual byte count because a client can lie
    in a header.
    """
    settings = get_settings()
    limit = settings.max_upload_size_mb * 1024 * 1024

    try:
        declared = int(request.headers.get("content-length", "0") or "0")
    except ValueError:
        declared = 0
    if declared > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum size of {settings.max_upload_size_mb} MB.",
        )

    data = file.file.read(limit + 1)  # one extra byte proves oversize without a second read
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the maximum size of {settings.max_upload_size_mb} MB.",
        )
    return data


def _validate_upload(filename: str, data: bytes) -> str:
    """Sanitize the name, verify the type allowlist and sniff the content.

    Returns the canonical MIME type from the allowlist — never the client's claim.
    Raises 400/415 HTTP errors for anything that fails.
    """
    display_name = filename.strip()
    if not display_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required.",
        )
    try:
        allowed = resolve_type(display_name)
        verify_content_matches(allowed, data[:_SNIFF_BYTES])
    except UploadRejected as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    return allowed.mime_type


async def _ensure_unique_in_workspace(session, workspace_id: uuid.UUID, checksum: str) -> None:
    existing = (
        await session.execute(
            select(Document.id).where(
                Document.workspace_id == workspace_id,
                Document.checksum == checksum,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A document with identical content already exists in this workspace.",
        )


def _invalidate_workspace_caches(workspace_id: uuid.UUID) -> None:
    """Invalidate routing cache and workspace knowledge after a document change.

    Call after any successful document state change (upload, approve, reject,
    delete) so the LLM router and knowledge context reflect the new reality.
    """
    invalidate_workspace_cache(workspace_id)
    invalidate_workspace_knowledge(workspace_id)


@router.post(
    "",
    response_model=UploadAcceptedResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document (owner: ingested immediately; member: pending)",
)
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,  # noqa: ARG001 — required by slowapi's decorator
    principal: CurrentPrincipal,
    file: Annotated[UploadFile, File(description="PDF, DOCX or CSV")],
    description: Annotated[str | None, Form(description="Optional description for the document")] = None,
) -> UploadAcceptedResponse:
    """Validate, store, and ingest a document for the principal's workspace.

    The workspace is the caller's default workspace from the verified JWT claim
    (Phase 2 architecture), and membership + role are resolved from the canonical
    ``members`` table at request time — never from the token (CLAUDE.md section 4).
    """
    workspace_id = principal.workspace_id
    role = await assert_workspace_role(workspace_id, principal)

    data = _read_upload(request, file)
    mime_type = _validate_upload(file.filename or "", data)
    checksum = hashlib.sha256(data).hexdigest()

    is_owner = role == "OWNER"

    user_desc = description.strip() if description else None
    if not is_owner:
        # Member upload: store the row PENDING, nothing more. No extraction, no
        # chunks — approval (Phase 4) is what makes a member's upload searchable,
        # and RLS structurally forbids this request from doing it itself.
        # If a description was provided, store it with the document.
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
            await _ensure_unique_in_workspace(session, workspace_id, checksum)
            row = (
                await session.execute(
                    insert(Document)
                    .values(
                        workspace_id=workspace_id,
                        uploaded_by=principal.user_id,
                        filename=file.filename or "unnamed",
                        mime_type=mime_type,
                        file_size=len(data),
                        checksum=checksum,
                        file_data=data,
                        status="PENDING",
                        description=user_desc,
                    )
                    .returning(Document)
                )
            ).scalar_one()
        logger.info(
            "Stored pending upload {file} for workspace {ws} (uploaded by {user})",
            file=row.filename,
            ws=workspace_id,
            user=principal.user_id,
        )
        return UploadAcceptedResponse(document=DocumentResponse.model_validate(row))

    # Owner upload: ingest inline. The CPU-bound extraction/embedding runs in a
    # worker thread so the event loop keeps serving other requests.
    try:
        prepared = await asyncio.to_thread(
            prepare_document,
            data,
            mime_type=mime_type,
            filename=file.filename or "unnamed",
        )
    except IngestionError as exc:
        # A document that cannot be read is a permanent failure: the row is stored
        # FAILED with a user-safe reason, and no chunks exist (section 7 invariant).
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
            await _ensure_unique_in_workspace(session, workspace_id, checksum)
            row = (
                await session.execute(
                    insert(Document)
                    .values(
                        workspace_id=workspace_id,
                        uploaded_by=principal.user_id,
                        filename=file.filename or "unnamed",
                        mime_type=mime_type,
                        file_size=len(data),
                        checksum=checksum,
                        file_data=data,
                        status="FAILED",
                        error_message=str(exc),
                    )
                    .returning(Document)
                )
            ).scalar_one()
        logger.warning(
            "Owner upload {file} failed ingestion for workspace {ws}: {reason}",
            file=row.filename,
            ws=workspace_id,
            reason=str(exc),
        )
        return UploadAcceptedResponse(document=DocumentResponse.model_validate(row))

    # Success: the document row and its chunks are written atomically, with the
    # status READY set before the chunk inserts (document_chunks_write policy
    # requires a READY document, same-transaction visibility).
    # If the user provided a description at upload time, store it directly
    # and skip auto-generation.
    user_desc = description.strip() if description else None
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        await _ensure_unique_in_workspace(session, workspace_id, checksum)
        row = (
            await session.execute(
                insert(Document)
                .values(
                    workspace_id=workspace_id,
                    uploaded_by=principal.user_id,
                    filename=file.filename or "unnamed",
                    mime_type=mime_type,
                    file_size=len(data),
                    checksum=checksum,
                    file_data=data,
                    status="READY",
                    description=user_desc,
                )
                .returning(Document)
            )
        ).scalar_one()

        await session.execute(
            insert(DocumentChunk),
            [
                {
                    "document_id": row.id,
                    "workspace_id": workspace_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_metadata": chunk.chunk_metadata,
                }
                for chunk in prepared.chunks
            ],
        )

    logger.info(
        "Ingested owner upload {file}: {n} chunks for workspace {ws}",
        file=row.filename,
        n=len(prepared.chunks),
        ws=workspace_id,
    )

    # Generate description asynchronously — best-effort, never blocks ingestion.
    # Skip if the user provided a description at upload time.
    if not user_desc:
        auto_description = await generate_document_description(
            prepared.chunks,
            filename=row.filename,
        )
        if auto_description:
            async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == row.id)
                    .values(description=auto_description)
                )
            logger.debug(
                "Generated description for {file}", file=row.filename,
        )

    _invalidate_workspace_caches(workspace_id)
    return UploadAcceptedResponse(
        document=DocumentResponse.model_validate(row),
        chunk_count=len(prepared.chunks),
    )


@router.get("", response_model=DocumentListResponse, summary="List workspace documents")
async def list_documents(
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: str | None = Query(None, alias="status"),
) -> DocumentListResponse:
    """List documents in the caller's workspace, newest first.

    RLS scopes the rows; the explicit ``workspace_id`` filter is the application's
    own copy of the rule (CLAUDE.md section 4) and can never widen what RLS allows.
    """
    workspace_id = principal.workspace_id
    await assert_workspace_role(workspace_id, principal)

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        stmt = (
            select(Document)
            .where(Document.workspace_id == workspace_id)
            .order_by(Document.created_at.desc())
        )
        count_stmt = select(func.count()).select_from(Document).where(
            Document.workspace_id == workspace_id
        )
        if status_filter:
            stmt = stmt.where(Document.status == status_filter)
            count_stmt = count_stmt.where(Document.status == status_filter)

        rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars()
        documents = [DocumentResponse.model_validate(row) for row in rows]
        total = (await session.execute(count_stmt)).scalar_one()

    return DocumentListResponse(documents=documents, total=total)


@router.get("/{document_id}", response_model=DocumentResponse, summary="Document detail")
async def get_document(principal: CurrentPrincipal, document_id: uuid.UUID) -> DocumentResponse:
    workspace_id = principal.workspace_id
    await assert_workspace_role(workspace_id, principal)

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        row = (
            await session.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                )
            )
        ).scalar_one_or_none()

    # RLS already scoped the query; "not visible" and "does not exist" are the same
    # 404, which is what avoids confirming another workspace's document IDs.
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse.model_validate(row)


class ApproveDocumentRequest(BaseModel):
    """Optional fields when approving a pending document."""
    model_config = ConfigDict(extra="forbid")
    description: str | None = Field(default=None, description="Optional description for the document")


@router.post(
    "/{document_id}/approve",
    response_model=UploadAcceptedResponse,
    summary="Approve a pending document (owner only)",
)
async def approve_document(
    principal: CurrentPrincipal,
    document_id: uuid.UUID,
    payload: ApproveDocumentRequest | None = None,
) -> UploadAcceptedResponse:
    """Approve a member's PENDING upload: ingest inline and flip to READY.

    Owner-only, enforced server-side (CLAUDE.md section 4 — role is resolved from
    the ``members`` table, never from the token). Only a PENDING document can be
    approved; READY/REJECTED/FAILED are terminal and return 409.

    Ingestion reuses the exact Phase 3 pipeline (:func:`prepare_document`), so an
    approved member upload produces chunks identical in shape to an owner upload.
    The state flip and the chunk inserts happen in ONE transaction, with the
    status set READY *before* the chunk inserts — the ``document_chunks_write``
    policy (migration 0008) only permits chunk writes for a READY document with
    same-transaction visibility. If ingestion fails, the row goes to FAILED with a
    user-safe ``error_message`` and zero chunks (section 7 invariant).

    Idempotency: the transition is guarded by ``WHERE status = 'PENDING'`` inside
    the transaction, so a concurrent or repeated approve cannot double-ingest —
    the second caller gets 409 and no chunks are duplicated.
    """
    workspace_id = principal.workspace_id
    await assert_workspace_role(workspace_id, principal, "OWNER")

    # Resolve the document (workspace-scoped; RLS + explicit filter). 404 keeps
    # cross-workspace IDs non-enumerable, matching the other endpoints.
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        row = (
            await session.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                )
            )
        ).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    if row.status != "PENDING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only pending documents can be approved.",
        )

    # Extract → chunk → embed, off the event loop, before any write happens. A
    # failure here means the row becomes FAILED — never READY with zero chunks.
    try:
        prepared = await asyncio.to_thread(
            prepare_document,
            row.file_data,
            mime_type=row.mime_type,
            filename=row.filename,
        )
    except IngestionError as exc:
        async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
            updated = (
                await session.execute(
                    update(Document)
                    .where(
                        Document.id == document_id,
                        Document.workspace_id == workspace_id,
                        Document.status == "PENDING",
                    )
                    .values(status="FAILED", error_message=str(exc))
                    .returning(Document)
                )
            ).scalar_one_or_none()
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is no longer pending.",
            ) from None
        logger.warning(
            "Approval of document {doc} failed ingestion for workspace {ws}: {reason}",
            doc=document_id,
            ws=workspace_id,
            reason=str(exc),
        )
        return UploadAcceptedResponse(document=DocumentResponse.model_validate(updated))

    # Success: flip PENDING → READY and insert chunks in one transaction. The
    # guarded UPDATE doubles as the idempotency check: if another request already
    # transitioned this document, no row is returned and nothing is written.
    # If the owner provided a description, store it and skip auto-generation.
    approve_desc = payload.description.strip() if payload and payload.description else None
    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        update_values: dict[str, Any] = {
            "status": "READY",
            "error_message": None,
            "approved_at": func.now(),
        }
        if approve_desc is not None:
            update_values["description"] = approve_desc
        updated = (
            await session.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                    Document.status == "PENDING",
                )
                .values(**update_values)
                .returning(Document)
            )
        ).scalar_one_or_none()
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is no longer pending.",
            )

        await session.execute(
            insert(DocumentChunk),
            [
                {
                    "document_id": updated.id,
                    "workspace_id": workspace_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "chunk_metadata": chunk.chunk_metadata,
                }
                for chunk in prepared.chunks
            ],
        )

    logger.info(
        "Approved document {doc} for workspace {ws}: {n} chunks ingested",
        doc=document_id,
        n=len(prepared.chunks),
        ws=workspace_id,
    )

    # Generate description asynchronously — best-effort, never blocks approval.
    # Skip if the owner provided a description at approval time.
    if not approve_desc:
        auto_description = await generate_document_description(
            prepared.chunks,
            filename=updated.filename,
        )
        if auto_description:
            async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == document_id)
                    .values(description=auto_description)
                )
            logger.debug(
                "Generated description for approved doc {doc}", doc=document_id,
            )

    _invalidate_workspace_caches(workspace_id)
    return UploadAcceptedResponse(
        document=DocumentResponse.model_validate(updated),
        chunk_count=len(prepared.chunks),
    )


@router.post(
    "/{document_id}/reject",
    response_model=DocumentResponse,
    summary="Reject a pending document (owner only)",
)
async def reject_document(
    principal: CurrentPrincipal,
    document_id: uuid.UUID,
) -> DocumentResponse:
    """Reject a member's PENDING upload: REJECTED, never ingested.

    Owner-only, enforced server-side. Only PENDING can be rejected — a rejected
    document is permanent (CLAUDE.md section 5) and structurally has zero chunks:
    rejection never runs extraction, and RLS forbids chunk writes for anything
    other than a READY document.

    Idempotent: the guarded ``WHERE status = 'PENDING'`` UPDATE means a second
    reject is a 409, never a corrupted state.
    """
    workspace_id = principal.workspace_id
    await assert_workspace_role(workspace_id, principal, "OWNER")

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        row = (
            await session.execute(
                select(Document).where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        if row.status != "PENDING":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only pending documents can be rejected.",
            )

        updated = (
            await session.execute(
                update(Document)
                .where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                    Document.status == "PENDING",
                )
                .values(status="REJECTED")
                .returning(Document)
            )
        ).scalar_one_or_none()
        if updated is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Document is no longer pending.",
            )

    logger.info(
        "Rejected document {doc} for workspace {ws} at the request of user {user}",
        doc=document_id,
        ws=workspace_id,
        user=principal.user_id,
    )
    _invalidate_workspace_caches(workspace_id)
    return DocumentResponse.model_validate(updated)


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(principal: CurrentPrincipal, document_id: uuid.UUID) -> None:
    """Delete a document.

    Members may delete only their own uploads; owners may delete any document in
    the workspace (CLAUDE.md section 4). Chunks go via the ON DELETE CASCADE from
    ``document_chunks.document_id`` — no orphaned chunk can survive its parent.
    """
    workspace_id = principal.workspace_id
    role = await assert_workspace_role(workspace_id, principal)

    async with tenant_session(workspace_id=workspace_id, user_id=principal.user_id) as session:
        row = (
            await session.execute(
                select(Document.uploaded_by).where(
                    Document.id == document_id,
                    Document.workspace_id == workspace_id,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        if role != "OWNER" and row != principal.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete documents you uploaded.",
            )

        # Explicit chunk delete plus the FK cascade: the cascade is the guarantee,
        # this makes the intent visible and survives a future FK change.
        await session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.workspace_id == workspace_id,
            )
        )
        await session.execute(
            delete(Document).where(
                Document.id == document_id,
                Document.workspace_id == workspace_id,
            )
        )

    logger.info(
        "Deleted document {doc} for workspace {ws} at the request of user {user}",
        doc=document_id,
        ws=workspace_id,
        user=principal.user_id,
    )
    _invalidate_workspace_caches(workspace_id)


__all__ = ["router"]

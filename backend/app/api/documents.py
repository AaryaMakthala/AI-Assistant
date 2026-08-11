"""Document upload, status and lifecycle endpoints.

The upload handler does three cheap things — validate, store, enqueue — and returns. It
never parses a document: that happens in a Celery worker, so a malicious file cannot hang
the API process (CLAUDE.md 4.2).

Deletion is the other side of that contract. An uploaded document owns four resources —
a row, its chunks and their vectors, a dead-letter trail, and the bytes on disk — and
Phase 10 requires all four to go. See :func:`delete_document` for the ordering, which is
chosen so a crash midway leaks disk rather than leaving a row pointing at nothing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    Depends,
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

from app.api.dependencies import require_role
from app.api.workspace_deps import assert_workspace_role
from app.db.legacy_models import TERMINAL_DOCUMENT_STATUSES, Document, DocumentChunk
from app.security.auth import CurrentPrincipal, Principal
from app.security.rate_limit import UPLOAD_RATE_LIMIT, limiter
from app.security.rls import tenant_session
from app.security.uploads import UploadRejected, storage_path_for, stream_to_storage

router = APIRouter(prefix="/documents", tags=["documents"])

#: Accepted `visibility` values, as a type so FastAPI rejects anything else at the edge.
DocumentVisibility = Literal["org", "personal"]
#: `all` is the default listing: everything RLS lets the caller see, unsplit.
DocumentScope = Literal["org", "personal", "all"]


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: str
    page_count: int | None = None
    chunk_count: int | None = None
    word_count: int | None = None
    error_message: str | None = None
    #: Owner and organization, so a client never has to infer either from its own session.
    org_id: uuid.UUID
    #: Null for documents uploaded before workspaces existed, or outside one.
    workspace_id: uuid.UUID | None = None
    uploaded_by: uuid.UUID
    #: 'org' — readable across the organization. 'personal' — readable by its uploader and
    #: by owners/admins. Returned so the client can split the library without guessing.
    visibility: str
    created_at: datetime
    updated_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


class DocumentStatusResponse(BaseModel):
    """The polling view: everything needed to render progress, and nothing else.

    Separate from :class:`DocumentResponse` because it is fetched every couple of seconds
    per in-flight document. It carries derived booleans so the client never has to
    reimplement which statuses are terminal — a rule that would then need changing in two
    places.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    filename: str
    #: Whether the bytes are stored. Always true here: a document row only exists after a
    #: successful upload, so this is a constant the client can render without special-casing.
    upload_status: str = "complete"
    processing_status: str
    #: True once ingestion has reached `ready` or `failed` and will not change on its own.
    is_terminal: bool
    #: True when chunks exist and are searchable.
    is_indexed: bool
    chunk_count: int | None = None
    page_count: int | None = None
    word_count: int | None = None
    retry_count: int
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None


class UploadAcceptedResponse(BaseModel):
    document: DocumentResponse
    task_id: str | None = Field(
        default=None, description="Celery task tracking ingestion, if it was queued."
    )


class VisibilityUpdate(BaseModel):
    """Body of a visibility change. A model rather than a bare string so the allowed values
    are validated by Pydantic before any query runs (CLAUDE.md 4.7)."""

    visibility: DocumentVisibility


#: Roles permitted to publish organization-wide documents and to administer everyone's.
_ADMIN_ROLES = ("owner", "admin")


def _is_admin(principal: CurrentPrincipal) -> bool:
    return principal.role in _ADMIN_ROLES


@router.post(
    "",
    response_model=UploadAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
)
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,  # noqa: ARG001 — required by slowapi's decorator
    principal: CurrentPrincipal,
    file: Annotated[UploadFile, File(description="pdf, docx, csv, xlsx or txt")],
    workspace_id: Annotated[uuid.UUID | None, Form()] = None,
    visibility: Annotated[DocumentVisibility, Form()] = "personal",
) -> UploadAcceptedResponse:
    """Validate and store an upload, then queue ingestion.

    Returns 202: the document exists and is queued, but has no chunks yet. The client
    polls the status endpoint rather than waiting on the response.

    Defaults to `personal`: an upload whose intended audience was not stated should reach
    the smallest one. Publishing org-wide is an owner/admin action and is refused outright
    rather than quietly downgraded, so the uploader learns their file did not go where
    they expected (CLAUDE.md 4.6).
    """
    # Checked before the bytes are read: a caller who may not write here should not be
    # able to spend the server's disk and time finding that out.
    if visibility == "org" and not _is_admin(principal):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only an owner or admin can publish a document to the whole organization.",
        )

    if workspace_id is not None:
        await assert_workspace_role(workspace_id, principal, "owner", "admin", "editor")

    try:
        stored = await stream_to_storage(file, filename=file.filename or "")
    except UploadRejected as exc:
        # A rejected upload is a client error worth surfacing precisely — the user needs
        # to know whether the file was too big or simply the wrong type.
        logger.info(
            "Upload rejected for org {org}: {reason}", org=principal.org_id, reason=exc.message
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        row = (
            await session.execute(
                insert(Document)
                .values(
                    org_id=principal.org_id,
                    uploaded_by=principal.user_id,
                    filename=stored.filename,
                    storage_key=stored.storage_key,
                    mime_type=stored.mime_type,
                    size_bytes=stored.size_bytes,
                    status="pending",
                    workspace_id=workspace_id,
                    visibility=visibility,
                )
                .returning(Document)
            )
        ).scalar_one()

    document = DocumentResponse.model_validate(row)

    # Imported here so the API process never pays for importing the worker's dependencies.
    from app.workers.ingestion import ingest_document

    task_id: str | None = None
    try:
        task = ingest_document.delay(str(document.id), str(principal.org_id))
        task_id = task.id
    except Exception as exc:
        # The broker being down must not lose the upload: the row stays `pending`, the
        # reaper re-queues it once the broker returns, and the client is not told the file
        # was rejected when in fact it is stored and waiting.
        logger.opt(exception=exc).error(
            "Could not queue ingestion for document {doc}", doc=document.id
        )

    if task_id is not None:
        # Recorded so a delete can revoke work that is still queued. Best-effort: losing
        # this write costs a revoke, not correctness, so it must not fail the upload.
        try:
            async with tenant_session(
                org_id=principal.org_id, user_id=principal.user_id, role=principal.role
            ) as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == document.id, Document.org_id == principal.org_id)
                    .values(task_id=task_id)
                )
        except Exception as exc:
            logger.opt(exception=exc).warning(
                "Could not record task id for document {doc}", doc=document.id
            )

    return UploadAcceptedResponse(document=document, task_id=task_id)


@router.get("", response_model=DocumentListResponse, summary="List this org's documents")
async def list_documents(
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    workspace_id: uuid.UUID | None = None,
    scope: DocumentScope = "all",
) -> DocumentListResponse:
    """Documents visible to the caller's organization, newest first.

    RLS scopes the rows; no `org_id` filter appears in the query because one in
    application code would be the *second* place that rule lives and could drift from the
    policy that actually enforces it.

    `workspace_id` narrows the list further, and requires membership of that workspace —
    otherwise the filter would be a way to read a workspace one has not joined.

    `scope` splits what RLS already returned into the two sections the library renders. It
    narrows and never widens: `personal` means the caller's own uploads, so an owner
    browsing "My Docs" sees their own files rather than everyone's. Requesting a scope can
    therefore never reveal a row that omitting it would have hidden.
    """
    if workspace_id is not None:
        await assert_workspace_role(workspace_id, principal)

    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        stmt = select(Document).order_by(Document.created_at.desc())
        count_stmt = select(func.count()).select_from(Document)
        if workspace_id is not None:
            stmt = stmt.where(Document.workspace_id == workspace_id)
            count_stmt = count_stmt.where(Document.workspace_id == workspace_id)
        if scope == "org":
            stmt = stmt.where(Document.visibility == "org")
            count_stmt = count_stmt.where(Document.visibility == "org")
        elif scope == "personal":
            personal = (Document.visibility == "personal") & (
                Document.uploaded_by == principal.user_id
            )
            stmt = stmt.where(personal)
            count_stmt = count_stmt.where(personal)

        rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars()
        documents = [DocumentResponse.model_validate(row) for row in rows]
        # Counted under the same policy, so it can never report rows the caller cannot see.
        total = (await session.execute(count_stmt)).scalar_one()
    return DocumentListResponse(documents=documents, total=total)


@router.get("/{document_id}", response_model=DocumentResponse, summary="Document detail")
async def get_document(principal: CurrentPrincipal, document_id: uuid.UUID) -> DocumentResponse:
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        row = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()

    # RLS already filtered by org, so "not visible" and "does not exist" are the same
    # case here — and 404 for both is what avoids confirming another org's document IDs.
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return DocumentResponse.model_validate(row)


@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Ingestion progress for one document",
)
async def get_document_status(
    principal: CurrentPrincipal, document_id: uuid.UUID
) -> DocumentStatusResponse:
    """The polling endpoint. Cheap by design — one indexed lookup, no joins."""
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        row = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return DocumentStatusResponse(
        id=row.id,
        org_id=row.org_id,
        filename=row.filename,
        processing_status=row.status,
        is_terminal=row.status in TERMINAL_DOCUMENT_STATUSES,
        # Chunks are what retrieval searches, so a `ready` document with none of them is
        # not actually queryable and must not claim to be.
        is_indexed=row.status == "ready" and bool(row.chunk_count),
        chunk_count=row.chunk_count,
        page_count=row.page_count,
        word_count=row.word_count,
        retry_count=row.retry_count,
        error_message=row.error_message,
        created_at=row.created_at,
        updated_at=row.updated_at,
        processing_started_at=row.processing_started_at,
        processing_completed_at=row.processing_completed_at,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and everything derived from it",
)
async def delete_document(principal: CurrentPrincipal, document_id: uuid.UUID) -> None:
    """Remove a document, its chunks and vectors, its failure trail, and its bytes.

    **Ordering.** The row is deleted first, inside a transaction that cascades to chunks
    and dead-letter rows, and only then are the bytes unlinked. That order is deliberate:
    a crash between the two leaks an orphaned file, which is recoverable and invisible to
    the user, whereas the reverse order would leave a row and a set of vectors pointing at
    a file that no longer exists — a document that appears in the library, appears
    searchable, and cannot be read.

    Postgres does the cascade, not the ORM. `document_chunks` and `ingestion_failures`
    both carry `ON DELETE CASCADE` on a composite key including `org_id`, so the vectors
    cannot survive their document and cannot be removed by anyone else's request.

    Any queued ingestion is revoked first, so a worker cannot resurrect chunks for a
    document that is about to stop existing.

    A member may read an org-wide document but may not delete one they did not upload;
    owners and admins may delete anything. The `documents_delete` policy from migration
    0007 enforces the same rule, so a mistake here narrows what the endpoint reaches
    rather than what the database returns.
    """
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        # SELECT then DELETE in one transaction: the storage key is needed after the row is
        # gone, and RLS constrains both statements, so this cannot read another org's key.
        row = (
            await session.execute(
                select(Document.storage_key, Document.task_id, Document.uploaded_by).where(
                    Document.id == document_id
                )
            )
        ).first()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

        storage_key, task_id, uploaded_by = row
        if uploaded_by != principal.user_id and not _is_admin(principal):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only delete documents you uploaded.",
            )
        _revoke_task(task_id, document_id)

        # Chunks explicitly as well as by cascade. The cascade is the guarantee; this is
        # the one that runs when a future migration changes the FK and nobody notices.
        await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        deleted = (
            await session.execute(delete(Document).where(Document.id == document_id))
        ).rowcount

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    _unlink_stored_file(storage_key, document_id)
    logger.info(
        "Deleted document {doc} for org {org} at the request of user {user}",
        doc=document_id,
        org=principal.org_id,
        user=principal.user_id,
    )


@router.patch(
    "/{document_id}/visibility",
    response_model=DocumentResponse,
    summary="Publish a document org-wide, or return it to personal (owners and admins)",
)
async def update_document_visibility(
    principal: Annotated[Principal, Depends(require_role(*_ADMIN_ROLES))],
    document_id: uuid.UUID,
    body: VisibilityUpdate,
) -> DocumentResponse:
    """Change who can read a document.

    Restricted to owners and admins, which is what makes "promote to company doc" possible
    over a member's personal upload — the same oversight the select policy grants them
    (CLAUDE.md 4.6). Gated twice: `require_role` rejects a member before any query runs,
    and `documents_update` constrains the row regardless.

    Demotion back to `personal` is the same operation in reverse, and is deliberately
    allowed: publishing by mistake must be undoable. The document keeps its original
    `uploaded_by`, so promoting a colleague's file never reassigns authorship.
    """
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        row = (
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(visibility=body.visibility)
                .returning(Document)
            )
        ).scalar_one_or_none()

    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    logger.info(
        "Set document {doc} visibility to {vis} for org {org} at the request of user {user}",
        doc=document_id,
        vis=body.visibility,
        org=principal.org_id,
        user=principal.user_id,
    )
    return DocumentResponse.model_validate(row)


def _revoke_task(task_id: str | None, document_id: uuid.UUID) -> None:
    """Best-effort revocation of queued ingestion for a document being deleted.

    Failure is tolerated and logged: a worker that runs anyway finds no document row —
    `_ingest` returns `missing` for exactly this case — and the delete is still correct.
    """
    if not task_id:
        return
    try:
        from app.workers.celery_app import celery_app

        celery_app.control.revoke(task_id, terminate=False)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            "Could not revoke ingestion task for document {doc}", doc=document_id
        )


def _unlink_stored_file(storage_key: uuid.UUID, document_id: uuid.UUID) -> None:
    """Remove the stored bytes. The path is derived from a generated UUID, never a name.

    A failure here leaves an orphaned file and is logged rather than raised: the database
    is already consistent by this point, and turning a completed delete into a 500 would
    invite a retry that finds nothing to delete and 404s.
    """
    try:
        storage_path_for(storage_key).unlink(missing_ok=True)
    except OSError as exc:
        logger.opt(exception=exc).error(
            "Deleted document {doc} but could not remove its stored file", doc=document_id
        )


@router.post(
    "/{document_id}/reprocess",
    response_model=UploadAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run ingestion for a document",
)
async def reprocess_document(
    principal: CurrentPrincipal, document_id: uuid.UUID
) -> UploadAcceptedResponse:
    """Queue ingestion again for a document that failed or was left behind.

    The stored bytes are still on disk, so this re-runs extraction rather than asking the
    user to upload the file a second time. Ingestion is idempotent — it replaces the
    document's chunks — so this is safe to call on a document that partially succeeded.

    Refused while a job is already in flight: a second worker on the same document would
    race the first one's chunk delete-and-insert.
    """
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        row = (
            await session.execute(select(Document).where(Document.id == document_id))
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        if row.status == "processing":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This document is already being processed.",
            )
        await session.execute(
            update(Document)
            .where(Document.id == document_id, Document.org_id == principal.org_id)
            .values(status="pending", error_message=None, retry_count=0)
        )
        document = DocumentResponse.model_validate(row)

    from app.workers.ingestion import ingest_document

    try:
        task = ingest_document.delay(str(document_id), str(principal.org_id))
    except Exception as exc:
        logger.opt(exception=exc).error(
            "Could not queue reprocessing for document {doc}", doc=document_id
        )
        # Left `pending` rather than failed: the row is in the state the reaper re-queues,
        # so the request being unable to reach the broker only delays the work.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Processing could not be queued right now. It will be retried shortly.",
        ) from exc

    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id, Document.org_id == principal.org_id)
            .values(task_id=task.id)
        )

    return UploadAcceptedResponse(
        document=document.model_copy(update={"status": "pending", "error_message": None}),
        task_id=task.id,
    )


__all__ = ["router"]

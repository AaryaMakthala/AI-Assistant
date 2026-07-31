"""Document upload and status endpoints.

The handler does three cheap things — validate, store, enqueue — and returns. It never
parses a document: that happens in a Celery worker, so a malicious file cannot hang the
API process (CLAUDE.md 4.2).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import insert, select

from app.db.models import Document
from app.security.auth import CurrentPrincipal
from app.security.rate_limit import UPLOAD_RATE_LIMIT, limiter
from app.security.rls import tenant_session
from app.security.uploads import UploadRejected, stream_to_storage

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: str
    page_count: int | None = None
    error_message: str | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class UploadAcceptedResponse(BaseModel):
    document: DocumentResponse
    task_id: str | None = Field(
        default=None, description="Celery task tracking ingestion, if it was queued."
    )


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
) -> UploadAcceptedResponse:
    """Validate and store an upload, then queue ingestion.

    Returns 202: the document exists and is queued, but has no chunks yet. The client
    polls the status endpoint rather than waiting on the response.
    """
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
                )
                .returning(Document)
            )
        ).one()

    document = DocumentResponse.model_validate(row)

    # Imported here so the API process never pays for importing the worker's dependencies.
    from app.workers.ingestion import ingest_document

    task_id: str | None = None
    try:
        task = ingest_document.delay(str(document.id), str(principal.org_id))
        task_id = task.id
    except Exception as exc:
        # The broker being down must not lose the upload: the row stays `pending` and can
        # be re-queued, rather than the client being told the file was never accepted.
        logger.opt(exception=exc).error(
            "Could not queue ingestion for document {doc}", doc=document.id
        )

    return UploadAcceptedResponse(document=document, task_id=task_id)


@router.get("", response_model=DocumentListResponse, summary="List this org's documents")
async def list_documents(
    principal: CurrentPrincipal,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentListResponse:
    async with tenant_session(
        org_id=principal.org_id, user_id=principal.user_id, role=principal.role
    ) as session:
        rows = (
            await session.execute(
                select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
            )
        ).scalars()
        documents = [DocumentResponse.model_validate(row) for row in rows]
    return DocumentListResponse(documents=documents)


@router.get("/{document_id}", response_model=DocumentResponse, summary="Ingestion status")
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


__all__ = ["router"]

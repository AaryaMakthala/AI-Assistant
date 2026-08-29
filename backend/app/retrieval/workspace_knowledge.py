"""Per-workspace knowledge file for the LLM router.

Builds a structured description of what each workspace contains and
supports, sourced from the existing database tables (documents, members).
This knowledge is injected into the LLM router's system prompt so its
routing decisions reflect real workspace capabilities.

No new DB table is needed — the knowledge is derived from existing tables
and cached in-memory with a TTL.  When a document is added/removed/approved,
the cache is invalidated for that workspace.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from loguru import logger
from sqlalchemy import func, select

from app.db.models import Document, Member


# ---------------------------------------------------------------------------
# Knowledge data structure
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceKnowledge:
    """Structured description of a workspace's contents and capabilities."""

    #: Workspace ID (for cache keying).
    workspace_id: uuid.UUID
    #: Document titles currently indexed (READY status only).
    document_titles: list[str] = field(default_factory=list)
    #: Number of READY documents.
    document_count: int = 0
    #: Number of ACTIVE members.
    member_count: int = 0
    #: Whether the workspace has any documents at all.
    has_documents: bool = False
    #: Timestamp when this knowledge was built (for TTL).
    built_at: float = field(default_factory=time.monotonic)

    def to_prompt_context(self) -> str:
        """Render as a concise string for injection into the LLM router prompt."""
        lines: list[str] = []

        if self.has_documents:
            lines.append(f"Documents indexed ({self.document_count}):")
            # Show up to 20 titles; beyond that, summarize.
            shown = self.document_titles[:20]
            for title in shown:
                lines.append(f"  - {title}")
            if len(self.document_titles) > 20:
                lines.append(f"  ... and {len(self.document_titles) - 20} more")
        else:
            lines.append("No documents currently indexed in this workspace.")

        lines.append(f"Active members: {self.member_count}")
        lines.append(
            "Available metadata queries: document count, document list, "
            "member count, member list, role lookup, admin/owner lookup."
        )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

#: How long a knowledge entry stays valid (seconds).
_KNOWLEDGE_TTL_SECONDS = 300.0  # 5 minutes

_cache: dict[uuid.UUID, WorkspaceKnowledge] = {}


def invalidate_workspace_knowledge(workspace_id: uuid.UUID) -> None:
    """Remove the cached knowledge for a workspace.

    Call this when documents are added, removed, approved, or rejected.
    """
    _cache.pop(workspace_id, None)
    logger.debug("Invalidated workspace knowledge cache for {ws}", ws=workspace_id)


async def get_workspace_knowledge(
    session,
    workspace_id: uuid.UUID,
) -> WorkspaceKnowledge:
    """Build (or return cached) workspace knowledge for the LLM router.

    Parameters
    ----------
    session:
        An async SQLAlchemy session (must not be used outside this call).
    workspace_id:
        The workspace to build knowledge for.
    """
    now = time.monotonic()

    # Check cache.
    cached = _cache.get(workspace_id)
    if cached is not None and (now - cached.built_at) < _KNOWLEDGE_TTL_SECONDS:
        return cached

    # Build from DB.
    try:
        # Document titles (READY only — these are what can be searched).
        doc_rows = (
            await session.execute(
                select(Document.filename).where(
                    Document.workspace_id == workspace_id,
                    Document.status == "READY",
                )
            )
        ).all()
        doc_titles = [row.filename for row in doc_rows]
        doc_count = len(doc_titles)

        # Member count (ACTIVE only).
        member_count = (
            await session.execute(
                select(func.count()).select_from(Member).where(
                    Member.workspace_id == workspace_id,
                    Member.status == "ACTIVE",
                )
            )
        ).scalar_one()

        knowledge = WorkspaceKnowledge(
            workspace_id=workspace_id,
            document_titles=doc_titles,
            document_count=doc_count,
            member_count=member_count,
            has_documents=doc_count > 0,
            built_at=now,
        )

        _cache[workspace_id] = knowledge
        logger.debug(
            "Built workspace knowledge for {ws}: {n_docs} docs, {n_members} members",
            ws=workspace_id,
            n_docs=doc_count,
            n_members=member_count,
        )
        return knowledge

    except Exception as exc:
        # If DB query fails, return a minimal knowledge object so routing
        # still works (the LLM router just won't have workspace context).
        logger.warning(
            "Failed to build workspace knowledge for {ws}: {error}",
            ws=workspace_id,
            error=str(exc)[:200],
        )
        return WorkspaceKnowledge(
            workspace_id=workspace_id,
            has_documents=False,
            built_at=now,
        )


__all__ = [
    "WorkspaceKnowledge",
    "get_workspace_knowledge",
    "invalidate_workspace_knowledge",
]

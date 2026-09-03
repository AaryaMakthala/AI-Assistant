"""Declarative bases for the two model layers.

``Base`` is the canonical metadata: exactly the tables in CLAUDE.md section 7, and the
only metadata Alembic autogenerate should ever see (``alembic/env.py`` targets it).

``LegacyBase`` exists only to keep the retired org-centric models importable while their
consumers are rebuilt in later phases. It is deliberately separate: the two
architectures collide on five table names (workspaces, documents, document_chunks,
chat_sessions, chat_messages), so a shared metadata would raise a duplicate-table error
the moment both sets are imported. Its tables must never enter ``Base.metadata``.
"""

from sqlalchemy import Column, MetaData, Table, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Canonical model metadata — the target schema of CLAUDE.md section 7."""


# ---------------------------------------------------------------------------
# Stub for ``auth.users`` — the Supabase-managed auth schema.
#
# Multiple application tables (workspaces, members, documents, chat_sessions,
# invitations) declare ``ForeignKey("auth.users.id")``.  Supabase owns the
# ``auth`` schema and the ``users`` table within it, so the table is never
# reflected by Alembic autogenerate and is absent from our ORM metadata.
#
# Without a stub, SQLAlchemy cannot resolve the FK target at flush time and
# raises ``NoReferencedTableError``.  Registering a minimal ``Table`` object
# here lets the ORM resolve the reference while keeping Alembic from trying
# to create or drop the table (Alembic only sees tables declared through
# declarative models on ``Base.metadata``, not raw ``Table`` objects — but to
# be safe we set ``extend_existing=True`` and keep the column list minimal).
# ---------------------------------------------------------------------------

Table(
    "users",
    Base.metadata,
    Column("id", UUID(as_uuid=True), primary_key=True),
    schema="auth",
    extend_existing=True,
)


class LegacyBase(DeclarativeBase):
    """Retired org-centric models, isolated from the canonical metadata."""

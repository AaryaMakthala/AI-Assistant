"""Declarative bases for the two model layers.

``Base`` is the canonical metadata: exactly the tables in CLAUDE.md section 7, and the
only metadata Alembic autogenerate should ever see (``alembic/env.py`` targets it).

``LegacyBase`` exists only to keep the retired org-centric models importable while their
consumers are rebuilt in later phases. It is deliberately separate: the two
architectures collide on five table names (workspaces, documents, document_chunks,
chat_sessions, chat_messages), so a shared metadata would raise a duplicate-table error
the moment both sets are imported. Its tables must never enter ``Base.metadata``.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Canonical model metadata — the target schema of CLAUDE.md section 7."""


class LegacyBase(DeclarativeBase):
    """Retired org-centric models, isolated from the canonical metadata."""

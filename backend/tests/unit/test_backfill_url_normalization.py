"""Focused tests for the backfill script's async PostgreSQL URL normalization.

The backfill script builds a SQLAlchemy async engine, but bare ``postgresql://``
URLs select the sync psycopg2 driver (not installed in this project — it uses
asyncpg). These tests pin the normalization and that an engine builds from the
normalized URL without connecting or importing psycopg2.
"""

from sqlalchemy.ext.asyncio import create_async_engine

from scripts.backfill_embeddings import _normalize_async_database_url


def test_normalizes_psycopg2_style_postgresql_url() -> None:
    out = _normalize_async_database_url(
        "postgresql://test:test@localhost:5432/officebrain"
    )
    assert out == "postgresql+asyncpg://test:test@localhost:5432/officebrain"


def test_normalizes_bare_postgres_url() -> None:
    out = _normalize_async_database_url(
        "postgres://test:test@db.example:5432/officebrain"
    )
    assert out == "postgresql+asyncpg://test:test@db.example:5432/officebrain"


def test_leaves_explicit_async_url_unchanged() -> None:
    url = "postgresql+asyncpg://test:test@localhost:5432/officebrain"
    assert _normalize_async_database_url(url) == url


def test_engine_builds_from_normalized_url_without_connecting() -> None:
    """Constructing the async engine must not require psycopg2 or open a connection."""
    engine = create_async_engine(
        _normalize_async_database_url(
            "postgresql://test:test@localhost:5432/officebrain"
        )
    )
    assert engine.url.drivername == "postgresql+asyncpg"
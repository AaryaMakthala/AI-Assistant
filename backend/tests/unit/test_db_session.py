"""Engine/session wiring. No live database required — Phase 2 owns the schema."""

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.base import Base
from app.db.session import dispose_engine, get_engine, get_session_factory


@pytest.fixture(autouse=True)
async def _reset_engine(valid_env: None):
    await dispose_engine()
    yield
    await dispose_engine()


async def test_engine_is_async_and_reused() -> None:
    engine = get_engine()

    assert isinstance(engine, AsyncEngine)
    assert get_engine() is engine


async def test_engine_uses_configured_url() -> None:
    url = get_engine().url

    assert url.drivername == "postgresql+asyncpg"
    assert url.database == "test"


async def test_engine_password_is_masked_in_repr() -> None:
    """A logged engine URL must not reveal the database password."""
    assert "test:test@" not in repr(get_engine().url)


async def test_session_factory_produces_async_sessions() -> None:
    async with get_session_factory()() as session:
        assert session.is_active
        assert session.bind is get_engine()


async def test_dispose_resets_engine() -> None:
    first = get_engine()
    await dispose_engine()

    assert get_engine() is not first


def test_declarative_base_has_metadata() -> None:
    assert Base.metadata is not None

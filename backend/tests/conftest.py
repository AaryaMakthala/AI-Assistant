"""Shared fixtures. Tests must never depend on a real .env being present."""

from collections.abc import Iterator

import pytest

from app.config import Settings, get_settings

_TEST_ENV = {
    "ENVIRONMENT": "development",
    "DEBUG": "false",
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost:5432/test",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_ANON_KEY": "test-anon-key",
    "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    "GEMINI_API_KEY": "test-gemini-key",
    "GROQ_API_KEY": "test-groq-key",
    "REDIS_URL": "redis://localhost:6379/0",
    "JWT_SECRET": "test-jwt-secret-that-is-long-enough-to-pass-validation",
}


@pytest.fixture(autouse=True)
def _ignore_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ignore any developer .env so results don't vary by machine."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in _TEST_ENV:
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def valid_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A complete, valid configuration."""
    for key, value in _TEST_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

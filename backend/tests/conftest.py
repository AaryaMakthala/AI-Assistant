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
    "LLM_PROVIDER": "test-provider",
    "LLM_MODEL": "test-model",
    "LLM_API_KEY": "test-llm-api-key",
    "JWT_SECRET": "test-jwt-secret-that-is-long-enough-to-pass-validation",
}

#: Legacy variables that a developer's shell or .env may still export. They are not
#: part of the Section 13 contract: they must be absent so the tests can prove the new
#: configuration does not need them (and that their presence is inert).
_LEGACY_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
)


@pytest.fixture(autouse=True)
def _ignore_dotenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ignore any developer .env so results don't vary by machine."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key in (*_TEST_ENV, *_LEGACY_ENV_KEYS):
        monkeypatch.delenv(key, raising=False)
    # Clear every remaining config-affecting variable (GITHUB_TOKEN, SENTRY_DSN, ...)
    # so results cannot vary with a developer's shell or user environment. The field's
    # env name may come from a validation_alias (e.g. embedding_dim ← EMBEDDING_DIMENSION)
    # rather than the uppercased field name, so clear the alias too.
    for field_name, field_info in Settings.model_fields.items():
        monkeypatch.delenv(field_name.upper(), raising=False)
        alias = field_info.validation_alias
        if isinstance(alias, str):
            monkeypatch.delenv(alias.upper(), raising=False)
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


@pytest.fixture(autouse=True)
def _isolate_sentry_client() -> Iterator[None]:
    """Leave no Sentry client behind.

    `sentry_sdk.init` installs a *process-global* client, so a test that enables Sentry
    would otherwise keep transmitting into the previous test's captured-event list — and
    a test asserting that reporting is off would see the client the last one left running.

    `dsn=""` rather than `dsn=None`: the SDK falls back to reading `SENTRY_DSN` from the
    environment when the argument is None, which is precisely what these tests set.

    The reference to `init` is captured *before* yielding. Tests that record events
    monkeypatch `sentry_sdk.init` to force a transport in, and that patch is still in
    place during teardown — resetting through it would reinstall an active client and
    defeat the isolation this fixture exists to provide.
    """
    import sentry_sdk

    pristine_init = sentry_sdk.init
    yield
    pristine_init(dsn="")

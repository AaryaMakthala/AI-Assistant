"""Phase 1 acceptance: missing config fails loudly and clearly, not silently."""

import pytest

from app.config import get_settings


def test_missing_env_exits_with_clear_message(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        get_settings()

    assert exc_info.value.code == 1
    stderr = capsys.readouterr().err
    assert "FATAL" in stderr
    assert "DATABASE_URL is required but not set." in stderr
    assert "JWT_SECRET is required but not set." in stderr
    assert ".env.example" in stderr


def test_partial_env_names_only_the_missing_vars(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db")

    with pytest.raises(SystemExit):
        get_settings()

    stderr = capsys.readouterr().err
    assert "DATABASE_URL is required" not in stderr
    assert "GEMINI_API_KEY is required but not set." in stderr


def test_short_jwt_secret_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], valid_env: None
) -> None:
    monkeypatch.setenv("JWT_SECRET", "too-short")

    with pytest.raises(SystemExit):
        get_settings()

    assert "JWT_SECRET" in capsys.readouterr().err


def test_valid_env_loads(valid_env: None) -> None:
    settings = get_settings()

    assert settings.environment == "development"
    assert settings.is_production is False
    assert settings.jwt_secret.get_secret_value().startswith("test-jwt-secret")


def test_secrets_are_not_exposed_in_repr(valid_env: None) -> None:
    """A leaked settings repr in a log or traceback must not reveal key material."""
    settings = get_settings()
    dumped = repr(settings) + str(settings.model_dump())

    assert "test-gemini-key" not in dumped
    assert "test-service-role-key" not in dumped
    assert "test-jwt-secret-that-is-long-enough-to-pass-validation" not in dumped


def test_optional_settings_default_to_none(valid_env: None) -> None:
    settings = get_settings()

    assert settings.sentry_dsn is None
    assert settings.github_token is None


def test_settings_are_cached(valid_env: None) -> None:
    assert get_settings() is get_settings()


def test_unknown_environment_is_rejected(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")

    with pytest.raises(SystemExit):
        get_settings()

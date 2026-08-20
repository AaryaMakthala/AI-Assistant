"""Phase 1A acceptance: Section 13 configuration loads, validates, and never leaks.

The contract under test is CLAUDE.md Section 13: exactly the documented variable set,
with legacy provider-specific variables (GEMINI_API_KEY, GROQ_API_KEY, REDIS_URL)
required by nobody and accepted only for compatibility.
"""

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
    assert "LLM_API_KEY is required but not set." in stderr


def test_short_jwt_secret_is_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], valid_env: None
) -> None:
    monkeypatch.setenv("JWT_SECRET", "too-short")

    with pytest.raises(SystemExit):
        get_settings()

    assert "JWT_SECRET" in capsys.readouterr().err


def test_empty_llm_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """`LLM_API_KEY=` in .env is missing-in-practice and must fail at boot."""
    monkeypatch.setenv("LLM_API_KEY", "")

    with pytest.raises(SystemExit):
        get_settings()


def test_valid_env_loads(valid_env: None) -> None:
    settings = get_settings()

    assert settings.environment == "development"
    assert settings.is_production is False
    assert settings.jwt_secret.get_secret_value().startswith("test-jwt-secret")
    assert settings.llm_provider == "test-provider"
    assert settings.llm_model == "test-model"
    assert settings.llm_api_key.get_secret_value() == "test-llm-api-key"


def test_secrets_are_not_exposed_in_repr(valid_env: None) -> None:
    """A leaked settings repr in a log or traceback must not reveal key material."""
    settings = get_settings()
    dumped = repr(settings) + str(settings.model_dump())

    assert "test-llm-api-key" not in dumped
    assert "test-service-role-key" not in dumped
    assert "test-jwt-secret-that-is-long-enough-to-pass-validation" not in dumped


def test_legacy_provider_variables_are_not_required(valid_env: None) -> None:
    """A Section 13 environment must load without any legacy provider variables."""
    settings = get_settings()

    assert settings.gemini_api_key is None
    assert settings.groq_api_key is None
    assert settings.redis_url is None
    assert settings.llm_base_url is None


def test_legacy_provider_variables_still_accepted(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """Backwards compatibility: an old .env that sets them keeps loading unchanged."""
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "legacy-groq-key")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    settings = get_settings()

    assert settings.gemini_api_key.get_secret_value() == "legacy-gemini-key"
    assert settings.groq_api_key.get_secret_value() == "legacy-groq-key"
    assert str(settings.redis_url) == "redis://localhost:6379/0"


def test_section13_defaults(valid_env: None) -> None:
    settings = get_settings()

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.embedding_dim == 384
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.retrieval_candidate_count == 15
    assert settings.retrieval_final_count == 8
    assert settings.retrieval_relevance_threshold == 0.3
    assert settings.max_upload_size_mb == 10


def test_embedding_dimension_env_var_maps_to_field(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "768")

    assert get_settings().embedding_dim == 768


def test_invalid_embedding_dimension_rejected(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    monkeypatch.setenv("EMBEDDING_DIMENSION", "0")

    with pytest.raises(SystemExit):
        get_settings()


def test_invalid_retrieval_candidate_count_rejected(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_COUNT", "0")

    with pytest.raises(SystemExit):
        get_settings()


def test_candidate_count_below_final_count_rejected(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], valid_env: None
) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_COUNT", "5")
    monkeypatch.setenv("RETRIEVAL_FINAL_COUNT", "8")

    with pytest.raises(SystemExit):
        get_settings()

    assert "RETRIEVAL_CANDIDATE_COUNT must be >= RETRIEVAL_FINAL_COUNT" in capsys.readouterr().err


def test_candidate_count_equal_to_final_count_allowed(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_COUNT", "8")
    monkeypatch.setenv("RETRIEVAL_FINAL_COUNT", "8")

    assert get_settings().retrieval_candidate_count == 8


def test_relevance_threshold_out_of_range_rejected(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    monkeypatch.setenv("RETRIEVAL_RELEVANCE_THRESHOLD", "1.5")

    with pytest.raises(SystemExit):
        get_settings()

    monkeypatch.setenv("RETRIEVAL_RELEVANCE_THRESHOLD", "-0.1")

    with pytest.raises(SystemExit):
        get_settings()


def test_max_upload_size_rejects_zero(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "0")

    with pytest.raises(SystemExit):
        get_settings()


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


# ── CORS origins parsing ────────────────────────────────────────────────


def test_cors_default_is_localhost(valid_env: None) -> None:
    """Without CORS_ALLOW_ORIGINS set, the default covers local dev."""
    settings = get_settings()
    assert settings.cors_allow_origins == ["http://localhost:3000"]


def test_cors_comma_separated_string_parses_correctly(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """The .env.example documents comma-separated origins; the parser must
    accept that format without requiring JSON.
    """
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        "https://app.vercel.app, https://staging.vercel.app",
    )
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.cors_allow_origins == [
        "https://app.vercel.app",
        "https://staging.vercel.app",
    ]


def test_cors_json_array_still_works(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """A JSON array (the Pydantic v2 native format) must also parse."""
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        '["https://a.com","https://b.com"]',
    )
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.cors_allow_origins == ["https://a.com", "https://b.com"]


def test_cors_single_origin(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """A single origin without a comma should work."""
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "https://myapp.vercel.app")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.cors_allow_origins == ["https://myapp.vercel.app"]


def test_cors_empty_string_yields_empty_list(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """An explicitly empty CORS_ALLOW_ORIGINS results in no origins."""
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "")
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.cors_allow_origins == []

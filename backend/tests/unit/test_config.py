"""Phase 1A acceptance: Section 13 configuration loads, validates, and never leaks.

The contract under test: GEMINI_API_KEY / GROQ_API_KEY are the primary way to
configure the LLM.  LLM_PROVIDER / LLM_MODEL / LLM_API_KEY are optional overrides
auto-derived when absent.  JWT_SECRET has a development default and is not required
in production (modern Supabase uses ES256/RS256 via JWKS).
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
    assert ".env.example" in stderr


def test_partial_env_names_only_the_missing_vars(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@localhost:5432/db")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")

    with pytest.raises(SystemExit):
        get_settings()

    stderr = capsys.readouterr().err
    assert "DATABASE_URL is required" not in stderr
    # Without LLM config, the model validator raises an error.
    assert "No LLM provider configured" in stderr


def test_empty_gemini_key_is_rejected(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """GEMINI_API_KEY= in .env is missing-in-practice and must fail at boot."""
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(SystemExit):
        get_settings()


def test_empty_groq_key_is_rejected(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """GROQ_API_KEY= in .env is missing-in-practice and must fail at boot."""
    monkeypatch.setenv("GROQ_API_KEY", "")

    with pytest.raises(SystemExit):
        get_settings()


def test_valid_env_loads(valid_env: None) -> None:
    settings = get_settings()

    assert settings.environment == "development"
    assert settings.is_production is False
    # jwt_secret uses the default dev value when not explicitly set.
    assert settings.jwt_secret.get_secret_value() == (
        "dev-only-jwt-secret-replace-in-production-if-using-hs256"
    )
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


def test_max_upload_size_rejects_zero(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
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


def test_cors_json_array_still_works(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """A JSON array (the Pydantic v2 native format) must also parse."""
    monkeypatch.setenv(
        "CORS_ALLOW_ORIGINS",
        '["https://a.com","https://b.com"]',
    )
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.cors_allow_origins == ["https://a.com", "https://b.com"]


def test_cors_single_origin(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
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


# ── Provider key auto-derivation ────────────────────────────────────────


def test_gemini_key_derives_llm_config(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """GEMINI_API_KEY alone should derive provider, model, base_url, and api_key."""
    # Remove explicit LLM overrides so the auto-derivation kicks in.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "my-gemini-key-1234567890")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-3.6-flash"
    assert settings.llm_api_key.get_secret_value() == "my-gemini-key-1234567890"
    assert settings.llm_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"


def test_groq_key_derives_llm_config(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """GROQ_API_KEY alone should derive provider, model, base_url, and api_key."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "my-groq-key-abcdef123456")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.llm_provider == "groq"
    assert settings.llm_model == "qwen/qwen3.6-27b"
    assert settings.llm_api_key.get_secret_value() == "my-groq-key-abcdef123456"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"


def test_explicit_llm_fields_override_presets(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """Explicit LLM_MODEL should survive when a provider key is also set."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "my-gemini-key-1234567890")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
    get_settings.cache_clear()

    settings = get_settings()

    # Provider derived from key, but model kept from explicit override.
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-2.5-pro"


def test_no_llm_config_at_all_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], valid_env: None
) -> None:
    """No provider key and no explicit LLM config must fail at boot."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_settings.cache_clear()

    with pytest.raises(SystemExit):
        get_settings()

    assert "No LLM provider configured" in capsys.readouterr().err


def test_gemini_key_not_exposed_in_repr(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """GEMINI_API_KEY must not appear in settings repr or dump."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "my-gemini-key-1234567890")
    get_settings.cache_clear()

    settings = get_settings()
    dumped = repr(settings) + str(settings.model_dump())

    assert "my-gemini-key-1234567890" not in dumped


# ── Fallback chain order ──────────────────────────────────────────────


def test_fallback_chain_order_groq_primary(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """When all three provider keys are set, chain order is Groq → OpenRouter → Gemini."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-1234567890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key-1234567890")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-1234567890")
    get_settings.cache_clear()

    settings = get_settings()
    chain = settings.fallback_chain_configs

    assert len(chain) == 3
    assert chain[0]["name"] == "groq"
    assert chain[1]["name"] == "openrouter"
    assert chain[2]["name"] == "gemini"


def test_fallback_chain_groq_only(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """When only GROQ_API_KEY is set, chain has one entry."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-1234567890")
    get_settings.cache_clear()

    settings = get_settings()
    chain = settings.fallback_chain_configs

    assert len(chain) == 1
    assert chain[0]["name"] == "groq"


def test_fallback_chain_groq_openrouter(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """When GROQ_API_KEY and OPENROUTER_API_KEY are set, chain has two entries in order."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-key-1234567890")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key-1234567890")
    get_settings.cache_clear()

    settings = get_settings()
    chain = settings.fallback_chain_configs

    assert len(chain) == 2
    assert chain[0]["name"] == "groq"
    assert chain[1]["name"] == "openrouter"

"""Application settings. All secrets come from .env — never hardcoded (CLAUDE.md 4.1)."""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    Field,
    PostgresDsn,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import EnvSettingsSource

_ENV_EXAMPLE_HINT = (
    "Copy .env.example to .env at the repo root and fill in the values listed above.\n"
    "See CLAUDE.md section 13 for the full variable list."
)

# ---------------------------------------------------------------------------
# Provider presets — the base URL and default model for each supported provider.
# All providers expose OpenAI-compatible chat-completions endpoints.
# Model identifiers are internal only; user-facing output uses generic names
# ("primary", "fallback", "secondary_fallback").
# ---------------------------------------------------------------------------

# User-facing display names for providers — used in API responses and logs.
# Internal model identifiers stay internal; these generic names are shown to users.
# Fallback chain order: Groq (primary) → OpenRouter (fallback) → Gemini (secondary_fallback).
_PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "groq": "primary",
    "openrouter": "fallback",
    "gemini": "secondary_fallback",
}


_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.6-27b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "google/gemini-2.5-flash",
    },
}


class _CorsEnvSettingsSource(EnvSettingsSource):
    """Env source that accepts comma-separated CORS origins from .env files.

    Pydantic v2 ``BaseSettings`` calls ``json.loads()`` on every ``list[str]``
    env value, which breaks the comma-separated format documented in
    ``.env.example``.  This override intercepts that single field before
    JSON decoding and splits on commas instead.
    """

    def prepare_field_value(
        self,
        field_name: str,
        field: Any,
        field_value: Any,
        value_is_complex: bool,
    ) -> Any:
        if field_name == "cors_allow_origins" and isinstance(field_value, str):
            if field_value.startswith("["):
                # Already a JSON array — let the default handler parse it.
                return super().prepare_field_value(
                    field_name, field, field_value, value_is_complex
                )
            # Comma-separated format (the .env.example convention).
            return [
                origin.strip()
                for origin in field_value.split(",")
                if origin.strip()
            ]
        return super().prepare_field_value(
            field_name, field, field_value, value_is_complex
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    database_url: PostgresDsn
    supabase_url: str
    supabase_anon_key: SecretStr
    supabase_service_role_key: SecretStr

    # --- LLM provider keys (direct, no abstraction) ---
    # The application supports Groq, OpenRouter, and Gemini via their direct
    # API keys.  The fallback chain is sequential:
    # Groq (primary) → OpenRouter (fallback) → Gemini (secondary fallback).
    # The generic LLM_PROVIDER/LLM_MODEL/LLM_API_KEY fields below can
    # override the primary provider when set explicitly.
    gemini_api_key: SecretStr | None = Field(default=None, min_length=1)
    groq_api_key: SecretStr | None = Field(default=None, min_length=1)
    openrouter_api_key: SecretStr | None = Field(default=None, min_length=1)

    # --- Per-provider model overrides (optional) ---
    # These override the default model for each provider in the fallback chain.
    # When unset, the provider preset default is used.
    openrouter_model: str | None = None

    # --- Generic LLM fields (optional when a provider key is set) ---
    # These are auto-derived from the provider key via a model validator.
    # They can also be set explicitly to override a preset (e.g. a custom
    # model name or a non-standard base URL).
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None

    # --- Auth ---
    # Modern Supabase signs tokens with asymmetric keys (ES256/RS256) verified
    # via JWKS — no shared secret needed.  jwt_secret is a fallback for legacy
    # HS256 projects and for locally minted test tokens.  A default value is
    # provided so production deployments that use ES256 only do not need to
    # supply this variable at all.
    jwt_secret: SecretStr = SecretStr(
        "dev-only-jwt-secret-replace-in-production-if-using-hs256"
    )

    sentry_dsn: str | None = None
    github_token: SecretStr | None = None

    # --- Observability ---

    #: Fraction of requests sampled for performance tracing. Errors are always captured
    #: regardless of this: it governs transactions only. Defaults off, because tracing every
    #: request on a free Sentry plan exhausts the quota long before an incident happens.
    sentry_traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    #: Version string attached to every event, so a spike can be attributed to a deploy.
    #: Usually the commit SHA, supplied by the platform's build environment.
    sentry_release: str | None = None
    #: Loguru level at or above which a log line becomes a Sentry *event*. Lines below it
    #: still attach as breadcrumbs, which is what gives an event its leading context.
    sentry_event_level: Literal["WARNING", "ERROR", "CRITICAL"] = "ERROR"

    #: Total tokens in one chat turn above which a warning is emitted and reported to Sentry
    #: (CLAUDE.md section 7, cost/quota overrun). 0 disables the check.
    token_usage_alert_threshold: int = Field(default=0, ge=0)

    cors_allow_origins: list[str] = Field(default=["http://localhost:3000"])

    # --- Document ingestion (CLAUDE.md 4.2, Phase 3) ---

    #: Where uploaded bytes land, named by generated UUID — never by user-supplied name.
    upload_dir: Path = Path("var/uploads")
    #: Hard ceiling enforced while streaming, before the file is fully on disk.
    max_upload_bytes: int = 25 * 1024 * 1024
    #: Guards against a zip bomb: an OOXML part that expands beyond this is refused.
    max_extracted_bytes: int = 200 * 1024 * 1024

    #: Pinned. Changing this invalidates every stored vector and requires a full
    #: re-embed — never a mix (CLAUDE.md 14, risk register).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    #: The env var is EMBEDDING_DIMENSION per CLAUDE.md 13; the Python attribute stays
    #: `embedding_dim` because existing consumers (app/rag/embeddings.py) read it by
    #: that name and are out of this phase's scope to rename.
    embedding_dim: int = Field(
        default=384, ge=1, validation_alias="EMBEDDING_DIMENSION"
    )

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Background job hardening ---

    #: How often the reaper sweeps for ingestion jobs that will never report back. Cheap
    #: (two indexed queries), so the interval is set by how long a user should stare at a
    #: spinner before the truth arrives, not by cost.
    ingestion_reap_interval_seconds: int = 300
    #: Kill switch for the sweep.
    ingestion_reaper_enabled: bool = True
    #: Attempts to embed a batch before falling back to embedding its chunks one at a time.
    embedding_max_attempts: int = 2
    #: Chunks embedded per batch. Larger is faster per chunk but loses more work to a
    #: single failure and holds more of the model's activations in memory at once.
    embedding_batch_size: int = 32

    # --- Retrieval and generation ---

    #: Chunks fetched per question. Beyond roughly this many, recall gains flatten while
    #: the odds of burying the relevant passage in noise keep rising.
    retrieval_top_k: int = 6
    #: Cosine distance above which a chunk is treated as unrelated. pgvector reports
    #: distance, not similarity: 0 is identical, 2 is opposite.
    retrieval_max_distance: float = 0.75
    #: Ceiling on context assembled into one prompt, in characters.
    retrieval_max_context_chars: int = 12_000

    # --- Retrieval tuning (CLAUDE.md 8, 13) ---

    #: Hybrid-retrieval candidates merged by RRF and fed to the reranker (pre-rerank
    #: cap, ~15). Must be >= retrieval_final_count; enforced by a model validator below.
    retrieval_candidate_count: int = Field(default=15, ge=1)
    #: Chunks actually passed to the LLM after reranking (post-rerank cap, 5–8).
    retrieval_final_count: int = Field(default=8, ge=1)
    #: Layer-1 grounding threshold (CLAUDE.md 8.3): if the top reranked chunk scores
    #: below this, the LLM is never called and the question is refused honestly.
    retrieval_relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- Phase B-2: Absolute grounding thresholds (cross-encoder logits) ---

    #: Absolute minimum rerank score for an overview query's top chunk.
    #: Cross-encoder scores are raw logits (range ~[-12, +12]), NOT probabilities.
    #: Calibrated: Kanban overview top=-3.80, clearly irrelevant is ~-8.
    #: This is independent of retrieval_relevance_threshold (which governs fact-lookup).
    overview_min_score: float = Field(
        default=-4.0,
        description="Absolute minimum rerank score for overview grounding",
    )
    #: Absolute minimum mean score across top chunks for overview aggregate grounding.
    #: Calibrated: Kanban overview top-3 mean ~-7.16, clearly irrelevant mean ~-8.7.
    overview_aggregate_min: float = Field(
        default=-7.5,
        description="Absolute minimum mean of top overview scores",
    )
    #: Confidence threshold above which document targeting is considered high-confidence
    #: and grounds with a relaxed score floor.  Range [0.0, 1.0].
    doc_target_high_confidence: float = Field(
        default=0.90, ge=0.0, le=1.0,
        description="Confidence threshold for high-confidence doc-target grounding",
    )
    #: Absolute minimum rerank score when high-confidence document targeting applies.
    #: More relaxed than overview_min_score because the doc target is strong evidence.
    #: Calibrated: Aarya resume top_score=-0.477 passes; clearly irrelevant ~-8 does not.
    doc_target_relaxed_score: float = Field(
        default=-3.0,
        description="Absolute min score when high-confidence doc-target relaxes grounding",
    )
    #: Permissive floor for filename-matched queries.  When filename matching finds
    #: a document and chunks from it are in the final set, we ground regardless of
    #: the reranker score — the filename IS the evidence.  Set very low to accommodate
    #: cases where the reranker scores the content poorly (e.g. "do you have any resume"
    #: scores -11 against resume content).  Clearly irrelevant chunks still fail
    #: because they won't be from the filename-matched document.
    filename_match_relaxed_score: float = Field(
        default=-15.0,
        description="Permissive floor when filename match + target chunks present",
    )

    # --- Reranker (CLAUDE.md 2) ---

    #: Local cross-encoder, run via sentence-transformers. Pinned like the embedding
    #: model: changing it changes the score scale the grounding threshold was tuned on.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Uploads (CLAUDE.md 6) ---

    #: Per-file ceiling in MB, enforced from Content-Length before the file is read
    #: into memory.
    max_upload_size_mb: int = Field(default=10, ge=1)

    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 4096
    #: Time budget for the whole generation. A stalled provider must surface as an error
    #: rather than an open connection the client waits on indefinitely.
    llm_timeout_seconds: float = 60.0

    #: Turns of prior conversation replayed into the prompt.
    chat_history_limit: int = 10

    # --- Workspace & conversation memory ---

    #: Maximum tokens of conversation history injected into the prompt.
    memory_max_tokens: int = 4000
    #: Token count above which old messages are summarized rather than replayed verbatim.
    memory_summary_threshold: int = 2000
    #: Maximum number of recent messages (both roles) to keep in the context window.
    memory_recent_window: int = 20
    #: Which memory strategy to use: "window" (newest N), "summary" (summarize old),
    #: or "hybrid" (summary of old + verbatim recent).
    memory_strategy: str = "hybrid"

    # --- JWT verification (CLAUDE.md 4.6) ---

    #: Supabase now signs session tokens with rotatable asymmetric keys (ES256) published
    #: at the project's JWKS endpoint. `jwt_secret` remains for legacy HS256 projects and
    #: for locally minted test tokens; both are accepted so a migration needs no downtime.
    jwt_algorithms: list[str] = ["ES256", "RS256", "HS256"]
    #: How long a fetched signing key is trusted before it is re-fetched.
    jwks_cache_seconds: int = 600

    @model_validator(mode="after")
    def _derive_llm_config(self) -> "Settings":
        """Auto-derive LLM_PROVIDER/MODEL/API_KEY from a provider key when not set.

        When a provider key (GEMINI_API_KEY, GROQ_API_KEY, or
        OPENROUTER_API_KEY) is present but the generic LLM fields are not, the
        validator fills them in from the provider preset.  Explicit LLM_* values
        always take precedence — this is a convenience, not an override of
        intentional configuration.
        """
        # If the generic fields are already fully specified, nothing to derive.
        if self.llm_provider and self.llm_model and self.llm_api_key:
            return self

        # Determine which provider key is set (priority order for primary).
        # Chain order: Groq (primary) → OpenRouter (fallback) → Gemini (secondary fallback).
        active_key: SecretStr | None = None
        provider_name: str | None = None
        if self.groq_api_key:
            active_key = self.groq_api_key
            provider_name = "groq"
        elif self.openrouter_api_key:
            active_key = self.openrouter_api_key
            provider_name = "openrouter"
        elif self.gemini_api_key:
            active_key = self.gemini_api_key
            provider_name = "gemini"

        if active_key is None:
            # No provider key and no generic fields — this is a configuration error.
            # Raise a clear message instead of letting the app crash later.
            raise ValueError(
                "No LLM provider configured. Set GEMINI_API_KEY, GROQ_API_KEY, "
                "or OPENROUTER_API_KEY, "
                "or provide LLM_PROVIDER + LLM_MODEL + LLM_API_KEY explicitly."
            )

        preset = _PROVIDER_PRESETS[provider_name]

        # Derive any missing generic field from the preset.  Explicit values
        # (e.g. LLM_MODEL=custom-model) are kept as-is.
        if not self.llm_provider:
            self.llm_provider = provider_name
        if not self.llm_model:
            self.llm_model = preset["model"]
        if not self.llm_api_key:
            self.llm_api_key = active_key
        if not self.llm_base_url:
            self.llm_base_url = preset["base_url"]

        return self

    @property
    def fallback_chain_configs(self) -> list[dict[str, str | None]]:
        """Build the ordered fallback chain from configured provider keys.

        Returns a list of dicts, each with keys: name, api_key, model, base_url.
        Only providers whose API key is configured are included.  The order is
        Groq (primary) → OpenRouter (fallback) → Gemini (secondary fallback).
        """
        chain: list[dict[str, str | None]] = []

        # Primary: Groq (or explicit LLM_* override when provider is groq)
        if self.groq_api_key:
            chain.append({
                "name": "groq",
                "api_key": self.groq_api_key.get_secret_value(),
                "model": self.llm_model if self.llm_provider == "groq" else _PROVIDER_PRESETS["groq"]["model"],
                "base_url": self.llm_base_url if self.llm_provider == "groq" else _PROVIDER_PRESETS["groq"]["base_url"],
            })

        # Fallback: OpenRouter (if configured)
        if self.openrouter_api_key:
            chain.append({
                "name": "openrouter",
                "api_key": self.openrouter_api_key.get_secret_value(),
                "model": self.openrouter_model or _PROVIDER_PRESETS["openrouter"]["model"],
                "base_url": _PROVIDER_PRESETS["openrouter"]["base_url"],
            })

        # Secondary fallback: Gemini (if configured)
        if self.gemini_api_key and self.llm_provider != "gemini":
            chain.append({
                "name": "gemini",
                "api_key": self.gemini_api_key.get_secret_value(),
                "model": _PROVIDER_PRESETS["gemini"]["model"],
                "base_url": _PROVIDER_PRESETS["gemini"]["base_url"],
            })

        return chain

    @model_validator(mode="after")
    def _retrieval_counts_are_consistent(self) -> "Settings":
        """The pre-rerank candidate pool must be at least as large as the final set."""
        if self.retrieval_candidate_count < self.retrieval_final_count:
            raise ValueError(
                "RETRIEVAL_CANDIDATE_COUNT must be >= RETRIEVAL_FINAL_COUNT "
                f"(got {self.retrieval_candidate_count} < {self.retrieval_final_count})."
            )
        return self

    @property
    def jwks_url(self) -> str:
        return f"{str(self.supabase_url).rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def sentry_enabled(self) -> bool:
        """Whether error reporting should be wired up at all.

        A blank DSN is the normal state for a developer machine, so it disables Sentry
        rather than failing — but it does so explicitly here instead of relying on the SDK's
        own no-DSN behaviour, so every call site can branch on one readable predicate.
        """
        return bool(self.sentry_dsn and self.sentry_dsn.strip())

    @field_validator("database_url", mode="before")
    @classmethod
    def _require_async_driver(cls, value: object) -> object:
        """Force the asyncpg driver onto a bare `postgresql://` URL.

        Supabase, Railway and most managed providers hand out the sync form, and pasting
        it verbatim is the obvious thing to do. Without a driver the URL resolves to
        psycopg2, which this project does not install — so the whole app would fail at
        first connection with a ModuleNotFoundError that says nothing about the cause.
        """
        if isinstance(value, str) and value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: EnvSettingsSource,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        """Replace the default env source with one that handles comma-separated
        CORS origins, which is the format documented in .env.example.
        """
        cors_source = _CorsEnvSettingsSource(
            settings_cls,
            env_prefix=env_settings.env_prefix,
        )
        return (
            init_settings,
            cors_source,
            dotenv_settings,
            file_secret_settings,
        )


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["", "=" * 72, "FATAL: application configuration is invalid.", "=" * 72, ""]
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"])
        env_var = field.upper()
        if error["type"] == "missing":
            lines.append(f"  {env_var} is required but not set.")
        elif env_var:
            lines.append(f"  {env_var}: {error['msg']}")
        else:
            # A model-level error (e.g. the retrieval-count relationship) has no field
            # location, so its message must stand alone.
            lines.append(f"  {error['msg']}")
    lines.extend(["", _ENV_EXAMPLE_HINT, "=" * 72, ""])
    return "\n".join(lines)


@lru_cache
def get_settings() -> Settings:
    """Load settings, exiting with a readable message rather than a raw traceback."""
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        print(_format_validation_error(exc), file=sys.stderr)  # noqa: T201
        raise SystemExit(1) from exc

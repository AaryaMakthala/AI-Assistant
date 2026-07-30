"""Application settings. All secrets come from .env — never hardcoded (CLAUDE.md 4.1)."""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ENV_EXAMPLE_HINT = (
    "Copy .env.example to .env at the repo root and fill in the values listed above.\n"
    "See CLAUDE.md section 9 for the full variable list."
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

    gemini_api_key: SecretStr
    groq_api_key: SecretStr

    redis_url: RedisDsn

    jwt_secret: SecretStr = Field(min_length=32)

    sentry_dsn: str | None = None
    langsmith_api_key: SecretStr | None = None
    github_token: SecretStr | None = None

    cors_allow_origins: list[str] = ["http://localhost:3000"]

    # --- Document ingestion (CLAUDE.md 4.2, Phase 3) ---

    #: Where uploaded bytes land, named by generated UUID — never by user-supplied name.
    upload_dir: Path = Path("var/uploads")
    #: Hard ceiling enforced while streaming, before the file is fully on disk.
    max_upload_bytes: int = 25 * 1024 * 1024
    #: Guards against a zip bomb: an OOXML part that expands beyond this is refused.
    max_extracted_bytes: int = 200 * 1024 * 1024

    #: Pinned. Changing this invalidates every stored vector and requires a full
    #: re-embed — never a mix (CLAUDE.md section 7, Risk 1).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Retrieval and generation (Phase 4) ---

    #: Chunks fetched per question. Beyond roughly this many, recall gains flatten while
    #: the odds of burying the relevant passage in noise keep rising.
    retrieval_top_k: int = 6
    #: Cosine distance above which a chunk is treated as unrelated. pgvector reports
    #: distance, not similarity: 0 is identical, 2 is opposite. Without a ceiling the
    #: nearest neighbours of an off-topic question still come back and read as sources.
    retrieval_max_distance: float = 0.75
    #: Ceiling on context assembled into one prompt, in characters.
    retrieval_max_context_chars: int = 12_000

    llm_model: str = "gemini-2.0-flash"
    llm_fallback_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.2
    llm_max_output_tokens: int = 1024
    #: Time budget for the whole generation. A stalled provider must surface as an error
    #: rather than an open connection the client waits on indefinitely.
    llm_timeout_seconds: float = 60.0

    #: Turns of prior conversation replayed into the prompt.
    chat_history_limit: int = 10

    # --- Guarded SQL agent (CLAUDE.md 4.3, Phase 5) ---

    #: Hard ceiling on rows any generated query may return. Injected into the query as a
    #: LIMIT rather than trusted to the model, and re-checked after generation.
    sql_max_rows: int = 500
    #: Postgres `statement_timeout` for a generated query. A model can easily produce a
    #: cross join that is valid, allowed, and would run for hours.
    sql_query_timeout_ms: int = 5_000
    #: NOLOGIN group role holding column-level SELECT on the allowlisted tables and nothing
    #: else. Every generated query runs after `SET LOCAL ROLE` to this, so the database
    #: refuses a write even if every layer above it were bypassed.
    sql_agent_role: str = "app_sql_agent"
    #: Attempts allowed to produce a query that passes validation. A rejected query is fed
    #: back once with the reason; beyond that the model is usually looping on the same idea.
    sql_max_generation_attempts: int = 2

    # --- Multi-agent orchestration (Phase 7) ---

    #: Hard cap on routing passes for one question (CLAUDE.md section 7, "agent loops
    #: forever"). A counter in graph state, not an instruction in a prompt — a model cannot
    #: decline to respect it. The graph is currently acyclic so this is never reached; it
    #: exists so adding a cycle later cannot produce an unbounded loop.
    agent_max_iterations: int = 3
    #: LangGraph's own step ceiling, an independent backstop to the counter above. Counts
    #: every node execution, so it must exceed the longest legitimate path: router plus a
    #: three-way fan-out plus synthesis.
    agent_recursion_limit: int = 12

    # --- JWT verification (CLAUDE.md 4.6) ---

    #: Supabase now signs session tokens with rotatable asymmetric keys (ES256) published
    #: at the project's JWKS endpoint. `jwt_secret` remains for legacy HS256 projects and
    #: for locally minted test tokens; both are accepted so a migration needs no downtime.
    jwt_algorithms: list[str] = ["ES256", "RS256", "HS256"]
    #: How long a fetched signing key is trusted before it is re-fetched. Rotation is
    #: expected, so an unbounded cache would eventually reject every live token.
    jwks_cache_seconds: int = 600

    @property
    def jwks_url(self) -> str:
        return f"{str(self.supabase_url).rstrip('/')}/auth/v1/.well-known/jwks.json"

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

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

    @property
    def celery_broker_url(self) -> str:
        return str(self.redis_url)

    @property
    def celery_result_backend(self) -> str:
        return str(self.redis_url)


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["", "=" * 72, "FATAL: application configuration is invalid.", "=" * 72, ""]
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"])
        env_var = field.upper()
        if error["type"] == "missing":
            lines.append(f"  {env_var} is required but not set.")
        else:
            lines.append(f"  {env_var}: {error['msg']}")
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

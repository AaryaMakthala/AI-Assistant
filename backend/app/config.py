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

    # --- Observability (Phase 11) ---

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

    #: Master switch for LangSmith. Off by default: agent traces contain the user's question
    #: and the retrieved document text, so shipping them to a third party is a decision that
    #: has to be made deliberately (CLAUDE.md section 2, observability).
    langsmith_tracing_enabled: bool = False
    langsmith_project: str = "enterprise-ai-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    #: Required to trace a production environment. Without it, tracing self-disables when
    #: `ENVIRONMENT=production` even if everything else is configured — the "dev/staging
    #: only" rule enforced in code rather than left to whoever writes the deploy config.
    langsmith_allow_production: bool = False
    #: Send run inputs/outputs to LangSmith. Forced off in production regardless, so an
    #: approved production trace still records structure and timings without content.
    langsmith_trace_io: bool = True

    #: Total tokens in one chat turn above which a warning is emitted and reported to Sentry
    #: (CLAUDE.md section 7, cost/quota overrun). 0 disables the check.
    token_usage_alert_threshold: int = Field(default=0, ge=0)

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

    # --- Background job hardening (Phase 10) ---

    #: How often the reaper sweeps for ingestion jobs that will never report back. Cheap
    #: (two indexed queries), so the interval is set by how long a user should stare at a
    #: spinner before the truth arrives, not by cost.
    ingestion_reap_interval_seconds: int = 300
    #: Kill switch for the sweep. Set false when running a one-off worker against a shared
    #: database, where another environment's in-flight jobs would look abandoned.
    ingestion_reaper_enabled: bool = True
    #: Attempts to embed a batch before falling back to embedding its chunks one at a time.
    embedding_max_attempts: int = 2
    #: Chunks embedded per batch. Larger is faster per chunk but loses more work to a
    #: single failure and holds more of the model's activations in memory at once.
    embedding_batch_size: int = 32

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

    #: Failover order, tried left to right. Groq leads: it is the faster of the two at
    #: this size and its free tier is the more generous, so Gemini is held in reserve for
    #: when Groq rate-limits. Configurable so the order can be changed without a deploy.
    #: A plain comma-separated string rather than a list, because pydantic-settings decodes
    #: a list-typed env var as JSON and would reject the obvious `groq,gemini`.
    llm_provider_order: str = "groq,gemini"
    llm_groq_model: str = "llama-3.3-70b-versatile"
    llm_gemini_model: str = "gemini-2.0-flash"
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
    def llm_providers(self) -> list[str]:
        """The failover order as a list of provider names, lowercased and de-duplicated.

        Unknown names are left in: rejecting them here would fail at import time with a
        pydantic error, whereas the router can name the offending provider and list the
        ones it does know.
        """
        seen: list[str] = []
        for part in self.llm_provider_order.split(","):
            name = part.strip().lower()
            if name and name not in seen:
                seen.append(name)
        return seen

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

    @property
    def langsmith_enabled(self) -> bool:
        """Whether agent traces may be shipped to LangSmith.

        Three conditions, all required. The production one is the reason this is a property
        rather than a bare flag: CLAUDE.md's rule is that production user data does not reach
        a third-party trace tool without a deliberate review, and a rule that lives only in a
        deploy checklist is a rule that gets skipped. `langsmith_allow_production` is the
        recorded outcome of that review.
        """
        if not (self.langsmith_tracing_enabled and self.langsmith_api_key):
            return False
        return self.langsmith_allow_production or not self.is_production

    @property
    def langsmith_records_io(self) -> bool:
        """Whether run inputs and outputs travel with a trace.

        Forced off in production even when tracing is approved there: the structure and
        timings of a run are what debugging needs, while the question text and retrieved
        passages are the part that carries tenant data.
        """
        return self.langsmith_trace_io and not self.is_production

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

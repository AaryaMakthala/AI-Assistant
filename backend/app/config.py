"""Application settings. All secrets come from .env — never hardcoded (CLAUDE.md 4.1)."""

import sys
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, ValidationError
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

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


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

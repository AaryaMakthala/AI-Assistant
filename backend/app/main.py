from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.api.documents import router as documents_router
from app.config import get_settings
from app.db.session import dispose_engine
from app.errors import register_exception_handlers, register_request_context
from app.logging_config import configure_logging
from app.security.rate_limit import register_rate_limiting


class HealthResponse(BaseModel):
    status: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info("Starting up in {environment} mode", environment=settings.environment)
    yield
    await dispose_engine()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()
    # Configure before anything can log: Loguru's default handler has diagnose=True,
    # which would dump local variables (and any secrets in them) into tracebacks.
    configure_logging(
        level="DEBUG" if settings.debug else "INFO",
        serialize=settings.is_production,
    )
    app = FastAPI(
        title="Enterprise AI Knowledge Intelligence Agent",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )

    register_request_context(app)
    register_rate_limiting(app)
    register_exception_handlers(app)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    app.include_router(documents_router)
    return app


def __getattr__(name: str) -> FastAPI:
    """Build the ASGI app on first access so importing this module needs no config."""
    if name == "app":
        return create_app()
    raise AttributeError(name)

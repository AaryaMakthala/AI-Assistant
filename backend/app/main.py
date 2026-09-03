from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.api.auth import router as auth_router
from app.api.chat_v2 import router as grounded_chat_router
from app.api.demo import router as demo_router
from app.api.documents_v2 import router as documents_router
from app.api.workspaces import accept_router
from app.api.workspaces import router as workspaces_router
from app.config import get_settings
from app.db.session import dispose_engine
from app.errors import register_exception_handlers, register_request_context
from app.logging_config import configure_logging
from app.observability import configure_sentry, configure_tracing
from app.observability.sentry import sentry_is_active
from app.observability.tracing import tracing_is_enabled
from app.security.rate_limit import register_rate_limiting


class HealthResponse(BaseModel):
    status: str
    environment: str
    #: Whether errors reach Sentry from this process. Reported so a deploy can confirm
    #: observability is live without waiting for something to break — the alternative is
    #: discovering an unset DSN during the incident it was meant to help with.
    error_reporting: bool = False
    #: Whether agent traces are being shipped. Expected false in production.
    agent_tracing: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info("Starting up in {environment} mode", environment=settings.environment)

    # Auto-apply pending Alembic migrations on startup so a missed
    # ``alembic upgrade head`` can never silently break an endpoint.
    if settings.environment != "test":
        try:
            import subprocess
            import sys
            from pathlib import Path
            _backend_dir = str(Path(__file__).resolve().parent.parent)
            result = subprocess.run(
                [sys.executable, "-m", "alembic", "upgrade", "head"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=_backend_dir,
            )
            if result.returncode == 0:
                # Only log if there were pending migrations (non-empty stdout).
                output = result.stdout.strip()
                if output and "Running upgrade" in output:
                    logger.info("Applied pending migrations: {output}", output=output)
            else:
                logger.error(
                    "Migration failed (exit code {code}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
                    code=result.returncode,
                    stdout=result.stdout[:2000] if result.stdout else "(empty)",
                    stderr=result.stderr[:2000] if result.stderr else "(empty)",
                )
        except Exception as exc:
            logger.exception("Migration runner error: {error}", error=exc)

    # Seed the demo workspace (idempotent) and start the cleanup task.
    cleanup_task = None
    if settings.demo_enabled and settings.environment != "test":
        try:
            from app.demo.seed import seed_demo_workspace
            ws_id = await seed_demo_workspace()
            if ws_id is not None:
                logger.info("Demo workspace ready: {ws}", ws=ws_id)
        except Exception as exc:
            logger.exception("Demo seed failed (non-fatal): {error}", error=exc)

        # Schedule periodic guest cleanup.
        import asyncio

        async def _run_cleanup_loop():
            from app.demo.cleanup import cleanup_demo_guests

            interval = settings.demo_guest_ttl_hours * 3600  # Run once per TTL period
            # First cleanup after a short delay to avoid startup contention.
            await asyncio.sleep(60)
            while True:
                try:
                    removed = await cleanup_demo_guests()
                    if removed > 0:
                        logger.info("Demo cleanup: removed {n} guest(s)", n=removed)
                except Exception as exc:
                    logger.warning("Demo cleanup failed: {error}", error=str(exc)[:200])
                await asyncio.sleep(interval)

        cleanup_task = asyncio.create_task(_run_cleanup_loop())

    yield

    # Cancel the cleanup task on shutdown.
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

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
    # Observability comes up next, and before the app object exists: Sentry's Starlette
    # integration patches Starlette at init time, so an app constructed first would not be
    # instrumented. Both calls are no-ops when unconfigured.
    configure_sentry(settings, component="api")
    configure_tracing(settings)

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
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Workspace-ID"],
    )

    register_request_context(app)
    register_rate_limiting(app)
    register_exception_handlers(app)

    # Workspace-switching middleware: reads X-Workspace-ID header and stores
    # the override in request state for get_principal to pick up.
    from app.security.workspace_switch import workspace_switch_middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    app.add_middleware(BaseHTTPMiddleware, dispatch=workspace_switch_middleware)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            environment=settings.environment,
            error_reporting=sentry_is_active(),
            agent_tracing=tracing_is_enabled(),
        )

    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(grounded_chat_router)
    app.include_router(workspaces_router)
    app.include_router(accept_router)
    app.include_router(demo_router)
    return app


def __getattr__(name: str) -> FastAPI:
    """Build the ASGI app on first access so importing this module needs no config."""
    if name == "app":
        return create_app()
    raise AttributeError(name)

"""Global exception handling. Clients get an opaque reference; details stay server-side."""

import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.logging_config import request_id_var


def _error_body(detail: str, request_id: str) -> dict[str, str]:
    return {"detail": detail, "request_id": request_id}


class RequestContextMiddleware:
    """Assign each request an ID, in the scope so error handlers at any layer can read it.

    Pure ASGI rather than BaseHTTPMiddleware: Starlette's ServerErrorMiddleware wraps the
    user middleware stack, so a ContextVar reset there would leave 500 handlers without an ID.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1").strip()
        request_id = incoming[:64] if incoming else uuid.uuid4().hex

        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)

        async def send_with_header(message: Message) -> None:
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_header)
        finally:
            request_id_var.reset(token)


def _request_id_of(request: Request) -> str:
    return request.scope.get("state", {}).get("request_id") or request_id_var.get()


def register_request_context(app: FastAPI) -> None:
    app.add_middleware(RequestContextMiddleware)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        request_id = _request_id_of(request)
        logger.warning("HTTP {status}: {detail}", status=exc.status_code, detail=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(str(exc.detail), request_id),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = _request_id_of(request)
        logger.warning("Request validation failed: {errors}", errors=exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_error_body("Request validation failed.", request_id),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id_of(request)
        logger.opt(exception=exc).error("Unhandled exception on {path}", path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body(
                "Internal server error. Quote the request_id when reporting this.", request_id
            ),
            headers={"x-request-id": request_id},
        )

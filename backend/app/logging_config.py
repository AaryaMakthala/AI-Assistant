"""Structured logging via Loguru, with request-ID correlation."""

import logging
import sys
from contextvars import ContextVar

from loguru import logger

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class _InterceptHandler(logging.Handler):
    """Route stdlib logging (uvicorn, sqlalchemy) through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _add_request_id(record: dict) -> None:
    record["extra"].setdefault("request_id", request_id_var.get())


def configure_logging(*, level: str = "INFO", serialize: bool = False) -> None:
    logger.remove()
    logger.configure(patcher=_add_request_id)
    logger.add(
        sys.stderr,
        level=level,
        serialize=serialize,
        backtrace=False,
        diagnose=False,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
            "<cyan>{extra[request_id]}</cyan> | <level>{message}</level>"
        ),
    )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "sqlalchemy.engine"):
        logging.getLogger(name).handlers = [_InterceptHandler()]
        logging.getLogger(name).propagate = False

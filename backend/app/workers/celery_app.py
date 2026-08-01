"""Celery application.

All slow or hostile work runs here rather than in a request handler: a malformed PDF
that hangs a parser costs a replaceable worker, not the API process (CLAUDE.md 4.2).

Observability is initialised per *worker process*, not at import (Phase 11). Celery's
default prefork pool forks after the module is imported, and a Sentry client created
before the fork holds a background transport thread that does not survive it — the child
would look configured and silently transmit nothing. `worker_process_init` fires inside
each child, which is the only place that is correct.
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import (
    beat_init,
    task_postrun,
    task_prerun,
    worker_process_init,
)
from loguru import logger

from app.config import get_settings
from app.logging_config import configure_logging, request_id_var
from app.observability import (
    clear_observability_context,
    configure_sentry,
    configure_tracing,
)
from app.observability.sentry import set_tag


def create_celery() -> Celery:
    settings = get_settings()
    app = Celery("enterprise_ai_agent", broker=settings.celery_broker_url)
    app.conf.update(
        result_backend=settings.celery_result_backend,
        task_serializer="json",
        result_serializer="json",
        # JSON only: pickle would let anything able to write to the broker execute
        # arbitrary code in a worker.
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # A hung parser is killed rather than occupying a worker forever. The soft limit
        # raises an exception first, so the task can still mark the document failed.
        task_soft_time_limit=10 * 60,
        task_time_limit=12 * 60,
        # Ingestion is idempotent per document, so a task lost to a crashed worker is
        # safe to redeliver; late ack is what makes that redelivery happen.
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        result_expires=24 * 3600,
        # A task whose worker dies is redelivered, but only if the worker was lost
        # cleanly. `reject_on_worker_lost` covers the harder case — a SIGKILL from an OOM
        # killer — by requeuing rather than silently dropping the message. The reaper in
        # app/workers/maintenance.py exists because even this is not total.
        task_reject_on_worker_lost=True,
        beat_schedule={
            "reap-stuck-documents": {
                "task": "app.workers.maintenance.reap_stuck_documents",
                "schedule": float(settings.ingestion_reap_interval_seconds),
                # Expire a missed tick rather than letting beat catch up: if the worker
                # was down for an hour, running twelve identical sweeps on restart is
                # pure duplication — the next scheduled one covers the same ground.
                "options": {"expires": float(settings.ingestion_reap_interval_seconds)},
            },
        },
    )
    app.autodiscover_tasks(["app.workers"])
    return app


celery_app = create_celery()


def _init_observability(component: str) -> None:
    settings = get_settings()
    configure_logging(
        level="DEBUG" if settings.debug else "INFO",
        serialize=settings.is_production,
    )
    configure_sentry(settings, component=component)
    configure_tracing(settings)
    logger.info("Observability configured for celery {component}", component=component)


@worker_process_init.connect
def _configure_worker_process(**_: Any) -> None:
    """Bring observability up inside each forked worker child."""
    _init_observability("worker")


@beat_init.connect
def _configure_beat(**_: Any) -> None:
    """Beat runs in its own process and never forks, so it configures itself directly."""
    _init_observability("beat")


@task_prerun.connect
def _bind_task_context(task_id: str | None = None, task: Any = None, **_: Any) -> None:
    """Correlate a task's logs and errors with the task itself.

    The Celery task id takes the place of a request id here: a worker has no HTTP request,
    but the same need to join a log line to the error report beside it. The id is written
    into the same ContextVar the log format already reads, so a worker's output is shaped
    exactly like the API's.
    """
    if task_id:
        request_id_var.set(task_id)
        set_tag("request_id", task_id)
    name = getattr(task, "name", None)
    if name:
        set_tag("task_name", name)


@task_postrun.connect
def _unbind_task_context(**_: Any) -> None:
    """Forget the finished task's identity.

    Load-bearing rather than tidy: a prefork worker reuses its process for task after task,
    so an org id left behind here would be attached to the *next* document's errors — a
    cross-tenant mix-up in the error store even though the data path itself is scoped.
    """
    request_id_var.set("-")
    clear_observability_context()


__all__ = ["celery_app", "create_celery"]

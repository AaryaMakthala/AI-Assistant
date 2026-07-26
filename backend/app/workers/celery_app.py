"""Celery application.

All slow or hostile work runs here rather than in a request handler: a malformed PDF
that hangs a parser costs a replaceable worker, not the API process (CLAUDE.md 4.2).
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings


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
    )
    app.autodiscover_tasks(["app.workers"])
    return app


celery_app = create_celery()

__all__ = ["celery_app", "create_celery"]

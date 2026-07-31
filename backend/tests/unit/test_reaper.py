"""The stuck-job reaper: the backstop for work that stops without reporting.

Retries cover a task that fails. Nothing covers a worker that is killed — the row stays
`processing` and the UI spins forever. The reaper's whole value is in its cutoff being
conservative: reaping a job that is merely slow destroys work a user is waiting on, so
these tests pin the boundary from both sides.

The database is not involved here. The cutoff arithmetic and the requeue decision are the
parts with logic in them; the queries around them are exercised by the integration suite.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.workers import maintenance
from app.workers.celery_app import celery_app
from app.workers.maintenance import _ABANDON_MARGIN, _abandoned_before, _requeue

pytestmark = pytest.mark.usefixtures("valid_env")


def test_cutoff_is_older_than_the_hard_time_limit() -> None:
    """A job cannot be abandoned before Celery would itself have killed it."""
    cutoff = _abandoned_before()
    hard_limit = timedelta(seconds=celery_app.conf.task_time_limit)

    assert cutoff < datetime.now(UTC) - hard_limit


def test_cutoff_includes_the_safety_margin() -> None:
    """The margin covers queue wait — without it a queued job would be reaped alive."""
    cutoff = _abandoned_before()
    expected = (
        datetime.now(UTC) - timedelta(seconds=celery_app.conf.task_time_limit) - _ABANDON_MARGIN
    )

    # Compared loosely: both sides call now() a moment apart.
    assert abs((cutoff - expected).total_seconds()) < 5


def test_a_job_that_just_started_is_not_past_the_cutoff() -> None:
    just_started = datetime.now(UTC)

    assert just_started > _abandoned_before()


def test_a_job_running_just_under_the_limit_is_not_past_the_cutoff() -> None:
    """The case that matters: a slow but live job must survive the sweep."""
    hard_limit = celery_app.conf.task_time_limit
    still_running = datetime.now(UTC) - timedelta(seconds=hard_limit - 60)

    assert still_running > _abandoned_before()


def test_a_long_abandoned_job_is_past_the_cutoff() -> None:
    abandoned = datetime.now(UTC) - timedelta(hours=3)

    assert abandoned < _abandoned_before()


def test_requeue_reports_success_when_the_broker_accepts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[tuple[str, str]] = []

    class FakeTask:
        def delay(self, document_id: str, org_id: str) -> None:
            sent.append((document_id, org_id))

    import app.workers.ingestion as ingestion

    monkeypatch.setattr(ingestion, "ingest_document", FakeTask())

    document_id, org_id = uuid.uuid4(), uuid.uuid4()
    assert _requeue(document_id, org_id) is True
    assert sent == [(str(document_id), str(org_id))]


def test_requeue_reports_failure_when_the_broker_is_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broker outage must leave the document pending, not crash the sweep."""

    class DeadTask:
        def delay(self, document_id: str, org_id: str) -> None:
            raise ConnectionError("broker unreachable")

    import app.workers.ingestion as ingestion

    monkeypatch.setattr(ingestion, "ingest_document", DeadTask())

    assert _requeue(uuid.uuid4(), uuid.uuid4()) is False


def test_reaper_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kill switch must short-circuit before touching the database."""
    monkeypatch.setenv("INGESTION_REAPER_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        result = maintenance.reap_stuck_documents()
        assert result == {"failed": 0, "requeued": 0, "skipped": True}
    finally:
        get_settings.cache_clear()


def test_beat_schedule_registers_the_reaper() -> None:
    """A reaper nobody schedules is a reaper that never runs."""
    schedule = celery_app.conf.beat_schedule

    assert "reap-stuck-documents" in schedule
    assert (
        schedule["reap-stuck-documents"]["task"] == "app.workers.maintenance.reap_stuck_documents"
    )

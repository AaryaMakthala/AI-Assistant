"""Phase 11: Sentry initialisation and the context attached to an error.

These exercise the real SDK against an in-process transport, so what is asserted is the
event as it would have been transmitted — not a mock's record of a call. The redaction tests
in tests/security/test_observability_redaction.py cover the other half of the phase's
acceptance criterion.
"""

from __future__ import annotations

import builtins
import uuid
from collections.abc import Iterator
from typing import Any

import pytest
import sentry_sdk
from sentry_sdk.transport import Transport

from app.config import get_settings
from app.observability import sentry as sentry_module
from app.observability.context import (
    bind_principal,
    bind_request,
    clear_observability_context,
    observability_tags,
)
from app.observability.sentry import capture_exception, configure_sentry, sentry_is_active

TEST_DSN = "https://publickey@o0.ingest.sentry.io/1"


class _RecordingTransport(Transport):
    """Collects events instead of sending them, so a test inspects the real payload.

    Reads the event back out of the envelope the SDK built, which is what would go over the
    wire — including everything `before_send` and the scrubber did to it.
    """

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        super().__init__()
        self._sink = sink

    def capture_envelope(self, envelope: Any) -> None:
        event = envelope.get_event()
        if event is not None:
            self._sink.append(event)

    def flush(self, timeout: float, callback: Any = None) -> None:
        return None

    def kill(self) -> None:
        return None


@pytest.fixture
def captured_events(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Initialise Sentry against an in-process transport and yield what it would send."""
    events: list[dict[str, Any]] = []
    real_init = sentry_sdk.init

    def init_with_capture(**kwargs: Any) -> Any:
        kwargs["transport"] = _RecordingTransport(events)
        return real_init(**kwargs)

    monkeypatch.setattr(sentry_sdk, "init", init_with_capture)
    yield events
    clear_observability_context()


@pytest.fixture
def sentry_on(
    monkeypatch: pytest.MonkeyPatch, valid_env: None, captured_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    monkeypatch.setenv("SENTRY_DSN", TEST_DSN)
    get_settings.cache_clear()
    assert configure_sentry(component="test") is True
    return captured_events


def test_sentry_stays_off_without_a_dsn(valid_env: None) -> None:
    assert get_settings().sentry_enabled is False
    assert configure_sentry() is False


def test_blank_dsn_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch, valid_env: None) -> None:
    """A DSN of whitespace is the shape a half-filled .env produces."""
    monkeypatch.setenv("SENTRY_DSN", "   ")
    get_settings.cache_clear()

    assert get_settings().sentry_enabled is False
    assert configure_sentry() is False


def test_configure_reports_enabled_and_activates_the_client(
    sentry_on: list[dict[str, Any]],
) -> None:
    assert sentry_is_active() is True


def test_a_deliberate_error_reaches_sentry(sentry_on: list[dict[str, Any]]) -> None:
    """The Phase 11 acceptance criterion, first half."""
    try:
        raise RuntimeError("deliberate failure for verification")
    except RuntimeError as exc:
        capture_exception(exc)
    sentry_sdk.flush()

    assert len(sentry_on) == 1
    values = sentry_on[0]["exception"]["values"]
    assert values[0]["type"] == "RuntimeError"
    assert values[0]["value"] == "deliberate failure for verification"


def test_the_event_carries_environment_and_component(sentry_on: list[dict[str, Any]]) -> None:
    capture_exception(RuntimeError("boom"))
    sentry_sdk.flush()

    event = sentry_on[0]
    assert event["environment"] == "development"
    assert event["tags"]["component"] == "test"


def test_request_id_is_attached_as_a_tag(sentry_on: list[dict[str, Any]]) -> None:
    """The correlation key: the same id the client sees in X-Request-ID."""
    bind_request("req-abc-123")
    capture_exception(RuntimeError("boom"))
    sentry_sdk.flush()

    assert sentry_on[0]["tags"]["request_id"] == "req-abc-123"


def test_principal_is_attached_as_opaque_ids(sentry_on: list[dict[str, Any]]) -> None:
    user_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    bind_principal(user_id=user_id, workspace_id=workspace_id)
    capture_exception(RuntimeError("boom"))
    sentry_sdk.flush()

    event = sentry_on[0]
    assert event["user"] == {"id": str(user_id)}
    assert event["tags"]["workspace_id"] == str(workspace_id)


def test_capture_exception_accepts_extra_tags(sentry_on: list[dict[str, Any]]) -> None:
    capture_exception(RuntimeError("boom"), subsystem="ingestion")
    sentry_sdk.flush()

    assert sentry_on[0]["tags"]["subsystem"] == "ingestion"


def test_extra_tags_do_not_leak_into_later_events(sentry_on: list[dict[str, Any]]) -> None:
    """capture_exception uses an isolated scope, so one call's tags are not the next one's."""
    capture_exception(RuntimeError("first"), subsystem="ingestion")
    capture_exception(ValueError("second"))
    sentry_sdk.flush()

    assert "subsystem" not in sentry_on[1].get("tags", {})


def test_capture_exception_is_a_no_op_when_sentry_is_off(valid_env: None) -> None:
    assert capture_exception(RuntimeError("boom")) is None


def test_capture_exception_never_raises(
    monkeypatch: pytest.MonkeyPatch, sentry_on: list[dict[str, Any]]
) -> None:
    """Reporting an error must not become the error."""

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("sentry itself is broken")

    monkeypatch.setattr(sentry_sdk, "capture_exception", explode)

    assert capture_exception(RuntimeError("boom")) is None


def test_observability_tags_reflect_the_bound_principal(valid_env: None) -> None:
    workspace_id = uuid.uuid4()
    bind_principal(user_id=uuid.uuid4(), workspace_id=workspace_id)
    try:
        tags = observability_tags()
        assert tags["workspace_id"] == str(workspace_id)
    finally:
        clear_observability_context()


def test_observability_tags_omit_unset_values(valid_env: None) -> None:
    clear_observability_context()
    tags = observability_tags()

    assert "user_id" not in tags
    assert "workspace_id" not in tags
    # "-" is the log format's placeholder for "no request", and is not a real id.
    assert tags.get("request_id") != "-"


def test_clear_forgets_the_principal(valid_env: None) -> None:
    bind_principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
    clear_observability_context()

    assert "workspace_id" not in observability_tags()


def test_missing_sdk_disables_reporting_rather_than_crashing(
    monkeypatch: pytest.MonkeyPatch, valid_env: None
) -> None:
    """A deployment without sentry-sdk installed must still boot."""
    monkeypatch.setenv("SENTRY_DSN", TEST_DSN)
    get_settings.cache_clear()

    real_import = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("sentry_sdk"):
            raise ImportError("no sentry_sdk")
        return real_import(name, *args, **kwargs)

    # Restored before the test returns rather than by monkeypatch's teardown, which runs
    # after the autouse fixture that needs to import sentry_sdk to reset the client.
    monkeypatch.setattr(builtins, "__import__", blocked)
    try:
        assert configure_sentry() is False
        assert sentry_module.sentry_is_active() is False
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)

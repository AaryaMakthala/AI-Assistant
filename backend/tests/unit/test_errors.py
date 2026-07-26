"""Phase 1 acceptance: the global handler never leaks internals to the client."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import create_app

SECRET_MARKER = "super-secret-connection-string"  # noqa: S105


@pytest.fixture
def client(valid_env: None) -> TestClient:
    app = create_app()

    @app.get("/boom")
    async def _boom() -> None:
        raise RuntimeError(f"database failure at {SECRET_MARKER}")

    @app.get("/teapot")
    async def _teapot() -> None:
        raise HTTPException(status_code=418, detail="I'm a teapot")

    # raise_server_exceptions=False so the app's own handler runs, as in production
    return TestClient(app, raise_server_exceptions=False)


def test_unhandled_exception_returns_500_without_stack_trace(client: TestClient) -> None:
    response = client.get("/boom")
    body = response.text

    assert response.status_code == 500
    assert SECRET_MARKER not in body
    assert "RuntimeError" not in body
    assert "Traceback" not in body
    assert "app/main.py" not in body


def test_unhandled_exception_returns_correlatable_request_id(client: TestClient) -> None:
    response = client.get("/boom")
    payload = response.json()

    assert payload["detail"] == ("Internal server error. Quote the request_id when reporting this.")
    assert payload["request_id"]
    assert response.headers["x-request-id"] == payload["request_id"]


def test_client_supplied_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})

    assert response.headers["x-request-id"] == "trace-abc-123"


def test_request_ids_are_unique_per_request(client: TestClient) -> None:
    first = client.get("/boom").json()["request_id"]
    second = client.get("/boom").json()["request_id"]

    assert first != second


def test_http_exception_detail_is_preserved(client: TestClient) -> None:
    response = client.get("/teapot")

    assert response.status_code == 418
    assert response.json()["detail"] == "I'm a teapot"


def test_unknown_route_returns_structured_404(client: TestClient) -> None:
    response = client.get("/no-such-route")
    payload = response.json()

    assert response.status_code == 404
    assert "request_id" in payload


def test_health_still_works(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

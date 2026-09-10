"""Focused test for the production liveness endpoint (UptimeRobot-style probe).

``GET /health`` answers a single question: is the FastAPI process alive and
able to accept HTTP requests? It must never depend on the database, the RAG/
LLM pipeline, or authentication, so a temporary dependency outage can never
report the whole service as down.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(valid_env: None) -> TestClient:
    # Built with no lifespan (no migration runner, no demo seeding, no engine
    # init) and no database available: if /health ever needed any of those,
    # this probe would fail loudly instead of returning 200.
    return TestClient(create_app(), raise_server_exceptions=False)


def test_health_is_liveness_probe(client: TestClient) -> None:
    # UptimeRobot parity: an ordinary HTTP GET with no Authorization header,
    # no cookie, and no X-Workspace-ID header must return the health contract.
    response = client.get("/health")

    assert response.status_code == 200
    # Exact, minimal payload — no environment, tenant, document, or pipeline data.
    assert response.json() == {"status": "ok"}
    # No authentication transport is sent, and none is required.
    assert response.request.headers.get("authorization") is None
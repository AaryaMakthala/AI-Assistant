"""Live integration smoke test for the LLM intent router.

Hits the real ``route_with_llm`` (not mocked) against specific inputs that
previously failed.  Skips automatically when no LLM provider API key is
present in the environment so CI without credentials stays green.

Run with:
    pytest tests/integration/test_live_routing.py -v
"""

from __future__ import annotations

import os

import pytest


def _live_integration_enabled() -> bool:
    """Check whether live integration tests should run.

    Requires ENABLE_LIVE_INTEGRATION=1 to be set, because these tests
    hit real LLM providers and cost money / require valid API keys.
    The conftest injects fake provider keys that would cause 401 errors,
    so we need an explicit opt-in gate.
    """
    return os.environ.get("ENABLE_LIVE_INTEGRATION", "").strip() == "1"


# Skip the entire module unless explicitly enabled.
pytestmark = [
    pytest.mark.usefixtures("valid_env"),
    pytest.mark.skipif(
        not _live_integration_enabled(),
        reason="Set ENABLE_LIVE_INTEGRATION=1 to run live LLM integration tests",
    ),
]


# -----------------------------------------------------------------------
# Inputs that previously failed or needed verification.
# Each tuple: (query, expected_route)
# -----------------------------------------------------------------------
_ROUTING_CASES: list[tuple[str, str]] = [
    ("nameste", "GREETING"),
    ("who are you", "IDENTITY_ASSISTANT"),
    ("my name is aarya", "IDENTITY_USER"),
    ("what are the file names", "METADATA"),
    ("can i add someone", "PERMISSIONS"),
    ("what is my company name", "METADATA"),
    ("crns what does it contains", "DOCUMENT_CONTENT"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_route"),
    _ROUTING_CASES,
    ids=[q for q, _ in _ROUTING_CASES],
)
async def test_live_router_classifies_correctly(
    query: str,
    expected_route: str,
) -> None:
    """The real LLM router must classify each input to the expected route."""
    from app.retrieval.llm_router import route_with_llm

    result = await route_with_llm(query=query)

    assert result.status == "success", (
        f"Router returned degraded status for '{query}': {result.reasoning}"
    )
    assert result.route == expected_route, (
        f"Query '{query}' routed to {result.route!r}, expected {expected_route!r} "
        f"(reasoning: {result.reasoning})"
    )
    assert result.confidence > 0.0, (
        f"Query '{query}' has zero confidence: {result.reasoning}"
    )


@pytest.mark.asyncio
async def test_live_router_never_returns_empty() -> None:
    """An empty response from the LLM must be caught and mapped to NEEDS_CLARIFICATION."""
    from app.retrieval.llm_router import route_with_llm

    # A very short, unambiguous greeting should succeed.
    result = await route_with_llm(query="hi")
    assert result.route in {
        "GREETING",
        "NEEDS_CLARIFICATION",
    }, f"Unexpected route: {result.route}"


@pytest.mark.asyncio
async def test_live_router_returns_valid_route() -> None:
    """Every route string must be in the valid taxonomy."""
    from app.retrieval.llm_router import _VALID_ROUTES, route_with_llm

    result = await route_with_llm(query="tell me about the vacation policy")
    assert result.route in _VALID_ROUTES, f"Invalid route: {result.route}"

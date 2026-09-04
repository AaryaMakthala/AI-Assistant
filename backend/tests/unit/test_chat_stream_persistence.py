"""Regression test: streaming chat persists one conversation under ONE session.

Covers the session-persistence fix in ``_stream_chat`` (POST /chat, SSE):
the session to append to is resolved once per request — reusing
``payload.session_id`` when it is valid — and every persistence branch
(RAG/document-content, non-document intents, clarification) writes under
that one id.

The test drives two requests back-to-back the way the frontend does:
  1. a document-content question (RAG branch) — creates the session,
  2. an out-of-scope question (non-document branch) — must reuse it,
with the session_id from the first response's "session" event sent on the
second request.

Expected after the fix: exactly one ChatSession row, four ChatMessage rows
(2 user + 2 assistant) all pointing at it, and a "session" SSE event only
on the request that actually created the session.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.sql import Select
from sqlalchemy.sql.dml import Insert

import app.api.chat_v2 as chat_module
from app.api.dependencies import get_generic_llm
from app.llm.base import Completion, Message, TokenUsage
from app.retrieval.llm_router import RouteResult
from app.retrieval.pipeline import RetrievalResult, RetrievedChunk
from app.security.auth import Principal, get_principal
from tests.unit.conftest import FakeResult

pytestmark = pytest.mark.usefixtures("valid_env")

MSG_DOC = "What does the handbook say about remote work?"
MSG_OOS = "how many planets orbit the sun"


# ---------------------------------------------------------------------------
# Recording in-memory DB — stands in for chat_sessions / chat_messages
# ---------------------------------------------------------------------------


class _RecordingDb:
    """In-memory stand-in for the tables _stream_chat persists to.

    Records every ChatSession/ChatMessage row the endpoint inserts, and answers
    the ownership / history selects the way a real DB holding exactly one
    conversation would.  Everything else (documents, workspace reads) returns
    empty results.
    """

    def __init__(self) -> None:
        self.sessions: list[uuid.UUID] = []
        self.messages: list[dict[str, Any]] = []

    async def execute(self, stmt: Any) -> FakeResult:
        if isinstance(stmt, Insert):
            table = stmt.table.name
            if table == "chat_sessions":
                session_id = uuid.uuid4()
                self.sessions.append(session_id)
                return FakeResult(scalar=session_id)
            if table == "chat_messages":
                params = stmt.compile().params
                self.messages.append(
                    {
                        "session_id": params["session_id"],
                        "role": params["role"],
                        "content": params["content"],
                    }
                )
                return FakeResult()
            return FakeResult()
        if isinstance(stmt, Select):
            tables = [f.name for f in stmt.get_final_froms()]
            if "chat_sessions" in tables:
                # Ownership / most-recent lookup — there is exactly one
                # conversation in this test, so report it when one exists.
                return FakeResult(scalar=self.sessions[-1] if self.sessions else None)
            if "chat_messages" in tables:
                rows = [
                    SimpleNamespace(role=m["role"], content=m["content"])
                    for m in self.messages
                ]
                return FakeResult(rows=rows)
            return FakeResult()
        return FakeResult()

    async def __aenter__(self) -> _RecordingDb:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubLLM:
    """Scripted generation provider (only the RAG turn calls it)."""

    name = "test-provider"
    model = "test-model"

    async def stream(
        self, messages: list[Message], *, completion: Completion
    ) -> AsyncIterator[str]:
        completion.provider = self.name
        completion.model = self.model
        completion.text = "Remote work is allowed two days per week."
        completion.usage = TokenUsage(prompt_tokens=5, completion_tokens=5)
        yield completion.text


async def _router(*, query: str, history: list | None = None, **kw: Any) -> RouteResult:
    """Deterministic intent router: the second message is out-of-scope."""
    if "planet" in query.lower():
        return RouteResult(route="OUT_OF_SCOPE", confidence=0.9, reasoning="test")
    return RouteResult(route="DOCUMENT_CONTENT", confidence=0.9, reasoning="test")


async def _retrieve(
    session: Any, *, query: str, workspace_id: uuid.UUID, **kwargs: Any
) -> RetrievalResult:
    chunk = RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename="handbook.pdf",
        content="Remote work is allowed two days per week.",
        page_number=1,
        section_title="Remote Work",
        chunk_index=0,
        rrf_score=0.1,
        rerank_score=0.9,
    )
    return RetrievalResult(chunks=[chunk], grounded=True, top_score=0.9)


@asynccontextmanager
async def _fake_rls_tenant_session(**kwargs: Any) -> AsyncIterator[None]:
    # Workspace-knowledge lookup inside classify_intent goes through the real
    # rls.tenant_session; yield a session whose reads are handled by the
    # patched get_workspace_knowledge below (which ignores it).
    yield None


async def _fake_get_workspace_knowledge(db: Any, workspace_id: uuid.UUID) -> Any:
    class _Knowledge:
        def to_prompt_context(self) -> str:
            return ""

    return _Knowledge()


def _parse_sse(body: str) -> list[dict[str, str]]:
    """Parse the typed SSE envelope into (event, data) blocks."""
    events: list[dict[str, str]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data = line.split(":", 1)[1].strip()
        events.append({"event": ev, "data": data})
    return events


def _post(
    client: TestClient, payload: dict[str, Any]
) -> tuple[int, list[dict[str, str]]]:
    with client.stream("POST", "/chat", json=payload) as resp:
        body = resp.read().decode()
        return resp.status_code, _parse_sse(body)


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


def test_stream_persistence_single_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two /chat turns → exactly 1 ChatSession row with all 4 messages under it."""
    from app.main import create_app

    db = _RecordingDb()
    principal = Principal(user_id=uuid.uuid4(), workspace_id=uuid.uuid4())
    app = create_app()
    app.dependency_overrides[get_principal] = lambda: principal
    app.dependency_overrides[get_generic_llm] = lambda: _StubLLM()

    async def _member(*args: Any, **kwargs: Any) -> str:
        return "OWNER"

    monkeypatch.setattr(chat_module, "assert_workspace_role", _member)
    monkeypatch.setattr(chat_module, "tenant_session", lambda **kw: db)
    monkeypatch.setattr(chat_module, "retrieve", _retrieve)
    monkeypatch.setattr("app.retrieval.llm_router.route_with_llm", _router)
    # Keep classify_intent's workspace-knowledge lookup off the real database.
    monkeypatch.setattr(
        "app.security.rls.tenant_session", _fake_rls_tenant_session
    )
    monkeypatch.setattr(
        "app.retrieval.workspace_knowledge.get_workspace_knowledge",
        _fake_get_workspace_knowledge,
    )

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            # Turn 1: document-content question → RAG branch → creates session.
            status1, events1 = _post(client, {"message": MSG_DOC})
            assert status1 == 200
            session_events = [e for e in events1 if e["event"] == "session"]
            assert len(session_events) == 1, "first turn must announce the new session"
            session_id = json.loads(session_events[0]["data"])["session_id"]
            assert [e["event"] for e in events1].count("done") == 1

            # Turn 2: out-of-scope question → non-document branch → must reuse
            # the same session (frontend sends the adopted session_id back).
            status2, events2 = _post(
                client, {"message": MSG_OOS, "session_id": session_id}
            )
            assert status2 == 200
            assert not [e for e in events2 if e["event"] == "session"], (
                "continuation turn must NOT emit another session event"
            )
            assert [e["event"] for e in events2].count("done") == 1
    finally:
        app.dependency_overrides.clear()

    # Exactly one ChatSession row, and every message under it.
    assert db.sessions == [uuid.UUID(session_id)], (
        f"expected exactly 1 ChatSession row, got {len(db.sessions)}"
    )
    assert len(db.messages) == 4, f"expected 4 messages, got {len(db.messages)}"
    assert all(m["session_id"] == uuid.UUID(session_id) for m in db.messages)
    assert [m["role"] for m in db.messages] == [
        "user", "assistant", "user", "assistant",
    ]
    assert db.messages[0]["content"] == MSG_DOC
    assert db.messages[2]["content"] == MSG_OOS
    assert all(m["content"] for m in db.messages)

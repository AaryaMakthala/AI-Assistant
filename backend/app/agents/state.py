"""The state one question carries through the supervisor graph (Phase 7).

Every node reads this and returns a partial update; LangGraph merges the updates. Two
details in here are load-bearing rather than incidental:

**Each sub-agent owns its own output key.** `rag`, `sql` and `mcp` run concurrently after
routing, and LangGraph raises `InvalidUpdateError` if two concurrent nodes write the same
key without a reducer. Giving each its own slot means the fan-out needs no locking and no
reducer that would have to guess how to combine two agents' findings.

**`steps` is the audit trail, and it accumulates.** It carries the `operator.add` reducer so
concurrent branches append rather than overwrite. The UI's "thinking/routing" indicator
(CLAUDE.md section 6) reads it, and so does the log line for a request that went wrong.

`Principal` is carried in state rather than read from a global: the graph is driven from a
request handler that already verified a JWT, and every scoped query underneath needs the
same identity. It is never derived from anything the model produced (CLAUDE.md 4.6).
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from app.llm.base import Message
from app.rag.retrieval import RetrievedChunk
from app.security.auth import Principal

#: The three capabilities the supervisor can route to. `direct` is the fourth outcome and
#: not a sub-agent: it means no source is needed (a greeting, a question about the
#: assistant itself), so no retrieval or query is run at all.
Route = Literal["documents", "business_data", "external", "direct"]

ROUTES: tuple[Route, ...] = ("documents", "business_data", "external", "direct")

#: Route → the node that serves it.
ROUTE_NODES: dict[Route, str] = {
    "documents": "rag_agent",
    "business_data": "sql_agent",
    "external": "mcp_agent",
}


@dataclass
class AgentOutcome:
    """What one sub-agent produced, in a shape the synthesis node can read uniformly.

    `ok` is not simply `bool(content)`: an agent that correctly found nothing is a success
    with no content, while one that failed is a failure — and synthesis has to tell the
    difference to avoid presenting an outage as an absence of data.
    """

    #: Which agent this came from, for the trace and for error messages.
    source: str
    ok: bool = False
    #: Evidence for the synthesis prompt. Already fenced as untrusted where it originated
    #: outside the system (CLAUDE.md 4.4).
    content: str = ""
    #: Operator-facing reason this agent produced nothing. Safe to show a user.
    detail: str = ""

    @property
    def has_content(self) -> bool:
        return bool(self.content.strip())


class SupervisorState(TypedDict, total=False):
    """The graph's working memory for one question."""

    # --- Inputs, set once before the graph runs ---
    question: str
    principal: Principal
    history: list[Message]

    # --- Routing ---
    routes: list[Route]
    route_reason: str
    #: Incremented by the routing node on every pass. The max-iteration guard reads this;
    #: see `app/agents/graph.py`.
    iterations: int

    # --- Sub-agent results. One key each, so the fan-out never collides. ---
    rag: AgentOutcome
    sql: AgentOutcome
    mcp: AgentOutcome
    #: Chunks the RAG agent retrieved, kept separately so citations resolve against the
    #: real retrieved set rather than against anything the model claimed.
    chunks: list[RetrievedChunk]

    # --- Output ---
    answer: str
    #: Human-readable trace of what ran, in order. Appended to by every node.
    steps: Annotated[list[str], operator.add]


def initial_state(
    *, question: str, principal: Principal, history: list[Message] | None = None
) -> SupervisorState:
    """A fresh state for one question.

    Every accumulating key is initialized explicitly. LangGraph tolerates a missing key on
    a `total=False` TypedDict, but a node that does `state["steps"] + [...]` would then
    raise a KeyError on the first pass rather than on some later one.
    """
    return SupervisorState(
        question=question,
        principal=principal,
        history=history or [],
        routes=[],
        route_reason="",
        iterations=0,
        chunks=[],
        answer="",
        steps=[],
    )


def outcomes(state: SupervisorState) -> list[AgentOutcome]:
    """Every sub-agent result present in `state`, in a stable order."""
    return [state[key] for key in ("rag", "sql", "mcp") if state.get(key) is not None]  # type: ignore[literal-required]


__all__ = [
    "ROUTES",
    "ROUTE_NODES",
    "AgentOutcome",
    "Route",
    "SupervisorState",
    "initial_state",
    "outcomes",
]

"""LangGraph multi-agent orchestration (Phase 7).

A supervisor routes one question to the RAG, SQL and MCP sub-agents — one, several, or none —
and then synthesizes a single answer from whatever they returned.

    START → router ─┬→ rag_agent ─┐
                    ├→ sql_agent ─┼→ synthesis → END
                    ├→ mcp_agent ─┘
                    └────────────→ (direct: no lookup needed)

The security boundaries from earlier phases are reused rather than re-implemented, which is
the point of the wrapper design in `sub_agents.py`:

- SQL still passes through Phase 5's sqlglot validation, allowlist, read-only role, row cap,
  timeout and audit log. This package adds no SQL path.
- MCP tools stay read-only, and identity is bound out of band via `acting_as`, never as a
  tool argument (CLAUDE.md 4.6).
- Everything a sub-agent returns is fenced as untrusted data before it reaches the synthesis
  prompt, and the router — the component that decides which tools run — never sees retrieved
  content at all (CLAUDE.md 4.4).
- Termination is a counter in state plus LangGraph's `recursion_limit`, not a prompt
  instruction (CLAUDE.md section 7).

Import `app.agents.graph` for the runnable entry points; the submodules are imported directly
so that pulling in the supervisor does not drag in sentence-transformers via the RAG path
until a caller actually needs it.
"""

from app.agents.state import (
    ROUTE_NODES,
    ROUTES,
    AgentOutcome,
    Route,
    SupervisorState,
    initial_state,
    outcomes,
)

__all__ = [
    "ROUTES",
    "ROUTE_NODES",
    "AgentOutcome",
    "Route",
    "SupervisorState",
    "initial_state",
    "outcomes",
]

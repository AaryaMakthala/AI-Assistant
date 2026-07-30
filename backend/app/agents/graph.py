"""The supervisor graph: assembly and execution (Phase 7).

Shape — router fans out to whichever sub-agents were selected, all of them converge on
synthesis:

    START → router ─┬→ rag_agent ─┐
                    ├→ sql_agent ─┼→ synthesis → END
                    ├→ mcp_agent ─┘
                    └────────────→ (straight to synthesis for `direct`)

Acyclic by construction. A supervisor that can re-route after seeing results is the usual next
step, and deliberately not taken here: it would mean the component choosing tools gets to read
retrieved content, which CLAUDE.md 4.4 forbids. The iteration counter in `supervisor.py` and
the `recursion_limit` set below both exist so that adding a cycle later cannot loop forever.

The graph is compiled once and cached. Compilation validates the topology, so doing it per
request would burn CPU re-proving something that cannot change between requests. Nothing
request-scoped is captured: the LLM router is bound per node via a closure over the argument
passed to :func:`build_graph`, and identity travels in state.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.agents.state import SupervisorState, initial_state
from app.agents.sub_agents import mcp_agent, rag_agent, sql_agent
from app.agents.supervisor import route_to_agents, routing_node, synthesis_node
from app.config import get_settings
from app.llm.base import LLMRouterProtocol, Message
from app.security.auth import Principal

#: Every node the router may fan out to, plus synthesis. LangGraph needs the full set of
#: possible destinations for a conditional edge so it can validate the graph up front.
_FAN_OUT_TARGETS = ["rag_agent", "sql_agent", "mcp_agent", "synthesis"]


def _write_token(token: str) -> None:
    """Push one synthesized token onto LangGraph's custom stream channel.

    Looked up lazily inside the call: `get_stream_writer()` resolves the writer for the node
    currently executing, so it must run inside that node rather than at graph-build time. When
    the graph is invoked rather than streamed there is no writer, and the token is dropped —
    which is correct, since nobody is listening.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"type": "token", "text": token})
    except RuntimeError:
        # No active stream writer: this is an ainvoke() run, not a stream.
        pass


def build_graph(llm: LLMRouterProtocol, *, emit: Callable[[str], None] | None = None) -> Any:
    """Compile the supervisor graph.

    `emit`, when given, receives each synthesized token as it arrives. Passed in at build time
    rather than read from state because LangGraph serializes state and a callable is not
    serializable.

    Nodes are bound with explicit `async def` closures rather than `functools.partial`.
    LangGraph inspects each node's signature to decide what to pass it, and `inspect.signature`
    rejects a partial whose keyword does not match the wrapped function — so a partial turns a
    signature mismatch into an obscure error at graph-build time. A closure takes exactly the
    one argument LangGraph supplies, which is also what makes these nodes substitutable in
    isolation.
    """
    graph = StateGraph(SupervisorState)

    async def router_node(state: SupervisorState) -> dict[str, object]:
        return await routing_node(state, llm)

    async def rag_node(state: SupervisorState) -> dict[str, object]:
        return await rag_agent(state, llm)

    async def sql_node(state: SupervisorState) -> dict[str, object]:
        return await sql_agent(state, llm)

    async def mcp_node(state: SupervisorState) -> dict[str, object]:
        return await mcp_agent(state, llm)

    async def synth_node(state: SupervisorState) -> dict[str, object]:
        return await synthesis_node(state, llm, emit=emit)

    graph.add_node("router", router_node)
    graph.add_node("rag_agent", rag_node)
    graph.add_node("sql_agent", sql_node)
    graph.add_node("mcp_agent", mcp_node)
    graph.add_node("synthesis", synth_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_to_agents, _FAN_OUT_TARGETS)
    for node in ("rag_agent", "sql_agent", "mcp_agent"):
        graph.add_edge(node, "synthesis")
    graph.add_edge("synthesis", END)

    return graph.compile()


async def run_supervisor(
    llm: LLMRouterProtocol,
    *,
    question: str,
    principal: Principal,
    history: list[Message] | None = None,
    emit: Callable[[str], None] | None = None,
) -> SupervisorState:
    """Answer one question through the graph, returning the final state.

    Raises only what synthesis raises — a sub-agent failure is recorded in state as a failed
    outcome instead, so a partly-successful answer still reaches the user.
    """
    settings = get_settings()
    graph = build_graph(llm, emit=emit)

    final: SupervisorState = await graph.ainvoke(
        initial_state(question=question, principal=principal, history=history),
        # Independent of the state counter in supervisor.py, and cheap insurance against a
        # topology mistake in a future phase.
        config={"recursion_limit": settings.agent_recursion_limit},
    )

    logger.info(
        "Supervisor finished for user {user}: {steps}",
        user=principal.user_id,
        steps=" | ".join(final.get("steps", [])),
    )
    return final


async def stream_supervisor(
    llm: LLMRouterProtocol,
    *,
    question: str,
    principal: Principal,
    history: list[Message] | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Run the graph, yielding `(kind, payload)` as it progresses.

    Kinds:
      - `step`   — one trace line, as a node completes; drives the UI's routing indicator
      - `token`  — one token of the final answer
      - `final`  — the completed state, once

    Tokens reach the caller through LangGraph's own custom-stream channel rather than a queue
    this module drains between state snapshots. That distinction is the difference between real
    streaming and the appearance of it: draining a queue only when a node boundary produces a new
    snapshot would hold every token until synthesis had finished, then release them in one burst.
    """
    graph = build_graph(llm, emit=_write_token)
    settings = get_settings()

    seen_steps = 0
    final_state: SupervisorState | None = None

    async for mode, chunk in graph.astream(
        initial_state(question=question, principal=principal, history=history),
        config={"recursion_limit": settings.agent_recursion_limit},
        # Both channels on one iterator, so tokens and trace lines arrive in the order they
        # actually happened.
        stream_mode=["custom", "values"],
    ):
        if mode == "custom":
            if isinstance(chunk, dict) and chunk.get("type") == "token":
                yield "token", chunk["text"]
            continue

        state: SupervisorState = chunk
        # Trace lines explain what the tokens that follow were built from, so they go first.
        steps = state.get("steps", [])
        for step in steps[seen_steps:]:
            yield "step", step
        seen_steps = len(steps)
        final_state = state

    if final_state is not None:
        yield "final", final_state


__all__ = ["build_graph", "run_supervisor", "stream_supervisor"]

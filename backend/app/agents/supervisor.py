"""The supervisor: routing, synthesis, and the iteration guard (Phase 7).

The supervisor decides *which* sources answer a question and then writes the final answer over
whatever they returned. It never queries anything itself.

**Routing is a classification, not an agent loop.** The router sees the user's question and
emits a source list; it cannot call tools, and it cannot see retrieved content. That is the
structural half of CLAUDE.md 4.4 — the component choosing which tools run is blind to the text
that would want to influence that choice. An unparseable or empty reply falls back to
`documents`, the narrowest useful default, rather than to everything.

**The iteration guard is a hard counter, not a prompt instruction** (CLAUDE.md section 7, "agent
loops forever"). Two independent limits:

1. `iterations` in state, incremented by the routing node and checked before any fan-out. Past
   `agent_max_iterations`, the graph is forced to synthesize with what it has.
2. LangGraph's own `recursion_limit` on the compiled graph, as a backstop in case a future edge
   creates a cycle this module's counter does not see.

The current graph is acyclic, so limit 1 is never reached today. It exists because the cheap
time to add a termination condition is before the graph has cycles, not after.
"""

from __future__ import annotations

from loguru import logger

from app.agents.prompts import (
    build_direct_messages,
    build_router_messages,
    build_synthesis_messages,
    parse_routes,
)
from app.agents.state import ROUTE_NODES, ROUTES, AgentOutcome, Route, SupervisorState, outcomes
from app.config import get_settings
from app.llm.base import Completion, LLMError, LLMRouterProtocol
from app.rag.pipeline import resolve_citations

#: Used when the router's reply cannot be parsed. Documents are the broadest knowledge source
#: and the cheapest wrong guess: a search that finds nothing costs one embedding, whereas
#: defaulting to every source runs a SQL generation and a GitHub call for a greeting.
_FALLBACK_ROUTE: Route = "documents"


async def routing_node(state: SupervisorState, llm: LLMRouterProtocol) -> dict[str, object]:
    """Classify the question into one or more sources.

    Non-streaming: a partial classification is meaningless, and the reply is one short line.
    """
    iterations = state.get("iterations", 0) + 1
    question = state["question"]
    settings = get_settings()

    if iterations > settings.agent_max_iterations:
        # Guard tripped. Reached only if a future edge routes back here; the answer is built
        # from whatever the earlier passes gathered rather than looping again.
        logger.warning(
            "Supervisor iteration guard tripped at {n} for user {user}",
            n=iterations,
            user=state["principal"].user_id,
        )
        return {
            "iterations": iterations,
            "routes": [],
            "route_reason": "iteration limit reached",
            "steps": [f"router: iteration limit ({settings.agent_max_iterations}) reached"],
        }

    messages = build_router_messages(question=question, history=state.get("history") or [])

    try:
        completion = Completion()
        text = ""
        async for token in llm.stream(messages, completion=completion):
            text += token
    except LLMError as exc:
        # Without a router the question is still answerable from documents. Failing the whole
        # request because the *classifier* was unavailable would be a needless outage.
        logger.warning("Routing failed, defaulting to documents: {err}", err=exc)
        return {
            "iterations": iterations,
            "routes": [_FALLBACK_ROUTE],
            "route_reason": "router unavailable; defaulted to documents",
            "steps": ["router: unavailable, defaulted to documents"],
        }

    candidates, reason = parse_routes(text)
    # Matched exactly against the known routes: the model's output selects from a fixed set
    # and can never name a node that does not exist.
    routes = [name for name in ROUTES if name in candidates]

    if not routes:
        logger.info("Unparseable routing reply {reply!r}; defaulting", reply=text[:120])
        routes = [_FALLBACK_ROUTE]
        reason = reason or "could not classify; defaulted to documents"

    # 'direct' is exclusive. If the model paired it with a real source it misunderstood the
    # question, and the real source is the safer half to keep.
    if "direct" in routes and len(routes) > 1:
        routes = [name for name in routes if name != "direct"]

    logger.info(
        "Routed question for user {user} to {routes}",
        user=state["principal"].user_id,
        routes=routes,
    )
    return {
        "iterations": iterations,
        "routes": routes,
        "route_reason": reason,
        "steps": [f"router: {', '.join(routes)}" + (f" ({reason})" if reason else "")],
    }


def route_to_agents(state: SupervisorState) -> list[str]:
    """Which sub-agent nodes to run next — LangGraph's conditional-edge function.

    Returning a list is how LangGraph fans out: every named node runs concurrently, and the
    graph converges on synthesis afterwards.
    """
    routes = state.get("routes") or []
    nodes = [ROUTE_NODES[route] for route in routes if route in ROUTE_NODES]
    # 'direct', an empty list, or a tripped guard all mean "nothing to gather" — go straight
    # to the node that writes the answer.
    return nodes or ["synthesis"]


async def synthesis_node(
    state: SupervisorState, llm: LLMRouterProtocol, *, emit: object = None
) -> dict[str, object]:
    """Write the final answer over whatever the sub-agents gathered.

    Streams through `emit` when one is supplied — a callable taking one token — so the API layer
    can forward tokens to the client as they arrive. `emit` is passed in rather than reaching for
    LangGraph's stream writer directly so this node stays callable outside a graph run.

    Synthesis has no tool access. It is the last node, and the material it reads is untrusted, so
    the two facts have to line up: nothing it produces can cause another lookup.

    Citations are resolved here rather than by the RAG node, because only this node knows the
    final text. They are resolved against `chunks` — what was actually retrieved and sent — so a
    bracketed number the model invented resolves to nothing instead of rendering as a source
    (the Phase 4 guarantee, preserved through the graph).
    """
    question = state["question"]
    results = [outcome for outcome in outcomes(state) if outcome is not None]
    history = state.get("history") or []
    routes = state.get("routes") or []
    chunks = state.get("chunks") or []

    # `direct` means the router judged that no lookup was needed. A different prompt, because
    # the synthesis prompt's rules are written around supplied evidence that does not exist here.
    if not results and "direct" in routes:
        messages = build_direct_messages(question=question, history=history)
        step = "synthesis: direct reply, no sources consulted"
    else:
        messages = build_synthesis_messages(question=question, results=results, history=history)
        step = f"synthesis: composed from {len(results)} source(s)"

    completion = Completion()
    text = ""
    try:
        async for token in llm.stream(messages, completion=completion):
            text += token
            if emit is not None:
                emit(token)  # type: ignore[operator]
    except LLMError:
        # Re-raised: unlike a sub-agent, this is the node that produces the answer. There is
        # nothing to degrade to, and inventing a reply here would be the worst option.
        logger.error("Synthesis failed for user {user}", user=state["principal"].user_id)
        raise

    citations = [citation.as_dict() for citation in resolve_citations(text, chunks)]

    return {
        "answer": text,
        "citations": citations,
        "steps": [step],
        # Carried out so the caller can report usage and provider without re-deriving them.
        # Declared in SupervisorState — an undeclared key would be dropped here silently.
        "completion": completion,
    }


def failure_outcome(source: str, detail: str) -> AgentOutcome:
    """A failed sub-agent result, for callers assembling state by hand."""
    return AgentOutcome(source=source, ok=False, detail=detail)


__all__ = [
    "failure_outcome",
    "route_to_agents",
    "routing_node",
    "synthesis_node",
]

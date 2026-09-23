"""MVP LangGraph conversation graph wiring.

Implements the routing matrix in `design.md` — "Routing matrix" and
"Active-flow routing": `router` decides intent (or defers to an active
flow), `consent_check` gates financial data access, `financial_analyst`/
`offer_simulator` do the actual work, `responder` drafts the reply, and
`compliance_guard` always runs last before `END`.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from apps.agent.llm.factory import LLMFactory
from apps.agent.nodes.compliance_guard import make_compliance_guard_node
from apps.agent.nodes.consent_check import make_consent_check_node
from apps.agent.nodes.financial_analyst import make_financial_analyst_node
from apps.agent.nodes.offer_simulator import make_offer_simulator_node
from apps.agent.nodes.responder import make_responder_node
from apps.agent.nodes.router import make_router_node
from apps.agent.repositories.customers import CustomerRepository
from apps.agent.state import ConversationState

_GATED_INTENTS = {"loan_simulation", "profile_analysis"}


def _route_after_router(state: ConversationState) -> str:
    active_flow = state.get("active_flow", "none")
    if active_flow == "consent_confirmation":
        return "consent_check"
    if active_flow == "slot_filling":
        return "offer_simulator"

    intent = state.get("intent")
    if intent in _GATED_INTENTS:
        return "consent_check"
    return "responder"


def _route_after_consent_check(state: ConversationState) -> str:
    if state.get("active_flow") == "consent_confirmation":
        return "responder"
    if state.get("consent_refused"):
        return "responder"

    intent = state.get("intent")
    if intent == "loan_simulation":
        return "offer_simulator"
    if intent == "profile_analysis":
        return "financial_analyst"
    return "responder"


def build_graph(
    llm_factory: LLMFactory,
    customer_repository: CustomerRepository,
    checkpointer: BaseCheckpointSaver[str] | None = None,
) -> CompiledStateGraph[ConversationState, None, ConversationState, ConversationState]:
    """Build (and compile) the MVP conversation graph."""
    graph: StateGraph[ConversationState, None, ConversationState, ConversationState] = StateGraph(
        ConversationState
    )

    # LangGraph's `add_node` overloads don't cleanly resolve against a
    # `TypedDict` node signature in this version's stubs, even though the
    # runtime behavior is correct (exercised by tests/agent/test_graph.py,
    # including the real-Postgres checkpointer test).
    graph.add_node("router", make_router_node(llm_factory))  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "consent_check", make_consent_check_node(customer_repository)
    )
    graph.add_node(  # type: ignore[call-overload]
        "financial_analyst", make_financial_analyst_node(customer_repository)
    )
    graph.add_node(  # type: ignore[call-overload]
        "offer_simulator", make_offer_simulator_node(llm_factory)
    )
    graph.add_node("responder", make_responder_node(llm_factory))  # type: ignore[call-overload]
    graph.add_node(  # type: ignore[call-overload]
        "compliance_guard", make_compliance_guard_node(llm_factory)
    )

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "consent_check": "consent_check",
            "offer_simulator": "offer_simulator",
            "responder": "responder",
        },
    )
    graph.add_conditional_edges(
        "consent_check",
        _route_after_consent_check,
        {
            "responder": "responder",
            "offer_simulator": "offer_simulator",
            "financial_analyst": "financial_analyst",
        },
    )
    graph.add_edge("financial_analyst", "responder")
    graph.add_edge("offer_simulator", "responder")
    graph.add_edge("responder", "compliance_guard")
    graph.add_edge("compliance_guard", END)

    return graph.compile(checkpointer=checkpointer)


NodeName = Literal[
    "router",
    "consent_check",
    "financial_analyst",
    "offer_simulator",
    "responder",
    "compliance_guard",
]
GRAPH_NODE_ORDER: tuple[NodeName, ...] = (
    "router",
    "consent_check",
    "financial_analyst",
    "offer_simulator",
    "responder",
    "compliance_guard",
)
"""Every possible node name, for the API layer to enumerate `node_started`/
`node_finished` events without importing graph internals (task group 8)."""

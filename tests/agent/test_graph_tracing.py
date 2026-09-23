"""End-to-end: tracing wired into a real graph turn via `LLMFactory(enable_tracing=True)`."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from apps.agent.graph import build_graph
from apps.agent.llm.factory import LLMFactory, Tier
from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.port import LLMPort
from apps.agent.llm.settings import Settings
from apps.agent.nodes.compliance_guard import ApprovalPromiseCheck
from apps.agent.nodes.router import RouterDecision
from apps.agent.observability.tracing import FakeTracer, reset_current_turn, set_current_turn
from apps.agent.state import ConversationState
from tests.agent.nodes.fakes import StubCustomerRepository, fake_knowledge_agent_node


class _TracedScriptedLLMFactory(LLMFactory):
    """Like `ScriptedLLMFactory`, but with `enable_tracing=True` so
    `for_node`/`fallback_for_node` actually wrap results in `TracedLLM`."""

    def __init__(self, fast: FakeLLM, smart: FakeLLM) -> None:
        super().__init__(Settings(llm_provider="fake"), enable_tracing=True)
        self._fast = fast
        self._smart = smart

    def for_alias(self, tier: Tier) -> LLMPort:
        return self._fast if tier == "fast" else self._smart


async def test_out_of_scope_turn_produces_one_node_span_per_node_and_one_llm_span_per_call() -> (
    None
):
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="out_of_scope"),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    factory = _TracedScriptedLLMFactory(fast=fast_llm, smart=FakeLLM())
    app = build_graph(
        factory, StubCustomerRepository(None), fake_knowledge_agent_node, checkpointer=MemorySaver()
    )

    tracer = FakeTracer()
    config: RunnableConfig = {"configurable": {"thread_id": "trace-turn-1"}}

    with tracer.turn("trace-turn-1") as turn:
        token = set_current_turn(turn)
        try:
            await app.ainvoke(
                ConversationState(
                    customer_id="cust-1", messages=[HumanMessage(content="qual a capital?")]
                ),
                config=config,
            )
        finally:
            reset_current_turn(token)

    # out_of_scope short-circuits straight to responder: router -> responder -> compliance_guard.
    assert turn.node_span_count == 3
    assert {span.name for span in turn.spans if span.kind == "node"} == {
        "router",
        "responder",
        "compliance_guard",
    }
    # router makes one structured LLM call; responder makes no LLM call for
    # out-of-scope (fixed template, no framing call needed); compliance_guard
    # makes one (the approval-promise check).
    assert turn.llm_span_count == 2
    assert len(tracer.turns) == 1

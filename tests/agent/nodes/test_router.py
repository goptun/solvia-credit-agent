"""Router: intent classification, and active-flow skip routing."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.router import RouterDecision, make_router_node
from apps.agent.state import ConversationState, initial_state
from tests.agent.nodes.fakes import ScriptedLLMFactory


def _state_with_message(text: str, **overrides: object) -> ConversationState:
    state = initial_state("cust-0001")
    state["messages"] = [HumanMessage(content=text)]
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


@pytest.mark.parametrize(
    "intent",
    [
        "product_question",
        "regulatory_question",
        "loan_simulation",
        "profile_analysis",
        "complaint",
        "out_of_scope",
    ],
)
async def test_router_classifies_each_intent_branch(intent: str) -> None:
    fast_llm = FakeLLM(responses=[RouterDecision(intent=intent)])  # type: ignore[arg-type]
    node = make_router_node(ScriptedLLMFactory(fast=fast_llm))

    updates = await node(_state_with_message("mensagem qualquer"))

    assert updates["intent"] == intent


async def test_router_classifies_a_regulatory_sounding_message() -> None:
    fast_llm = FakeLLM(responses=[RouterDecision(intent="regulatory_question")])
    node = make_router_node(ScriptedLLMFactory(fast=fast_llm))
    message = "Quais são meus direitos segundo o código de defesa do consumidor?"

    updates = await node(_state_with_message(message))

    assert updates["intent"] == "regulatory_question"


async def test_active_flow_consent_confirmation_skips_reclassification() -> None:
    fast_llm = FakeLLM(responses=[RouterDecision(intent="out_of_scope", is_new_request=False)])
    node = make_router_node(ScriptedLLMFactory(fast=fast_llm))
    state = _state_with_message(
        "sim, autorizo", active_flow="consent_confirmation", pending_intent="loan_simulation"
    )

    updates = await node(state)

    assert updates == {}


async def test_active_flow_slot_filling_skips_reclassification() -> None:
    fast_llm = FakeLLM(responses=[RouterDecision(intent="out_of_scope", is_new_request=False)])
    node = make_router_node(ScriptedLLMFactory(fast=fast_llm))
    state = _state_with_message(
        "24 meses", active_flow="slot_filling", pending_intent="loan_simulation"
    )

    updates = await node(state)

    assert updates == {}


async def test_new_request_signal_clears_the_active_flow() -> None:
    fast_llm = FakeLLM(responses=[RouterDecision(intent="product_question", is_new_request=True)])
    node = make_router_node(ScriptedLLMFactory(fast=fast_llm))
    state = _state_with_message(
        "na verdade, quais produtos voces tem?",
        active_flow="consent_confirmation",
        pending_intent="loan_simulation",
    )

    updates = await node(state)

    assert updates["intent"] == "product_question"
    assert updates["active_flow"] == "none"
    assert updates["pending_intent"] is None

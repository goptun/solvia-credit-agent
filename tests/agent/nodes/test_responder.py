"""responder: per-intent reply drafting, with numbers always templated."""

from __future__ import annotations

from decimal import Decimal

from langchain_core.messages import AIMessage

from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.responder import make_responder_node
from apps.agent.state import (
    ConversationState,
    CustomerProfileSummary,
    SimulationSummary,
    initial_state,
)
from tests.agent.nodes.fakes import ScriptedLLMFactory


def _reply(state: ConversationState) -> str:
    draft_reply = state.get("draft_reply")
    assert draft_reply is not None
    return draft_reply


async def test_consent_ask_reply_requests_authorization() -> None:
    node = make_responder_node(ScriptedLLMFactory())
    state = initial_state("cust-1")
    state["consent_prompt"] = "ask"
    state["pending_intent"] = "loan_simulation"

    updates = await node(state)

    assert "autoriz" in _reply(updates).lower()


async def test_consent_reask_reply_is_the_fixed_reask_message() -> None:
    node = make_responder_node(ScriptedLLMFactory())
    state = initial_state("cust-1")
    state["consent_prompt"] = "reask"

    updates = await node(state)

    assert "não entendi" in _reply(updates).lower()


async def test_consent_refused_reply_acknowledges_and_offers_alternative() -> None:
    node = make_responder_node(ScriptedLLMFactory())
    state = initial_state("cust-1")
    state["consent_refused"] = True

    updates = await node(state)

    assert "sem a autorização" in _reply(updates).lower()


async def test_out_of_scope_reply_is_a_fixed_polite_refusal() -> None:
    node = make_responder_node(ScriptedLLMFactory())
    state = initial_state("cust-1")
    state["intent"] = "out_of_scope"

    updates = await node(state)

    assert "fora do que posso ajudar" in _reply(updates)


async def test_complaint_reply_mentions_future_human_handoff() -> None:
    node = make_responder_node(ScriptedLLMFactory())
    state = initial_state("cust-1")
    state["intent"] = "complaint"

    updates = await node(state)

    assert "futura" in _reply(updates)


async def test_missing_slot_reply_asks_the_targeted_question() -> None:
    node = make_responder_node(ScriptedLLMFactory())
    state = initial_state("cust-1")
    state["next_missing_slot"] = "term_months"

    updates = await node(state)

    assert "meses" in _reply(updates).lower()


async def test_loan_simulation_reply_renders_numbers_via_template_never_via_the_llm() -> None:
    smart_llm = FakeLLM(responses=[AIMessage(content="Aqui está o resultado da sua simulação:")])
    node = make_responder_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["intent"] = "loan_simulation"
    state["simulation_result"] = SimulationSummary(
        amortization_type="PRICE",
        principal=Decimal("1000.00"),
        term_months=12,
        cet_annual=Decimal("0.35"),
        first_installment=Decimal("95.00"),
        total_paid=Decimal("1140.00"),
    )

    updates = await node(state)

    reply = _reply(updates)
    assert "Aqui está o resultado" in reply
    assert "1.140,00" in reply
    assert smart_llm.calls
    for call_messages in smart_llm.calls:
        for message in call_messages:
            assert "1140" not in str(message.content)
            assert "1000" not in str(message.content)


async def test_profile_analysis_reply_renders_numbers_via_template() -> None:
    smart_llm = FakeLLM(responses=[AIMessage(content="Aqui está o seu resumo financeiro:")])
    node = make_responder_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["intent"] = "profile_analysis"
    state["customer_profile"] = CustomerProfileSummary(
        income=Decimal("3000.00"),
        debt_to_income_ratio=Decimal("0.25"),
        spending_categories={"FOOD": Decimal("-400.00")},
    )

    updates = await node(state)

    reply = _reply(updates)
    assert "3.000,00" in reply
    assert "25,00%" in reply

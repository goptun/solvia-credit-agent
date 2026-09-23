"""Full graph wiring: routing paths end-to-end (fake LLM, in-memory
checkpointer) and a real-Postgres integration test for resume-after-
restart (task 7.9)."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from apps.agent.checkpointer import postgres_checkpointer
from apps.agent.graph import build_graph
from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.compliance_guard import ApprovalPromiseCheck
from apps.agent.nodes.offer_simulator import SlotExtraction
from apps.agent.nodes.router import RouterDecision
from apps.agent.state import ConversationState
from apps.agent.synthetic_data.models import (
    Consent,
    ConsentStatus,
    Customer,
    CustomerProfile,
)
from tests.agent.nodes.fakes import ScriptedLLMFactory, StubCustomerRepository

_DATABASE_URL = os.environ.get("DATABASE_URL")


def _customer(status: ConsentStatus) -> Customer:
    return Customer(
        customer_id="cust-1",
        name="Ana Souza",
        cpf="111.222.333-44",
        profile=CustomerProfile.SALARIED,
        accounts=[],
        credit_cards=[],
        transactions=[],
        consent=Consent(
            consent_id="consent-1",
            customer_id="cust-1",
            status=status,
            scope=[],
            granted_at=None,
            expires_at=None,
        ),
    )


def _reply(state: ConversationState | dict[str, Any]) -> str:
    draft_reply = state.get("draft_reply")
    assert draft_reply is not None
    return draft_reply


async def test_out_of_scope_flow_end_to_end() -> None:
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="out_of_scope"),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    factory = ScriptedLLMFactory(fast=fast_llm)
    app = build_graph(factory, StubCustomerRepository(None), checkpointer=MemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "t-oos"}}
    result = await app.ainvoke(
        ConversationState(
            customer_id="cust-1",
            messages=[HumanMessage(content="qual a capital da frança?")],
        ),
        config=config,
    )

    reply = _reply(result)
    assert "fora do que posso ajudar" in reply
    assert "demonstração" not in reply  # no disclaimer outside sim/analysis


async def test_loan_simulation_with_valid_consent_completes_in_one_turn() -> None:
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="loan_simulation"),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    smart_llm = FakeLLM(
        responses=[
            SlotExtraction(amount=Decimal("3000"), term_months=6, amortization_type="PRICE"),
            AIMessage(content="Aqui está sua simulação:"),
        ]
    )
    factory = ScriptedLLMFactory(fast=fast_llm, smart=smart_llm)
    repo = StubCustomerRepository(_customer(ConsentStatus.VALID))
    app = build_graph(factory, repo, checkpointer=MemorySaver())

    config: RunnableConfig = {"configurable": {"thread_id": "t-sim"}}
    result = await app.ainvoke(
        ConversationState(
            customer_id="cust-1",
            messages=[HumanMessage(content="quero 3000 reais em 6 meses, Price")],
        ),
        config=config,
    )

    reply = _reply(result)
    assert "Aqui está sua simulação" in reply
    assert "CET" in reply
    assert "demonstração" in reply  # disclaimer present for simulations


async def test_missing_consent_then_confirmation_resumes_the_loan_simulation() -> None:
    # fast_llm backs both `router` and `compliance_guard` (both fast-tier),
    # so responses must be queued in real call order across both turns:
    # turn 1 -> router, compliance_guard; turn 2 -> router, compliance_guard.
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="loan_simulation"),
            ApprovalPromiseCheck(promises_approval=False),
            RouterDecision(intent="loan_simulation", is_new_request=False),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    smart_llm = FakeLLM(
        responses=[
            SlotExtraction(amount=Decimal("1000"), term_months=12, amortization_type="SAC"),
            AIMessage(content="Segue sua simulação:"),
        ]
    )
    factory = ScriptedLLMFactory(fast=fast_llm, smart=smart_llm)
    repo = StubCustomerRepository(_customer(ConsentStatus.MISSING))
    app = build_graph(factory, repo, checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": "t-consent"}}

    first_turn = await app.ainvoke(
        ConversationState(
            customer_id="cust-1",
            messages=[HumanMessage(content="quero simular um empréstimo")],
        ),
        config=config,
    )
    assert "autoriz" in _reply(first_turn).lower()
    assert first_turn["active_flow"] == "consent_confirmation"

    second_turn = await app.ainvoke(
        ConversationState(customer_id="cust-1", messages=[HumanMessage(content="sim, autorizo")]),
        config=config,
    )

    assert second_turn["consent_status"] == "just_granted"
    assert "Segue sua simulação" in _reply(second_turn)
    assert second_turn["active_flow"] == "none"


@pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")
async def test_postgres_checkpointer_resumes_after_a_simulated_process_restart() -> None:
    assert _DATABASE_URL is not None
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="loan_simulation"),
            ApprovalPromiseCheck(promises_approval=False),
            RouterDecision(intent="loan_simulation", is_new_request=False),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    smart_llm = FakeLLM(
        responses=[
            SlotExtraction(amount=Decimal("2000"), term_months=24, amortization_type="PRICE"),
            AIMessage(content="Aqui está:"),
        ]
    )
    factory = ScriptedLLMFactory(fast=fast_llm, smart=smart_llm)
    repo = StubCustomerRepository(_customer(ConsentStatus.MISSING))
    config: RunnableConfig = {"configurable": {"thread_id": "t-restart-1"}}

    async with postgres_checkpointer(_DATABASE_URL) as checkpointer:
        # "Process 1": start the conversation, hit the consent gate.
        app_before_restart = build_graph(factory, repo, checkpointer=checkpointer)
        first_turn = await app_before_restart.ainvoke(
            ConversationState(
                customer_id="cust-1",
                messages=[HumanMessage(content="quero simular um empréstimo")],
            ),
            config=config,
        )
        assert first_turn["active_flow"] == "consent_confirmation"

    # "Process 2": a brand-new graph instance and a brand-new checkpointer
    # connection — nothing here is shared in memory with process 1.
    async with postgres_checkpointer(_DATABASE_URL) as checkpointer_after_restart:
        app_after_restart = build_graph(factory, repo, checkpointer=checkpointer_after_restart)
        resumed_state = await app_after_restart.aget_state(config)

        assert resumed_state.values["customer_id"] == "cust-1"
        assert resumed_state.values["active_flow"] == "consent_confirmation"
        assert resumed_state.values["pending_intent"] == "loan_simulation"

        second_turn = await app_after_restart.ainvoke(
            ConversationState(
                customer_id="cust-1", messages=[HumanMessage(content="sim, autorizo")]
            ),
            config=config,
        )

        assert second_turn["customer_id"] == "cust-1"  # bound customer id survived the restart
        assert second_turn["consent_status"] == "just_granted"
        assert "Aqui está" in _reply(second_turn)

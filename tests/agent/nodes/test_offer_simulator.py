"""offer_simulator: slot extraction, follow-up questions, and the
deterministic simulation call."""

from __future__ import annotations

from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apps.agent.llm.errors import StructuredOutputError
from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.offer_simulator import SlotExtraction, make_offer_simulator_node
from apps.agent.state import SimulationSlots, initial_state
from tests.agent.nodes.fakes import ScriptedLLMFactory


async def test_incomplete_slots_trigger_a_follow_up_question() -> None:
    smart_llm = FakeLLM(responses=[SlotExtraction(amount=Decimal("5000"))])
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["messages"] = [HumanMessage(content="quero simular 5000 reais")]

    updates = await node(state)

    assert updates["active_flow"] == "slot_filling"
    assert updates["next_missing_slot"] in ("term_months", "amortization_type")
    assert "simulation_result" not in updates


async def test_slot_answer_is_merged_with_existing_slots_and_completes_the_simulation() -> None:
    smart_llm = FakeLLM(responses=[SlotExtraction(term_months=24)])
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["simulation_slots"] = SimulationSlots(amount=Decimal("5000"), amortization_type="PRICE")
    state["messages"] = [HumanMessage(content="24 meses")]

    updates = await node(state)

    assert updates["simulation_slots"].term_months == 24
    assert updates["simulation_result"] is not None
    assert updates["active_flow"] == "none"
    assert updates["pending_intent"] is None


async def test_llm_never_produces_the_numeric_simulation_result() -> None:
    smart_llm = FakeLLM(
        responses=[SlotExtraction(amount=Decimal("2000"), term_months=6, amortization_type="SAC")]
    )
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["messages"] = [HumanMessage(content="2000 reais em 6 meses, SAC")]

    updates = await node(state)

    result = updates["simulation_result"]
    assert result is not None
    # These numbers only exist because the deterministic `simulate()` tool
    # computed them — SlotExtraction (what the LLM returns) has no such fields.
    assert result.cet_annual > 0
    assert result.total_paid > result.principal
    assert not hasattr(SlotExtraction, "cet_annual")


async def test_slot_extraction_uses_the_json_fallback_when_the_model_returns_no_tool_call() -> None:
    extraction = SlotExtraction(amount=Decimal("5000"))
    smart_llm = FakeLLM(responses=[None, AIMessage(content=extraction.model_dump_json())])
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["messages"] = [HumanMessage(content="quero simular 5000 reais")]

    updates = await node(state)

    assert updates["simulation_slots"].amount == Decimal("5000")
    assert updates["active_flow"] == "slot_filling"


async def test_slot_extraction_raises_instead_of_using_none_when_both_paths_fail() -> None:
    smart_llm = FakeLLM(responses=[None, AIMessage(content="sem json")])
    fast_llm = FakeLLM(responses=[None, AIMessage(content="sem json")])
    node = make_offer_simulator_node(ScriptedLLMFactory(fast=fast_llm, smart=smart_llm))
    state = initial_state("cust-1")
    state["messages"] = [HumanMessage(content="quero simular")]

    with pytest.raises(StructuredOutputError):
        await node(state)


# --- fix: a missing amount/term must stay unset, never a placeholder 0 ------


def test_slot_extraction_normalizes_a_zero_amount_to_none() -> None:
    assert SlotExtraction(amount=Decimal("0")).amount is None


def test_slot_extraction_normalizes_a_negative_amount_to_none() -> None:
    assert SlotExtraction(amount=Decimal("-500")).amount is None


def test_slot_extraction_normalizes_a_zero_or_negative_term_to_none() -> None:
    assert SlotExtraction(term_months=0).term_months is None
    assert SlotExtraction(term_months=-6).term_months is None


def test_slot_extraction_normalizes_an_unsupported_amortization_type_to_none() -> None:
    assert SlotExtraction(amortization_type="FIBONACCI").amortization_type is None


def test_slot_extraction_still_accepts_a_valid_amortization_type_case_insensitively() -> None:
    assert SlotExtraction(amortization_type="price").amortization_type == "PRICE"


async def test_a_zero_amount_from_the_llm_does_not_overwrite_an_already_valid_slot() -> None:
    """Without normalization, `0 is not None` would make `_merge_slots`
    overwrite a slot already collected on an earlier turn with a bogus `0`."""
    smart_llm = FakeLLM(responses=[SlotExtraction(amount=Decimal("0"), term_months=24)])
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["simulation_slots"] = SimulationSlots(amount=Decimal("5000"), amortization_type="PRICE")
    state["messages"] = [HumanMessage(content="24 meses")]

    updates = await node(state)

    assert updates["simulation_slots"].amount == Decimal("5000")
    assert updates["simulation_result"] is not None


async def test_an_invalid_amortization_type_does_not_overwrite_an_already_valid_slot() -> None:
    smart_llm = FakeLLM(responses=[SlotExtraction(amortization_type="FIBONACCI", term_months=24)])
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["simulation_slots"] = SimulationSlots(amount=Decimal("5000"), amortization_type="PRICE")
    state["messages"] = [HumanMessage(content="24 meses")]

    updates = await node(state)

    assert updates["simulation_slots"].amortization_type == "PRICE"
    assert updates["simulation_result"] is not None


async def test_a_zero_amount_with_no_prior_slot_asks_the_follow_up_instead_of_inventing() -> None:
    smart_llm = FakeLLM(responses=[SlotExtraction(amount=Decimal("0"), term_months=6)])
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["messages"] = [HumanMessage(content="quero simular")]

    updates = await node(state)

    assert updates["simulation_slots"].amount is None
    assert updates["active_flow"] == "slot_filling"
    assert updates["next_missing_slot"] == "amount"


async def test_an_unsupported_amortization_type_triggers_the_follow_up_question() -> None:
    smart_llm = FakeLLM(
        responses=[
            SlotExtraction(amount=Decimal("3000"), term_months=12, amortization_type="FIBONACCI")
        ]
    )
    node = make_offer_simulator_node(ScriptedLLMFactory(smart=smart_llm))
    state = initial_state("cust-1")
    state["messages"] = [HumanMessage(content="3000 reais em 12 meses, tipo fibonacci")]

    updates = await node(state)

    assert updates["simulation_slots"].amortization_type is None
    assert updates["active_flow"] == "slot_filling"
    assert updates["next_missing_slot"] == "amortization_type"
    assert "simulation_result" not in updates

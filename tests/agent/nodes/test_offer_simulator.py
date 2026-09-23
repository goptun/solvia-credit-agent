"""offer_simulator: slot extraction, follow-up questions, and the
deterministic simulation call."""

from __future__ import annotations

from decimal import Decimal

from langchain_core.messages import HumanMessage

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

"""offer_simulator node: structured-output slot extraction, then the
deterministic Price/SAC simulation tool.

The LLM only ever extracts amount/term/amortization-type slots; it never
computes the simulation numbers themselves (see
`specs/conversation-graph/spec.md` — "Simulation slot extraction" and
"Credit offer simulation").
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.structured import ainvoke_structured
from apps.agent.state import ConversationState, SimulationSlots, SimulationSummary
from apps.agent.tools.simulation import simulate

OfferSimulatorNode = Callable[[ConversationState], Awaitable[ConversationState]]

_EXTRACTION_INSTRUCTIONS = (
    "Extraia os dados de simulação de empréstimo desta conversa: valor solicitado "
    "(amount), prazo em meses (term_months), e tipo de amortização "
    "(amortization_type: PRICE ou SAC). Retorne null para qualquer dado que não "
    "tenha sido informado."
)


class SlotExtraction(BaseModel):
    amount: Decimal | None = None
    term_months: int | None = None
    amortization_type: str | None = None


def _merge_slots(current: SimulationSlots, extracted: SlotExtraction) -> SimulationSlots:
    return SimulationSlots(
        amount=extracted.amount if extracted.amount is not None else current.amount,
        term_months=(
            extracted.term_months if extracted.term_months is not None else current.term_months
        ),
        amortization_type=(
            extracted.amortization_type.upper()
            if extracted.amortization_type
            else current.amortization_type
        ),
    )


def make_offer_simulator_node(llm_factory: LLMFactory) -> OfferSimulatorNode:
    async def offer_simulator_node(state: ConversationState) -> ConversationState:
        current_slots = state.get("simulation_slots") or SimulationSlots()

        llm = llm_factory.for_node("offer_simulator")
        fallback_llm = llm_factory.fallback_for_node("offer_simulator")
        known = (
            f"amount={current_slots.amount}, term_months={current_slots.term_months}, "
            f"amortization_type={current_slots.amortization_type}"
        )
        prompt = [
            *state.get("messages", []),
            HumanMessage(content=f"{_EXTRACTION_INSTRUCTIONS}\n\nDados já conhecidos: {known}"),
        ]
        extraction = await ainvoke_structured(llm, prompt, SlotExtraction, fallback=fallback_llm)
        merged = _merge_slots(current_slots, extraction)

        if not merged.is_complete:
            return ConversationState(
                simulation_slots=merged,
                active_flow="slot_filling",
                pending_intent="loan_simulation",
                next_missing_slot=merged.missing_fields()[0],
            )

        assert merged.amount is not None
        assert merged.term_months is not None
        assert merged.amortization_type is not None
        result = simulate(merged.amount, merged.term_months, merged.amortization_type)

        summary = SimulationSummary(
            amortization_type=result.amortization_type.value,
            principal=result.principal,
            term_months=result.term_months,
            cet_annual=result.cet_annual,
            first_installment=result.installments[0].payment,
            total_paid=result.total_paid,
        )
        return ConversationState(
            simulation_slots=merged,
            simulation_result=summary,
            active_flow="none",
            pending_intent=None,
            next_missing_slot=None,
        )

    return offer_simulator_node

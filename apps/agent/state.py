"""Typed conversation state for the Solvia agent graph.

See `openspec/changes/add-solvia-foundation-mvp/design.md` —
"Conversation graph and state" and "Routing matrix". A plain
`TypedDict` (rather than a Pydantic model) is the LangGraph state
schema itself, matching LangGraph's most common partial-update pattern;
the nested pieces below are Pydantic models for their own validation.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

Intent = Literal[
    "product_question",
    "regulatory_question",
    "loan_simulation",
    "profile_analysis",
    "complaint",
    "out_of_scope",
]
ActiveFlow = Literal["none", "consent_confirmation", "slot_filling"]
ConsentStatusValue = Literal["valid", "missing", "expired", "just_granted"]

VALID_AMORTIZATION_TYPES = {"PRICE", "SAC"}
"""The only amortization types the simulation tool supports (see
`apps.agent.tools.simulation`). Shared with `offer_simulator`'s slot
extraction, so an unsupported value is never merged into state as if it
were a real slot value."""


class SimulationSlots(BaseModel):
    """Loan simulation slots extracted from the conversation so far."""

    model_config = ConfigDict(frozen=True)

    amount: Decimal | None = None
    term_months: int | None = None
    amortization_type: str | None = None

    def missing_fields(self) -> list[str]:
        """Which slots are still missing or fail basic validation."""
        missing = []
        if self.amount is None or self.amount <= 0:
            missing.append("amount")
        if self.term_months is None or self.term_months <= 0:
            missing.append("term_months")
        if self.amortization_type not in VALID_AMORTIZATION_TYPES:
            missing.append("amortization_type")
        return missing

    @property
    def is_complete(self) -> bool:
        return not self.missing_fields()


class CustomerProfileSummary(BaseModel):
    """Financial-analyst output: income, DTI, and spending categories."""

    model_config = ConfigDict(frozen=True)

    income: Decimal
    debt_to_income_ratio: Decimal
    spending_categories: dict[str, Decimal] = Field(default_factory=dict)


class SimulationSummary(BaseModel):
    """A compact, template-ready view of a completed simulation."""

    model_config = ConfigDict(frozen=True)

    amortization_type: str
    principal: Decimal
    term_months: int
    cet_annual: Decimal
    first_installment: Decimal
    total_paid: Decimal


class ConversationState(TypedDict, total=False):
    """The full LangGraph state. Nodes return partial updates; missing
    keys mean "unchanged" for that turn."""

    customer_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Intent | None
    pending_intent: Intent | None
    active_flow: ActiveFlow
    consent_status: ConsentStatusValue
    simulation_slots: SimulationSlots
    customer_profile: CustomerProfileSummary | None
    simulation_result: SimulationSummary | None
    draft_reply: str | None
    compliance_flags: list[str]
    trace_id: str | None
    consent_prompt: Literal["ask", "reask"] | None
    """Set by `consent_check` so `responder` knows whether to ask for
    authorization for the first time or re-ask after an ambiguous reply."""
    consent_refused: bool
    """Set by `consent_check` when the customer's reply clearly refuses
    consent, so `responder` can acknowledge it."""
    next_missing_slot: str | None
    """Set by `offer_simulator` to the single slot (amount/term_months/
    amortization_type) `responder` should ask about next."""


def initial_state(customer_id: str) -> ConversationState:
    """The state a brand-new conversation starts from."""
    return ConversationState(
        customer_id=customer_id,
        messages=[],
        intent=None,
        pending_intent=None,
        active_flow="none",
        consent_status="missing",
        simulation_slots=SimulationSlots(),
        customer_profile=None,
        simulation_result=None,
        draft_reply=None,
        compliance_flags=[],
        trace_id=None,
        consent_prompt=None,
        consent_refused=False,
        next_missing_slot=None,
    )

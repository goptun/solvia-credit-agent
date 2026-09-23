"""consent_check: deterministic gate, confirmation detection, and resume."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from apps.agent.nodes.consent_check import make_consent_check_node
from apps.agent.state import initial_state
from apps.agent.synthetic_data.models import (
    Consent,
    ConsentStatus,
    Customer,
    CustomerProfile,
)
from tests.agent.nodes.fakes import StubCustomerRepository


def _customer_with_consent(status: ConsentStatus) -> Customer:
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


async def test_missing_consent_asks_for_authorization() -> None:
    repo = StubCustomerRepository(_customer_with_consent(ConsentStatus.MISSING))
    node = make_consent_check_node(repo)
    state = initial_state("cust-1")
    state["intent"] = "loan_simulation"

    updates = await node(state)

    assert updates["active_flow"] == "consent_confirmation"
    assert updates["pending_intent"] == "loan_simulation"
    assert updates["consent_prompt"] == "ask"


async def test_expired_consent_asks_for_authorization() -> None:
    repo = StubCustomerRepository(_customer_with_consent(ConsentStatus.EXPIRED))
    node = make_consent_check_node(repo)
    state = initial_state("cust-1")
    state["intent"] = "profile_analysis"

    updates = await node(state)

    assert updates["active_flow"] == "consent_confirmation"
    assert updates["consent_status"] == "expired"


async def test_valid_consent_proceeds_without_asking() -> None:
    repo = StubCustomerRepository(_customer_with_consent(ConsentStatus.VALID))
    node = make_consent_check_node(repo)
    state = initial_state("cust-1")
    state["intent"] = "profile_analysis"

    updates = await node(state)

    assert updates == {"consent_status": "valid"}


async def test_explicit_confirmation_grants_consent_and_resumes_pending_intent() -> None:
    node = make_consent_check_node(StubCustomerRepository(None))
    state = initial_state("cust-1")
    state["active_flow"] = "consent_confirmation"
    state["pending_intent"] = "loan_simulation"
    state["messages"] = [HumanMessage(content="sim, autorizo")]

    updates = await node(state)

    assert updates["consent_status"] == "just_granted"
    assert updates["active_flow"] == "none"
    assert updates["intent"] == "loan_simulation"


async def test_clear_refusal_clears_the_flow() -> None:
    node = make_consent_check_node(StubCustomerRepository(None))
    state = initial_state("cust-1")
    state["active_flow"] = "consent_confirmation"
    state["pending_intent"] = "loan_simulation"
    state["messages"] = [HumanMessage(content="não autorizo")]

    updates = await node(state)

    assert updates["active_flow"] == "none"
    assert updates["consent_refused"] is True


async def test_ambiguous_reply_reasks_without_changing_status() -> None:
    node = make_consent_check_node(StubCustomerRepository(None))
    state = initial_state("cust-1")
    state["active_flow"] = "consent_confirmation"
    state["messages"] = [HumanMessage(content="não sei, talvez")]

    updates = await node(state)

    assert updates == {"consent_prompt": "reask"}

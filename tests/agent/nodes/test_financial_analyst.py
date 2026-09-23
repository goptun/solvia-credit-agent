"""financial_analyst: delegates entirely to the deterministic tools."""

from __future__ import annotations

from apps.agent.nodes.financial_analyst import make_financial_analyst_node
from apps.agent.state import initial_state
from apps.agent.synthetic_data.generator import generate_customers
from tests.agent.nodes.fakes import StubCustomerRepository


async def test_produces_a_profile_summary_from_real_synthetic_data() -> None:
    customer = generate_customers(count=1)[0]
    node = make_financial_analyst_node(StubCustomerRepository(customer))
    state = initial_state(customer.customer_id)

    updates = await node(state)

    profile = updates["customer_profile"]
    assert profile is not None
    assert profile.income >= 0
    assert profile.debt_to_income_ratio >= 0
    assert isinstance(profile.spending_categories, dict)


async def test_unknown_customer_returns_no_update() -> None:
    node = make_financial_analyst_node(StubCustomerRepository(None))
    state = initial_state("cust-unknown")

    updates = await node(state)

    assert updates == {}

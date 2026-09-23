"""Typed conversation state construction and validation."""

from __future__ import annotations

from decimal import Decimal

from apps.agent.state import (
    CustomerProfileSummary,
    SimulationSlots,
    SimulationSummary,
    initial_state,
)


def test_initial_state_has_expected_defaults() -> None:
    state = initial_state("cust-0001")

    assert state["customer_id"] == "cust-0001"
    assert state["messages"] == []
    assert state["intent"] is None
    assert state["active_flow"] == "none"
    assert state["consent_status"] == "missing"
    assert state["simulation_slots"] == SimulationSlots()
    assert state["compliance_flags"] == []


def test_simulation_slots_reports_missing_fields() -> None:
    slots = SimulationSlots(amount=Decimal("1000"))

    assert "term_months" in slots.missing_fields()
    assert "amortization_type" in slots.missing_fields()
    assert not slots.is_complete


def test_simulation_slots_complete_when_all_valid() -> None:
    slots = SimulationSlots(amount=Decimal("1000"), term_months=12, amortization_type="PRICE")

    assert slots.is_complete
    assert slots.missing_fields() == []


def test_simulation_slots_invalid_amount_is_still_missing() -> None:
    slots = SimulationSlots(amount=Decimal("-1"), term_months=12, amortization_type="PRICE")

    assert "amount" in slots.missing_fields()


def test_customer_profile_summary_holds_spending_categories() -> None:
    summary = CustomerProfileSummary(
        income=Decimal("3000.00"),
        debt_to_income_ratio=Decimal("0.25"),
        spending_categories={"FOOD": Decimal("-400.00")},
    )

    assert summary.spending_categories["FOOD"] == Decimal("-400.00")


def test_simulation_summary_is_a_plain_snapshot() -> None:
    summary = SimulationSummary(
        amortization_type="PRICE",
        principal=Decimal("1000.00"),
        term_months=12,
        cet_annual=Decimal("0.35"),
        first_installment=Decimal("100.00"),
        total_paid=Decimal("1200.00"),
    )

    assert summary.amortization_type == "PRICE"

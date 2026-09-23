"""Debt-to-income ratio calculation, including zero-income edge cases."""

from __future__ import annotations

from decimal import Decimal

from apps.agent.synthetic_data.models import CreditCard
from apps.agent.tools.dti import (
    debt_to_income_ratio,
    estimate_monthly_debt_payments,
)


def test_ratio_computed_from_income_and_debt() -> None:
    ratio = debt_to_income_ratio(Decimal("4000.00"), Decimal("1200.00"))

    assert ratio == Decimal("0.3000")


def test_zero_income_with_debt_is_fully_indebted() -> None:
    ratio = debt_to_income_ratio(Decimal("0"), Decimal("500.00"))

    assert ratio == Decimal("1.0000")


def test_zero_income_with_no_debt_is_zero() -> None:
    ratio = debt_to_income_ratio(Decimal("0"), Decimal("0"))

    assert ratio == Decimal("0.0000")


def test_estimate_monthly_debt_payments_from_credit_cards() -> None:
    cards = [
        CreditCard(
            card_id="card-1",
            customer_id="cust-1",
            credit_limit=Decimal("5000.00"),
            balance=Decimal("2000.00"),
            due_day=10,
        ),
        CreditCard(
            card_id="card-2",
            customer_id="cust-1",
            credit_limit=Decimal("3000.00"),
            balance=Decimal("1000.00"),
            due_day=15,
        ),
    ]

    payments = estimate_monthly_debt_payments(cards)

    # (2000 + 1000) * 0.15
    assert payments == Decimal("450.00")


def test_estimate_monthly_debt_payments_with_no_cards_is_zero() -> None:
    assert estimate_monthly_debt_payments([]) == Decimal("0.00")

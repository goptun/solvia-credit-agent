"""Deterministic debt-to-income ratio.

See `specs/credit-simulation/spec.md` — "Debt-to-income ratio".
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal

from apps.agent.synthetic_data.models import CreditCard

MINIMUM_PAYMENT_RATE = Decimal("0.15")
"""Approximate minimum monthly payment as a fraction of a credit card
balance, used to estimate monthly debt obligations from card data."""


def estimate_monthly_debt_payments(credit_cards: Sequence[CreditCard]) -> Decimal:
    """Estimate monthly debt obligations from outstanding credit card balances."""
    total_balance = sum((card.balance for card in credit_cards), Decimal("0"))
    return (total_balance * MINIMUM_PAYMENT_RATE).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def debt_to_income_ratio(monthly_income: Decimal, monthly_debt_payments: Decimal) -> Decimal:
    """Ratio of monthly debt obligations to monthly income.

    A customer with no income and any debt is treated as fully
    indebted (ratio of 1); a customer with no income and no debt has a
    ratio of 0 — both edge cases avoid a division by zero.
    """
    if monthly_income <= 0:
        return Decimal("1.0000") if monthly_debt_payments > 0 else Decimal("0.0000")
    ratio = monthly_debt_payments / monthly_income
    return ratio.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

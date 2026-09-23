"""Deterministic income estimation from a customer's transaction history.

See `specs/credit-simulation/spec.md` — "Income estimation from
transaction history".
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal

from apps.agent.synthetic_data.models import Transaction, TransactionType

RECURRING_INCOME_CATEGORY_CODES = frozenset({"SALARIO", "PRO_LABORE"})
"""Transaction category codes treated as recurring income credits."""


def estimate_income(transactions: Sequence[Transaction]) -> Decimal:
    """Average monthly income from recurring income-coded credits.

    Returns `Decimal("0")` when the history has no recurring credit —
    the caller decides how to handle a zero-income customer.
    """
    income_amounts = [
        txn.amount
        for txn in transactions
        if txn.type == TransactionType.CREDIT
        and txn.category_code in RECURRING_INCOME_CATEGORY_CODES
    ]
    if not income_amounts:
        return Decimal("0")

    average = sum(income_amounts, Decimal("0")) / len(income_amounts)
    return average.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

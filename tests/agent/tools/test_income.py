"""Income estimation from recurring credit transactions."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from apps.agent.synthetic_data.models import Transaction, TransactionType
from apps.agent.tools.income import estimate_income


def _txn(
    category_code: str, amount: str, txn_type: TransactionType = TransactionType.CREDIT
) -> Transaction:
    return Transaction(
        transaction_id=f"txn-{category_code}-{amount}",
        account_id="acct-1",
        posted_at=date(2026, 1, 1),
        amount=Decimal(amount),
        type=txn_type,
        category_code=category_code,
        description="",
    )


def test_income_averages_recurring_salary_credits() -> None:
    transactions = [
        _txn("SALARIO", "3000.00"),
        _txn("SALARIO", "3200.00"),
        _txn("SUPERMERCADO", "-400.00", TransactionType.DEBIT),
    ]

    income = estimate_income(transactions)

    assert income == Decimal("3100.00")


def test_income_averages_recurring_pro_labore_credits() -> None:
    transactions = [_txn("PRO_LABORE", "2000.00"), _txn("PRO_LABORE", "2500.00")]

    income = estimate_income(transactions)

    assert income == Decimal("2250.00")


def test_no_recurring_credits_returns_zero() -> None:
    transactions = [
        _txn("PIX_RECEBIDO", "150.00"),
        _txn("SUPERMERCADO", "-80.00", TransactionType.DEBIT),
    ]

    income = estimate_income(transactions)

    assert income == Decimal("0")


def test_empty_history_returns_zero() -> None:
    assert estimate_income([]) == Decimal("0")

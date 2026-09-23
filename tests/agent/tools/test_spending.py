"""Deterministic spending categorization, including the fallback case."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from apps.agent.synthetic_data.models import SpendingCategory, Transaction, TransactionType
from apps.agent.tools.spending import categorize_spending, categorize_transaction


def _txn(category_code: str, description: str = "", amount: str = "-100.00") -> Transaction:
    return Transaction(
        transaction_id=f"txn-{category_code}",
        account_id="acct-1",
        posted_at=date(2026, 1, 10),
        amount=Decimal(amount),
        type=TransactionType.DEBIT,
        category_code=category_code,
        description=description,
    )


def test_known_category_code_maps_directly() -> None:
    assert categorize_transaction(_txn("ALUGUEL")) == SpendingCategory.HOUSING
    assert categorize_transaction(_txn("SUPERMERCADO")) == SpendingCategory.FOOD
    assert categorize_transaction(_txn("PAGAMENTO_FATURA_CARTAO")) == SpendingCategory.DEBT_PAYMENT


def test_unknown_code_falls_back_to_description_keyword() -> None:
    txn = _txn("OUTROS", description="Pagamento de farmacia popular")

    assert categorize_transaction(txn) == SpendingCategory.HEALTH


def test_unrecognized_transaction_falls_back_to_uncategorized() -> None:
    txn = _txn("PIX_ENVIADO", description="Pix enviado para Fulano")

    assert categorize_transaction(txn) == SpendingCategory.UNCATEGORIZED


def test_categorize_spending_sums_debits_per_category() -> None:
    transactions = [
        _txn("SUPERMERCADO", amount="-200.00"),
        _txn("SUPERMERCADO", amount="-150.00"),
        _txn("ALUGUEL", amount="-1200.00"),
        Transaction(
            transaction_id="txn-income",
            account_id="acct-1",
            posted_at=date(2026, 1, 5),
            amount=Decimal("3000.00"),
            type=TransactionType.CREDIT,
            category_code="SALARIO",
            description="",
        ),
    ]

    totals = categorize_spending(transactions)

    assert totals[SpendingCategory.FOOD] == Decimal("-350.00")
    assert totals[SpendingCategory.HOUSING] == Decimal("-1200.00")
    assert SpendingCategory.INCOME not in totals  # income is a credit, not a spending debit

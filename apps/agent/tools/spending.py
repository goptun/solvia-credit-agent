"""Deterministic spending categorization from transaction codes/descriptions.

See `specs/credit-simulation/spec.md` — "Deterministic spending
categorization".
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from apps.agent.synthetic_data.models import SpendingCategory, Transaction, TransactionType

_CATEGORY_CODE_MAP: dict[str, SpendingCategory] = {
    "SALARIO": SpendingCategory.INCOME,
    "PRO_LABORE": SpendingCategory.INCOME,
    "ALUGUEL": SpendingCategory.HOUSING,
    "CONDOMINIO": SpendingCategory.HOUSING,
    "SUPERMERCADO": SpendingCategory.FOOD,
    "RESTAURANTE": SpendingCategory.FOOD,
    "TRANSPORTE_APP": SpendingCategory.TRANSPORT,
    "COMBUSTIVEL": SpendingCategory.TRANSPORT,
    "FARMACIA": SpendingCategory.HEALTH,
    "PLANO_SAUDE": SpendingCategory.HEALTH,
    "STREAMING": SpendingCategory.LEISURE,
    "CINEMA": SpendingCategory.LEISURE,
    "PAGAMENTO_FATURA_CARTAO": SpendingCategory.DEBT_PAYMENT,
    "EMPRESTIMO_PARCELA": SpendingCategory.DEBT_PAYMENT,
}

_DESCRIPTION_KEYWORDS: list[tuple[str, SpendingCategory]] = [
    ("aluguel", SpendingCategory.HOUSING),
    ("condominio", SpendingCategory.HOUSING),
    ("supermercado", SpendingCategory.FOOD),
    ("restaurante", SpendingCategory.FOOD),
    ("farmacia", SpendingCategory.HEALTH),
    ("plano de saude", SpendingCategory.HEALTH),
    ("streaming", SpendingCategory.LEISURE),
    ("cinema", SpendingCategory.LEISURE),
    ("fatura", SpendingCategory.DEBT_PAYMENT),
    ("emprestimo", SpendingCategory.DEBT_PAYMENT),
    ("salario", SpendingCategory.INCOME),
    ("pro-labore", SpendingCategory.INCOME),
]


def categorize_transaction(transaction: Transaction) -> SpendingCategory:
    """Categorize a single transaction, falling back to `UNCATEGORIZED`
    when neither its category code nor its description match a known
    pattern (see spec — "Unrecognized transaction falls back to a
    default category")."""
    known = _CATEGORY_CODE_MAP.get(transaction.category_code)
    if known is not None:
        return known

    description = transaction.description.lower()
    for keyword, category in _DESCRIPTION_KEYWORDS:
        if keyword in description:
            return category

    return SpendingCategory.UNCATEGORIZED


def categorize_spending(transactions: Sequence[Transaction]) -> dict[SpendingCategory, Decimal]:
    """Sum debit transaction amounts per spending category."""
    totals: dict[SpendingCategory, Decimal] = {}
    for txn in transactions:
        if txn.type != TransactionType.DEBIT:
            continue
        category = categorize_transaction(txn)
        totals[category] = totals.get(category, Decimal("0")) + txn.amount
    return totals

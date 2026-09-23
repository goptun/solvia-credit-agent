"""Reproducible synthetic Open Finance Brasil dataset generator.

Deterministic from a fixed seed and the fixed `REFERENCE_DATE` in
`apps.agent.config.reference_date` — never from the system clock or
`random` module state outside a seeded `random.Random` instance (see
`specs/synthetic-data/spec.md`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from apps.agent.config.reference_date import REFERENCE_DATE
from apps.agent.synthetic_data.models import (
    Account,
    AccountType,
    Consent,
    ConsentStatus,
    CreditCard,
    Customer,
    CustomerProfile,
    Transaction,
    TransactionType,
)

DEFAULT_SEED = 20260601
DEFAULT_CUSTOMER_COUNT = 200

_PROFILES = [
    CustomerProfile.SALARIED,
    CustomerProfile.SELF_EMPLOYED,
    CustomerProfile.OVER_INDEBTED,
    CustomerProfile.THIN_FILE,
]

_FIRST_NAMES = [
    "Ana",
    "Bruno",
    "Carla",
    "Diego",
    "Elisa",
    "Fabio",
    "Gabriela",
    "Heitor",
    "Isabela",
    "Joao",
    "Larissa",
    "Marcos",
    "Nicole",
    "Otavio",
    "Patricia",
    "Rafael",
    "Sabrina",
    "Thiago",
    "Vanessa",
    "Wesley",
]
_LAST_NAMES = [
    "Almeida",
    "Barbosa",
    "Cardoso",
    "Duarte",
    "Esteves",
    "Ferreira",
    "Gomes",
    "Henriques",
    "Iglesias",
    "Junqueira",
    "Lopes",
    "Martins",
    "Nogueira",
    "Oliveira",
    "Pereira",
    "Queiroz",
    "Ramos",
    "Souza",
    "Teixeira",
    "Vieira",
]


@dataclass(frozen=True)
class _MonthlyExpense:
    category_code: str
    description: str
    amount_range: tuple[Decimal, Decimal]


_EXPENSES: dict[CustomerProfile, list[_MonthlyExpense]] = {
    CustomerProfile.SALARIED: [
        _MonthlyExpense("ALUGUEL", "Pagamento de aluguel", (Decimal("1200"), Decimal("1800"))),
        _MonthlyExpense("SUPERMERCADO", "Compra em supermercado", (Decimal("300"), Decimal("600"))),
        _MonthlyExpense("TRANSPORTE_APP", "Corrida de aplicativo", (Decimal("40"), Decimal("120"))),
        _MonthlyExpense("STREAMING", "Assinatura de streaming", (Decimal("30"), Decimal("60"))),
    ],
    CustomerProfile.SELF_EMPLOYED: [
        _MonthlyExpense("ALUGUEL", "Pagamento de aluguel", (Decimal("900"), Decimal("1500"))),
        _MonthlyExpense("SUPERMERCADO", "Compra em supermercado", (Decimal("250"), Decimal("500"))),
        _MonthlyExpense("COMBUSTIVEL", "Abastecimento", (Decimal("150"), Decimal("350"))),
        _MonthlyExpense("FARMACIA", "Compra em farmacia", (Decimal("40"), Decimal("150"))),
    ],
    CustomerProfile.OVER_INDEBTED: [
        _MonthlyExpense("ALUGUEL", "Pagamento de aluguel", (Decimal("900"), Decimal("1400"))),
        _MonthlyExpense(
            "PAGAMENTO_FATURA_CARTAO",
            "Pagamento de fatura do cartao",
            (Decimal("400"), Decimal("900")),
        ),
        _MonthlyExpense(
            "EMPRESTIMO_PARCELA",
            "Parcela de emprestimo consignado",
            (Decimal("500"), Decimal("1100")),
        ),
        _MonthlyExpense("SUPERMERCADO", "Compra em supermercado", (Decimal("300"), Decimal("550"))),
    ],
    CustomerProfile.THIN_FILE: [
        _MonthlyExpense("SUPERMERCADO", "Compra em supermercado", (Decimal("150"), Decimal("300"))),
    ],
}


def _rand_decimal(rng: random.Random, low: Decimal, high: Decimal) -> Decimal:
    value = rng.uniform(float(low), float(high))
    return Decimal(str(round(value, 2)))


def _synthetic_cpf(rng: random.Random) -> str:
    digits = [str(rng.randint(0, 9)) for _ in range(11)]
    raw = "".join(digits)
    return f"{raw[0:3]}.{raw[3:6]}.{raw[6:9]}-{raw[9:11]}"


def _customer_name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _consent_status_for_index(index: int) -> ConsentStatus:
    """~70% valid, ~15% missing, ~15% expired — guarantees coverage of
    all three statuses well within `DEFAULT_CUSTOMER_COUNT`."""
    bucket = index % 10
    if bucket < 7:
        return ConsentStatus.VALID
    if bucket == 7:
        return ConsentStatus.MISSING
    return ConsentStatus.EXPIRED


def _generate_consent(customer_id: str, status: ConsentStatus) -> Consent:
    if status == ConsentStatus.MISSING:
        return Consent(
            consent_id=f"consent-{customer_id}",
            customer_id=customer_id,
            status=status,
            scope=[],
            granted_at=None,
            expires_at=None,
        )
    if status == ConsentStatus.VALID:
        return Consent(
            consent_id=f"consent-{customer_id}",
            customer_id=customer_id,
            status=status,
            scope=["ACCOUNTS_READ", "CREDIT_CARDS_READ"],
            granted_at=REFERENCE_DATE - timedelta(days=90),
            expires_at=REFERENCE_DATE + timedelta(days=275),
        )
    return Consent(
        consent_id=f"consent-{customer_id}",
        customer_id=customer_id,
        status=status,
        scope=["ACCOUNTS_READ", "CREDIT_CARDS_READ"],
        granted_at=REFERENCE_DATE - timedelta(days=400),
        expires_at=REFERENCE_DATE - timedelta(days=35),
    )


def _generate_accounts(
    rng: random.Random, customer_id: str, profile: CustomerProfile
) -> list[Account]:
    opened_days_ago = 60 if profile == CustomerProfile.THIN_FILE else rng.randint(400, 1800)
    balance_range = {
        CustomerProfile.SALARIED: (Decimal("500"), Decimal("8000")),
        CustomerProfile.SELF_EMPLOYED: (Decimal("100"), Decimal("5000")),
        CustomerProfile.OVER_INDEBTED: (Decimal("-800"), Decimal("200")),
        CustomerProfile.THIN_FILE: (Decimal("50"), Decimal("600")),
    }[profile]
    return [
        Account(
            account_id=f"acct-{customer_id}",
            customer_id=customer_id,
            type=AccountType.CHECKING,
            balance=_rand_decimal(rng, *balance_range),
            opened_at=REFERENCE_DATE - timedelta(days=opened_days_ago),
        )
    ]


def _income_transactions(
    rng: random.Random, account_id: str, profile: CustomerProfile, months: int
) -> list[Transaction]:
    if profile == CustomerProfile.THIN_FILE:
        return []

    code, description, income_range = {
        CustomerProfile.SALARIED: (
            "SALARIO",
            "Pagamento de salario",
            (Decimal("2500"), Decimal("9000")),
        ),
        CustomerProfile.SELF_EMPLOYED: (
            "PRO_LABORE",
            "Recebimento de pro-labore",
            (Decimal("1500"), Decimal("6000")),
        ),
        CustomerProfile.OVER_INDEBTED: (
            "SALARIO",
            "Pagamento de salario",
            (Decimal("1800"), Decimal("4000")),
        ),
    }[profile]

    transactions = []
    for month_offset in range(months):
        posted_at = REFERENCE_DATE.replace(day=5) - timedelta(days=30 * month_offset)
        amount = _rand_decimal(rng, *income_range)
        transactions.append(
            Transaction(
                transaction_id=f"txn-{account_id}-income-{month_offset}",
                account_id=account_id,
                posted_at=posted_at,
                amount=amount,
                type=TransactionType.CREDIT,
                category_code=code,
                description=description,
            )
        )
    return transactions


def _expense_transactions(
    rng: random.Random, account_id: str, profile: CustomerProfile, months: int
) -> list[Transaction]:
    transactions = []
    for month_offset in range(months):
        for expense_index, expense in enumerate(_EXPENSES[profile]):
            posted_at = REFERENCE_DATE.replace(day=15) - timedelta(days=30 * month_offset)
            amount = _rand_decimal(rng, *expense.amount_range)
            transactions.append(
                Transaction(
                    transaction_id=f"txn-{account_id}-exp-{month_offset}-{expense_index}",
                    account_id=account_id,
                    posted_at=posted_at,
                    amount=amount,
                    type=TransactionType.DEBIT,
                    category_code=expense.category_code,
                    description=expense.description,
                )
            )
    return transactions


def _generate_credit_cards(
    rng: random.Random, customer_id: str, profile: CustomerProfile
) -> list[CreditCard]:
    if profile == CustomerProfile.THIN_FILE:
        return []

    limit_range, utilization_range = {
        CustomerProfile.SALARIED: (
            (Decimal("2000"), Decimal("8000")),
            (Decimal("0.1"), Decimal("0.5")),
        ),
        CustomerProfile.SELF_EMPLOYED: (
            (Decimal("1500"), Decimal("5000")),
            (Decimal("0.1"), Decimal("0.6")),
        ),
        CustomerProfile.OVER_INDEBTED: (
            (Decimal("1000"), Decimal("4000")),
            (Decimal("0.85"), Decimal("1.0")),
        ),
    }[profile]

    limit = _rand_decimal(rng, *limit_range)
    utilization = _rand_decimal(rng, *utilization_range)
    balance = (limit * utilization).quantize(Decimal("0.01"))
    return [
        CreditCard(
            card_id=f"card-{customer_id}",
            customer_id=customer_id,
            credit_limit=limit,
            balance=balance,
            due_day=rng.randint(1, 28),
        )
    ]


def _generate_customer(rng: random.Random, index: int) -> Customer:
    customer_id = f"cust-{index:04d}"
    profile = _PROFILES[index % len(_PROFILES)]
    accounts = _generate_accounts(rng, customer_id, profile)
    account_id = accounts[0].account_id
    months = 2 if profile == CustomerProfile.THIN_FILE else 6
    transactions = _income_transactions(rng, account_id, profile, months) + _expense_transactions(
        rng, account_id, profile, months
    )

    return Customer(
        customer_id=customer_id,
        name=_customer_name(rng),
        cpf=_synthetic_cpf(rng),
        profile=profile,
        accounts=accounts,
        credit_cards=_generate_credit_cards(rng, customer_id, profile),
        transactions=transactions,
        consent=_generate_consent(customer_id, _consent_status_for_index(index)),
    )


def generate_customers(
    seed: int = DEFAULT_SEED, count: int = DEFAULT_CUSTOMER_COUNT
) -> list[Customer]:
    """Generate `count` synthetic customers, deterministic for a given `seed`.

    All dates are derived from `REFERENCE_DATE`; nothing here reads the
    system clock.
    """
    rng = random.Random(seed)
    return [_generate_customer(rng, index) for index in range(count)]


def reference_date() -> date:
    """The fixed reference date every generated date is derived from."""
    return REFERENCE_DATE

"""Open Finance Brasil–shaped models validate their required fields."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

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


def test_account_requires_type_balance_and_currency() -> None:
    account = Account(
        account_id="acct-1",
        customer_id="cust-1",
        type=AccountType.CHECKING,
        balance=Decimal("1234.56"),
        opened_at=date(2024, 1, 1),
    )
    assert account.currency == "BRL"
    assert account.type == AccountType.CHECKING


def test_account_missing_required_field_raises() -> None:
    with pytest.raises(ValidationError):
        Account.model_validate({"account_id": "acct-1", "customer_id": "cust-1", "balance": "10.0"})


def test_consent_captures_status_and_scope() -> None:
    consent = Consent(
        consent_id="consent-1",
        customer_id="cust-1",
        status=ConsentStatus.VALID,
        scope=["ACCOUNTS_READ"],
        granted_at=date(2026, 1, 1),
        expires_at=date(2026, 12, 31),
    )
    assert consent.status == ConsentStatus.VALID
    assert "ACCOUNTS_READ" in consent.scope


def test_transaction_requires_type_and_amount() -> None:
    txn = Transaction(
        transaction_id="txn-1",
        account_id="acct-1",
        posted_at=date(2026, 1, 5),
        amount=Decimal("-99.90"),
        type=TransactionType.DEBIT,
        category_code="SUPERMERCADO",
        description="Compra em supermercado",
    )
    assert txn.type == TransactionType.DEBIT


def test_credit_card_requires_limit_and_balance() -> None:
    card = CreditCard(
        card_id="card-1",
        customer_id="cust-1",
        credit_limit=Decimal("2000.00"),
        balance=Decimal("500.00"),
        due_day=10,
    )
    assert card.credit_limit == Decimal("2000.00")


def test_customer_bundles_accounts_cards_transactions_and_consent() -> None:
    account = Account(
        account_id="acct-1",
        customer_id="cust-1",
        type=AccountType.CHECKING,
        balance=Decimal("100.00"),
        opened_at=date(2024, 1, 1),
    )
    consent = Consent(
        consent_id="consent-1",
        customer_id="cust-1",
        status=ConsentStatus.MISSING,
        scope=[],
        granted_at=None,
        expires_at=None,
    )
    customer = Customer(
        customer_id="cust-1",
        name="Ana Souza",
        cpf="111.222.333-44",
        profile=CustomerProfile.SALARIED,
        accounts=[account],
        credit_cards=[],
        transactions=[],
        consent=consent,
    )
    assert customer.profile == CustomerProfile.SALARIED
    assert customer.accounts == [account]

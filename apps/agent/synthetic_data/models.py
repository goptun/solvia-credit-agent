"""Open Finance Brasil–shaped Pydantic models for synthetic data.

Field shapes follow the public Open Finance Brasil API specs (accounts,
transactions, credit cards, consents) closely enough for the MVP's own
tools to consume without translation — see
`specs/synthetic-data/spec.md`.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class CustomerProfile(StrEnum):
    SALARIED = "SALARIED"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    OVER_INDEBTED = "OVER_INDEBTED"
    THIN_FILE = "THIN_FILE"


class ConsentStatus(StrEnum):
    VALID = "valid"
    MISSING = "missing"
    EXPIRED = "expired"


class AccountType(StrEnum):
    CHECKING = "CONTA_DEPOSITO_A_VISTA"
    SAVINGS = "CONTA_POUPANCA"


class TransactionType(StrEnum):
    CREDIT = "CREDITO"
    DEBIT = "DEBITO"


class SpendingCategory(StrEnum):
    INCOME = "INCOME"
    HOUSING = "HOUSING"
    FOOD = "FOOD"
    TRANSPORT = "TRANSPORT"
    HEALTH = "HEALTH"
    LEISURE = "LEISURE"
    DEBT_PAYMENT = "DEBT_PAYMENT"
    UNCATEGORIZED = "UNCATEGORIZED"


class Consent(BaseModel):
    model_config = ConfigDict(frozen=True)

    consent_id: str
    customer_id: str
    status: ConsentStatus
    scope: list[str]
    granted_at: date | None
    expires_at: date | None


class Account(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    customer_id: str
    type: AccountType
    currency: str = "BRL"
    balance: Decimal
    opened_at: date


class Transaction(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    account_id: str
    posted_at: date
    amount: Decimal
    type: TransactionType
    category_code: str
    description: str


class CreditCard(BaseModel):
    model_config = ConfigDict(frozen=True)

    card_id: str
    customer_id: str
    credit_limit: Decimal
    balance: Decimal
    due_day: int


class Customer(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_id: str
    name: str
    cpf: str
    profile: CustomerProfile
    accounts: list[Account]
    credit_cards: list[CreditCard]
    transactions: list[Transaction]
    consent: Consent

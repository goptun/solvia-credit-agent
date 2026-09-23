"""Result shapes returned by the credit-simulation tools."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from apps.agent.config.catalog import AmortizationType


@dataclass(frozen=True)
class Installment:
    number: int
    payment: Decimal
    principal: Decimal
    interest: Decimal
    balance: Decimal


@dataclass(frozen=True)
class SimulationResult:
    amortization_type: AmortizationType
    principal: Decimal
    monthly_interest_rate: Decimal
    term_months: int
    iof_total: Decimal
    origination_fee: Decimal
    net_released: Decimal
    installments: list[Installment]
    total_paid: Decimal
    cet_annual: Decimal

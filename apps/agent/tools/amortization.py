"""Price and SAC amortization schedules.

Pure math, parameterized by an explicit monthly rate so it can be
verified independently against the standard closed-form formulas in
tests, before `apps.agent.tools.simulation` wires in the product
catalog (see `specs/credit-simulation/spec.md` — "Price/SAC amortization
simulation with CET").
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from apps.agent.tools.exceptions import SimulationInputError
from apps.agent.tools.models import Installment


def _round_money(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _validate(amount: Decimal, monthly_rate: Decimal, term_months: int) -> None:
    if amount <= 0:
        raise SimulationInputError("loan amount must be positive")
    if term_months <= 0:
        raise SimulationInputError("term must be a positive number of months")
    if monthly_rate < 0:
        raise SimulationInputError("interest rate must not be negative")


def price_schedule(amount: Decimal, monthly_rate: Decimal, term_months: int) -> list[Installment]:
    """Constant-installment ("Tabela Price") amortization schedule.

    Installment = P * i / (1 - (1 + i) ** -n), the standard annuity
    formula. The final installment's principal is adjusted to pay the
    remaining balance off exactly, absorbing rounding drift.
    """
    _validate(amount, monthly_rate, term_months)

    if monthly_rate == 0:
        installment = _round_money(amount / term_months)
    else:
        denominator = Decimal(1) - (Decimal(1) + monthly_rate) ** (-term_months)
        installment = _round_money(amount * monthly_rate / denominator)

    schedule: list[Installment] = []
    balance = amount
    for period in range(1, term_months + 1):
        interest = _round_money(balance * monthly_rate)
        if period == term_months:
            principal_payment = balance
        else:
            principal_payment = _round_money(installment - interest)
        payment = _round_money(principal_payment + interest)
        balance = _round_money(balance - principal_payment)
        schedule.append(
            Installment(
                number=period,
                payment=payment,
                principal=principal_payment,
                interest=interest,
                balance=balance,
            )
        )
    return schedule


def sac_schedule(amount: Decimal, monthly_rate: Decimal, term_months: int) -> list[Installment]:
    """Constant-amortization ("SAC") schedule — installments decrease
    over time as the outstanding balance (and its interest) shrinks."""
    _validate(amount, monthly_rate, term_months)

    base_amortization = _round_money(amount / term_months)

    schedule: list[Installment] = []
    balance = amount
    for period in range(1, term_months + 1):
        interest = _round_money(balance * monthly_rate)
        principal_payment = balance if period == term_months else base_amortization
        payment = _round_money(principal_payment + interest)
        balance = _round_money(balance - principal_payment)
        schedule.append(
            Installment(
                number=period,
                payment=payment,
                principal=principal_payment,
                interest=interest,
                balance=balance,
            )
        )
    return schedule

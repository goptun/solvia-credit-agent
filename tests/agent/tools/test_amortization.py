"""Price and SAC schedules against hand-computed reference cases."""

from __future__ import annotations

from decimal import Decimal

import pytest

from apps.agent.tools.amortization import price_schedule, sac_schedule
from apps.agent.tools.exceptions import SimulationInputError


def test_price_schedule_matches_the_standard_annuity_formula() -> None:
    """Reference case: P=1000, i=1% a.m., n=3.

    installment = P * i / (1 - (1 + i) ** -n) = 1000 * 0.01 / (1 - 1.01**-3)
    = 340.02 (rounded to cents), verified independently against the
    standard closed-form Price/annuity formula.
    """
    schedule = price_schedule(Decimal("1000"), Decimal("0.01"), 3)

    assert len(schedule) == 3
    assert schedule[0].payment == Decimal("340.02")
    assert schedule[0].interest == Decimal("10.00")
    assert schedule[0].principal == Decimal("330.02")
    assert schedule[0].balance == Decimal("669.98")
    assert schedule[1].payment == Decimal("340.02")
    assert schedule[2].balance == Decimal("0.00")
    # Constant installment across all periods, modulo the final
    # rounding-drift cent absorbed by the last installment.
    assert abs(schedule[2].payment - schedule[0].payment) <= Decimal("0.01")


def test_price_schedule_final_installment_zeroes_the_balance() -> None:
    schedule = price_schedule(Decimal("5000"), Decimal("0.025"), 12)

    assert schedule[-1].balance == Decimal("0.00")


def test_sac_schedule_matches_the_constant_amortization_formula() -> None:
    """Reference case: P=1200, i=2% a.m., n=3, amortization=400 (constant).

    Period 1: interest = 1200*0.02 = 24.00, payment = 424.00, balance = 800.00
    Period 2: interest =  800*0.02 = 16.00, payment = 416.00, balance = 400.00
    Period 3: interest =  400*0.02 =  8.00, payment = 408.00, balance =   0.00
    """
    schedule = sac_schedule(Decimal("1200"), Decimal("0.02"), 3)

    assert schedule[0].payment == Decimal("424.00")
    assert schedule[1].payment == Decimal("416.00")
    assert schedule[2].payment == Decimal("408.00")
    assert schedule[2].balance == Decimal("0.00")


def test_sac_installments_decrease_over_time() -> None:
    schedule = sac_schedule(Decimal("6000"), Decimal("0.03"), 6)

    payments = [installment.payment for installment in schedule]
    assert payments == sorted(payments, reverse=True)
    assert payments[0] > payments[-1]


@pytest.mark.parametrize("schedule_fn", [price_schedule, sac_schedule])
def test_non_positive_amount_is_rejected(schedule_fn: object) -> None:
    with pytest.raises(SimulationInputError):
        schedule_fn(Decimal("0"), Decimal("0.02"), 12)  # type: ignore[operator]


@pytest.mark.parametrize("schedule_fn", [price_schedule, sac_schedule])
def test_non_positive_term_is_rejected(schedule_fn: object) -> None:
    with pytest.raises(SimulationInputError):
        schedule_fn(Decimal("1000"), Decimal("0.02"), 0)  # type: ignore[operator]


@pytest.mark.parametrize("schedule_fn", [price_schedule, sac_schedule])
def test_negative_rate_is_rejected(schedule_fn: object) -> None:
    with pytest.raises(SimulationInputError):
        schedule_fn(Decimal("1000"), Decimal("-0.01"), 12)  # type: ignore[operator]

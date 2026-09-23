"""CET (annualized IRR) against reference cases solved independently of
the bisection implementation — a single-installment case has a direct
closed-form solution, and a two-installment case is solved via the
quadratic formula. Both are documented in the comments below.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from apps.agent.tools.cet import annual_cet
from apps.agent.tools.exceptions import CetCalculationError


def test_single_installment_matches_closed_form_reference() -> None:
    """net_released = P / (1 + r)  =>  r = P / net_released - 1 (exact).

    net_released = 1000, payment = 1100 => monthly r = 0.10 exactly,
    annual CET = 1.10**12 - 1 = 2.138428... -> 2.1384 (rounded to 4dp).
    """
    cet = annual_cet(Decimal("1000"), [Decimal("1100")])

    assert cet == Decimal("2.1384")


def test_two_installments_matches_quadratic_formula_reference() -> None:
    """net_released = P1/(1+r) + P2/(1+r)**2. With x = 1/(1+r):

        P2*x**2 + P1*x - net_released = 0
        x = (-P1 + sqrt(P1**2 + 4*P2*net_released)) / (2*P2)

    net_released = 1000, P1 = P2 = 550 => x solved via the quadratic
    formula gives monthly r ~= 0.0659646..., annual CET ~= 1.1524
    (rounded to 4dp) — computed independently of `annual_cet`'s
    bisection search.
    """
    cet = annual_cet(Decimal("1000"), [Decimal("550"), Decimal("550")])

    assert cet == Decimal("1.1524")


def test_cet_reflects_iof_and_fees_via_a_lower_net_released() -> None:
    """A lower net_released (more IOF/fees taken out up front) for the
    same payments must produce a strictly higher CET than a higher
    net_released, since less money was actually released to the
    customer for the same repayment stream."""
    payments = [Decimal("400"), Decimal("400"), Decimal("400")]

    cet_high_net = annual_cet(Decimal("1100"), payments)
    cet_low_net = annual_cet(Decimal("1000"), payments)

    assert cet_low_net > cet_high_net


def test_net_released_above_total_payments_raises_instead_of_reporting_200_percent() -> None:
    """Degenerate cash flow: more was "released" than will ever be repaid
    — no rate in [0%, 200% a.m.] reconciles that, so this must raise
    rather than silently reporting a fixed 200% CET."""
    with pytest.raises(CetCalculationError):
        annual_cet(Decimal("2000"), [Decimal("500")])


def test_non_positive_net_released_raises() -> None:
    """Another degenerate case: nothing (or a negative amount) was
    actually released to the customer."""
    with pytest.raises(CetCalculationError):
        annual_cet(Decimal("-100"), [Decimal("50")])

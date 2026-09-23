"""CET (total effective cost) as the annualized IRR of the net cash flow.

CET is the internal rate of return `r` (monthly) that solves:

    net_released = sum(payment_i / (1 + r) ** i for i in 1..n)

i.e. the discount rate that makes the amount actually released to the
customer (loan amount minus IOF and fees) equal to the present value of
the installment payments. The annualized figure reported to the
customer is `(1 + r) ** 12 - 1` (see
`specs/credit-simulation/spec.md` — "CET as annualized IRR of the net
cash flow").

Solved by bisection (not Newton-Raphson) because the net present value
function is monotonic and well-behaved over the loan's plausible rate
range, and bisection needs no derivative and never overshoots.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal

from apps.agent.tools.exceptions import CetCalculationError

_MAX_MONTHLY_RATE = Decimal("2")
"""Upper search bound (200% a.m.) — far above any plausible result;
only used to bracket the root for bisection."""
_BISECTION_ITERATIONS = 200
_TOLERANCE = Decimal("0.0000001")


def _net_present_value(
    rate: Decimal, net_released: Decimal, payments: Sequence[Decimal]
) -> Decimal:
    total = net_released
    for period, payment in enumerate(payments, start=1):
        total -= payment / ((Decimal(1) + rate) ** period)
    return total


def _monthly_irr(net_released: Decimal, payments: Sequence[Decimal]) -> Decimal:
    low, high = Decimal("0"), _MAX_MONTHLY_RATE
    f_low = _net_present_value(low, net_released, payments)
    f_high = _net_present_value(high, net_released, payments)

    # NPV is decreasing in rate for a positive net_released and positive
    # payments; if the bracket doesn't already contain a sign change, no
    # rate in [0%, 200% a.m.] reconciles net_released with the payment
    # schedule — a degenerate cash flow, not a real (if extreme) CET.
    # Reporting a fixed 200% here would silently misreport the result.
    if f_low * f_high > 0:
        raise CetCalculationError(
            "no sign change in the IRR search bracket [0%, 200% a.m.] — "
            "net_released and the payment schedule do not reconcile to any "
            "rate in range (net_released="
            f"{net_released}, total_payments={sum(payments, Decimal('0'))})"
        )

    for _ in range(_BISECTION_ITERATIONS):
        mid = (low + high) / 2
        f_mid = _net_present_value(mid, net_released, payments)
        if abs(f_mid) < _TOLERANCE:
            return mid
        if (f_low < 0) != (f_mid < 0):
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


def annual_cet(net_released: Decimal, payments: Sequence[Decimal]) -> Decimal:
    """Annualized CET from the net amount released and the installment schedule."""
    monthly_rate = _monthly_irr(net_released, payments)
    annual_rate = (Decimal(1) + monthly_rate) ** 12 - Decimal(1)
    return annual_rate.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

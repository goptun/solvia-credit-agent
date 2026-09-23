"""IOF and origination-fee cost computation, and the net amount released.

IOF = fixed rate (on principal) + daily rate (on principal, capped at a
maximum number of days) — both configurable per
`apps.agent.config.catalog.ProductTerms` (see
`specs/credit-simulation/spec.md` — "CET as annualized IRR").
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from apps.agent.config.catalog import DAYS_PER_MONTH, ProductTerms


def compute_iof(principal: Decimal, term_months: int, terms: ProductTerms) -> Decimal:
    """Total IOF: fixed rate on principal + daily rate on principal,
    capped at `terms.iof_daily_cap_days`."""
    term_days = min(term_months * DAYS_PER_MONTH, terms.iof_daily_cap_days)
    fixed_component = principal * terms.iof_fixed_rate
    daily_component = principal * terms.iof_daily_rate * term_days
    return (fixed_component + daily_component).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def compute_net_released(
    principal: Decimal, iof_total: Decimal, origination_fee: Decimal
) -> Decimal:
    """The amount actually released to the customer, after IOF and fees."""
    return (principal - iof_total - origination_fee).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

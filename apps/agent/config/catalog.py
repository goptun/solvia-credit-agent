"""Fictional product catalog: interest rate, IOF, and fees.

The single source of commercial terms for credit simulations. The LLM
never supplies or alters these values (see
`specs/credit-simulation/spec.md` — "Fictional product catalog as the
source of commercial terms").

IOF parameters (fixed rate + capped daily rate) follow the structure of
the current Brazilian regulation for personal credit operations; the
values are configurable here, not hardcoded into the calculation logic.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class AmortizationType(StrEnum):
    PRICE = "PRICE"
    SAC = "SAC"


class ProductTerms(BaseModel):
    model_config = ConfigDict(frozen=True)

    monthly_interest_rate: Decimal
    iof_fixed_rate: Decimal
    iof_daily_rate: Decimal
    iof_daily_cap_days: int
    origination_fee: Decimal


CATALOG: dict[AmortizationType, ProductTerms] = {
    AmortizationType.PRICE: ProductTerms(
        monthly_interest_rate=Decimal("0.025"),
        iof_fixed_rate=Decimal("0.0038"),
        iof_daily_rate=Decimal("0.000082"),
        iof_daily_cap_days=365,
        origination_fee=Decimal("50.00"),
    ),
    AmortizationType.SAC: ProductTerms(
        monthly_interest_rate=Decimal("0.025"),
        iof_fixed_rate=Decimal("0.0038"),
        iof_daily_rate=Decimal("0.000082"),
        iof_daily_cap_days=365,
        origination_fee=Decimal("50.00"),
    ),
}

DAYS_PER_MONTH = 30
"""Approximation used to convert a term in months to days for the IOF
daily-rate component, capped at `ProductTerms.iof_daily_cap_days`."""


def get_product_terms(amortization_type: AmortizationType) -> ProductTerms:
    """Look up the catalog terms for an amortization type."""
    return CATALOG[amortization_type]

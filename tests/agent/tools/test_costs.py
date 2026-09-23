"""IOF and origination-fee cost computation."""

from __future__ import annotations

from decimal import Decimal

from apps.agent.config.catalog import ProductTerms
from apps.agent.tools.costs import compute_iof, compute_net_released


def test_iof_combines_fixed_and_daily_components_capped_at_term_days() -> None:
    terms = ProductTerms(
        monthly_interest_rate=Decimal("0.02"),
        iof_fixed_rate=Decimal("0.01"),
        iof_daily_rate=Decimal("0.0001"),
        iof_daily_cap_days=365,
        origination_fee=Decimal("0"),
    )

    # principal=1000, term=3 months=90 days (below cap)
    # fixed: 1000*0.01=10.00, daily: 1000*0.0001*90=9.00 -> total 19.00
    iof = compute_iof(Decimal("1000"), 3, terms)

    assert iof == Decimal("19.00")


def test_iof_daily_component_is_capped_at_max_days() -> None:
    terms = ProductTerms(
        monthly_interest_rate=Decimal("0.02"),
        iof_fixed_rate=Decimal("0.01"),
        iof_daily_rate=Decimal("0.0001"),
        iof_daily_cap_days=60,
        origination_fee=Decimal("0"),
    )

    # term=6 months=180 days, capped at 60 days
    # fixed: 1000*0.01=10.00, daily: 1000*0.0001*60=6.00 -> total 16.00
    iof = compute_iof(Decimal("1000"), 6, terms)

    assert iof == Decimal("16.00")


def test_net_released_subtracts_iof_and_fee_from_principal() -> None:
    net_released = compute_net_released(Decimal("1000.00"), Decimal("19.00"), Decimal("50.00"))

    assert net_released == Decimal("931.00")

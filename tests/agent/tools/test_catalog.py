"""The product catalog is the only source of rate/IOF/fee values."""

from __future__ import annotations

from decimal import Decimal

from apps.agent.config.catalog import AmortizationType, get_product_terms


def test_catalog_has_terms_for_every_amortization_type() -> None:
    for amortization_type in AmortizationType:
        terms = get_product_terms(amortization_type)
        assert terms.monthly_interest_rate > 0
        assert terms.iof_fixed_rate >= 0
        assert terms.iof_daily_rate >= 0
        assert terms.iof_daily_cap_days > 0
        assert terms.origination_fee >= 0


def test_catalog_terms_are_configurable_values_not_hardcoded_per_call() -> None:
    terms = get_product_terms(AmortizationType.PRICE)

    assert terms.monthly_interest_rate == Decimal("0.025")
    assert terms.iof_fixed_rate == Decimal("0.0038")
    assert terms.iof_daily_cap_days == 365

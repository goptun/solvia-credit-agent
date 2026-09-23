"""End-to-end simulation: catalog-sourced terms, both amortization types,
and input validation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from apps.agent.config.catalog import AmortizationType, get_product_terms
from apps.agent.tools.exceptions import SimulationInputError
from apps.agent.tools.simulation import simulate


def test_price_simulation_returns_schedule_and_cet_from_catalog_terms() -> None:
    result = simulate(Decimal("5000"), 12, AmortizationType.PRICE)

    terms = get_product_terms(AmortizationType.PRICE)
    assert result.monthly_interest_rate == terms.monthly_interest_rate
    assert len(result.installments) == 12
    assert result.installments[-1].balance == Decimal("0.00")
    assert result.cet_annual > 0
    assert result.net_released < result.principal  # IOF/fees were deducted


def test_sac_simulation_returns_decreasing_installments_and_cet() -> None:
    result = simulate(Decimal("5000"), 12, AmortizationType.SAC)

    payments = [installment.payment for installment in result.installments]
    assert payments == sorted(payments, reverse=True)
    assert result.cet_annual > 0


def test_simulate_accepts_amortization_type_as_a_raw_string() -> None:
    result = simulate(Decimal("1000"), 6, "PRICE")

    assert result.amortization_type == AmortizationType.PRICE


def test_non_positive_amount_is_rejected() -> None:
    with pytest.raises(SimulationInputError):
        simulate(Decimal("0"), 12, AmortizationType.PRICE)


def test_non_positive_term_is_rejected() -> None:
    with pytest.raises(SimulationInputError):
        simulate(Decimal("1000"), 0, AmortizationType.PRICE)


def test_unrecognized_amortization_type_is_rejected() -> None:
    with pytest.raises(SimulationInputError):
        simulate(Decimal("1000"), 12, "HYBRID")


def test_llm_never_supplies_the_simulation_numbers() -> None:
    """The tool alone determines every numeric value in the result —
    the caller only supplies amount/term/type."""
    result = simulate(Decimal("2000"), 6, AmortizationType.SAC)

    assert result.total_paid == sum((i.payment for i in result.installments), Decimal("0"))

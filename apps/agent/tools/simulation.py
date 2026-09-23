"""Credit offer simulation: the tool the agent graph actually calls.

Wires the pure amortization math, the CET/IRR calculation, and the
fictional product catalog together. This is the only entry point graph
nodes should use — never `price_schedule`/`sac_schedule` directly — so
commercial terms always come from the catalog (see
`specs/credit-simulation/spec.md`).
"""

from __future__ import annotations

from decimal import Decimal

from apps.agent.config.catalog import AmortizationType, get_product_terms
from apps.agent.tools.amortization import price_schedule, sac_schedule
from apps.agent.tools.cet import annual_cet
from apps.agent.tools.costs import compute_iof, compute_net_released
from apps.agent.tools.exceptions import SimulationInputError
from apps.agent.tools.models import SimulationResult


def simulate(
    amount: Decimal, term_months: int, amortization_type: AmortizationType | str
) -> SimulationResult:
    """Simulate a credit offer for `amortization_type`, sourcing the
    interest rate, IOF, and fees from the product catalog.

    `amortization_type` accepts a raw string too (e.g. straight from LLM
    slot extraction) and is validated against the known catalog types.
    """
    if amount <= 0:
        raise SimulationInputError("loan amount must be positive")
    if term_months <= 0:
        raise SimulationInputError("term must be a positive number of months")
    try:
        amortization_type = AmortizationType(amortization_type)
    except ValueError as exc:
        raise SimulationInputError(
            f"unrecognized amortization type: {amortization_type!r}"
        ) from exc

    terms = get_product_terms(amortization_type)
    rate = terms.monthly_interest_rate

    schedule_fn = price_schedule if amortization_type == AmortizationType.PRICE else sac_schedule
    installments = schedule_fn(amount, rate, term_months)

    iof_total = compute_iof(amount, term_months, terms)
    net_released = compute_net_released(amount, iof_total, terms.origination_fee)
    cet = annual_cet(net_released, [installment.payment for installment in installments])
    total_paid = sum((installment.payment for installment in installments), Decimal("0"))

    return SimulationResult(
        amortization_type=amortization_type,
        principal=amount,
        monthly_interest_rate=rate,
        term_months=term_months,
        iof_total=iof_total,
        origination_fee=terms.origination_fee,
        net_released=net_released,
        installments=installments,
        total_paid=total_paid,
        cet_annual=cet,
    )

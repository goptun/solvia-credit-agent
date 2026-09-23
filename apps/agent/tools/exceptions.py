"""Domain exceptions for the deterministic financial tools."""

from __future__ import annotations


class SimulationInputError(ValueError):
    """Raised for invalid simulation inputs.

    Covers a non-positive loan amount, a non-positive term, a negative
    interest rate, or an unrecognized amortization type (see
    `specs/credit-simulation/spec.md` — "Invalid simulation inputs are
    rejected").
    """


class CetCalculationError(ValueError):
    """Raised when the CET/IRR bisection search cannot bracket a root —
    i.e. the net present value has the same sign at both ends of the
    search range (0%-200% a.m.). This means the net cash flow is
    degenerate (e.g. `net_released` at or above the total of the
    payments, so no positive monthly rate reconciles them), not that a
    real rate exists but bisection missed it — returning a fixed
    fallback rate in that case would silently misreport the CET.
    """

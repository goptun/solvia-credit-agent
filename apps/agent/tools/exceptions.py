"""Domain exceptions for the deterministic financial tools."""

from __future__ import annotations


class SimulationInputError(ValueError):
    """Raised for invalid simulation inputs.

    Covers a non-positive loan amount, a non-positive term, a negative
    interest rate, or an unrecognized amortization type (see
    `specs/credit-simulation/spec.md` — "Invalid simulation inputs are
    rejected").
    """

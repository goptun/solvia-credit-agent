"""Default regression tolerances (design.md, Decision 6).

Proportions and MRR: ±0.03 overall (about two items of ~75), ±0.06 per
stratum (smaller samples). Deterministic metrics: zero — any change fails."""

from __future__ import annotations

from evals.core.run import MetricValue

OVERALL_TOLERANCE = 0.03
STRATUM_TOLERANCE = 0.06

_STRATUM_INFIXES = (".style.", ".document.", ".difficulty.", ".category.")
_STRATUM_SUFFIXES = (".far", ".near_miss")


def is_stratum(name: str) -> bool:
    """A breakdown of a headline metric over a subset of the dataset (by style,
    document, difficulty, category, or far/near-miss negatives)."""
    return any(infix in name for infix in _STRATUM_INFIXES) or name.endswith(_STRATUM_SUFFIXES)


def default_tolerance(name: str, metric: MetricValue) -> float:
    if metric.deterministic:
        return 0.0
    return STRATUM_TOLERANCE if is_stratum(name) else OVERALL_TOLERANCE

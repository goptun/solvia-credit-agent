"""Default regression tolerances (design.md, Decision 6).

Proportions and MRR: ±0.03 overall (about two items of ~75), ±0.06 per
stratum (smaller samples). Deterministic metrics: zero — any change fails."""

from __future__ import annotations

from evals.core.run import MetricValue

OVERALL_TOLERANCE = 0.03
STRATUM_TOLERANCE = 0.06


def default_tolerance(name: str, metric: MetricValue) -> float:
    if metric.deterministic:
        return 0.0
    return STRATUM_TOLERANCE if "." in name else OVERALL_TOLERANCE

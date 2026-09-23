"""Confidence intervals for small samples.

Proportions use the Wilson score interval (well behaved at 0/n and n/n);
means use a seeded percentile bootstrap so a report is reproducible.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

_Z_95 = 1.959963984540054
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class Proportion:
    """`successes` out of `n`, with its 95% Wilson interval."""

    successes: int
    n: int
    ci: Interval

    @property
    def value(self) -> float:
        return self.successes / self.n if self.n else 0.0


@dataclass(frozen=True)
class MeanEstimate:
    """A mean over `n` values with its 95% bootstrap interval."""

    value: float
    n: int
    ci: Interval


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> Interval:
    if n <= 0:
        return Interval(0.0, 1.0)
    if not 0 <= successes <= n:
        raise ValueError(f"successes must be within 0..n, got {successes}/{n}")
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return Interval(max(0.0, centre - margin), min(1.0, centre + margin))


def proportion(successes: int, n: int) -> Proportion:
    return Proportion(successes=successes, n=n, ci=wilson_interval(successes, n))


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile (`q` in 0..1); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_mean(
    values: Sequence[float], seed: int, resamples: int = BOOTSTRAP_RESAMPLES
) -> MeanEstimate:
    if not values:
        return MeanEstimate(0.0, 0, Interval(0.0, 0.0))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    low = means[int(0.025 * (resamples - 1))]
    high = means[int(math.ceil(0.975 * (resamples - 1)))]
    return MeanEstimate(sum(values) / n, n, Interval(low, high))

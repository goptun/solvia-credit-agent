"""Wilson and bootstrap intervals, checked against hand-computed values."""

from __future__ import annotations

import pytest

from evals.core.stats import bootstrap_mean, proportion, quantile, wilson_interval


def test_wilson_for_zero_successes() -> None:
    interval = wilson_interval(0, 10)

    assert interval.low == 0.0
    assert interval.high == pytest.approx(0.2775, abs=1e-3)


def test_wilson_for_all_successes() -> None:
    interval = wilson_interval(10, 10)

    assert interval.low == pytest.approx(0.7225, abs=1e-3)
    assert interval.high == pytest.approx(1.0)


def test_wilson_for_seven_of_twenty_five() -> None:
    interval = wilson_interval(7, 25)

    assert interval.low == pytest.approx(0.1428, abs=1e-3)
    assert interval.high == pytest.approx(0.4758, abs=1e-3)


def test_wilson_with_no_observations_is_the_whole_range() -> None:
    interval = wilson_interval(0, 0)

    assert (interval.low, interval.high) == (0.0, 1.0)


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_proportion_value_and_interval() -> None:
    result = proportion(7, 25)

    assert result.value == pytest.approx(0.28)
    assert result.ci.low < result.value < result.ci.high


def test_quantile_interpolates() -> None:
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert quantile([], 0.5) == 0.0


def test_bootstrap_is_reproducible_for_a_seed_and_differs_across_seeds() -> None:
    values = [0.0, 1.0, 0.5, 1.0, 0.25, 0.0, 1.0, 0.5]

    first = bootstrap_mean(values, seed=42, resamples=500)
    second = bootstrap_mean(values, seed=42, resamples=500)
    other = bootstrap_mean(values, seed=7, resamples=500)

    assert first == second
    assert first.value == pytest.approx(sum(values) / len(values))
    assert first.ci.low <= first.value <= first.ci.high
    assert (first.ci.low, first.ci.high) != (other.ci.low, other.ci.high)


def test_bootstrap_of_no_values_is_empty() -> None:
    estimate = bootstrap_mean([], seed=1)

    assert estimate.n == 0

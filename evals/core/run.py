"""The machine-readable record of an evaluation run.

Offline runs are reproducible byte for byte, so the record carries no
timestamp (a baseline adds its own creation time when it is written)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from evals.core.baseline import DatasetRef, Direction
from evals.core.stats import MeanEstimate, Proportion


class MetricValue(BaseModel):
    """One metric of a suite, with its 95% interval."""

    model_config = ConfigDict(frozen=True)

    value: float
    n: int
    direction: Direction
    ci_low: float | None = None
    ci_high: float | None = None
    deterministic: bool = False
    """Depends only on code and data (no model, no floats that vary by machine):
    any change at all is a change, so its baseline tolerance is zero."""


def proportion_metric(
    value: Proportion, direction: Direction, *, deterministic: bool = False
) -> MetricValue:
    return MetricValue(
        value=round(value.value, 6),
        n=value.n,
        direction=direction,
        ci_low=round(value.ci.low, 6),
        ci_high=round(value.ci.high, 6),
        deterministic=deterministic,
    )


def mean_metric(value: MeanEstimate, direction: Direction) -> MetricValue:
    return MetricValue(
        value=round(value.value, 6),
        n=value.n,
        direction=direction,
        ci_low=round(value.ci.low, 6),
        ci_high=round(value.ci.high, 6),
    )


class SuiteResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    datasets: list[DatasetRef]
    metrics: dict[str, MetricValue]
    details: dict[str, Any] = {}
    """Supporting, JSON-serializable data (failed item ids, tables)."""


class RunRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["offline", "live"]
    git_sha: str
    git_dirty: bool
    seed: int
    environment: dict[str, str]
    suites: dict[str, SuiteResult]
    sample_fraction: float | None = None
    """Set when only a stratified fraction of each dataset ran."""
    incomplete: bool = False
    """The run stopped before covering every selected item (e.g. the budget)."""
    contaminated: bool = False
    contamination_reasons: list[str] = []
    model_sets_unset: bool = False
    """Live only: no approved expected-model set, so the mix cannot be judged."""
    aliases: dict[str, str] = {}
    budget: dict[str, int] = {}
    """Live: `max_calls`, `used`, `estimated_typical`, `estimated_pessimistic`."""
    resolved_model_mix: dict[str, dict[str, int]] = {}

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

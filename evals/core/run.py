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

    def to_json(self) -> str:
        return self.model_dump_json(indent=2) + "\n"

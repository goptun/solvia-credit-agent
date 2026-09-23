"""Baseline records and the tolerance-based regression comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

Direction = Literal["higher", "lower"]
Status = Literal["ok", "improved", "regression", "changed", "missing", "new"]

_EPSILON = 1e-9
"""Absorbs float noise so a change of exactly the tolerance is not a regression."""

FAILING: frozenset[str] = frozenset({"regression", "changed", "missing"})


class DatasetRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: int
    sha256: str


class MetricBaseline(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: float
    n: int
    direction: Direction
    tolerance: float
    ci_low: float | None = None
    ci_high: float | None = None


class Baseline(BaseModel):
    """A committed baseline for one suite (design.md, Decision 6)."""

    model_config = ConfigDict(frozen=True)

    suite: str
    mode: Literal["offline", "live"]
    git_sha: str
    datasets: list[DatasetRef]
    created_at: str
    environment: dict[str, str] = {}
    aliases: dict[str, str] = {}
    resolved_model_mix: dict[str, dict[str, int]] = {}
    metrics: dict[str, MetricBaseline]


@dataclass(frozen=True)
class Comparison:
    metric: str
    baseline: float | None
    new: float | None
    tolerance: float
    diff: float | None
    status: Status


def _status(baseline: MetricBaseline, new: float) -> Status:
    diff = new - baseline.value
    if baseline.tolerance == 0.0:
        return "ok" if diff == 0.0 else "changed"
    worse = -diff if baseline.direction == "higher" else diff
    if worse > baseline.tolerance + _EPSILON:
        return "regression"
    if -worse > baseline.tolerance + _EPSILON:
        return "improved"
    return "ok"


def compare(baseline: dict[str, MetricBaseline], new: dict[str, float]) -> list[Comparison]:
    """One row per metric, in baseline order then any new metrics. A metric
    with zero tolerance fails on any change at all (deterministic checks)."""
    rows = []
    for name, reference in baseline.items():
        if name not in new:
            rows.append(
                Comparison(name, reference.value, None, reference.tolerance, None, "missing")
            )
            continue
        rows.append(
            Comparison(
                name,
                reference.value,
                new[name],
                reference.tolerance,
                new[name] - reference.value,
                _status(reference, new[name]),
            )
        )
    for name, value in new.items():
        if name not in baseline:
            rows.append(Comparison(name, None, value, 0.0, None, "new"))
    return rows


def has_regression(rows: list[Comparison]) -> bool:
    return any(row.status in FAILING for row in rows)


def _cell(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def render_table(rows: list[Comparison]) -> str:
    lines = [
        "| metric | baseline | new | tolerance | diff | status |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.metric} | {_cell(row.baseline)} | {_cell(row.new)} | "
            f"{row.tolerance:.4f} | {_cell(row.diff, signed=True)} | {row.status} |"
        )
    improved = [row.metric for row in rows if row.status == "improved"]
    if improved:
        lines.append("")
        lines.append(
            "Improved beyond tolerance: "
            + ", ".join(improved)
            + " — consider updating the baseline (`python -m evals baseline update`)."
        )
    return "\n".join(lines)

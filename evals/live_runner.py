"""The live runner: sequential, paced, budgeted, contamination-checked.

Runs the selected live suites item by item through one instrumented factory,
so the call budget and the pacing are exact. Never raises for gateway trouble:
a run that could not finish is returned marked incomplete/contaminated."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from evals.adapters.instrumentation import CallRecorder
from evals.core.baseline import DatasetRef
from evals.core.budget import CallBudget, Estimate, check_estimate, estimate_calls
from evals.core.contamination import ModelSets, assess_contamination
from evals.core.operational import CallRecord, model_mix, node_metrics
from evals.core.run import MetricValue, RunRecord, SuiteResult, proportion_metric
from evals.core.sampling import sample_fraction
from evals.datasets import LoadedDataset, load_dataset
from evals.suites.live import LIVE_SUITES, LiveContext, LiveSuite

OPERATIONAL_SUITE = "operational"


class UnknownLiveSuite(ValueError):
    """A requested suite name is not a live suite."""


class Clock(Protocol):
    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class RealClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(frozen=True)
class PlannedSuite:
    suite: LiveSuite[Any, Any]
    loaded: LoadedDataset
    items: Sequence[Any]


@dataclass(frozen=True)
class RunSettings:
    seed: int
    pacing_seconds: float
    max_error_share: float
    max_fallback_share: float
    sample: float | None = None


@dataclass(frozen=True)
class RunMeta:
    git_sha: str
    git_dirty: bool
    environment: dict[str, str]
    aliases: dict[str, str]


def build_plan(
    names: Sequence[str],
    *,
    sample: float | None,
    seed: int,
    suites: Mapping[str, LiveSuite[Any, Any]] | None = None,
) -> list[PlannedSuite]:
    """The items each suite will run: everything, or a stratified seeded
    fraction (same strata as `review-sample`)."""
    available = suites if suites is not None else LIVE_SUITES
    unknown = [name for name in names if name not in available]
    if unknown:
        raise UnknownLiveSuite(
            f"unknown live suite(s): {', '.join(unknown)} (live suites: {', '.join(available)})"
        )
    loaded_by_name: dict[str, LoadedDataset] = {}
    plan = []
    for name in names:
        suite = available[name]
        loaded = loaded_by_name.setdefault(suite.dataset, load_dataset(suite.dataset))
        items = suite.items(loaded)
        if sample is not None:
            items = sample_fraction(items, suite.stratum(loaded), sample, seed)
        plan.append(PlannedSuite(suite, loaded, list(items)))
    return plan


def estimate_plan(plan: Sequence[PlannedSuite]) -> Estimate:
    counts = {planned.suite.name: len(planned.items) for planned in plan}
    grounding = next((p.items for p in plan if p.suite.name == "grounding"), [])
    answerable = sum(1 for item in grounding if item.kind == "answerable")
    return estimate_calls(
        router=counts.get("router", 0),
        slots=counts.get("slots", 0),
        compliance_llm=counts.get("compliance-llm", 0),
        grounding_answerable=answerable,
        grounding_unanswerable=len(grounding) - answerable,
    )


def _dataset_ref(loaded: LoadedDataset) -> DatasetRef:
    return DatasetRef(name=loaded.name, version=loaded.version, sha256=loaded.sha256)


async def _pace(clock: Clock, last_start: float | None, interval: float) -> float:
    """Wait until `interval` has passed since the previous item started;
    returns the new item's start time."""
    if last_start is not None:
        wait = interval - (clock.monotonic() - last_start)
        if wait > 0:
            await clock.sleep(wait)
    return clock.monotonic()


def _operational_metrics(records: Sequence[CallRecord]) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {}
    for node, stats in node_metrics(records).items():
        metrics[f"{node}.no_tool_call_rate"] = proportion_metric(stats.no_tool_call_rate, "lower")
        metrics[f"{node}.json_fallback_rate"] = proportion_metric(stats.json_fallback_rate, "lower")
        for label, value in (
            ("latency_p50_seconds", stats.latency_p50),
            ("latency_p95_seconds", stats.latency_p95),
            ("attempts_mean", stats.attempts_mean),
        ):
            metrics[f"{node}.{label}"] = MetricValue(
                value=round(value, 3), n=stats.calls, direction="lower"
            )
    return metrics


def _call_log(records: Sequence[CallRecord]) -> list[dict[str, Any]]:
    return [
        {
            "operation": r.operation_id,
            "node": r.node,
            "alias": r.alias,
            "model": r.model,
            "status": r.status,
            "kind": r.kind,
            "latency_seconds": round(r.latency_seconds, 3),
        }
        for r in records
    ]


async def run_live(
    plan: Sequence[PlannedSuite],
    ctx: LiveContext,
    recorder: CallRecorder,
    budget: CallBudget,
    settings: RunSettings,
    meta: RunMeta,
    model_sets: Mapping[str, ModelSets] | None,
    clock: Clock | None = None,
) -> RunRecord:
    """Run the planned suites. The caller has already refused an over-budget
    estimate (`check_estimate`)."""
    clock = clock or RealClock()
    estimate = estimate_plan(plan)
    completed: dict[str, list[tuple[Any, Any]]] = {p.suite.name: [] for p in plan}
    unavailable: dict[str, dict[str, str]] = {p.suite.name: {} for p in plan}
    interrupted = False
    last_start: float | None = None

    try:
        for planned in plan:
            suite = planned.suite
            for item in planned.items:
                if not budget.can_start_item():
                    break
                last_start = await _pace(clock, last_start, settings.pacing_seconds)
                try:
                    outcome = await suite.run_item(item, ctx)
                except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
                    unavailable[suite.name][suite.item_id(item)] = type(exc).__name__
                    continue
                if outcome is None:
                    unavailable[suite.name].setdefault(suite.item_id(item), "unavailable")
                else:
                    completed[suite.name].append((item, outcome))
            if budget.stopped_by_budget:
                break
    except (asyncio.CancelledError, KeyboardInterrupt):
        interrupted = True

    incomplete = budget.stopped_by_budget or any(
        len(completed[p.suite.name]) + len(unavailable[p.suite.name]) < len(p.items) for p in plan
    )
    suites: dict[str, SuiteResult] = {}
    for planned in plan:
        name = planned.suite.name
        summary = planned.suite.summarize(completed[name], list(unavailable[name]), settings.seed)
        suites[name] = SuiteResult(
            datasets=[
                DatasetRef(
                    name=planned.loaded.name,
                    version=planned.loaded.version,
                    sha256=planned.loaded.sha256,
                )
            ],
            metrics=summary.metrics,
            details={
                **summary.details,
                "items_selected": len(planned.items),
                "items_scored": len(completed[name]),
                "unavailable": unavailable[name],
            },
        )
    suites[OPERATIONAL_SUITE] = SuiteResult(
        datasets=[],
        metrics=_operational_metrics(recorder.records),
        details={"calls": _call_log(recorder.records)},
    )

    verdict = assess_contamination(
        recorder.records,
        model_sets,
        max_error_share=settings.max_error_share,
        max_fallback_share=settings.max_fallback_share,
        incomplete=incomplete,
        interrupted=interrupted,
    )
    return RunRecord(
        mode="live",
        git_sha=meta.git_sha,
        git_dirty=meta.git_dirty,
        seed=settings.seed,
        environment=meta.environment,
        suites=suites,
        sample_fraction=settings.sample,
        incomplete=incomplete,
        contaminated=verdict.contaminated,
        contamination_reasons=list(verdict.reasons),
        model_sets_unset=verdict.model_sets_unset,
        aliases=meta.aliases,
        budget={
            "max_calls": budget.max_calls,
            "used": budget.used,
            "estimated_typical": estimate.typical,
            "estimated_pessimistic": estimate.pessimistic,
        },
        resolved_model_mix=model_mix(recorder.records),
    )


def refuse_over_budget(plan: Sequence[PlannedSuite], max_calls: int) -> Estimate:
    """The pre-run estimate; raises `BudgetExceeded` when it cannot fit."""
    estimate = estimate_plan(plan)
    check_estimate(estimate, max_calls)
    return estimate

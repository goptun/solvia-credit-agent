"""Eligibility of a run to become a committed baseline (design.md, Decision 6).

Pure: the facts about the repository (HEAD, dirtiness, what changed since a
commit) are passed in, so every refusal is unit-testable."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from evals.core.approval import Approval, approval_problem
from evals.core.baseline import Baseline, MetricBaseline
from evals.core.run import RunRecord
from evals.core.tolerances import default_tolerance

Source = Literal["run", "artifact"]

BASELINES_PREFIX = "evals/baselines/"
RETRIEVAL_SUITE = "retrieval"
CI_ARCHITECTURES = frozenset({"x86_64", "amd64"})


@dataclass(frozen=True)
class RepoState:
    head_sha: str
    dirty: bool
    approvals: Mapping[str, Approval]
    changed_since: Callable[[str], list[str] | None]
    """Files changed between a commit and HEAD; `None` if it is not an ancestor."""


def _quality_problems(record: RunRecord) -> list[str]:
    problems = []
    if record.contaminated:
        reasons = "; ".join(record.contamination_reasons) or "no reason recorded"
        problems.append(f"the run is contaminated ({reasons})")
    if record.incomplete:
        problems.append("the run is incomplete")
    if record.sample_fraction is not None:
        problems.append(f"the run is a {record.sample_fraction:.0%} sample, not the full suite")
    if record.mode == "live" and record.model_sets_unset:
        problems.append("no expected model set is approved for the gateway aliases")
    return problems


def _source_problems(record: RunRecord, suite: str, source: Source, repo: RepoState) -> list[str]:
    problems = []
    if repo.dirty:
        problems.append("the working tree is dirty")
    if record.mode == "offline" and suite == RETRIEVAL_SUITE and source == "run":
        problems.append(
            "the offline retrieval baseline must come from the CI artifact "
            "(--from-artifact), not from a local run"
        )
    if source == "run":
        if record.git_dirty:
            problems.append("the run was produced on a dirty tree")
        if record.git_sha != repo.head_sha:
            problems.append(f"the run's commit {record.git_sha[:7]} is not HEAD")
        return problems
    if record.environment.get("ci") != "true":
        problems.append("the artifact was not produced by CI")
    if record.environment.get("arch") not in CI_ARCHITECTURES:
        problems.append(f"the artifact ran on {record.environment.get('arch')!r}, not x86_64")
    changed = repo.changed_since(record.git_sha)
    if changed is None:
        problems.append(f"the artifact's commit {record.git_sha[:7]} is not an ancestor of HEAD")
    else:
        outside = [path for path in changed if not path.startswith(BASELINES_PREFIX)]
        if outside:
            problems.append(
                f"files changed since the artifact's commit: {', '.join(outside[:5])} "
                f"(only {BASELINES_PREFIX} may differ)"
            )
    return problems


def baseline_problems(
    record: RunRecord, suite: str, mode: Literal["offline", "live"], source: Source, repo: RepoState
) -> list[str]:
    """Every reason the run cannot become the `suite` baseline; empty if eligible."""
    if record.mode != mode:
        return [f"the run mode is {record.mode!r}, expected {mode!r}"]
    result = record.suites.get(suite)
    if result is None:
        return [f"the run has no suite {suite!r} (has: {', '.join(record.suites) or 'none'})"]
    problems = _quality_problems(record) + _source_problems(record, suite, source, repo)
    for dataset in result.datasets:
        problem = approval_problem(dataset.name, dataset.version, dataset.sha256, repo.approvals)
        if problem is not None:
            problems.append(problem)
    return problems


def build_baseline(record: RunRecord, suite: str, created_at: str) -> Baseline:
    result = record.suites[suite]
    return Baseline(
        suite=suite,
        mode=record.mode,
        git_sha=record.git_sha,
        datasets=result.datasets,
        created_at=created_at,
        environment=record.environment,
        aliases=record.aliases,
        resolved_model_mix=record.resolved_model_mix,
        metrics={
            name: MetricBaseline(
                value=metric.value,
                n=metric.n,
                direction=metric.direction,
                tolerance=default_tolerance(name, metric),
                ci_low=metric.ci_low,
                ci_high=metric.ci_high,
            )
            for name, metric in result.metrics.items()
        },
    )

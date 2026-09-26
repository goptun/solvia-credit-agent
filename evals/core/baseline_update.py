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
_UNRELATED_TO_MEASUREMENT_PREFIXES = ("tests/", "evals/reports/", BASELINES_PREFIX, "docs/")
_UNRELATED_TO_MEASUREMENT_FILES = frozenset(
    {
        # The baseline/report tooling: it decides whether a run can become a
        # baseline and how a baseline is rendered, but it plays no part in
        # producing the suite's own result — changing it cannot retroactively
        # alter what a run measured.
        "evals/core/baseline_update.py",
        "evals/core/report.py",
    }
)
"""A live run's own commit necessarily predates the commit of its own report
(you cannot commit a file before the command that produces it has run): requiring
the run's commit to equal HEAD exactly would make a live run un-baselineable the
moment its report is committed. An explicit allowlist, not a broad `evals/`
exemption: application/rag code, datasets, `live_config.yaml`, and the suites
that actually score an item (`evals/suites/**`, `evals/core/*metrics*`, ...)
are never exempt — only paths that describe how a run is reported or tested,
never how one is scored, may differ since the run's commit."""
RETRIEVAL_SUITE = "retrieval"
CI_ARCHITECTURES = frozenset({"x86_64", "amd64"})


def _unrelated_to_measurement(path: str) -> bool:
    if path.startswith(_UNRELATED_TO_MEASUREMENT_PREFIXES):
        return True
    return path in _UNRELATED_TO_MEASUREMENT_FILES or path.endswith(".md")


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
            problems.extend(
                _ancestor_problems(record.git_sha, repo, "run", _unrelated_to_measurement)
            )
        return problems
    if record.environment.get("ci") != "true":
        problems.append("the artifact was not produced by CI")
    if record.environment.get("arch") not in CI_ARCHITECTURES:
        problems.append(f"the artifact ran on {record.environment.get('arch')!r}, not x86_64")
    problems.extend(
        _ancestor_problems(
            record.git_sha, repo, "artifact", lambda p: p.startswith(BASELINES_PREFIX)
        )
    )
    return problems


def _ancestor_problems(
    sha: str, repo: RepoState, label: str, is_allowed: Callable[[str], bool]
) -> list[str]:
    changed = repo.changed_since(sha)
    if changed is None:
        return [f"the {label}'s commit {sha[:7]} is not an ancestor of HEAD"]
    outside = [path for path in changed if not is_allowed(path)]
    if not outside:
        return []
    return [
        f"files changed since the {label}'s commit that could affect it: {', '.join(outside[:5])}"
    ]


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

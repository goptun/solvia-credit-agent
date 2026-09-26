"""The regression gate: compare a run against committed baselines."""

from __future__ import annotations

from dataclasses import dataclass

from evals.core.baseline import Baseline, compare, has_regression, render_table
from evals.core.run import RunRecord

RETRIEVAL_SUITE = "retrieval"

NO_BASELINE_NOTICE = (
    "NOTICE: no baseline recorded for suite {suite!r} yet — nothing to compare against "
    "(record one with `python -m evals baseline update`)."
)
DATASET_CHANGED = (
    "dataset {name!r} is v{new} ({new_hash}) but the baseline was recorded on "
    "v{old} ({old_hash}); update the baseline after the dataset review"
)


@dataclass(frozen=True)
class GateOutcome:
    text: str
    exit_code: int


def _informational(suite: str, record: RunRecord) -> bool:
    """Retrieval numbers depend on the CPU's float behavior: outside the CI
    environment they are shown but never fail."""
    return suite == RETRIEVAL_SUITE and record.environment.get("ci") != "true"


def gate(record: RunRecord, baselines: dict[str, Baseline]) -> GateOutcome:
    sections: list[str] = []
    failed = False
    for name, suite in record.suites.items():
        baseline = baselines.get(name)
        if baseline is None:
            sections.append(NO_BASELINE_NOTICE.format(suite=name))
            continue
        problems = [
            DATASET_CHANGED.format(
                name=new.name,
                new=new.version,
                new_hash=new.sha256[:8],
                old=old.version,
                old_hash=old.sha256[:8],
            )
            for new, old in zip(suite.datasets, baseline.datasets, strict=False)
            if new != old
        ]
        rows = compare(baseline.metrics, {k: v.value for k, v in suite.metrics.items()})
        regression = bool(problems) or has_regression(rows)
        informational = _informational(name, record)
        status = "regression" if regression else "ok"
        if regression and informational:
            status = "regression (informational — not the CI environment)"
        heading = f"## {name}: {status}"
        sections.append("\n\n".join([heading, *problems, render_table(rows)]))
        failed = failed or (regression and not informational)
    return GateOutcome("\n\n".join(sections) + "\n", 1 if failed else 0)

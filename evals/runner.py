"""Runs suites and assembles the run record (the `run` command)."""

from __future__ import annotations

from collections.abc import Callable

from evals.adapters.environment import environment, git_dirty, git_sha
from evals.core.run import RunRecord, SuiteResult
from evals.suites.compliance_offline import SUITE as COMPLIANCE
from evals.suites.compliance_offline import run_compliance_offline

OFFLINE_SUITES: dict[str, Callable[[], SuiteResult]] = {COMPLIANCE: run_compliance_offline}


class UnknownSuite(ValueError):
    """The requested suite does not exist in the requested mode."""


def run_offline(suites: list[str], seed: int) -> RunRecord:
    results: dict[str, SuiteResult] = {}
    for name in suites:
        if name not in OFFLINE_SUITES:
            raise UnknownSuite(f"no offline suite {name!r}; available: {sorted(OFFLINE_SUITES)}")
        results[name] = OFFLINE_SUITES[name]()
    return RunRecord(
        mode="offline",
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        seed=seed,
        environment=environment(),
        suites=results,
    )

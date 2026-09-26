"""Baseline eligibility (one test per refusal) and the regression gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pytest

from evals.cli import main
from evals.core.approval import Approval
from evals.core.baseline import Baseline, DatasetRef, MetricBaseline
from evals.core.baseline_update import RepoState, Source, baseline_problems, build_baseline
from evals.core.gate import gate
from evals.core.run import MetricValue, RunRecord, SuiteResult

HEAD = "a" * 40
CI_ENV = {"ci": "true", "os": "linux", "arch": "x86_64", "python": "3.12.0"}
LOCAL_ENV = {"ci": "false", "os": "darwin", "arch": "arm64", "python": "3.12.0"}
DATASET = DatasetRef(name="retrieval", version=2, sha256="h1")


def _record(
    suite: str = "retrieval",
    *,
    mode: Literal["offline", "live"] = "offline",
    sha: str = HEAD,
    env: dict[str, str] | None = None,
    recall: float = 0.7,
    **extra: Any,
) -> RunRecord:
    return RunRecord.model_validate(
        {
            "mode": mode,
            "git_sha": sha,
            "git_dirty": False,
            "seed": 42,
            "environment": env or LOCAL_ENV,
            "suites": {
                suite: SuiteResult(
                    datasets=[DATASET],
                    metrics={
                        "recall_at_k": MetricValue(
                            value=recall, n=75, direction="higher", ci_low=0.6, ci_high=0.8
                        ),
                        "recall_at_k.style.lexical": MetricValue(
                            value=0.8, n=40, direction="higher"
                        ),
                        "precision": MetricValue(
                            value=1.0, n=7, direction="higher", deterministic=True
                        ),
                    },
                )
            },
            **extra,
        }
    )


_APPROVED = {"retrieval": Approval(version=2, sha256="h1")}


def _repo(
    *,
    dirty: bool = False,
    changed: list[str] | None = None,
    approvals: dict[str, Approval] | None = None,
) -> RepoState:
    return RepoState(
        head_sha=HEAD,
        dirty=dirty,
        approvals=approvals
        if approvals is not None
        else {"retrieval": Approval(version=2, sha256="h1")},
        changed_since=lambda _commit: changed,
    )


def _problems(record: RunRecord, source: Source = "run", repo: RepoState | None = None) -> str:
    suite = next(iter(record.suites))
    found = baseline_problems(
        record,
        suite,
        record.mode,
        source,
        repo or _repo(changed=[]),
    )
    return " | ".join(found)


def test_an_eligible_compliance_run_has_no_problems() -> None:
    record = _record("compliance")
    repo = _repo(approvals={"retrieval": Approval(version=2, sha256="h1")})

    assert baseline_problems(record, "compliance", "offline", "run", repo) == []


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"contaminated": True, "contamination_reasons": ["429 share 20%"]}, "contaminated"),
        ({"incomplete": True}, "incomplete"),
        ({"sample_fraction": 0.3}, "30% sample"),
        ({"mode": "live", "model_sets_unset": True}, "expected model set"),
    ],
)
def test_quality_problems_are_refused(extra: dict[str, Any], expected: str) -> None:
    record = _record("router", **extra)
    repo = _repo(approvals={"retrieval": Approval(version=2, sha256="h1")})

    assert expected in " | ".join(baseline_problems(record, "router", record.mode, "run", repo))


def test_a_dirty_tree_is_refused() -> None:
    assert "working tree is dirty" in _problems(_record("compliance"), repo=_repo(dirty=True))


def test_a_run_whose_commit_is_not_an_ancestor_of_head_is_refused() -> None:
    record = _record("compliance", sha="b" * 40)

    assert "not an ancestor of HEAD" in _problems(record, repo=_repo(changed=None))


def test_a_run_followed_by_a_code_change_is_refused() -> None:
    record = _record("router", sha="b" * 40)
    repo = _repo(changed=["evals/reports/live-baseline-router.md", "apps/agent/nodes/router.py"])

    problems = _problems(record, repo=repo)

    assert "apps/agent/nodes/router.py" in problems
    assert "evals/reports/live-baseline-router.md" not in problems


def test_a_run_followed_only_by_its_own_report_commit_is_accepted() -> None:
    """A live run necessarily predates the commit of its own report — requiring
    the run's commit to equal HEAD exactly would make it un-baselineable the
    moment that report is committed."""
    record = _record("router", sha="b" * 40)
    repo = _repo(
        changed=[
            "evals/reports/live-baseline-router.json",
            "evals/reports/live-baseline-router.md",
        ]
    )

    assert _problems(record, repo=repo) == ""


def test_a_run_from_a_dirty_tree_is_refused() -> None:
    assert "dirty tree" in _problems(_record("compliance", git_dirty=True))


def test_an_unapproved_or_changed_dataset_is_refused() -> None:
    assert "no recorded approval" in _problems(_record("compliance"), repo=_repo(approvals={}))
    changed = {"retrieval": Approval(version=1, sha256="other")}
    assert "differs from the approved" in _problems(
        _record("compliance"), repo=_repo(approvals=changed)
    )


def test_a_mode_or_suite_mismatch_is_refused() -> None:
    record = _record("compliance")

    assert "expected 'live'" in " ".join(
        baseline_problems(record, "compliance", "live", "run", _repo())
    )
    assert "no suite 'router'" in " ".join(
        baseline_problems(record, "router", "offline", "run", _repo())
    )


def test_offline_retrieval_from_a_local_run_is_refused() -> None:
    assert "CI artifact" in _problems(_record("retrieval"), "run")


def test_a_local_arm_artifact_is_refused() -> None:
    problems = _problems(_record("retrieval", env=LOCAL_ENV), "artifact")

    assert "not produced by CI" in problems and "not x86_64" in problems


def test_an_artifact_whose_commit_is_not_an_ancestor_is_refused() -> None:
    record = _record("retrieval", sha="c" * 40, env=CI_ENV)

    assert "not an ancestor of HEAD" in _problems(record, "artifact", _repo(changed=None))


def test_an_artifact_followed_by_a_code_change_is_refused() -> None:
    record = _record("retrieval", sha="c" * 40, env=CI_ENV)
    repo = _repo(changed=["evals/baselines/compliance.json", "rag/retrieval/hybrid.py"])

    problems = _problems(record, "artifact", repo)

    assert (
        "rag/retrieval/hybrid.py" in problems and "evals/baselines/compliance.json" not in problems
    )


def test_a_ci_artifact_followed_only_by_baseline_commits_is_accepted() -> None:
    record = _record("retrieval", sha="c" * 40, env=CI_ENV)
    repo = _repo(changed=["evals/baselines/compliance.json"])

    assert _problems(record, "artifact", repo) == ""


def test_the_written_baseline_records_provenance_and_tolerances() -> None:
    record = _record(
        "router",
        mode="live",
        env=CI_ENV,
        aliases={"solvia-fast": "primary"},
        resolved_model_mix={"solvia-fast": {"primary": 30}},
    )

    baseline = build_baseline(record, "router", "2026-09-23T10:00:00Z")

    assert baseline.git_sha == HEAD
    assert baseline.datasets == [DATASET]
    assert baseline.aliases == {"solvia-fast": "primary"}
    assert baseline.resolved_model_mix == {"solvia-fast": {"primary": 30}}
    assert baseline.created_at == "2026-09-23T10:00:00Z"
    assert baseline.metrics["recall_at_k"].tolerance == 0.03
    assert baseline.metrics["recall_at_k.style.lexical"].tolerance == 0.06
    assert baseline.metrics["precision"].tolerance == 0.0


def _baseline(suite: str = "retrieval", recall: float = 0.7) -> Baseline:
    return Baseline(
        suite=suite,
        mode="offline",
        git_sha="deadbee",
        datasets=[DATASET],
        created_at="2026-09-23T10:00:00Z",
        metrics={
            "recall_at_k": MetricBaseline(value=recall, n=75, direction="higher", tolerance=0.03)
        },
    )


def test_the_gate_passes_within_tolerance_and_fails_on_regression() -> None:
    within = gate(_record(env=CI_ENV, recall=0.68), {"retrieval": _baseline()})
    regressed = gate(_record(env=CI_ENV, recall=0.6), {"retrieval": _baseline()})

    assert within.exit_code == 0
    assert regressed.exit_code == 1
    assert "| recall_at_k | 0.7000 | 0.6000 | 0.0300 | -0.1000 | regression |" in regressed.text


def test_an_improvement_beyond_tolerance_passes_with_a_note() -> None:
    outcome = gate(_record(env=CI_ENV, recall=0.8), {"retrieval": _baseline()})

    assert outcome.exit_code == 0
    assert "consider updating the baseline" in outcome.text


def test_a_local_retrieval_regression_is_informational_only() -> None:
    outcome = gate(_record(env=LOCAL_ENV, recall=0.4), {"retrieval": _baseline()})

    assert outcome.exit_code == 0
    assert "informational" in outcome.text


def test_a_local_non_retrieval_regression_still_fails() -> None:
    record = _record("compliance", env=LOCAL_ENV, recall=0.4)

    assert gate(record, {"compliance": _baseline("compliance")}).exit_code == 1


def test_no_baseline_passes_with_a_prominent_notice() -> None:
    outcome = gate(_record(env=CI_ENV), {})

    assert outcome.exit_code == 0
    assert outcome.text.startswith("NOTICE: no baseline recorded for suite 'retrieval'")


def test_a_baseline_on_another_dataset_version_fails_the_gate() -> None:
    baseline = _baseline().model_copy(
        update={"datasets": [DatasetRef(name="retrieval", version=1, sha256="old")]}
    )

    outcome = gate(_record(env=CI_ENV), {"retrieval": baseline})

    assert outcome.exit_code == 1
    assert "baseline was recorded on v1" in outcome.text


def test_run_compare_baseline_with_no_baseline_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "run", "--suite", "compliance", "--mode", "offline",
            "--compare-baseline", "--baselines-dir", str(tmp_path),
        ]
    )  # fmt: skip

    assert code == 0
    assert "NOTICE: no baseline recorded for suite 'compliance'" in capsys.readouterr().err

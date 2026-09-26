"""Report rendering (golden strings) and the README metrics block."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.cli import main
from evals.core.baseline import Baseline, DatasetRef, MetricBaseline
from evals.core.report import (
    README_END,
    README_START,
    MarkersNotFound,
    render_baselines,
    render_run,
    replace_block,
)
from evals.core.run import MetricValue, RunRecord, SuiteResult

_DATASET = DatasetRef(name="compliance", version=2, sha256="abc")


def _record(recall: float = 0.64) -> RunRecord:
    return RunRecord(
        mode="offline",
        git_sha="1234567",
        git_dirty=False,
        seed=42,
        environment={"ci": "false"},
        suites={
            "retrieval": SuiteResult(
                datasets=[_DATASET],
                metrics={
                    "recall_at_k": MetricValue(
                        value=recall, n=75, direction="higher", ci_low=0.53, ci_high=0.74
                    ),
                    "recall_at_k.style.lexical": MetricValue(
                        value=0.8, n=40, direction="higher", ci_low=0.66, ci_high=0.89
                    ),
                },
            )
        },
    )


def _baseline() -> Baseline:
    return Baseline(
        suite="retrieval",
        mode="offline",
        git_sha="deadbee",
        datasets=[_DATASET],
        created_at="2026-09-23T00:00:00Z",
        metrics={
            "recall_at_k": MetricBaseline(
                value=0.68, n=75, direction="higher", tolerance=0.03, ci_low=0.57, ci_high=0.77
            ),
            "recall_at_k.style.lexical": MetricBaseline(
                value=0.8, n=40, direction="higher", tolerance=0.06
            ),
        },
    )


GOLDEN_RUN = """_mode `offline`, commit `1234567`, seed 42_

## retrieval

Datasets: compliance v2

| metric | value | 95% CI | n |
|---|---|---|---|
| recall_at_k | 0.6400 | [0.530, 0.740] | 75 |
| recall_at_k.style.lexical | 0.8000 | [0.660, 0.890] | 40 |
"""

GOLDEN_DIFF = """
### retrieval vs baseline (deadbee)

| metric | baseline | new | tolerance | diff | status |
|---|---|---|---|---|---|
| recall_at_k | 0.6800 | 0.6400 | 0.0300 | -0.0400 | regression |
| recall_at_k.style.lexical | 0.8000 | 0.8000 | 0.0600 | +0.0000 | ok |
"""


def test_a_run_renders_a_metric_table_per_suite() -> None:
    assert render_run(_record()) == GOLDEN_RUN


def test_a_run_with_a_baseline_appends_the_diff_table() -> None:
    assert render_run(_record(), {"retrieval": _baseline()}) == GOLDEN_RUN + GOLDEN_DIFF


GOLDEN_BLOCK = """**retrieval** (offline, commit `deadbee`, compliance v2)

| metric | value | 95% CI | n |
|---|---|---|---|
| recall_at_k | 0.6800 | [0.570, 0.770] | 75 |"""


def test_the_readme_block_shows_only_headline_metrics() -> None:
    assert render_baselines([_baseline()]) == GOLDEN_BLOCK
    assert render_baselines([]) == "_No baselines recorded yet._"


def test_only_the_marked_block_is_rewritten() -> None:
    before = f"# Title\n\nintro\n\n{README_START}\nold\n{README_END}\n\n## After\n\ntext\n"

    after = replace_block(before, "NEW")

    assert after == f"# Title\n\nintro\n\n{README_START}\nNEW\n{README_END}\n\n## After\n\ntext\n"
    assert replace_block(after, "NEW") == after


def test_missing_or_reversed_markers_are_refused() -> None:
    with pytest.raises(MarkersNotFound):
        replace_block("no markers here", "x")
    with pytest.raises(MarkersNotFound):
        replace_block(f"{README_END}\n{README_START}", "x")


def test_the_report_command_prints_the_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = tmp_path / "run.json"
    run.write_text(_record().to_json(), encoding="utf-8")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(_baseline().model_dump_json(), encoding="utf-8")

    code = main(["report", "--run", str(run), "--baseline", str(baseline)])

    assert code == 0
    assert capsys.readouterr().out == GOLDEN_RUN + GOLDEN_DIFF


def test_update_readme_rewrites_only_the_block_from_committed_baselines(tmp_path: Path) -> None:
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    (baselines / "retrieval.json").write_text(_baseline().model_dump_json(), encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text(f"top\n{README_START}\nstale\n{README_END}\nbottom\n", encoding="utf-8")

    code = main(
        ["report", "--update-readme", "--readme", str(readme), "--baselines-dir", str(baselines)]
    )

    assert code == 0
    assert readme.read_text(encoding="utf-8") == (
        f"top\n{README_START}\n{GOLDEN_BLOCK}\n{README_END}\nbottom\n"
    )


def test_update_readme_without_markers_fails_and_leaves_the_file(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("nothing to replace\n", encoding="utf-8")

    code = main(["report", "--update-readme", "--readme", str(readme)])

    assert code == 2
    assert readme.read_text(encoding="utf-8") == "nothing to replace\n"


def test_report_needs_a_run_or_update_readme() -> None:
    assert main(["report"]) == 2
    assert json.loads(_record().to_json())["mode"] == "offline"


def test_only_breakdowns_count_as_strata_and_headline_dotted_names_are_kept() -> None:
    from evals.core.tolerances import is_stratum

    assert is_stratum("recall_at_k.style.lexical") and is_stratum("mrr.difficulty.hard")
    assert is_stratum("router.accuracy.category.clear") and is_stratum(
        "grounding.refusal_accuracy.far"
    )
    assert not is_stratum("keyword.precision") and not is_stratum("slots.exact_match.amount")
    assert not is_stratum("recall_at_k")

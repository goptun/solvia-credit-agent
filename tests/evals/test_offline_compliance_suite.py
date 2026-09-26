"""The offline compliance suite and the `run` command."""

from __future__ import annotations

from pathlib import Path

import pytest

import evals.runner as runner_module
from evals.cli import main
from evals.core.run import RunRecord
from evals.core.schemas import ApprovalItem, ComplianceDataset, PiiItem
from evals.datasets import LoadedDataset
from evals.runner import UnknownSuite, run_offline
from evals.suites.compliance_offline import run_compliance_offline


def _synthetic() -> LoadedDataset:
    dataset = ComplianceDataset(
        version=7,
        approval=[
            # keyword screen: flags (approval + guarantee word); strict: flags too
            ApprovalItem(id="C-1", text="Sua aprovação está garantida!", label="promise"),
            # both screens miss: no approval word
            ApprovalItem(id="C-2", text="O dinheiro cai na sua conta amanhã.", label="promise"),
            # strict flags an unhedged approval the keyword screen misses
            ApprovalItem(id="C-3", text="Você será aprovado rapidamente.", label="promise"),
            # explicit hedge: neither flags... except keyword (has "garant" + "aprova")
            ApprovalItem(
                id="C-4",
                text="Não posso garantir a aprovação.",
                label="hedge",
                tags=["mentions_approval"],
            ),
            ApprovalItem(id="C-5", text="Aqui está a simulação.", label="neutral"),
        ],
        pii=[
            PiiItem(id="P-1", text="CPF 000.000.000-00", expected_masked="CPF [DADO PROTEGIDO]"),
            # a `known_gap` case, decoupled from any real `mask_pii` behavior
            # (this suite-wiring test would otherwise silently stop
            # exercising the "known_gap still counts as a failure" path the
            # moment the masker gets better at some real case): the text has
            # no PII to mask, and the expected output is deliberately wrong.
            PiiItem(
                id="P-2",
                text="Nada de especial aqui.",
                expected_masked="Algo completamente diferente.",
                known_gap=True,
            ),
        ],
    )
    return LoadedDataset(name="compliance", version=7, sha256="f" * 64, dataset=dataset)


def test_screens_and_masking_are_scored_by_hand_computable_counts() -> None:
    result = run_compliance_offline(_synthetic())
    m = result.metrics

    # keyword: flags C-1 and C-4 -> tp=1 (C-1), fp=1 (C-4), fn=2 (C-2, C-3)
    assert (m["keyword.precision"].value, m["keyword.precision"].n) == (0.5, 2)
    assert m["keyword.recall"].value == pytest.approx(1 / 3)
    assert m["keyword.recall"].n == 3
    # strict: flags C-1 and C-3 (unhedged approval); C-4 has an explicit hedge phrase
    assert (m["strict.precision"].value, m["strict.precision"].n) == (1.0, 2)
    assert m["strict.recall"].value == pytest.approx(2 / 3)
    assert m["strict.recall"].n == 3
    assert m["strict.false_positive_rate"].value == 0.0
    # C-4 is the only hedge/neutral item that mentions approval; keyword flags it
    assert m["keyword.false_positive_rate_on_approval_mentions"].value == 1.0
    assert m["strict.false_positive_rate_on_approval_mentions"].value == 0.0
    # masking counts the known gap as a failure
    assert (m["masking.exact_match"].value, m["masking.exact_match"].n) == (0.5, 2)
    assert result.details["masking_failed_ids"] == ["P-2"]
    assert result.datasets[0].version == 7


def test_every_compliance_metric_is_deterministic_with_zero_tolerance_flag() -> None:
    result = run_compliance_offline(_synthetic())

    assert all(metric.deterministic for metric in result.metrics.values())


def test_the_run_record_carries_mode_commit_dataset_versions_and_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALS_GIT_SHA", "abc123")
    monkeypatch.setattr(runner_module, "git_dirty", lambda: False)

    record = run_offline(["compliance"], seed=42)

    assert record.mode == "offline"
    assert record.git_sha == "abc123"
    assert record.seed == 42
    assert record.suites["compliance"].datasets[0].name == "compliance"
    assert record.suites["compliance"].datasets[0].version >= 1
    assert set(record.environment) == {"ci", "os", "arch", "python"}


def test_two_consecutive_runs_are_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALS_GIT_SHA", "abc123")
    monkeypatch.setattr(runner_module, "git_dirty", lambda: False)

    first = run_offline(["compliance"], seed=42).to_json()
    second = run_offline(["compliance"], seed=42).to_json()

    assert first == second


def test_an_unknown_suite_is_reported() -> None:
    with pytest.raises(UnknownSuite):
        run_offline(["retrieval-typo"], seed=42)


def test_run_command_writes_valid_machine_readable_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALS_GIT_SHA", "abc123")
    output = tmp_path / "run.json"

    exit_code = main(["run", "--suite", "compliance", "--mode", "offline", "--output", str(output)])

    assert exit_code == 0
    record = RunRecord.model_validate_json(output.read_text(encoding="utf-8"))
    # The committed dataset has no known masking gap since
    # `fix/pii-masking-and-slot-validation`: this is now a full-marks check.
    assert record.suites["compliance"].metrics["masking.exact_match"].value == pytest.approx(1.0)


def test_run_command_rejects_an_unknown_suite(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["run", "--suite", "nope", "--mode", "offline"])

    assert exit_code == 2
    assert "no offline suite" in capsys.readouterr().err

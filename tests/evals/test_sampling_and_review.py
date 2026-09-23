"""Stratified sampling and the `review-sample` command."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import evals.cli as cli_module
import evals.datasets as datasets_module
from evals.cli import main
from evals.core.sampling import sample_fraction, stratified_sample
from evals.datasets import load_dataset
from evals.review import render_review

_ITEMS = [(f"i{n}", "a" if n < 10 else "b" if n < 14 else "c") for n in range(20)]


def test_same_seed_same_sample_and_different_seeds_differ() -> None:
    first = stratified_sample(_ITEMS, lambda i: i[1], 6, seed=42)
    again = stratified_sample(_ITEMS, lambda i: i[1], 6, seed=42)
    other = stratified_sample(_ITEMS, lambda i: i[1], 6, seed=7)

    assert first == again
    assert first != other
    assert len(first) == 6


def test_every_stratum_appears_when_the_sample_size_allows() -> None:
    sample = stratified_sample(_ITEMS, lambda i: i[1], 3, seed=1)

    assert {stratum for _, stratum in sample} == {"a", "b", "c"}


def test_a_sample_larger_than_the_dataset_returns_everything() -> None:
    assert stratified_sample(_ITEMS, lambda i: i[1], 100, seed=1) == _ITEMS


def test_sample_fraction_covers_strata_and_is_reproducible() -> None:
    sample = sample_fraction(_ITEMS, lambda i: i[1], 0.3, seed=42)

    assert len(sample) == 6
    assert {stratum for _, stratum in sample} == {"a", "b", "c"}
    assert sample == sample_fraction(_ITEMS, lambda i: i[1], 0.3, seed=42)
    with pytest.raises(ValueError):
        sample_fraction(_ITEMS, lambda i: i[1], 0.0, seed=1)


def _write_router(directory: Path) -> None:
    items = [
        {
            "id": f"T-{n}",
            "message": f"mensagem {n}",
            "active_flow": "none",
            "category": "clear",
            "expected": intent,
        }
        for n, intent in enumerate(
            ["loan_simulation", "complaint", "out_of_scope", "product_question"] * 3
        )
    ]
    (directory / "router.yaml").write_text(
        yaml.safe_dump({"version": 2, "items": items}, allow_unicode=True), encoding="utf-8"
    )


def test_render_review_prints_a_stratified_sample_with_provenance(tmp_path: Path) -> None:
    _write_router(tmp_path)
    loaded = load_dataset("router", tmp_path)

    text = render_review(loaded, n=4, seed=42)

    assert "router — version 2" in text
    assert "4 of 12 items" in text
    for intent in ("loan_simulation", "complaint", "out_of_scope", "product_question"):
        assert f"expected {intent}" in text  # every stratum shows up at n = #strata
    assert text == render_review(loaded, n=4, seed=42)


def test_review_sample_command_prints_the_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_router(tmp_path)
    monkeypatch.setattr(datasets_module, "DATASET_DIR", tmp_path)
    monkeypatch.setattr(cli_module, "load_dataset", lambda name: load_dataset(name, tmp_path))

    exit_code = main(["review-sample", "--dataset", "router", "--n", "4", "--seed", "42"])

    assert exit_code == 0
    assert "router — version 2" in capsys.readouterr().out

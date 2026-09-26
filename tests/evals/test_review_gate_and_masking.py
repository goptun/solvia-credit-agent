"""The dataset review gate (approved hashes) and the masking metric."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evals.core.approval import Approval, approval_problem
from evals.core.masking import masking_exact_match
from evals.core.schemas import ComplianceDataset, PiiItem
from evals.datasets import DATASET_NAMES, current_hashes, load_approvals, load_dataset

_APPROVED = {"retrieval": Approval(version=2, sha256="abc")}


def test_an_approved_dataset_passes() -> None:
    assert approval_problem("retrieval", 2, "abc", _APPROVED) is None


def test_a_changed_dataset_is_refused_and_says_to_review_again() -> None:
    problem = approval_problem("retrieval", 3, "def", _APPROVED)

    assert problem is not None
    assert "reviewed again" in problem


def test_a_dataset_with_no_approval_is_refused() -> None:
    problem = approval_problem("router", 1, "abc", _APPROVED)

    assert problem is not None
    assert "no recorded approval" in problem


def test_every_committed_dataset_matches_its_approved_hash() -> None:
    """Editing a dataset without re-approving it fails here — the review gate
    is mechanical, not a convention."""
    approvals = load_approvals()
    current = current_hashes()

    assert set(approvals) == set(DATASET_NAMES)
    problems = [
        problem
        for name in DATASET_NAMES
        if (
            problem := approval_problem(
                name, current[name].version, current[name].sha256, approvals
            )
        )
        is not None
    ]
    assert problems == []


def test_hashes_can_be_pasted_back_as_a_review_file(tmp_path: Path) -> None:
    approvals = {
        name: {"version": a.version, "sha256": a.sha256} for name, a in current_hashes().items()
    }
    path = tmp_path / "review.yaml"
    path.write_text(
        yaml.safe_dump({"approved_by": "m", "approved_on": "2026-01-01", "datasets": approvals})
    )

    assert load_approvals(path) == current_hashes()


def test_the_masking_metric_counts_known_gap_items_as_failures() -> None:
    """The dataset measures the masker, not only what it already handles:
    a `known_gap` item must lower the exact-match rate, never be filtered
    out. Exercised against a synthetic item rather than the committed
    dataset, whose masking gaps were fixed in
    `fix/pii-masking-and-slot-validation` (it currently has none) — this
    property must hold whenever a future gap is recorded, not only today."""
    items = [
        PiiItem(id="P-X", text="tudo bem", expected_masked="tudo bem"),
        PiiItem(id="P-Y", text="não bate", expected_masked="isso não bate", known_gap=True),
    ]

    result = masking_exact_match(items, lambda text: text)

    assert result.exact_match.n == 2  # nothing filtered out
    assert result.exact_match.successes == 1
    assert result.exact_match.value == pytest.approx(0.5)
    assert result.failed_ids == ("P-Y",)


def test_the_committed_compliance_dataset_has_no_known_masking_gap_today() -> None:
    dataset = load_dataset("compliance").dataset
    assert isinstance(dataset, ComplianceDataset)

    assert [item.id for item in dataset.pii if item.known_gap] == []


def test_the_masking_metric_reaches_100_percent_when_everything_is_masked_correctly() -> None:
    dataset = load_dataset("compliance").dataset
    assert isinstance(dataset, ComplianceDataset)
    expected = {item.text: item.expected_masked for item in dataset.pii}

    result = masking_exact_match(dataset.pii, lambda text: expected[text])

    assert result.exact_match.value == pytest.approx(1.0)
    assert result.failed_ids == ()

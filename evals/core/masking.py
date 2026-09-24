"""Masking exact-match metric."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evals.core.schemas import PiiItem
from evals.core.stats import Proportion, proportion


@dataclass(frozen=True)
class MaskingResult:
    exact_match: Proportion
    failed_ids: tuple[str, ...]


def masking_exact_match(items: Sequence[PiiItem], mask: Callable[[str], str]) -> MaskingResult:
    """Share of cases whose masked output equals the expected output exactly.

    Every item counts, including `known_gap` ones: a known masker bug is a
    failure the baseline must record, never a case to filter out."""
    failed = tuple(item.id for item in items if mask(item.text) != item.expected_masked)
    return MaskingResult(
        exact_match=proportion(len(items) - len(failed), len(items)), failed_ids=failed
    )

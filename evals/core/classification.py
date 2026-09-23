"""Classification, extraction and detection metrics (label comparisons
only — no LLM-judged scoring)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from evals.core.stats import Proportion, proportion


@dataclass(frozen=True)
class ClassificationCase:
    expected: str
    predicted: str
    acceptable: tuple[str, ...] = ()
    """Extra labels also counted correct (ambiguous items)."""


def is_correct(case: ClassificationCase) -> bool:
    return case.predicted == case.expected or case.predicted in case.acceptable


def accuracy(cases: Sequence[ClassificationCase]) -> Proportion:
    return proportion(sum(1 for case in cases if is_correct(case)), len(cases))


def confusion_matrix(
    labels: Sequence[str], cases: Sequence[ClassificationCase]
) -> dict[str, dict[str, int]]:
    """`matrix[expected][predicted]` counts. Every expected label has a
    row and every label a column; an unexpected predicted label gets its
    own column so nothing is silently dropped."""
    matrix: dict[str, dict[str, int]] = {label: dict.fromkeys(labels, 0) for label in labels}
    for case in cases:
        row = matrix.setdefault(case.expected, dict.fromkeys(labels, 0))
        row[case.predicted] = row.get(case.predicted, 0) + 1
    return matrix


@dataclass(frozen=True)
class SlotCase:
    expected: dict[str, str | None]
    predicted: dict[str, str | None]
    """`None` = the field is absent/unset."""


def slot_exact_match(cases: Sequence[SlotCase], fields: Sequence[str]) -> dict[str, Proportion]:
    result = {}
    for name in fields:
        matches = sum(1 for case in cases if case.expected.get(name) == case.predicted.get(name))
        result[name] = proportion(matches, len(cases))
    return result


def no_invented_value(cases: Sequence[SlotCase], fields: Sequence[str]) -> Proportion:
    """Among (item, field) pairs whose expected value is absent, the share
    the extraction left unset."""
    total = unset = 0
    for case in cases:
        for name in fields:
            if case.expected.get(name) is None:
                total += 1
                unset += case.predicted.get(name) is None
    return proportion(unset, total)


@dataclass(frozen=True)
class PrecisionRecall:
    precision: Proportion
    recall: Proportion
    false_positive_rate: Proportion
    tp: int
    fp: int
    fn: int
    tn: int


def precision_recall(outcomes: Sequence[tuple[bool, bool]]) -> PrecisionRecall:
    """`outcomes` are `(actually_positive, predicted_positive)` pairs."""
    counts: dict[tuple[bool, bool], int] = defaultdict(int)
    for actual, predicted in outcomes:
        counts[(actual, predicted)] += 1
    tp, fp = counts[(True, True)], counts[(False, True)]
    fn, tn = counts[(True, False)], counts[(False, False)]
    return PrecisionRecall(
        precision=proportion(tp, tp + fp),
        recall=proportion(tp, tp + fn),
        false_positive_rate=proportion(fp, fp + tn),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def exact_match_rate(pairs: Sequence[tuple[str, str]]) -> Proportion:
    """`pairs` are `(expected, actual)` strings compared exactly."""
    return proportion(sum(1 for expected, actual in pairs if expected == actual), len(pairs))

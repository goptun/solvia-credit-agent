"""Retrieval metrics: recall@k, MRR, similarity distributions and the
threshold trade-off table. Report only — choosing a threshold belongs to
a later change (design.md, Decision 3)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from evals.core.stats import (
    MeanEstimate,
    Proportion,
    bootstrap_mean,
    proportion,
    quantile,
)

THRESHOLDS: tuple[float, ...] = tuple(round(0.30 + 0.05 * i, 2) for i in range(13))
"""0.30 ... 0.90 in steps of 0.05."""


@dataclass(frozen=True)
class RankedChunk:
    document_id: str
    article_ref: str | None


@dataclass(frozen=True)
class AnswerableResult:
    question_id: str
    style: str
    document: str
    difficulty: str
    expected_refs: tuple[str, ...] | None
    """`None` for the product catalog, which has no article structure."""
    ranked: tuple[RankedChunk, ...]
    """The top-k retrieved chunks, best first."""
    best_similarity: float


@dataclass(frozen=True)
class UnanswerableResult:
    question_id: str
    distance: str
    best_similarity: float


def hit_rank(result: AnswerableResult) -> int | None:
    """1-based rank of the first chunk from the expected document with an
    acceptable article reference, or `None` if there is none."""
    acceptable = set(result.expected_refs) if result.expected_refs else None
    for position, chunk in enumerate(result.ranked, start=1):
        if chunk.document_id != result.document:
            continue
        if acceptable is None or chunk.article_ref in acceptable:
            return position
    return None


def recall_at_k(results: Sequence[AnswerableResult]) -> Proportion:
    hits = sum(1 for result in results if hit_rank(result) is not None)
    return proportion(hits, len(results))


def mrr(results: Sequence[AnswerableResult], seed: int) -> MeanEstimate:
    reciprocal_ranks = [
        1.0 / rank if (rank := hit_rank(result)) is not None else 0.0 for result in results
    ]
    return bootstrap_mean(reciprocal_ranks, seed)


@dataclass(frozen=True)
class RetrievalMetrics:
    recall: Proportion
    mrr: MeanEstimate


def retrieval_metrics(results: Sequence[AnswerableResult], seed: int) -> RetrievalMetrics:
    return RetrievalMetrics(recall=recall_at_k(results), mrr=mrr(results, seed))


def breakdown(
    results: Sequence[AnswerableResult],
    key: Callable[[AnswerableResult], str],
    seed: int,
) -> dict[str, RetrievalMetrics]:
    """Metrics per stratum (style, document, difficulty), keys sorted."""
    groups: dict[str, list[AnswerableResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    return {name: retrieval_metrics(groups[name], seed) for name in sorted(groups)}


@dataclass(frozen=True)
class SimilarityQuantiles:
    minimum: float
    q25: float
    median: float
    q75: float
    maximum: float


def similarity_quantiles(values: Sequence[float]) -> SimilarityQuantiles:
    return SimilarityQuantiles(
        minimum=quantile(values, 0.0),
        q25=quantile(values, 0.25),
        median=quantile(values, 0.5),
        q75=quantile(values, 0.75),
        maximum=quantile(values, 1.0),
    )


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    false_refusal: Proportion
    """Answerable questions whose best similarity is below the threshold."""
    correct_refusal: Proportion
    """Unanswerable questions whose best similarity is below the threshold."""


def threshold_table(
    answerable: Sequence[AnswerableResult],
    unanswerable: Sequence[UnanswerableResult],
    thresholds: Sequence[float] = THRESHOLDS,
) -> list[ThresholdRow]:
    rows = []
    for threshold in thresholds:
        refused_answerable = sum(1 for r in answerable if r.best_similarity < threshold)
        refused_unanswerable = sum(1 for r in unanswerable if r.best_similarity < threshold)
        rows.append(
            ThresholdRow(
                threshold=threshold,
                false_refusal=proportion(refused_answerable, len(answerable)),
                correct_refusal=proportion(refused_unanswerable, len(unanswerable)),
            )
        )
    return rows

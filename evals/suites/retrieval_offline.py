"""Offline retrieval suite: the production hybrid search over the committed
corpus fixture, scored with recall@k, MRR, similarity distributions and the
false-refusal vs unanswerable-refusal table across thresholds (report only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import psycopg

from evals.adapters.retrieval import retrieve
from evals.core.baseline import DatasetRef
from evals.core.embedding import EmbedQuery
from evals.core.retrieval import (
    AnswerableResult,
    RankedChunk,
    RetrievalMetrics,
    UnanswerableResult,
    breakdown,
    retrieval_metrics,
    similarity_quantiles,
    threshold_table,
)
from evals.core.run import MetricValue, SuiteResult, mean_metric, proportion_metric
from evals.core.schemas import RetrievalDataset
from evals.core.stats import proportion
from evals.datasets import LoadedDataset, load_dataset

SUITE = "retrieval"


def _best_similarity(chunks: list[Any]) -> float:
    """What the agent's refusal threshold looks at: the best raw vector
    similarity among the retrieved chunks (never the fused rank score)."""
    return max((chunk.vector_similarity for chunk in chunks), default=0.0)


def _metrics(prefix: str, value: RetrievalMetrics) -> dict[str, MetricValue]:
    return {
        f"recall_at_k{prefix}": proportion_metric(value.recall, "higher"),
        f"mrr{prefix}": mean_metric(value.mrr, "higher"),
    }


def run_retrieval_offline(
    conn: psycopg.Connection[Any],
    embed_query: EmbedQuery,
    *,
    top_k: int,
    min_relevance: float,
    seed: int,
    config: dict[str, Any],
    loaded: LoadedDataset | None = None,
) -> SuiteResult:
    dataset_loaded = loaded or load_dataset("retrieval")
    dataset = dataset_loaded.dataset
    assert isinstance(dataset, RetrievalDataset)

    answerable: list[AnswerableResult] = []
    unanswerable: list[UnanswerableResult] = []
    per_item: list[dict[str, Any]] = []
    for item in dataset.items:
        chunks = retrieve(conn, embed_query, item, top_k)
        best = _best_similarity(chunks)
        if item.kind == "answerable":
            assert item.document is not None and item.style is not None
            result = AnswerableResult(
                question_id=item.id,
                style=item.style,
                document=item.document,
                difficulty=item.difficulty,
                acceptable=item.acceptable(),
                ranked=tuple(RankedChunk(c.document_id, c.article_ref) for c in chunks),
                best_similarity=best,
            )
            answerable.append(result)
            per_item.append({"id": item.id, "best_similarity": round(best, 4)})
        else:
            assert item.distance is not None
            unanswerable.append(UnanswerableResult(item.id, item.distance, best))
            per_item.append({"id": item.id, "best_similarity": round(best, 4)})

    metrics: dict[str, MetricValue] = _metrics("", retrieval_metrics(answerable, seed))
    for stratum, key in (
        ("style", lambda r: r.style),
        ("document", lambda r: r.document),
        ("difficulty", lambda r: r.difficulty),
    ):
        for name, value in breakdown(answerable, key, seed).items():
            metrics.update(_metrics(f".{stratum}.{name}", value))

    refused_answerable = sum(1 for r in answerable if r.best_similarity < min_relevance)
    refused_unanswerable = sum(1 for r in unanswerable if r.best_similarity < min_relevance)
    metrics["false_refusal_rate_at_threshold"] = proportion_metric(
        proportion(refused_answerable, len(answerable)), "lower"
    )
    metrics["unanswerable_refusal_rate_at_threshold"] = proportion_metric(
        proportion(refused_unanswerable, len(unanswerable)), "higher"
    )

    table = threshold_table(answerable, unanswerable)
    return SuiteResult(
        datasets=[
            DatasetRef(
                name=dataset_loaded.name,
                version=dataset_loaded.version,
                sha256=dataset_loaded.sha256,
            )
        ],
        metrics=metrics,
        details={
            "config": {**config, "top_k": top_k, "min_relevance": min_relevance},
            "similarity_answerable": asdict(
                similarity_quantiles([r.best_similarity for r in answerable])
            ),
            "similarity_unanswerable": asdict(
                similarity_quantiles([r.best_similarity for r in unanswerable])
            ),
            "threshold_table": [
                {
                    "threshold": row.threshold,
                    "false_refusal": round(row.false_refusal.value, 4),
                    "correct_refusal": round(row.correct_refusal.value, 4),
                }
                for row in table
            ],
            "per_item": per_item,
        },
    )


RetrievalRunner = Callable[[], SuiteResult]

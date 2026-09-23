"""`python -m rag.eval.run` — retrieval-quality evaluation.

Reports recall@k and MRR over the answerable questions (broken down by
`style`: lexical vs. colloquial) and refusal accuracy over the
unanswerable ones (broken down by `distance`: far vs. near-miss) — see
`design.md` — "Retrieval quality evaluation". Requires the corpus to
already be ingested into `rag_chunks` (see `rag/ingest/__main__.py`
and `rag/migrate.py`).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg

from rag.corpus.manifest import SourceType
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.embeddings.port import EmbeddingsPort
from rag.eval.questions import (
    AnswerableQuestion,
    EvalQuestionSet,
    UnanswerableQuestion,
    load_questions,
)
from rag.retrieval.hybrid import hybrid_search
from rag.settings import get_rag_settings


@dataclass(frozen=True)
class AnswerableResult:
    style: str
    hit: bool
    reciprocal_rank: float


@dataclass(frozen=True)
class UnanswerableResult:
    distance: str
    refused: bool


@dataclass(frozen=True)
class EvalReport:
    overall_recall_at_k: float
    overall_mrr: float
    overall_refusal_accuracy: float
    recall_at_k_by_style: dict[str, float]
    mrr_by_style: dict[str, float]
    refusal_accuracy_by_distance: dict[str, float]


def _source_type_for(document_id: str) -> SourceType:
    return "product_catalog" if document_id == "product-catalog" else "regulation"


def evaluate_answerable(
    conn: psycopg.Connection[Any],
    embeddings: EmbeddingsPort,
    question: AnswerableQuestion,
    *,
    top_k: int,
) -> AnswerableResult:
    source_type = _source_type_for(question.expected_document_id)
    results = hybrid_search(
        conn,
        question.question,
        embeddings.embed_query(question.question),
        source_type=source_type,
        top_k=top_k,
    )
    expected_refs = set(question.expected_refs) if question.expected_refs else None

    rank = None
    for position, result in enumerate(results, start=1):
        if result.document_id != question.expected_document_id:
            continue
        if expected_refs is None or result.article_ref in expected_refs:
            rank = position
            break

    return AnswerableResult(
        style=question.style,
        hit=rank is not None,
        reciprocal_rank=(1.0 / rank) if rank else 0.0,
    )


def evaluate_unanswerable(
    conn: psycopg.Connection[Any],
    embeddings: EmbeddingsPort,
    question: UnanswerableQuestion,
    *,
    min_relevance_score: float,
) -> UnanswerableResult:
    results = hybrid_search(
        conn, question.question, embeddings.embed_query(question.question), top_k=1
    )
    top_similarity = results[0].vector_similarity if results else 0.0
    refused = top_similarity < min_relevance_score
    return UnanswerableResult(distance=question.distance, refused=refused)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_by(
    results: list[AnswerableResult],
) -> tuple[dict[str, float], dict[str, float]]:
    by_style: dict[str, list[AnswerableResult]] = defaultdict(list)
    for result in results:
        by_style[result.style].append(result)
    recall = {style: _mean([1.0 if r.hit else 0.0 for r in rs]) for style, rs in by_style.items()}
    mrr = {style: _mean([r.reciprocal_rank for r in rs]) for style, rs in by_style.items()}
    return recall, mrr


def run_eval(
    conn: psycopg.Connection[Any],
    embeddings: EmbeddingsPort,
    questions: EvalQuestionSet,
    *,
    top_k: int,
    min_relevance_score: float,
) -> EvalReport:
    answerable_results = [
        evaluate_answerable(conn, embeddings, q, top_k=top_k) for q in questions.answerable
    ]
    unanswerable_results = [
        evaluate_unanswerable(conn, embeddings, q, min_relevance_score=min_relevance_score)
        for q in questions.unanswerable
    ]

    recall_by_style, mrr_by_style = _group_by(answerable_results)

    by_distance: dict[str, list[UnanswerableResult]] = defaultdict(list)
    for result in unanswerable_results:
        by_distance[result.distance].append(result)
    refusal_by_distance = {
        distance: _mean([1.0 if r.refused else 0.0 for r in rs])
        for distance, rs in by_distance.items()
    }

    return EvalReport(
        overall_recall_at_k=_mean([1.0 if r.hit else 0.0 for r in answerable_results]),
        overall_mrr=_mean([r.reciprocal_rank for r in answerable_results]),
        overall_refusal_accuracy=_mean([1.0 if r.refused else 0.0 for r in unanswerable_results]),
        recall_at_k_by_style=recall_by_style,
        mrr_by_style=mrr_by_style,
        refusal_accuracy_by_distance=refusal_by_distance,
    )


def _print_report(report: EvalReport) -> None:
    print(f"recall@k (overall): {report.overall_recall_at_k:.2%}")
    for style, value in sorted(report.recall_at_k_by_style.items()):
        print(f"  recall@k [{style}]: {value:.2%}")
    print(f"MRR (overall): {report.overall_mrr:.3f}")
    for style, value in sorted(report.mrr_by_style.items()):
        print(f"  MRR [{style}]: {value:.3f}")
    print(f"refusal accuracy (overall): {report.overall_refusal_accuracy:.2%}")
    for distance, value in sorted(report.refusal_accuracy_by_distance.items()):
        print(f"  refusal accuracy [{distance}]: {value:.2%}")


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set", file=sys.stderr)
        return 1

    settings = get_rag_settings()
    embeddings = FastEmbedAdapter(settings.rag_embedding_model)
    questions = load_questions()

    with psycopg.connect(database_url, autocommit=True) as conn:
        report = run_eval(
            conn,
            embeddings,
            questions,
            top_k=settings.rag_top_k,
            min_relevance_score=settings.rag_min_relevance_score,
        )

    _print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

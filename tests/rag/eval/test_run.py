"""`rag.eval.run` against a small seeded pgvector-enabled Postgres, with
`FakeEmbeddings`. Skipped when no `DATABASE_URL` is configured."""

from __future__ import annotations

import os
from datetime import date

import psycopg
import pytest

from rag.corpus.manifest import ManifestDocument
from rag.embeddings.fake import FakeEmbeddings
from rag.eval.questions import AnswerableQuestion, EvalQuestionSet, UnanswerableQuestion
from rag.eval.run import run_eval
from rag.ingest.extracted_document import ExtractedDocument
from rag.ingest.indexing import index_document
from rag.migrate import migrate

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")

_EMBEDDINGS = FakeEmbeddings()


def _doc(doc_id: str, source_type: str, sha256: str) -> ManifestDocument:
    return ManifestDocument(
        id=doc_id,
        title=f"Test {doc_id}",
        norm="Lei nº 1/2020" if source_type == "regulation" else None,
        source_type=source_type,  # type: ignore[arg-type]
        url="https://planalto.gov.br/test.htm" if source_type == "regulation" else None,
        retrieved_at=date(2026, 1, 1),
        version_date=date(2020, 1, 1),
        sha256=sha256,
    )


@pytest.fixture(autouse=True)
def _seed() -> None:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)

    doc = _doc("test-run-eval-a", "regulation", "hash-run-eval-a")
    extracted = ExtractedDocument(
        text="Art. 1º Disposições exclusivas sobre criptomoedas e blockchain regulamentado.",
        amendment_notes=(),
    )
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        # A full wipe, not a scoped `WHERE document_id = ...` delete:
        # the eval's MRR/rank assertions below need to be the *only*
        # chunks `hybrid_search` sees. The OR-over-lexemes full-text
        # query (`rag/retrieval/hybrid.py` — "Full-text search: OR over
        # parsed lexemes") matches on shared vocabulary, not an exact
        # phrase, so leftover chunks from sibling test files sharing
        # common Portuguese words (e.g. "disposições", "sobre") can
        # otherwise enter the fused ranking and shift the target's
        # rank — a real isolation gap the old, stricter AND query
        # happened to mask.
        conn.execute("DELETE FROM rag_chunks")
        index_document(conn, doc, extracted, _EMBEDDINGS.embed_documents)


def test_run_eval_reports_a_hit_for_a_known_answerable_question_and_refusal_for_unrelated() -> None:
    assert _DATABASE_URL is not None
    questions = EvalQuestionSet(
        answerable=[
            AnswerableQuestion(
                question="Disposições exclusivas sobre criptomoedas e blockchain regulamentado.",
                expected_document_id="test-run-eval-a",
                expected_refs=["art. 1º"],
                evidence="criptomoedas",
                style="lexical",
            ),
            AnswerableQuestion(
                question="Um texto completamente diferente que não bate com nada indexado.",
                expected_document_id="test-run-eval-a",
                expected_refs=["art. 1º"],
                evidence="criptomoedas",
                style="colloquial",
            ),
        ],
        unanswerable=[
            UnanswerableQuestion(
                question="Pergunta totalmente fora do escopo do corpus.", distance="far"
            ),
        ],
    )

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        report = run_eval(
            conn, _EMBEDDINGS, questions, top_k=5, min_relevance_score=2.0
        )  # threshold above any FakeEmbeddings similarity forces a refusal

    assert report.recall_at_k_by_style["lexical"] == 1.0
    assert report.mrr_by_style["lexical"] == 1.0
    assert report.overall_refusal_accuracy == 1.0
    assert report.refusal_accuracy_by_distance["far"] == 1.0
    # "colloquial" uses FakeEmbeddings' hash-based vectors, which have
    # no real semantic relationship to the indexed text — asserting the
    # breakdown key exists (not a specific value) is what's under test.
    assert "colloquial" in report.recall_at_k_by_style

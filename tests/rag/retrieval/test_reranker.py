"""`rag.retrieval.reranker` — the `NoopReranker` default, and that
`hybrid_search` calls whichever reranker it's given unconditionally."""

from __future__ import annotations

import os
from datetime import date

import psycopg
import pytest

from rag.corpus.manifest import ManifestDocument
from rag.embeddings.fake import FakeEmbeddings
from rag.ingest.extracted_document import ExtractedDocument
from rag.ingest.indexing import index_document
from rag.migrate import migrate
from rag.retrieval.hybrid import hybrid_search
from rag.retrieval.reranker import NoopReranker, get_reranker
from rag.retrieval.retrieved_chunk import RetrievedChunk


def test_get_reranker_returns_noop_regardless_of_the_flag() -> None:
    assert isinstance(get_reranker(reranking_enabled=False), NoopReranker)
    assert isinstance(get_reranker(reranking_enabled=True), NoopReranker)


_DATABASE_URL = os.environ.get("DATABASE_URL")
_EMBEDDINGS = FakeEmbeddings()


def test_noop_reranker_returns_results_unchanged() -> None:
    a = RetrievedChunk(
        chunk_id="a",
        document_id="doc",
        norm=None,
        source_type="regulation",
        article_ref=None,
        hierarchy_path="",
        source_url=None,
        version_date=None,
        amendment_note=None,
        content="conteúdo",
        fused_score=1.0,
        vector_similarity=0.9,
        matched_fts=True,
    )

    assert NoopReranker().rerank("qualquer pergunta", [a]) == [a]


class _ReversingReranker:
    """A spy/fake reranker: reverses the list and records that it was
    called, so the test can prove `hybrid_search` actually invokes it
    rather than merely accepting the parameter."""

    def __init__(self) -> None:
        self.called_with_query: str | None = None

    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        self.called_with_query = query
        return list(reversed(results))


@pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")
def test_hybrid_search_calls_the_given_reranker_unconditionally() -> None:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)
    doc = ManifestDocument(
        id="test-rerank-a",
        title="Test",
        norm="Lei nº 1/2020",
        source_type="regulation",
        url="https://planalto.gov.br/test.htm",
        retrieved_at=date(2026, 1, 1),
        version_date=date(2020, 1, 1),
        sha256="hash-rerank",
    )
    extracted = ExtractedDocument(
        text="Art. 1º Texto sobre proteção de dados. § 1º Mais texto substantivo aqui.",
        amendment_notes=(),
    )
    reranker = _ReversingReranker()

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM rag_chunks WHERE document_id = 'test-rerank-a'")
        index_document(conn, doc, extracted, _EMBEDDINGS.embed_documents)

        without_reranking = hybrid_search(
            conn, "proteção de dados", _EMBEDDINGS.embed_query("proteção de dados"), top_k=10
        )
        with_reranking = hybrid_search(
            conn,
            "proteção de dados",
            _EMBEDDINGS.embed_query("proteção de dados"),
            top_k=10,
            reranker=reranker,
        )

    assert reranker.called_with_query == "proteção de dados"
    assert with_reranking == list(reversed(without_reranking))

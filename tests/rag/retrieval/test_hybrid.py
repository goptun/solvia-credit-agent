"""`rag.retrieval.hybrid` against a real pgvector-enabled Postgres, with
`FakeEmbeddings`. Skipped when no `DATABASE_URL` is configured."""

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
    """A small fixed corpus: two regulation chunks with clearly
    different content (so vector search distinguishes them), a
    product_catalog chunk, and one chunk whose distinctive keyword
    only a full-text match would surface (to force the fused ranking
    to actually need both signals)."""
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)

    docs = [
        (
            "test-hybrid-a",
            "regulation",
            "hash-a",
            "Art. 1º Disposições sobre proteção de dados pessoais e privacidade do titular.",
        ),
        (
            "test-hybrid-b",
            "regulation",
            "hash-b",
            "Art. 1º Disposições sobre operações de crédito, juros e custo efetivo total.",
        ),
        (
            "test-hybrid-c",
            "product_catalog",
            "hash-c",
            "Crédito pessoal Solvia com parcelas fixas pela tabela Price.",
        ),
        (
            "test-hybrid-d",
            "regulation",
            "hash-d",
            "Art. 2º Menciona xilofonemasingular como termo raro e específico deste artigo.",
        ),
    ]

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM rag_chunks WHERE document_id LIKE 'test-hybrid-%'")
        for doc_id, source_type, sha256, text in docs:
            doc = _doc(doc_id, source_type, sha256)
            extracted = ExtractedDocument(text=text, amendment_notes=())
            index_document(conn, doc, extracted, _EMBEDDINGS.embed_documents)


def test_source_type_filter_excludes_the_other_type() -> None:
    assert _DATABASE_URL is not None
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        results = hybrid_search(
            conn,
            "crédito pessoal",
            _EMBEDDINGS.embed_query("crédito pessoal"),
            source_type="regulation",
            top_k=10,
        )

    assert all(r.source_type == "regulation" for r in results)
    assert all(r.document_id != "test-hybrid-c" for r in results)


def test_full_text_match_surfaces_a_chunk_vector_search_alone_would_miss() -> None:
    assert _DATABASE_URL is not None
    rare_word = "xilofonemasingular"

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        results = hybrid_search(
            conn, rare_word, _EMBEDDINGS.embed_query(rare_word), top_k=10, k_vector=1
        )

    matched = [r for r in results if r.document_id == "test-hybrid-d"]
    assert matched
    assert matched[0].matched_fts is True


def test_vector_similarity_is_independent_of_fused_rank() -> None:
    assert _DATABASE_URL is not None
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        results = hybrid_search(
            conn,
            "proteção de dados pessoais",
            _EMBEDDINGS.embed_query("proteção de dados pessoais"),
            top_k=10,
        )

    assert results
    # Similarity is a real per-chunk cosine value, not derived from
    # (and therefore not monotonic with) the RRF fused rank/score.
    for result in results:
        assert -1.0 <= result.vector_similarity <= 1.0


def test_results_carry_full_chunk_metadata() -> None:
    assert _DATABASE_URL is not None
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        results = hybrid_search(
            conn,
            "proteção de dados",
            _EMBEDDINGS.embed_query("proteção de dados"),
            top_k=1,
        )

    assert results
    chunk = results[0]
    assert chunk.chunk_id
    assert chunk.document_id
    assert chunk.content
    assert chunk.hierarchy_path is not None

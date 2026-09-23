"""`rag.ingest.indexing` against a real pgvector-enabled Postgres.

Skipped when no `DATABASE_URL` is configured, same convention as
`tests/agent/test_graph.py`'s checkpointer integration test.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import date
from typing import Any

import psycopg
import pytest

from rag.corpus.manifest import ManifestDocument
from rag.ingest.extracted_document import ExtractedDocument
from rag.ingest.indexing import index_document
from rag.migrate import migrate

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")

_LONG_ARTICLE_TEXT = "Art. 1º Caput do artigo. " + " ".join(
    f"§ {n}º Texto substantivo do parágrafo número {n}, com conteúdo suficiente para "
    "testar a divisão em pedaços quando o orçamento de caracteres é pequeno."
    for n in range(1, 6)
)


def _fake_embed_documents(texts: Sequence[str]) -> list[list[float]]:
    return [[0.1] * 384 for _ in texts]


def _doc(doc_id: str, sha256: str) -> ManifestDocument:
    return ManifestDocument(
        id=doc_id,
        title=f"Test {doc_id}",
        norm="Lei nº 1/2020",
        source_type="regulation",
        url="https://planalto.gov.br/test.htm",
        retrieved_at=date(2026, 1, 1),
        version_date=date(2020, 1, 1),
        sha256=sha256,
    )


def _catalog_doc(doc_id: str) -> ManifestDocument:
    """A `product_catalog` document, whose manifest `sha256` is always
    `None` by design — see `rag/ingest/fetch.py`."""
    return ManifestDocument(
        id=doc_id,
        title=f"Test {doc_id}",
        norm=None,
        source_type="product_catalog",
        url=None,
        retrieved_at=date(2026, 1, 1),
        version_date=None,
        sha256=None,
    )


@pytest.fixture(autouse=True)
def _clean_table() -> None:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM rag_chunks WHERE document_id LIKE 'test-idx-%'")


def test_product_catalog_document_reindexes_idempotently_without_a_manifest_hash() -> None:
    """A `product_catalog` document has no manifest-pinned `sha256`
    (see `rag/ingest/fetch.py`) — `index_document` must still compute
    an effective hash from the extracted text itself, so unchanged
    reindexing skips and a real content change still reindexes."""
    assert _DATABASE_URL is not None
    doc = _catalog_doc("test-idx-catalog")
    extracted_v1 = ExtractedDocument(text="Catálogo fictício versão um.", amendment_notes=())

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        reindexed_first = index_document(conn, doc, extracted_v1, _fake_embed_documents)
        reindexed_unchanged = index_document(conn, doc, extracted_v1, _fake_embed_documents)

        extracted_v2 = ExtractedDocument(text="Catálogo fictício versão dois.", amendment_notes=())
        reindexed_changed = index_document(conn, doc, extracted_v2, _fake_embed_documents)

    assert reindexed_first is True
    assert reindexed_unchanged is False
    assert reindexed_changed is True


def _chunk_count(conn: psycopg.Connection[Any], document_id: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM rag_chunks WHERE document_id = %s", (document_id,)
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_reindexing_an_unchanged_document_is_a_no_op() -> None:
    assert _DATABASE_URL is not None
    doc = _doc("test-idx-a", sha256="hash-v1")
    extracted = ExtractedDocument(text=_LONG_ARTICLE_TEXT, amendment_notes=())

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        reindexed_first = index_document(conn, doc, extracted, _fake_embed_documents)
        count_after_first = _chunk_count(conn, doc.id)

        reindexed_second = index_document(conn, doc, extracted, _fake_embed_documents)
        count_after_second = _chunk_count(conn, doc.id)

    assert reindexed_first is True
    assert reindexed_second is False
    assert count_after_first == count_after_second


def test_a_changed_document_reindexes_without_touching_siblings() -> None:
    assert _DATABASE_URL is not None
    doc_a_v1 = _doc("test-idx-a", sha256="hash-v1")
    doc_b = _doc("test-idx-b", sha256="hash-b")
    extracted = ExtractedDocument(text=_LONG_ARTICLE_TEXT, amendment_notes=())

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        index_document(conn, doc_a_v1, extracted, _fake_embed_documents, chunk_max_chars=1500)
        index_document(conn, doc_b, extracted, _fake_embed_documents, chunk_max_chars=1500)
        count_a_before = _chunk_count(conn, doc_a_v1.id)
        count_b_before = _chunk_count(conn, doc_b.id)

        # Simulate a content change (new hash) that also happens to be
        # ingested with a smaller chunk budget, forcing more pieces.
        doc_a_v2 = _doc("test-idx-a", sha256="hash-v2")
        reindexed = index_document(
            conn, doc_a_v2, extracted, _fake_embed_documents, chunk_max_chars=100
        )
        count_a_after = _chunk_count(conn, doc_a_v1.id)
        count_b_after = _chunk_count(conn, doc_b.id)

    assert reindexed is True
    assert count_a_after != count_a_before
    assert count_b_after == count_b_before


def test_upsert_updates_content_for_an_existing_chunk_id() -> None:
    assert _DATABASE_URL is not None
    doc_v1 = _doc("test-idx-c", sha256="hash-v1")
    extracted_v1 = ExtractedDocument(text="Art. 1º Texto original.", amendment_notes=())

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        index_document(conn, doc_v1, extracted_v1, _fake_embed_documents)

        doc_v2 = _doc("test-idx-c", sha256="hash-v2")
        extracted_v2 = ExtractedDocument(text="Art. 1º Texto revisado.", amendment_notes=())
        index_document(conn, doc_v2, extracted_v2, _fake_embed_documents)

        row = conn.execute(
            "SELECT content FROM rag_chunks WHERE document_id = %s", (doc_v1.id,)
        ).fetchone()

    assert row is not None
    assert "revisado" in row[0]
    assert "original" not in row[0]

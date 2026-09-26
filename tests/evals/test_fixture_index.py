"""Indexing the committed fixture into pgvector with `FakeEmbeddings`.
Skipped when no `DATABASE_URL` is configured."""

from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from evals.adapters.fixture import index_fixture, read_fixture
from rag.embeddings.fake import FakeEmbeddings
from rag.migrate import migrate

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")


@pytest.fixture
def clean_index() -> Iterator[None]:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM rag_chunks")
    yield
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM rag_chunks")


def test_the_fixture_indexes_with_the_expected_counts(clean_index: None) -> None:
    assert _DATABASE_URL is not None
    fixture = read_fixture()
    embeddings = FakeEmbeddings()

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        indexed = index_fixture(conn, fixture, embeddings.embed_documents)
        total = conn.execute("SELECT count(*) FROM rag_chunks").fetchone()
        rows = conn.execute(
            "SELECT document_id, count(*) FROM rag_chunks GROUP BY document_id"
        ).fetchall()
        per_document: dict[str, int] = {str(row[0]): int(row[1]) for row in rows}
        second = index_fixture(conn, fixture, embeddings.embed_documents)

    expected: dict[str, int] = {}
    for record in fixture.records:
        expected[record.document_id] = expected.get(record.document_id, 0) + 1
    assert indexed == len(fixture.documents)
    assert total is not None and total[0] == len(fixture.records)
    assert per_document == expected
    assert second == 0  # idempotent: nothing to reindex

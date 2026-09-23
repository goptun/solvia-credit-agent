"""`rag.migrate` against a real pgvector-enabled Postgres.

Skipped when no `DATABASE_URL` is configured, same convention as
`tests/agent/test_graph.py`'s checkpointer integration test.
"""

from __future__ import annotations

import os

import psycopg
import pytest

from rag.migrate import migrate

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")


def test_migrate_creates_the_rag_chunks_table_and_indexes() -> None:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'rag_chunks'"
        ).fetchall()
        column_names = {row[0] for row in columns}

    assert {
        "chunk_id",
        "document_id",
        "document_hash",
        "source_type",
        "norm",
        "article_ref",
        "hierarchy_path",
        "source_url",
        "version_date",
        "amendment_note",
        "content",
        "embedding",
        "content_tsv",
    } <= column_names


def test_migrate_is_idempotent() -> None:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)
    migrate(_DATABASE_URL)  # must not raise on a second run


def test_unaccented_query_matches_accented_content() -> None:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)

    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        matched = conn.execute(
            "SELECT to_tsvector('portuguese_unaccent', %s) "
            "@@ to_tsquery('portuguese_unaccent', %s)",
            ("código", "codigo"),
        ).fetchone()

    assert matched is not None
    assert matched[0] is True

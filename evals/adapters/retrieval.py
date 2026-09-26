"""Retrieval through the production hybrid search, for the evaluation
harness (review evidence now, the offline retrieval suite later)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

import psycopg
from psycopg.conninfo import make_conninfo

from evals.adapters.fixture import Fixture, index_fixture
from evals.core.embedding import EmbedDocuments, EmbedQuery
from evals.core.schemas import RetrievalItem
from evals.review import EvidenceProvider
from rag.corpus.manifest import SourceType
from rag.migrate import migrate
from rag.retrieval.hybrid import hybrid_search
from rag.retrieval.retrieved_chunk import RetrievedChunk

EVIDENCE_TOP_K = 3
EVIDENCE_PREVIEW_CHARS = 200


def source_type_for(document: str | None) -> SourceType | None:
    """Source-type scope the agent applies per intent: answerable product
    questions search the catalog, regulatory ones the regulation; an
    unanswerable question has no intent to scope by, so searches all."""
    if document is None:
        return None
    return "product_catalog" if document == "product-catalog" else "regulation"


def retrieve(
    conn: psycopg.Connection[Any], embed_query: EmbedQuery, item: RetrievalItem, top_k: int
) -> list[RetrievedChunk]:
    return hybrid_search(
        conn,
        item.question,
        embed_query(item.question),
        source_type=source_type_for(item.document),
        top_k=top_k,
    )


def _preview(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:EVIDENCE_PREVIEW_CHARS]


def make_evidence_provider(
    conn: psycopg.Connection[Any], embed_query: EmbedQuery
) -> EvidenceProvider:
    """An `EvidenceProvider`: the top-3 chunks (document, article ref,
    similarity, first 200 characters) the retrieval index returns for an
    item — no gateway call, only the embedding model."""

    def provider(item: RetrievalItem) -> list[str]:
        chunks = retrieve(conn, embed_query, item, EVIDENCE_TOP_K)
        return [
            f"{rank}. {chunk.document_id} · {chunk.article_ref or '-'} · "
            f'sim {chunk.vector_similarity:.3f} · "{_preview(chunk.content)}"'
            for rank, chunk in enumerate(chunks, start=1)
        ]

    return provider


EVAL_SCHEMA_PREFIX = "evals_eval_"
"""The evaluation's own schemas: the fixture is indexed here, never into the
application's `rag_chunks`."""
HNSW_EF_SEARCH = 1000
"""pgvector's maximum: above the fixture's ~500 rows, so the HNSW scan is
effectively exact and approximate-index variance cannot move a metric."""


def eval_schema(embedding_id: str) -> str:
    """One schema per embedding model: the index is idempotent per document
    hash, so sharing a schema between models (or with the fake embeddings used
    in tests) would silently serve vectors from the wrong model."""
    slug = re.sub(r"[^a-z0-9]+", "_", embedding_id.lower()).strip("_")
    digest = hashlib.sha256(embedding_id.encode()).hexdigest()[:8]
    return f"{EVAL_SCHEMA_PREFIX}{slug[:24]}_{digest}"


def eval_conninfo(database_url: str, schema: str) -> str:
    """Connection string that resolves `rag_chunks` inside `schema`."""
    return make_conninfo(database_url, options=f"-c search_path={schema},public")


def prepare_eval_index(
    database_url: str,
    fixture: Fixture,
    embed_documents: EmbedDocuments,
    embedding_id: str,
) -> str:
    """Create the model's evaluation schema/table and index the fixture into
    it (idempotent per document hash). Returns the connection string to use."""
    schema = eval_schema(embedding_id)
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    conninfo = eval_conninfo(database_url, schema)
    migrate(conninfo)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        index_fixture(conn, fixture, embed_documents)
    return conninfo


def open_eval_connection(conninfo: str) -> psycopg.Connection[Any]:
    conn = psycopg.connect(conninfo, autocommit=True)
    conn.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    return conn

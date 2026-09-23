"""Retrieval through the production hybrid search, for the evaluation
harness (review evidence now, the offline retrieval suite later)."""

from __future__ import annotations

from typing import Any

import psycopg

from evals.core.embedding import EmbedQuery
from evals.core.schemas import RetrievalItem
from evals.review import EvidenceProvider
from rag.corpus.manifest import SourceType
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

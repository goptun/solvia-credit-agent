"""Async wrapper around `hybrid_search` for use from an async caller
(e.g. `apps/agent/nodes/knowledge_agent.py`'s `RetrieveFn`).

`hybrid_search` itself stays synchronous (plain `psycopg`, matching
every other `rag/` CLI tool) — this just runs it in a worker thread via
`asyncio.to_thread`, checking out its own connection from the pool per
call so concurrent requests don't contend on one connection.
"""

from __future__ import annotations

import asyncio

from psycopg_pool import ConnectionPool

from rag.corpus.manifest import SourceType
from rag.embeddings.port import Vector
from rag.retrieval.hybrid import hybrid_search
from rag.retrieval.reranker import get_reranker
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings


def _search_sync(
    pool: ConnectionPool,
    question: str,
    query_vector: Vector,
    source_type: SourceType | None,
    settings: RagSettings,
) -> list[RetrievedChunk]:
    reranker = get_reranker(settings.rag_reranking_enabled)
    with pool.connection() as conn:
        return hybrid_search(
            conn,
            question,
            query_vector,
            source_type=source_type,
            k_vector=settings.rag_k_vector,
            k_fts=settings.rag_k_fts,
            rrf_k=settings.rag_rrf_k,
            top_k=settings.rag_top_k,
            reranker=reranker,
        )


async def hybrid_search_async(
    pool: ConnectionPool,
    settings: RagSettings,
    question: str,
    query_vector: Vector,
    source_type: SourceType | None,
) -> list[RetrievedChunk]:
    return await asyncio.to_thread(
        _search_sync, pool, question, query_vector, source_type, settings
    )

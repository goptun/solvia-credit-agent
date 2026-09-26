"""Real retrieval dependencies for the live grounding suite: the fixture indexed
into the evaluation schema (never the application's index), searched with the
production hybrid search under the production retrieval settings."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from contextlib import contextmanager

from evals.adapters.fixture import read_fixture
from evals.adapters.retrieval import open_eval_connection, prepare_eval_index
from evals.suites.live import GroundingDeps
from rag.corpus.manifest import SourceType
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.embeddings.port import Vector
from rag.retrieval.hybrid import hybrid_search
from rag.retrieval.reranker import get_reranker
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import get_rag_settings


class MissingDatabase(RuntimeError):
    """The grounding suite needs a pgvector Postgres (`DATABASE_URL`)."""


@contextmanager
def grounding_dependencies() -> Iterator[GroundingDeps]:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise MissingDatabase("the grounding suite needs DATABASE_URL (a pgvector Postgres)")
    settings = get_rag_settings()
    embeddings = FastEmbedAdapter(
        settings.rag_embedding_model,
        threads=settings.rag_embedding_threads,
        batch_size=settings.rag_embedding_batch_size,
    )
    conninfo = prepare_eval_index(
        database_url, read_fixture(), embeddings.embed_documents, settings.rag_embedding_model
    )
    reranker = get_reranker(settings.rag_reranking_enabled)
    with open_eval_connection(conninfo) as conn:

        def search(
            question: str, vector: Vector, source_type: SourceType | None
        ) -> list[RetrievedChunk]:
            return hybrid_search(
                conn,
                question,
                vector,
                source_type=source_type,
                k_vector=settings.rag_k_vector,
                k_fts=settings.rag_k_fts,
                rrf_k=settings.rag_rrf_k,
                top_k=settings.rag_top_k,
                reranker=reranker,
            )

        async def retrieve(
            question: str, vector: Vector, source_type: SourceType | None
        ) -> list[RetrievedChunk]:
            return await asyncio.to_thread(search, question, vector, source_type)

        yield GroundingDeps(embeddings, retrieve, settings)

"""FastAPI application factory.

Run locally with: `uv run uvicorn apps.api.app:app --reload`.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from psycopg_pool import ConnectionPool

from apps.agent.checkpointer import postgres_checkpointer
from apps.agent.graph import build_graph
from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.settings import get_settings
from apps.agent.nodes.knowledge_agent import make_knowledge_agent_node
from apps.agent.observability.logging import configure_logging
from apps.agent.observability.tracing import build_tracer
from apps.agent.repositories.customers import InMemoryCustomerRepository
from apps.api.context import AppContext
from apps.api.routes import router
from apps.api.routes_knowledge import router as knowledge_router
from apps.api.settings import get_api_settings
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.migrate import migrate
from rag.retrieval.live import hybrid_search_async
from rag.settings import get_rag_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    llm_settings = get_settings()
    api_settings = get_api_settings()
    rag_settings = get_rag_settings()
    customer_repository = InMemoryCustomerRepository.from_fixtures()
    llm_factory = LLMFactory(llm_settings, enable_tracing=True)
    tracer = build_tracer()

    await asyncio.to_thread(migrate, api_settings.database_url)
    embeddings = FastEmbedAdapter(
        rag_settings.rag_embedding_model,
        threads=rag_settings.rag_embedding_threads,
        batch_size=rag_settings.rag_embedding_batch_size,
    )
    rag_pool = ConnectionPool(api_settings.database_url, open=True)
    retrieve = functools.partial(hybrid_search_async, rag_pool, rag_settings)
    knowledge_agent_node = make_knowledge_agent_node(
        llm_factory, embeddings, retrieve, rag_settings
    )

    try:
        async with postgres_checkpointer(api_settings.database_url) as checkpointer:
            graph = build_graph(
                llm_factory, customer_repository, knowledge_agent_node, checkpointer=checkpointer
            )
            app.state.context = AppContext(
                llm_factory=llm_factory,
                llm_settings=llm_settings,
                customer_repository=customer_repository,
                checkpointer=checkpointer,
                graph=graph,
                tracer=tracer,
                knowledge_embeddings=embeddings,
                knowledge_retrieve=retrieve,
                rag_settings=rag_settings,
            )
            yield
    finally:
        rag_pool.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Solvia API", lifespan=lifespan)
    app.include_router(router)
    app.include_router(knowledge_router)
    return app


app = create_app()

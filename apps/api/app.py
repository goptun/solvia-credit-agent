"""FastAPI application factory.

Run locally with: `uv run uvicorn apps.api.app:app --reload`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from apps.agent.checkpointer import postgres_checkpointer
from apps.agent.graph import build_graph
from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.settings import get_settings
from apps.agent.observability.logging import configure_logging
from apps.agent.observability.tracing import build_tracer
from apps.agent.repositories.customers import InMemoryCustomerRepository
from apps.api.context import AppContext
from apps.api.routes import router
from apps.api.settings import get_api_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    llm_settings = get_settings()
    api_settings = get_api_settings()
    customer_repository = InMemoryCustomerRepository.from_fixtures()
    llm_factory = LLMFactory(llm_settings, enable_tracing=True)
    tracer = build_tracer()

    async with postgres_checkpointer(api_settings.database_url) as checkpointer:
        graph = build_graph(llm_factory, customer_repository, checkpointer=checkpointer)
        app.state.context = AppContext(
            llm_factory=llm_factory,
            llm_settings=llm_settings,
            customer_repository=customer_repository,
            checkpointer=checkpointer,
            graph=graph,
            tracer=tracer,
        )
        yield


def create_app() -> FastAPI:
    app = FastAPI(title="Solvia API", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()

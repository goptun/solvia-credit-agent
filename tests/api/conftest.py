"""Shared test app: the real FastAPI app wired to fakes (no real Postgres,
no real gateway)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver

from apps.agent.graph import build_graph
from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.port import LLMPort
from apps.agent.llm.settings import Settings
from apps.agent.nodes.knowledge_agent import RetrieveFn
from apps.agent.observability.tracing import FakeTracer
from apps.agent.synthetic_data.models import Customer
from apps.api.context import AppContext
from apps.api.routes import router
from apps.api.routes_knowledge import router as knowledge_router
from rag.embeddings.fake import FakeEmbeddings
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings
from tests.agent.nodes.fakes import (
    ScriptedLLMFactory,
    StubCustomerRepository,
    fake_knowledge_agent_node,
)


async def _empty_retrieve(
    question: str, vector: list[float], source_type: object
) -> list[RetrievedChunk]:
    return []


def parse_sse(text: str) -> list[dict[str, Any]]:
    """Parse `event: X\\ndata: {...}\\n\\n` blocks into a list of payload dicts."""
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.splitlines()
        data_line = next(line for line in lines if line.startswith("data: "))
        events.append(json.loads(data_line.removeprefix("data: ")))
    return events


def build_test_app(
    fast_llm: LLMPort | None = None,
    smart_llm: LLMPort | None = None,
    customer: Customer | None = None,
    knowledge_retrieve: RetrieveFn | None = None,
    llm_turn_deadline_seconds: float = 45.0,
) -> FastAPI:
    factory = ScriptedLLMFactory(fast=fast_llm or FakeLLM(), smart=smart_llm or FakeLLM())
    repo = StubCustomerRepository(customer)
    graph = build_graph(factory, repo, fake_knowledge_agent_node, checkpointer=MemorySaver())

    app = FastAPI()
    app.include_router(router)
    app.include_router(knowledge_router)
    app.state.context = AppContext(
        llm_factory=factory,
        llm_settings=Settings(
            llm_provider="fake", llm_turn_deadline_seconds=llm_turn_deadline_seconds
        ),
        customer_repository=repo,
        checkpointer=MemorySaver(),
        graph=graph,
        tracer=FakeTracer(),
        knowledge_embeddings=FakeEmbeddings(),
        knowledge_retrieve=knowledge_retrieve or _empty_retrieve,
        rag_settings=RagSettings(),
    )
    return app

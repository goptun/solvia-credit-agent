"""Shared test app: the real FastAPI app wired to fakes (no real Postgres,
no real gateway)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver

from apps.agent.graph import build_graph
from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.settings import Settings
from apps.agent.synthetic_data.models import Customer
from apps.api.context import AppContext
from apps.api.routes import router
from tests.agent.nodes.fakes import ScriptedLLMFactory, StubCustomerRepository


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
    fast_llm: FakeLLM | None = None,
    smart_llm: FakeLLM | None = None,
    customer: Customer | None = None,
) -> FastAPI:
    factory = ScriptedLLMFactory(fast=fast_llm or FakeLLM(), smart=smart_llm or FakeLLM())
    repo = StubCustomerRepository(customer)
    graph = build_graph(factory, repo, checkpointer=MemorySaver())

    app = FastAPI()
    app.include_router(router)
    app.state.context = AppContext(
        llm_factory=factory,
        llm_settings=Settings(llm_provider="fake"),
        customer_repository=repo,
        checkpointer=MemorySaver(),
        graph=graph,
    )
    return app

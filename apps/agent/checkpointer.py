"""Postgres checkpointer construction.

Explicitly allowlists the custom Pydantic model types that appear in
`ConversationState` for msgpack deserialization, rather than relying on
LangGraph's permissive (deprecated) default — see
`design.md` — "Conversation graph and state".
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from apps.agent.state import CustomerProfileSummary, SimulationSlots, SimulationSummary

_ALLOWED_MSGPACK_MODULES = [SimulationSlots, CustomerProfileSummary, SimulationSummary]


def build_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)


@asynccontextmanager
async def postgres_checkpointer(conn_string: str) -> AsyncIterator[AsyncPostgresSaver]:
    """A ready-to-use `AsyncPostgresSaver`, with its tables migrated."""
    async with AsyncPostgresSaver.from_conn_string(conn_string, serde=build_serde()) as saver:
        await saver.setup()
        yield saver

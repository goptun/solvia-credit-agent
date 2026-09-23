"""Wraps an `LLMPort` so every call opens an LLM span on the current
turn's trace, if one is active — a no-op otherwise (e.g. in tests that
don't configure tracing).

See `specs/observability/spec.md` — "LangFuse tracing per turn and LLM
call".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from apps.agent.observability.tracing import get_current_turn


class TracedLLM:
    """Delegates every call to `inner`, opening an LLM span (named
    `call_name`) around it when a turn trace is active."""

    def __init__(self, inner: Any, call_name: str) -> None:
        self._inner = inner
        self._call_name = call_name

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        turn = get_current_turn()
        if turn is None:
            return await self._inner.ainvoke(messages, **kwargs)
        with turn.llm_span(self._call_name):
            return await self._inner.ainvoke(messages, **kwargs)

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> TracedLLM:
        return TracedLLM(self._inner.bind_tools(tools, **kwargs), self._call_name)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> TracedLLM:
        return TracedLLM(self._inner.with_structured_output(schema, **kwargs), self._call_name)

"""Deterministic fake LLM adapter used by the automated test suite.

Selected via `LLM_PROVIDER=fake`. It never makes a network call, so CI
never depends on the real gateway (see
`specs/llm-gateway/spec.md` — "Fake LLM for automated tests").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


class FakeStructuredLLM:
    """Fake adapter returned by `FakeLLM.with_structured_output(...)`."""

    def __init__(self, responses: Sequence[Any], schema: Any) -> None:
        self._responses = list(responses)
        self._schema = schema
        self._index = 0
        self.calls: list[Sequence[BaseMessage]] = []

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        self.calls.append(messages)
        if not self._responses:
            raise ValueError("FakeLLM has no structured responses configured")
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


class FakeLLM:
    """Returns pre-programmed `AIMessage` responses in order.

    With no responses configured, every call returns a plain "OK" reply
    — enough for tests that only care that *a* call happened.
    """

    def __init__(self, responses: Sequence[AIMessage] | None = None) -> None:
        self._responses = list(responses or [])
        self._index = 0
        self.calls: list[Sequence[BaseMessage]] = []

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        if not self._responses:
            return AIMessage(content="OK")
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeStructuredLLM:
        return FakeStructuredLLM(self._responses, schema)

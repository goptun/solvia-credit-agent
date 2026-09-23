"""Deterministic fake LLM adapter used by the automated test suite.

Selected via `LLM_PROVIDER=fake`. It never makes a network call, so CI
never depends on the real gateway (see
`specs/llm-gateway/spec.md` — "Fake LLM for automated tests").

A single `FakeLLM` instance can back several nodes that share the same
tier (e.g. `router` and `compliance_guard` are both `fast`), each using
a different `.with_structured_output(...)` schema — or a plain
`.ainvoke(...)` call. All of these share one ordered response queue and
one `.calls` log on the `FakeLLM` itself, so a scripted list of
responses is consumed strictly in real call order, regardless of which
node or schema asked.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


class FakeStructuredLLM:
    """A `.with_structured_output(schema)` view onto a parent `FakeLLM`.

    Delegates its response queue and call log to the parent, so it
    shares state with the parent's plain `.ainvoke(...)` and with any
    other schema's structured view.
    """

    def __init__(self, parent: FakeLLM, schema: Any) -> None:
        self._parent = parent
        self._schema = schema

    @property
    def calls(self) -> list[Sequence[BaseMessage]]:
        return self._parent.calls

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        self._parent.calls.append(messages)
        return self._parent._next_response()


class FakeLLM:
    """Returns pre-programmed responses, in order, to whichever caller
    asks next (plain `.ainvoke` or any `.with_structured_output(...)`
    view). With no responses configured, plain calls return "OK"."""

    def __init__(self, responses: Sequence[Any] | None = None) -> None:
        self._responses = list(responses or [])
        self._index = 0
        self.calls: list[Sequence[BaseMessage]] = []
        self._structured_views: dict[Any, FakeStructuredLLM] = {}

    def _next_response(self) -> Any:
        if not self._responses:
            raise ValueError("FakeLLM has no responses configured")
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.calls.append(messages)
        if not self._responses:
            return AIMessage(content="OK")
        response: AIMessage = self._next_response()
        return response

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> FakeLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeStructuredLLM:
        if schema not in self._structured_views:
            self._structured_views[schema] = FakeStructuredLLM(self, schema)
        return self._structured_views[schema]

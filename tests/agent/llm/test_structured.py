"""Structured output: native tool-calling first, JSON-mode fallback with
fence-stripping and one repair retry, and empty/length-completion retry
integrated into that fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel

from apps.agent.llm.errors import StructuredOutputError
from apps.agent.llm.structured import ainvoke_structured, strip_markdown_fences

MESSAGES = [HumanMessage(content="extraia os dados")]


class _Schema(BaseModel):
    value: str


class _AlwaysFailsStructured:
    """The `.with_structured_output(...)` view: native tool-calling never
    works for this fake LLM, forcing the JSON-mode fallback."""

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        raise ValueError("this model does not support function calling")


class _PlainTextThenStructuredFailsLLM:
    """`.ainvoke(...)` (plain/JSON-mode) returns each of `plain_responses`
    in order; `.with_structured_output(...)` always fails, so every test
    here exercises the JSON-mode fallback path specifically."""

    def __init__(self, plain_responses: Sequence[AIMessage]) -> None:
        self._responses = list(plain_responses)
        self.plain_calls = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.plain_calls += 1
        return self._responses[min(self.plain_calls - 1, len(self._responses) - 1)]

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _PlainTextThenStructuredFailsLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _AlwaysFailsStructured:
        return _AlwaysFailsStructured()


class _NativeStructuredView:
    def __init__(self, result: BaseModel) -> None:
        self._result = result

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> BaseModel:
        return self._result


class _NativeSucceedsLLM:
    """`.with_structured_output(...)` succeeds immediately — the JSON-mode
    fallback (this object's own `.ainvoke`) must never be exercised."""

    def __init__(self, result: BaseModel) -> None:
        self._result = result
        self.plain_calls = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.plain_calls += 1
        raise AssertionError("JSON-mode fallback should not run when native output succeeds")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _NativeSucceedsLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _NativeStructuredView:
        return _NativeStructuredView(self._result)


def test_strip_markdown_fences_removes_a_json_fence() -> None:
    assert strip_markdown_fences('```json\n{"value": "x"}\n```') == '{"value": "x"}'


def test_strip_markdown_fences_removes_a_bare_fence() -> None:
    assert strip_markdown_fences('```\n{"value": "x"}\n```') == '{"value": "x"}'


def test_strip_markdown_fences_leaves_plain_json_untouched() -> None:
    assert strip_markdown_fences('{"value": "x"}') == '{"value": "x"}'


async def test_native_tool_calling_succeeds_without_touching_json_mode() -> None:
    llm = _NativeSucceedsLLM(_Schema(value="native"))

    result = await ainvoke_structured(llm, MESSAGES, _Schema)

    assert result.value == "native"
    assert llm.plain_calls == 0


async def test_falls_back_to_json_mode_with_fenced_content_when_native_fails() -> None:
    llm = _PlainTextThenStructuredFailsLLM(
        [AIMessage(content='```json\n{"value": "from-json-mode"}\n```')]
    )

    result = await ainvoke_structured(llm, MESSAGES, _Schema)

    assert result.value == "from-json-mode"


async def test_json_mode_retries_through_an_empty_length_completion() -> None:
    """A fake that returns an empty completion with finish_reason="length"
    on its first plain call, then valid fenced JSON — task 1's retry
    mechanism and task 2's JSON-mode fallback composed together."""
    llm = _PlainTextThenStructuredFailsLLM(
        [
            AIMessage(content="", response_metadata={"finish_reason": "length"}),
            AIMessage(content='```json\n{"value": "recovered"}\n```'),
        ]
    )

    result = await ainvoke_structured(llm, MESSAGES, _Schema, max_retries=3)

    assert result.value == "recovered"
    assert llm.plain_calls == 2


async def test_invalid_json_triggers_exactly_one_repair_retry_then_succeeds() -> None:
    llm = _PlainTextThenStructuredFailsLLM(
        [
            AIMessage(content="isto não é JSON"),
            AIMessage(content='{"value": "repaired"}'),
        ]
    )

    result = await ainvoke_structured(llm, MESSAGES, _Schema, max_retries=1)

    assert result.value == "repaired"
    assert llm.plain_calls == 2


async def test_repair_retry_failing_again_raises_structured_output_error() -> None:
    llm = _PlainTextThenStructuredFailsLLM(
        [AIMessage(content="isto não é JSON"), AIMessage(content="ainda não é JSON")]
    )

    with pytest.raises(StructuredOutputError):
        await ainvoke_structured(llm, MESSAGES, _Schema, max_retries=1)

    assert llm.plain_calls == 2  # exactly one repair attempt, never more


async def test_degrades_to_fallback_tier_when_primary_fails_entirely() -> None:
    primary = _PlainTextThenStructuredFailsLLM(
        [AIMessage(content="nunca vai virar JSON"), AIMessage(content="continua não sendo")]
    )
    fallback = _NativeSucceedsLLM(_Schema(value="from-fallback-tier"))

    result = await ainvoke_structured(primary, MESSAGES, _Schema, fallback=fallback, max_retries=1)

    assert result.value == "from-fallback-tier"


async def test_no_fallback_and_primary_fails_entirely_raises() -> None:
    primary = _PlainTextThenStructuredFailsLLM(
        [AIMessage(content="nunca vai virar JSON"), AIMessage(content="continua não sendo")]
    )

    with pytest.raises(StructuredOutputError):
        await ainvoke_structured(primary, MESSAGES, _Schema, fallback=None, max_retries=1)


class _NoToolCallStructured:
    """The model answered without calling the tool: LangChain's tool
    parser returns `None` instead of raising."""

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        return None


class _NoToolCallLLM(_PlainTextThenStructuredFailsLLM):
    def with_structured_output(self, schema: Any, **kwargs: Any) -> _NoToolCallStructured:  # type: ignore[override]
        return _NoToolCallStructured()


async def test_no_tool_call_result_falls_through_to_json_mode_instead_of_returning_none() -> None:
    llm = _NoToolCallLLM([AIMessage(content='```json\n{"value": "ok"}\n```')])

    result = await ainvoke_structured(llm, MESSAGES, _Schema)

    assert result == _Schema(value="ok")
    assert llm.plain_calls == 1


async def test_no_tool_call_then_unusable_json_raises_and_never_returns_none() -> None:
    llm = _NoToolCallLLM([AIMessage(content="não sei responder")])

    with pytest.raises(StructuredOutputError):
        await ainvoke_structured(llm, MESSAGES, _Schema)

    assert llm.plain_calls == 2  # the JSON attempt plus exactly one repair


async def test_a_result_of_the_wrong_type_is_also_a_native_failure() -> None:
    class _WrongTypeStructured:
        async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
            return {"value": "not a schema instance"}

    class _WrongTypeLLM(_PlainTextThenStructuredFailsLLM):
        def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
            return _WrongTypeStructured()

    llm = _WrongTypeLLM([AIMessage(content='{"value": "from json"}')])

    result = await ainvoke_structured(llm, MESSAGES, _Schema)

    assert result == _Schema(value="from json")


async def test_the_json_mode_call_carries_the_schema() -> None:
    seen: list[str] = []

    class _RecordingLLM(_NoToolCallLLM):
        async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
            seen.append(str(messages[-1].content))
            return await super().ainvoke(messages, **kwargs)

    llm = _RecordingLLM([AIMessage(content='{"value": "ok"}')])

    await ainvoke_structured(llm, MESSAGES, _Schema)

    assert "JSON Schema" in seen[0]
    assert '"value"' in seen[0]

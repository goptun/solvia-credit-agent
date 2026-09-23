"""Timeout/retry, smart->fast degradation, and the friendly fallback message."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE, invoke_with_resilience


class ScriptedLLM:
    """Raises for the first `fail_times` calls, then returns `response`."""

    def __init__(self, fail_times: int, response: AIMessage | None = None) -> None:
        self.fail_times = fail_times
        self.response = response or AIMessage(content="ok")
        self.call_count = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise RuntimeError("transient gateway error")
        return self.response

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self


class AlwaysFailsLLM:
    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        raise RuntimeError("permanent gateway error")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> AlwaysFailsLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self


MESSAGES = [HumanMessage(content="oi")]


@pytest.mark.asyncio
async def test_transient_failure_is_retried_then_succeeds() -> None:
    llm = ScriptedLLM(fail_times=2, response=AIMessage(content="funcionou"))

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=3)

    assert result.message.content == "funcionou"
    assert llm.call_count == 3


@pytest.mark.asyncio
async def test_smart_tier_succeeds_without_touching_fallback() -> None:
    primary = ScriptedLLM(fail_times=0, response=AIMessage(content="smart ok"))
    fallback_calls: list[Any] = []

    class TrackedFallback(AlwaysFailsLLM):
        async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
            fallback_calls.append(messages)
            return await super().ainvoke(messages, **kwargs)

    result = await invoke_with_resilience(
        primary, MESSAGES, fallback=TrackedFallback(), max_retries=3
    )

    assert result.message.content == "smart ok"
    assert fallback_calls == []


@pytest.mark.asyncio
async def test_smart_fails_then_fast_fallback_succeeds() -> None:
    primary = AlwaysFailsLLM()
    fallback = ScriptedLLM(fail_times=0, response=AIMessage(content="fast ok"))

    result = await invoke_with_resilience(primary, MESSAGES, fallback=fallback, max_retries=2)

    assert result.message.content == "fast ok"


@pytest.mark.asyncio
async def test_both_tiers_fail_returns_friendly_unavailable_message() -> None:
    result = await invoke_with_resilience(
        AlwaysFailsLLM(), MESSAGES, fallback=AlwaysFailsLLM(), max_retries=2
    )

    assert result.message.content == UNAVAILABLE_MESSAGE
    assert result.resolved_model is None


@pytest.mark.asyncio
async def test_fast_tier_with_no_fallback_returns_unavailable_message() -> None:
    result = await invoke_with_resilience(AlwaysFailsLLM(), MESSAGES, fallback=None, max_retries=2)

    assert result.message.content == UNAVAILABLE_MESSAGE


@pytest.mark.asyncio
async def test_resolved_model_attached_when_present_in_metadata() -> None:
    message = AIMessage(content="oi", response_metadata={"model_name": "gemini-3.8-flash"})
    llm = ScriptedLLM(fail_times=0, response=message)

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=1)

    assert result.resolved_model == "gemini-3.8-flash"


@pytest.mark.asyncio
async def test_resolved_model_absent_when_not_in_metadata() -> None:
    message = AIMessage(content="oi", response_metadata={})
    llm = ScriptedLLM(fail_times=0, response=message)

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=1)

    assert result.resolved_model is None

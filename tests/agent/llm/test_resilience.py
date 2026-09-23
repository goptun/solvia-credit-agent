"""Timeout/retry (transient errors only), smart->fast degradation, the
friendly fallback message, and empty/length-completion retry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE, invoke_with_resilience


def _transient_error() -> httpx.TimeoutException:
    return httpx.TimeoutException("gateway timed out")


class ScriptedLLM:
    """Raises a transient error for the first `fail_times` calls, then
    returns `response`."""

    def __init__(self, fail_times: int, response: AIMessage | None = None) -> None:
        self.fail_times = fail_times
        self.response = response or AIMessage(content="ok")
        self.call_count = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise _transient_error()
        return self.response

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> ScriptedLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self


class AlwaysFailsLLM:
    """Raises a non-transient error on every call — never worth retrying."""

    def __init__(self) -> None:
        self.call_count = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.call_count += 1
        raise ValueError("permanent, non-transient gateway error")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> AlwaysFailsLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self


MESSAGES = [HumanMessage(content="oi")]


async def test_transient_failure_is_retried_then_succeeds() -> None:
    llm = ScriptedLLM(fail_times=2, response=AIMessage(content="funcionou"))

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=3)

    assert result.message.content == "funcionou"
    assert llm.call_count == 3


async def test_non_transient_failure_is_not_retried() -> None:
    llm = AlwaysFailsLLM()

    result = await invoke_with_resilience(llm, MESSAGES, fallback=None, max_retries=3)

    assert result.message.content == UNAVAILABLE_MESSAGE
    assert llm.call_count == 1  # no retries wasted on a non-transient error


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


async def test_smart_fails_then_fast_fallback_succeeds() -> None:
    primary = AlwaysFailsLLM()
    fallback = ScriptedLLM(fail_times=0, response=AIMessage(content="fast ok"))

    result = await invoke_with_resilience(primary, MESSAGES, fallback=fallback, max_retries=2)

    assert result.message.content == "fast ok"


async def test_both_tiers_fail_returns_friendly_unavailable_message() -> None:
    result = await invoke_with_resilience(
        AlwaysFailsLLM(), MESSAGES, fallback=AlwaysFailsLLM(), max_retries=2
    )

    assert result.message.content == UNAVAILABLE_MESSAGE
    assert result.resolved_model is None


async def test_fast_tier_with_no_fallback_returns_unavailable_message() -> None:
    result = await invoke_with_resilience(AlwaysFailsLLM(), MESSAGES, fallback=None, max_retries=2)

    assert result.message.content == UNAVAILABLE_MESSAGE


async def test_resolved_model_attached_when_present_in_metadata() -> None:
    message = AIMessage(content="oi", response_metadata={"model_name": "gemini-3.8-flash"})
    llm = ScriptedLLM(fail_times=0, response=message)

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=1)

    assert result.resolved_model == "gemini-3.8-flash"


async def test_resolved_model_absent_when_not_in_metadata() -> None:
    message = AIMessage(content="oi", response_metadata={})
    llm = ScriptedLLM(fail_times=0, response=message)

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=1)

    assert result.resolved_model is None


class _EmptyLengthThenOkLLM:
    """Returns an empty completion with finish_reason="length" for the
    first `fail_times` calls, then a real reply."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.call_count = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            return AIMessage(content="", response_metadata={"finish_reason": "length"})
        return AIMessage(content="resposta completa")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _EmptyLengthThenOkLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        return self


async def test_empty_length_completion_is_retried_then_succeeds() -> None:
    llm = _EmptyLengthThenOkLLM(fail_times=2)

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=3)

    assert result.message.content == "resposta completa"
    assert llm.call_count == 3


async def test_empty_length_completion_exhausts_retries_and_returns_unavailable() -> None:
    llm = _EmptyLengthThenOkLLM(fail_times=99)

    result = await invoke_with_resilience(llm, MESSAGES, fallback=None, max_retries=2)

    assert result.message.content == UNAVAILABLE_MESSAGE
    assert llm.call_count == 2


async def test_non_empty_content_with_finish_reason_length_is_not_retried() -> None:
    """`finish_reason == "length"` alone isn't the signal — only an
    *empty* completion is treated as retryable (a long-but-truncated
    reply is still useful and should not be discarded)."""
    message = AIMessage(
        content="uma resposta parcial", response_metadata={"finish_reason": "length"}
    )
    llm = ScriptedLLM(fail_times=0, response=message)

    result = await invoke_with_resilience(llm, MESSAGES, max_retries=3)

    assert result.message.content == "uma resposta parcial"
    assert llm.call_count == 1

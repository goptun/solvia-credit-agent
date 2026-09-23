"""Per-turn LLM deadline (`LLM_TURN_DEADLINE_SECONDS`): retries, the
JSON-mode fallback and smart -> fast degradation must all fit inside
one shared budget, with a deliberately slow fake LLM."""

from __future__ import annotations

import time

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.deadline import remaining_seconds, run_within_deadline, turn_deadline
from apps.agent.llm.errors import LLMDeadlineExceeded
from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE, invoke_with_resilience
from apps.agent.llm.structured import ainvoke_structured
from tests.agent.nodes.fakes import SlowLLM

_MESSAGES = [HumanMessage(content="oi")]


class _Schema(BaseModel):
    value: int


async def test_deadline_cuts_off_retries_json_fallback_and_tier_degradation() -> None:
    primary = SlowLLM(0.05, fail=True)
    fallback = SlowLLM(0.05, fail=True)

    started = time.monotonic()
    with turn_deadline(0.3), pytest.raises(LLMDeadlineExceeded):
        await ainvoke_structured(primary, _MESSAGES, _Schema, fallback=fallback)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    # Without a deadline this path makes 3 native + 3 JSON + 1 repair
    # attempts on the primary and again on the fallback (backoff waits
    # alone exceed the budget) — it must be cut short well before that.
    assert primary.calls + fallback.calls < 14


async def test_the_deadline_does_not_degrade_to_the_fallback_after_it_expires() -> None:
    primary = SlowLLM(1.0)
    fallback = SlowLLM(0.0, response=_Schema(value=1))

    with turn_deadline(0.1), pytest.raises(LLMDeadlineExceeded):
        await ainvoke_structured(primary, _MESSAGES, _Schema, fallback=fallback)

    assert fallback.calls == 0


async def test_invoke_with_resilience_returns_the_unavailable_reply_on_expiry() -> None:
    slow = SlowLLM(1.0)

    started = time.monotonic()
    with turn_deadline(0.1):
        result = await invoke_with_resilience(slow, _MESSAGES)

    assert result.message.content == UNAVAILABLE_MESSAGE
    assert time.monotonic() - started < 0.6


async def test_without_a_deadline_calls_are_unbounded_and_unchanged() -> None:
    fast = SlowLLM(0.01, response=_Schema(value=7))

    result = await ainvoke_structured(fast, _MESSAGES, _Schema)

    assert result == _Schema(value=7)
    assert remaining_seconds() is None


async def test_an_already_exhausted_deadline_never_calls_the_llm() -> None:
    llm = SlowLLM(0.0, response=_Schema(value=1))

    with turn_deadline(0.0), pytest.raises(LLMDeadlineExceeded):
        await ainvoke_structured(llm, _MESSAGES, _Schema)

    assert llm.calls == 0


async def test_a_nested_deadline_cannot_extend_the_outer_budget() -> None:
    with turn_deadline(0.5):
        with turn_deadline(100):
            remaining = remaining_seconds()
        assert remaining is not None
        assert remaining <= 0.5


async def test_a_timeout_that_is_not_the_deadline_is_not_reported_as_one() -> None:
    async def raises_timeout() -> None:
        raise TimeoutError("some other timeout")

    with turn_deadline(10), pytest.raises(TimeoutError) as info:
        await run_within_deadline(raises_timeout)

    assert not isinstance(info.value, LLMDeadlineExceeded)

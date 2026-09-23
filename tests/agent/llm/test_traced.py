"""TracedLLM opens exactly one LLM span per call, and never leaks
anything beyond the call name into the span (no settings, no secrets,
no infra details) — see `specs/observability/spec.md` — "Traces never
contain secrets or infra details"."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.traced import TracedLLM
from apps.agent.observability.tracing import reset_current_turn, set_current_turn


class _SpyTurn:
    def __init__(self) -> None:
        self.llm_span_calls: list[Any] = []

    @contextmanager
    def node_span(self, node_name: str):  # type: ignore[no-untyped-def]
        yield

    @contextmanager
    def llm_span(self, call_name: str):  # type: ignore[no-untyped-def]
        self.llm_span_calls.append(call_name)
        yield


async def test_traced_llm_opens_exactly_one_span_per_ainvoke_call() -> None:
    turn = _SpyTurn()
    token = set_current_turn(turn)
    try:
        traced = TracedLLM(FakeLLM(), call_name="router")
        await traced.ainvoke([HumanMessage(content="oi")])
        await traced.ainvoke([HumanMessage(content="de novo")])
    finally:
        reset_current_turn(token)

    assert turn.llm_span_calls == ["router", "router"]


async def test_traced_llm_span_carries_only_the_call_name_no_settings_or_secrets() -> None:
    """The span call receives a single positional string (the node/call
    name) — there is no code path for it to receive an API key, base
    URL, or any other settings-derived value."""
    turn = _SpyTurn()
    token = set_current_turn(turn)
    try:
        traced = TracedLLM(FakeLLM(), call_name="compliance_guard")
        await traced.ainvoke([HumanMessage(content="oi")])
    finally:
        reset_current_turn(token)

    assert turn.llm_span_calls == ["compliance_guard"]
    for call in turn.llm_span_calls:
        assert isinstance(call, str)
        assert "key" not in call.lower()
        assert "http" not in call.lower()


async def test_traced_llm_without_an_active_turn_is_a_pure_passthrough() -> None:
    fake = FakeLLM(responses=[AIMessage(content="ok")])
    traced = TracedLLM(fake, call_name="router")

    result = await traced.ainvoke([HumanMessage(content="oi")])

    assert result.content == "ok"


async def test_traced_llm_structured_output_is_also_traced() -> None:
    from pydantic import BaseModel

    class Decision(BaseModel):
        value: str

    turn = _SpyTurn()
    token = set_current_turn(turn)
    try:
        fake = FakeLLM(responses=[Decision(value="x")])
        traced = TracedLLM(fake, call_name="router")
        structured = traced.with_structured_output(Decision)
        result = await structured.ainvoke([HumanMessage(content="oi")])
    finally:
        reset_current_turn(token)

    assert result.value == "x"
    assert turn.llm_span_calls == ["router"]

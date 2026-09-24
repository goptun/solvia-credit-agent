"""The instrumented factory records every raw provider call, native
structured-output calls included."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from apps.agent.llm.port import LLMPort
from apps.agent.llm.settings import Settings
from apps.agent.llm.structured import ainvoke_structured
from apps.agent.nodes.router import RouterDecision
from evals.adapters.instrumentation import CallRecorder, InstrumentedLLMFactory
from evals.core.budget import CallBudget
from evals.core.operational import node_metrics


class QuotaError(Exception):
    status_code = 429


class ScriptedChat(BaseChatModel):
    """A chat model that plays back a script of messages/errors."""

    script: list[Any]
    position: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        return self.bind(tools=tools, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        item = self.script[self.position]
        self.position += 1
        if isinstance(item, Exception):
            raise item
        return ChatResult(generations=[ChatGeneration(message=item)])


def _tool_call(model: str, **args: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "RouterDecision", "args": args, "id": "call-1"}],
        response_metadata={"model_name": model},
    )


def _text(content: str, model: str) -> AIMessage:
    return AIMessage(content=content, response_metadata={"model_name": model})


class _Factory(InstrumentedLLMFactory):
    def __init__(self, recorder: CallRecorder, script: list[Any]) -> None:
        super().__init__(Settings(llm_provider="fake"), recorder)
        self._script = script

    def _build(self, model: str, max_tokens: int, timeout_seconds: float) -> LLMPort:
        chat = ScriptedChat(script=self._script)
        return chat  # type: ignore[return-value]


async def _route(factory: InstrumentedLLMFactory) -> RouterDecision:
    return await ainvoke_structured(
        factory.for_node("router"),
        [HumanMessage(content="oi")],
        RouterDecision,
        fallback=factory.fallback_for_node("router"),
    )


async def test_a_native_tool_call_records_the_resolved_model() -> None:
    recorder = CallRecorder()
    factory = _Factory(recorder, [_tool_call("gemini-primary", intent="complaint")])

    decision = await _route(factory)

    assert decision.intent == "complaint"
    [record] = recorder.records
    assert (record.node, record.alias, record.model) == ("router", "solvia-fast", "gemini-primary")
    assert record.status == "ok" and record.kind == "tool_call" and record.structured
    assert record.latency_seconds >= 0


async def test_a_completion_without_a_tool_call_is_followed_by_a_json_call_in_one_operation() -> (
    None
):
    recorder = CallRecorder()
    factory = _Factory(
        recorder,
        [_text("claro!", "gemini-lite"), _text('{"intent": "out_of_scope"}', "gemini-lite")],
    )

    decision = await _route(factory)

    assert decision.intent == "out_of_scope"
    first, second = recorder.records
    assert first.kind == "no_tool_call" and second.kind == "no_tool_call"
    assert first.operation_id == second.operation_id
    metrics = node_metrics(recorder.records)["router"]
    assert metrics.attempts_max == 2
    assert metrics.json_fallback_rate.value == 1.0


async def test_an_error_records_its_http_status_and_no_model() -> None:
    recorder = CallRecorder()
    factory = _Factory(recorder, [QuotaError("slow down")])
    llm = factory.for_node("router")

    with pytest.raises(QuotaError):
        await llm.ainvoke([HumanMessage(content="oi")])

    [record] = recorder.records
    assert record.status == "429" and record.kind == "error" and record.model is None


async def test_calls_are_counted_against_the_budget() -> None:
    budget = CallBudget(max_calls=10)
    recorder = CallRecorder(budget)
    factory = _Factory(recorder, [_tool_call("m", intent="complaint")])

    await _route(factory)

    assert budget.used == 1


async def test_the_smart_tier_fallback_shares_the_operation_and_names_the_fast_alias() -> None:
    recorder = CallRecorder()
    factory = _Factory(recorder, [_text("x", "m")])

    primary = factory.for_node("knowledge_agent")
    fallback = factory.fallback_for_node("knowledge_agent")
    assert fallback is not None
    await primary.ainvoke([HumanMessage(content="a")])

    fallback_chat: Any = fallback
    fallback_chat.script = [_text("y", "m2")]
    await fallback.ainvoke([HumanMessage(content="a")])

    smart, fast = recorder.records
    assert (smart.node, smart.alias) == ("knowledge_agent", "solvia-smart")
    assert (fast.node, fast.alias) == ("knowledge_agent_fallback", "solvia-fast")
    assert smart.operation_id == fast.operation_id

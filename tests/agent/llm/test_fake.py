"""FakeLLM drives node tests without any network call."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from apps.agent.llm.fake import FakeLLM


@pytest.mark.asyncio
async def test_fake_llm_returns_default_response_and_records_calls() -> None:
    llm = FakeLLM()
    messages = [HumanMessage(content="oi")]

    result = await llm.ainvoke(messages)

    assert isinstance(result, AIMessage)
    assert result.content == "OK"
    assert llm.calls == [messages]


@pytest.mark.asyncio
async def test_fake_llm_returns_configured_responses_in_order() -> None:
    responses = [AIMessage(content="primeira"), AIMessage(content="segunda")]
    llm = FakeLLM(responses=responses)

    first = await llm.ainvoke([HumanMessage(content="a")])
    second = await llm.ainvoke([HumanMessage(content="b")])

    assert first.content == "primeira"
    assert second.content == "segunda"


@pytest.mark.asyncio
async def test_fake_llm_structured_output_returns_configured_object() -> None:
    payload = {"amount": 1000}
    llm = FakeLLM(responses=[payload])

    structured = llm.with_structured_output(dict)
    result = await structured.ainvoke([HumanMessage(content="24 meses")])

    assert result == payload

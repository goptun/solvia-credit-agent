"""Test doubles for node-level tests.

`ScriptedLLMFactory` is a real `LLMFactory` subclass (so it satisfies
every node's type hint) whose tiers return whichever `FakeLLM` the test
configured, instead of a fresh empty one per call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import httpx
from langchain_core.messages import AIMessage, BaseMessage

from apps.agent.llm.factory import LLMFactory, Tier
from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.port import LLMPort
from apps.agent.llm.settings import Settings
from apps.agent.state import ConversationState
from apps.agent.synthetic_data.models import Customer


class SlowLLM:
    """An `LLMPort` double whose every call sleeps `delay` seconds and
    then either fails with a transient timeout (`fail=True`, so retries
    and fallbacks keep firing) or returns `response`. Counts calls in
    `.calls`, across plain and structured use."""

    def __init__(self, delay: float, *, fail: bool = False, response: Any = None) -> None:
        self.delay = delay
        self.fail = fail
        self.response = response if response is not None else AIMessage(content="{}")
        self.calls = 0

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise httpx.ReadTimeout("simulated slow gateway")
        return self.response

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> SlowLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> SlowLLM:
        return self


class ScriptedLLMFactory(LLMFactory):
    def __init__(self, fast: LLMPort | None = None, smart: LLMPort | None = None) -> None:
        super().__init__(Settings(llm_provider="fake"))
        self._fast: LLMPort = fast if fast is not None else FakeLLM()
        self._smart: LLMPort = smart if smart is not None else FakeLLM()

    def for_alias(self, tier: Tier) -> LLMPort:
        return self._fast if tier == "fast" else self._smart


async def fake_knowledge_agent_node(state: ConversationState) -> ConversationState:
    """A minimal `KnowledgeAgentNode` double for graph-level tests that
    don't exercise the `product_question`/`regulatory_question` path
    themselves (grounded-answer behavior is covered by
    `tests/agent/nodes/test_knowledge_agent.py`)."""
    return ConversationState(draft_reply="Resposta fictícia do knowledge_agent.")


class StubCustomerRepository:
    """Returns the same (possibly `None`) customer regardless of id."""

    def __init__(self, customer: Customer | None) -> None:
        self._customer = customer

    def get(self, customer_id: str) -> Customer | None:
        return self._customer

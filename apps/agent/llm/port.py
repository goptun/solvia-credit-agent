"""Provider-agnostic LLM interface every agent node calls through.

Adapters wrap a concrete LangChain chat model but expose only this
narrow surface, so node code never imports a provider SDK directly (see
`design.md` — "Module boundaries and dependency injection").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage


class LLMPort(Protocol):
    """Minimal chat-model surface agent nodes are allowed to use."""

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage: ...

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> LLMPort: ...

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any: ...

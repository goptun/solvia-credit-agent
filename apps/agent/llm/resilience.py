"""Application-level resilience on top of the LLM gateway.

Timeout + exponential-backoff retries, one-shot smart->fast degradation,
and a fixed "demo temporarily unavailable" fallback when both tiers
fail — the gateway's own model-alias fallback chain is a separate,
lower-level concern this layer does not duplicate (see `design.md` —
"LLM gateway strategy and provider abstraction").
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from apps.agent.llm.port import LLMPort

UNAVAILABLE_MESSAGE = (
    "Desculpe, o assistente está temporariamente indisponível. "
    "Por favor, tente novamente em alguns instantes."
)


@dataclass(frozen=True)
class LLMCallResult:
    """The outcome of a resilient LLM call.

    `resolved_model` carries the underlying model name from the
    gateway's response metadata, when present, for observability to
    attach to a trace span later (task group 9) — the application never
    branches logic on this value.
    """

    message: AIMessage
    resolved_model: str | None


def _extract_resolved_model(message: AIMessage) -> str | None:
    metadata = message.response_metadata or {}
    model = metadata.get("model_name") or metadata.get("model")
    return str(model) if model else None


async def call_with_retries(
    llm: Any,
    messages: Sequence[BaseMessage],
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Timeout/retry wrapper generic enough for both a plain chat model
    (`.ainvoke(...) -> AIMessage`) and a `.with_structured_output(...)`
    call (`.ainvoke(...) -> <schema instance>`).

    There is no universally safe fallback object for a structured call
    (unlike the fixed unavailable `AIMessage` for plain calls), so a
    structured-output call that exhausts retries raises instead of
    degrading — callers (the API layer, task group 8) turn that into
    the same friendly unavailable response at the turn level.
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max(max_retries, 1)),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    ):
        with attempt:
            return await llm.ainvoke(messages, **kwargs)
    raise RuntimeError("retry loop exited without a result")  # pragma: no cover


async def call_structured_with_resilience(
    primary: Any,
    messages: Sequence[BaseMessage],
    *,
    fallback: Any | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """Like `invoke_with_resilience`, for a `.with_structured_output(...)`
    call: degrade once to `fallback` on exhausted retries. There is no
    safe synthetic default for an arbitrary schema, so if `fallback` also
    fails (or there is none), the exception propagates — callers turn
    that into the turn-level unavailable response (task group 8).
    """
    try:
        return await call_with_retries(primary, messages, max_retries, **kwargs)
    except Exception:
        if fallback is None:
            raise

    return await call_with_retries(fallback, messages, max_retries, **kwargs)


async def invoke_with_resilience(
    primary: LLMPort,
    messages: Sequence[BaseMessage],
    *,
    fallback: LLMPort | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> LLMCallResult:
    """Call `primary`; degrade once to `fallback` on exhausted retries;
    return the fixed unavailable message instead of raising if both fail
    (or if `primary` has no fallback and fails).
    """
    try:
        message = await call_with_retries(primary, messages, max_retries, **kwargs)
        return LLMCallResult(message=message, resolved_model=_extract_resolved_model(message))
    except Exception:
        if fallback is None:
            return LLMCallResult(
                message=AIMessage(content=UNAVAILABLE_MESSAGE), resolved_model=None
            )

    try:
        message = await call_with_retries(fallback, messages, max_retries, **kwargs)
        return LLMCallResult(message=message, resolved_model=_extract_resolved_model(message))
    except Exception:
        return LLMCallResult(message=AIMessage(content=UNAVAILABLE_MESSAGE), resolved_model=None)

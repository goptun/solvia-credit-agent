"""Structured-output extraction: native tool-calling first, then a
JSON-mode fallback with Markdown-fence stripping, Pydantic validation,
and exactly one repair retry.

See `specs/llm-gateway/spec.md` and `design.md` — "Gateway's free-tier
Gemini models return inconsistent structured output": the gateway may
swap models mid-conversation, and a model that doesn't reliably support
native tool-calling can still return usable JSON, sometimes wrapped in
a Markdown code fence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from apps.agent.llm.deadline import run_within_deadline
from apps.agent.llm.errors import LLMDeadlineExceeded, StructuredOutputError
from apps.agent.llm.port import LLMPort
from apps.agent.llm.resilience import call_with_retries

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")

_REPAIR_INSTRUCTION = (
    "Sua última resposta não é um JSON válido para o formato esperado. "
    "Erro de parsing/validação: {error}. Responda novamente com apenas "
    "o objeto JSON válido, sem texto adicional e sem marcação Markdown."
)


def strip_markdown_fences(text: str) -> str:
    """Remove a leading/trailing ```json ... ``` (or bare ```) fence, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return _FENCE_PATTERN.sub("", stripped).strip()


def _content_text(message: Any) -> str:
    content = getattr(message, "content", message)
    return content if isinstance(content, str) else str(content)


def _try_parse[T: BaseModel](message: Any, schema: type[T]) -> T | None:
    text = strip_markdown_fences(_content_text(message))
    try:
        return schema.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError):
        return None


async def _json_mode_with_repair[T: BaseModel](
    llm: LLMPort,
    messages: Sequence[BaseMessage],
    schema: type[T],
    max_retries: int,
) -> T:
    response = await call_with_retries(llm, messages, max_retries)
    parsed = _try_parse(response, schema)
    if parsed is not None:
        return parsed

    repair_prompt = [
        *messages,
        HumanMessage(content=_REPAIR_INSTRUCTION.format(error="conteúdo não é um JSON válido")),
    ]
    repaired = await call_with_retries(llm, repair_prompt, max_retries=1)
    parsed = _try_parse(repaired, schema)
    if parsed is not None:
        return parsed

    raise StructuredOutputError(
        f"could not extract {schema.__name__} via JSON mode after one repair attempt"
    )


async def _native_then_json_mode[T: BaseModel](
    llm: LLMPort,
    messages: Sequence[BaseMessage],
    schema: type[T],
    max_retries: int,
) -> T:
    try:
        structured = llm.with_structured_output(schema, method="function_calling")
        result: T = await call_with_retries(structured, messages, max_retries)
        return result
    except Exception:
        # Any failure of the native path (transient-exhausted or not)
        # falls through to the JSON-mode fallback exactly once — see
        # `specs/llm-gateway/spec.md`.
        return await _json_mode_with_repair(llm, messages, schema, max_retries)


async def ainvoke_structured[T: BaseModel](
    primary: LLMPort,
    messages: Sequence[BaseMessage],
    schema: type[T],
    *,
    fallback: LLMPort | None = None,
    max_retries: int = 3,
) -> T:
    """Extract `schema` from `messages` via `primary`; degrade once to
    `fallback` (the smart->fast tier degradation) if `primary` fails
    entirely, including its own JSON-mode fallback attempt.
    """
    return await run_within_deadline(
        lambda: _ainvoke_structured_with_fallback(primary, messages, schema, fallback, max_retries)
    )


async def _ainvoke_structured_with_fallback[T: BaseModel](
    primary: LLMPort,
    messages: Sequence[BaseMessage],
    schema: type[T],
    fallback: LLMPort | None,
    max_retries: int,
) -> T:
    try:
        return await _native_then_json_mode(primary, messages, schema, max_retries)
    except LLMDeadlineExceeded:
        raise
    except Exception:
        if fallback is None:
            raise
    return await _native_then_json_mode(fallback, messages, schema, max_retries)

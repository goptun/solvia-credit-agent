"""OpenAI-compatible adapter — talks to the `9router` gateway.

Client construction uses only settings-derived values; nothing is
hardcoded (see `design.md` — "LLM gateway strategy and provider
abstraction").
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from apps.agent.llm.settings import Settings


def build_openai_compatible_llm(settings: Settings, model: str, max_tokens: int) -> ChatOpenAI:
    """Build a `ChatOpenAI` client pointed at the configured gateway alias."""
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=SecretStr(settings.llm_api_key),
        model=model,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # retries are handled by apps.agent.llm.resilience
        max_tokens=max_tokens,  # type: ignore[call-arg]  # accepted at runtime; missing from this version's __init__ stub
    )

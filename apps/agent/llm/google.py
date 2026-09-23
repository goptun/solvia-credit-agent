"""Google adapter: Gemini Developer API or Vertex AI, chosen by configuration.

Same `LLMPort` surface either way, so nodes and the factory never branch
on which one is active (see `design.md` — "LLM gateway strategy and
provider abstraction").
"""

from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from apps.agent.llm.settings import Settings


def build_google_llm(settings: Settings, model: str) -> ChatGoogleGenerativeAI:
    """Build a `ChatGoogleGenerativeAI` client for the mode selected in settings."""
    if settings.google_vertexai:
        return ChatGoogleGenerativeAI(
            model=model,
            vertexai=True,
            project=settings.google_project,
            location=settings.google_location,
        )
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
    )

"""Google adapter covers both the Gemini Developer API and Vertex AI modes."""

from __future__ import annotations

from apps.agent.llm.google import build_google_llm
from apps.agent.llm.settings import Settings


def test_gemini_developer_api_mode() -> None:
    settings = Settings(
        google_vertexai=False,
        google_api_key="dev-api-key",
    )

    llm = build_google_llm(settings, model="gemini-x", max_tokens=512)

    assert llm.model == "gemini-x"
    assert not llm.vertexai
    assert llm.max_output_tokens == 512


def test_vertex_ai_mode() -> None:
    settings = Settings(
        google_vertexai=True,
        google_project="solvia-project",
        google_location="us-central1",
    )

    llm = build_google_llm(settings, model="gemini-x", max_tokens=1024)

    assert llm.vertexai is True
    assert llm.project == "solvia-project"
    assert llm.location == "us-central1"
    assert llm.max_output_tokens == 1024

"""Settings load only from the environment, with no hardcoded values."""

from __future__ import annotations

import pytest

from apps.agent.llm.settings import Settings


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL_FAST",
        "LLM_MODEL_SMART",
        "GOOGLE_VERTEXAI",
        "GOOGLE_PROJECT",
        "GOOGLE_LOCATION",
        "GOOGLE_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_settings_loads_values_from_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "google")
    monkeypatch.setenv("LLM_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL_FAST", "custom-fast")
    monkeypatch.setenv("LLM_MODEL_SMART", "custom-smart")
    monkeypatch.setenv("GOOGLE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_PROJECT", "my-project")
    monkeypatch.setenv("GOOGLE_LOCATION", "us-central1")

    settings = Settings()

    assert settings.llm_provider == "google"
    assert settings.llm_base_url == "http://example.invalid/v1"
    assert settings.llm_api_key == "test-key"
    assert settings.llm_model_fast == "custom-fast"
    assert settings.llm_model_smart == "custom-smart"
    assert settings.google_vertexai is True
    assert settings.google_project == "my-project"
    assert settings.google_location == "us-central1"


def test_settings_defaults_use_the_gateway_aliases(clean_env: None) -> None:
    settings = Settings()

    assert settings.llm_model_fast == "solvia-fast"
    assert settings.llm_model_smart == "solvia-smart"
    assert settings.llm_provider == "openai_compatible"

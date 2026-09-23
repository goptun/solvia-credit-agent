"""OpenAI-compatible adapter construction uses only settings-derived values."""

from __future__ import annotations

from pydantic import SecretStr

from apps.agent.llm.openai_compatible import build_openai_compatible_llm
from apps.agent.llm.settings import Settings


def test_client_uses_only_settings_derived_values() -> None:
    settings = Settings(
        llm_base_url="http://gateway.invalid/v1",
        llm_api_key="secret-key",
        llm_timeout_seconds=12.5,
    )

    llm = build_openai_compatible_llm(settings, model="solvia-fast")

    assert llm.openai_api_base == "http://gateway.invalid/v1"
    assert isinstance(llm.openai_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "secret-key"
    assert llm.model_name == "solvia-fast"
    assert llm.request_timeout == 12.5

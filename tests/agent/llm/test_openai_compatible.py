"""OpenAI-compatible adapter construction uses only settings-derived values."""

from __future__ import annotations

from pydantic import SecretStr

from apps.agent.llm.openai_compatible import build_openai_compatible_llm
from apps.agent.llm.settings import Settings


def test_client_uses_only_settings_derived_values() -> None:
    settings = Settings(
        llm_base_url="http://gateway.invalid/v1",
        llm_api_key="secret-key",
    )

    llm = build_openai_compatible_llm(
        settings, model="solvia-fast", max_tokens=256, timeout_seconds=12.5
    )

    assert llm.openai_api_base == "http://gateway.invalid/v1"
    assert isinstance(llm.openai_api_key, SecretStr)
    assert llm.openai_api_key.get_secret_value() == "secret-key"
    assert llm.model_name == "solvia-fast"
    assert llm.request_timeout == 12.5
    assert llm.max_tokens == 256


def test_max_tokens_is_configurable_per_call() -> None:
    settings = Settings(llm_base_url="http://gateway.invalid/v1", llm_api_key="k")

    smart = build_openai_compatible_llm(
        settings, model="solvia-smart", max_tokens=1024, timeout_seconds=30.0
    )

    assert smart.max_tokens == 1024


def test_factory_applies_the_per_tier_timeout() -> None:
    from apps.agent.llm.factory import LLMFactory

    settings = Settings(
        llm_base_url="http://gateway.invalid/v1",
        llm_api_key="k",
        llm_timeout_seconds_fast=7.0,
        llm_timeout_seconds_smart=99.0,
    )
    factory = LLMFactory(settings)

    assert factory.for_alias("fast").request_timeout == 7.0  # type: ignore[attr-defined]
    assert factory.for_alias("smart").request_timeout == 99.0  # type: ignore[attr-defined]


def test_default_timeouts_are_per_tier_and_smart_is_not_shorter_than_fast() -> None:
    settings = Settings()

    assert settings.llm_timeout_seconds_fast == 45.0
    assert settings.llm_timeout_seconds_smart == 60.0
    assert settings.llm_timeout_seconds_smart >= settings.llm_timeout_seconds_fast

"""Application settings loaded from environment variables.

Nothing here is hardcoded: every LLM gateway/provider value comes from
the environment (or `.env` locally), per `design.md` — "LLM gateway
strategy and provider abstraction".
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven configuration for the LLM gateway and provider adapters."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: Literal["openai_compatible", "google", "fake"] = "openai_compatible"
    llm_base_url: str = "http://localhost:8080/v1"
    llm_api_key: str = ""
    llm_model_fast: str = "solvia-fast"
    llm_model_smart: str = "solvia-smart"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3

    google_vertexai: bool = False
    google_project: str | None = None
    google_location: str | None = None
    google_api_key: str | None = None


def get_settings() -> Settings:
    """Build `Settings` from the current environment."""
    return Settings()

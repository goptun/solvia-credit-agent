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
    llm_timeout_seconds_fast: float = 45.0
    llm_timeout_seconds_smart: float = 60.0
    """Per-attempt request timeouts by tier (see `docs/adr/ADR-005-llm-
    latency-and-timeouts.md`). Measured end-to-end `knowledge_agent`
    calls through the gateway (retries and JSON fallback included) had
    p50 ~43-44 s on both tiers, and a single 30 s timeout cut off calls
    that would have completed. These defaults are sized to let one
    attempt finish rather than to bound total latency, and per-attempt
    latency was not isolated in that measurement — a manual eval that
    must not be cut short overrides them (e.g. 120 s), and such an
    override is not the production value."""
    llm_turn_deadline_seconds: float = 45.0
    """Total LLM budget for one conversation turn: retries, the JSON-mode
    fallback and smart -> fast degradation must all fit inside it; on
    expiry the turn gets the friendly unavailable reply. Independent of
    the per-attempt tier timeouts above."""
    llm_max_retries: int = 3
    llm_max_tokens_fast: int = 256
    llm_max_tokens_smart: int = 1024
    """`smart` defaults well above `fast` because the gateway's smart-tier
    model spends `reasoning_tokens` out of this same budget before
    emitting visible content — a low value here reproduces the
    empty-completion finding in `docs/infra-assessment.md`."""

    google_vertexai: bool = False
    google_project: str | None = None
    google_location: str | None = None
    google_api_key: str | None = None


def get_settings() -> Settings:
    """Build `Settings` from the current environment."""
    return Settings()

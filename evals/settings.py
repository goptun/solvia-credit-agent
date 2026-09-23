"""Environment-driven configuration for the evaluation harness."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvalsSettings(BaseSettings):
    """Budget, pacing and contamination thresholds for live runs."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    evals_max_gateway_calls: int = 300
    """Hard budget of raw provider calls per live run, retries and fallbacks
    included (design.md, Decision 7)."""
    evals_pacing_seconds: float = 6.0
    """Minimum interval between the first calls of consecutive live items."""
    evals_max_error_share: float = 0.05
    """Share of calls answered `429`/`503`/timeout above which a run is contaminated."""
    evals_max_fallback_share: float = 0.10
    """Share of calls resolved outside the alias's primary model set above which a
    run is contaminated."""
    evals_seed: int = 42
    """Seed for sampling, review samples and bootstrap intervals."""


def get_evals_settings() -> EvalsSettings:
    """Build `EvalsSettings` from the current environment."""
    return EvalsSettings()

"""API-level settings: where the checkpointer and customer data live."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://solvia:solvia@localhost:5432/solvia"


def get_api_settings() -> ApiSettings:
    return ApiSettings()

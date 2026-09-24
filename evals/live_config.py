"""The approved model sets per gateway alias (`evals/live_config.yaml`)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from evals.core.contamination import ModelSets

LIVE_CONFIG_PATH = Path(__file__).parent / "live_config.yaml"


class AliasModels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    expected: list[str] = []
    primary: list[str] = []


class LiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    aliases: dict[str, AliasModels]


def load_model_sets(path: Path | None = None) -> dict[str, ModelSets]:
    raw = yaml.safe_load((path or LIVE_CONFIG_PATH).read_text(encoding="utf-8"))
    config = LiveConfig.model_validate(raw)
    return {
        alias: ModelSets(frozenset(models.expected), frozenset(models.primary))
        for alias, models in config.aliases.items()
    }

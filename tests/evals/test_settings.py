"""EvalsSettings: defaults and environment overrides."""

from __future__ import annotations

import pytest

from evals.settings import EvalsSettings, get_evals_settings


def test_defaults_match_the_design(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "EVALS_MAX_GATEWAY_CALLS",
        "EVALS_PACING_SECONDS",
        "EVALS_MAX_ERROR_SHARE",
        "EVALS_MAX_FALLBACK_SHARE",
        "EVALS_SEED",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = EvalsSettings(_env_file=None)  # type: ignore[call-arg]

    assert settings.evals_max_gateway_calls == 300
    assert settings.evals_pacing_seconds == 6.0
    assert settings.evals_max_error_share == 0.05
    assert settings.evals_max_fallback_share == 0.10
    assert settings.evals_seed == 42


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALS_MAX_GATEWAY_CALLS", "100")
    monkeypatch.setenv("EVALS_PACING_SECONDS", "1.5")
    monkeypatch.setenv("EVALS_SEED", "7")

    settings = get_evals_settings()

    assert settings.evals_max_gateway_calls == 100
    assert settings.evals_pacing_seconds == 1.5
    assert settings.evals_seed == 7

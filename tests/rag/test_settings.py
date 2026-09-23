"""RAG settings load only from the environment, with no hardcoded values."""

from __future__ import annotations

import pytest

from rag.settings import RagSettings

_ENV_KEYS = [
    "RAG_EMBEDDING_MODEL",
    "RAG_CHUNK_MAX_CHARS",
    "RAG_K_VECTOR",
    "RAG_K_FTS",
    "RAG_RRF_K",
    "RAG_TOP_K",
    "RAG_MIN_RELEVANCE_SCORE",
    "RAG_RERANKING_ENABLED",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_defaults_use_the_small_multilingual_model_and_reranking_disabled(
    clean_env: None,
) -> None:
    settings = RagSettings()

    assert settings.rag_embedding_model == "intfloat/multilingual-e5-small"
    assert settings.rag_reranking_enabled is False
    assert "bcb.gov.br" in settings.rag_official_domains
    assert "planalto.gov.br" in settings.rag_official_domains


def test_settings_loads_values_from_environment(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom/model")
    monkeypatch.setenv("RAG_CHUNK_MAX_CHARS", "2000")
    monkeypatch.setenv("RAG_K_VECTOR", "10")
    monkeypatch.setenv("RAG_K_FTS", "15")
    monkeypatch.setenv("RAG_RRF_K", "30")
    monkeypatch.setenv("RAG_TOP_K", "3")
    monkeypatch.setenv("RAG_MIN_RELEVANCE_SCORE", "0.6")
    monkeypatch.setenv("RAG_RERANKING_ENABLED", "true")

    settings = RagSettings()

    assert settings.rag_embedding_model == "custom/model"
    assert settings.rag_chunk_max_chars == 2000
    assert settings.rag_k_vector == 10
    assert settings.rag_k_fts == 15
    assert settings.rag_rrf_k == 30
    assert settings.rag_top_k == 3
    assert settings.rag_min_relevance_score == 0.6
    assert settings.rag_reranking_enabled is True

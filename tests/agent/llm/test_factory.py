"""LLMFactory resolves the configured fast/smart tier mapping table."""

from __future__ import annotations

import pytest

from apps.agent.llm.factory import NODE_TIER_MAP, LLMFactory
from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.settings import Settings


@pytest.fixture
def factory() -> LLMFactory:
    settings = Settings(llm_provider="fake")
    return LLMFactory(settings)


@pytest.mark.parametrize(
    ("node_name", "expected_tier"),
    [
        ("router", "fast"),
        ("financial_analyst", "smart"),
        ("offer_simulator", "smart"),
        ("responder", "smart"),
        ("compliance_guard", "fast"),
    ],
)
def test_tier_mapping_table_covers_every_mvp_node(node_name: str, expected_tier: str) -> None:
    assert NODE_TIER_MAP[node_name] == expected_tier


def test_for_node_builds_an_llm_for_every_mapped_node(factory: LLMFactory) -> None:
    for node_name in NODE_TIER_MAP:
        llm = factory.for_node(node_name)
        assert isinstance(llm, FakeLLM)


def test_consent_check_has_no_configured_tier(factory: LLMFactory) -> None:
    assert "consent_check" not in NODE_TIER_MAP
    with pytest.raises(KeyError):
        factory.for_node("consent_check")


def test_smart_tier_nodes_fall_back_to_fast(factory: LLMFactory) -> None:
    assert factory.fallback_for_node("financial_analyst") is not None
    assert factory.fallback_for_node("offer_simulator") is not None
    assert factory.fallback_for_node("responder") is not None


def test_fast_tier_nodes_have_no_fallback(factory: LLMFactory) -> None:
    assert factory.fallback_for_node("router") is None
    assert factory.fallback_for_node("compliance_guard") is None


def test_for_alias_passes_the_per_tier_max_tokens_to_the_built_client() -> None:
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://gateway.invalid/v1",
        llm_api_key="k",
        llm_max_tokens_fast=256,
        llm_max_tokens_smart=1024,
    )
    factory = LLMFactory(settings)

    fast_llm = factory.for_alias("fast")
    smart_llm = factory.for_alias("smart")

    assert fast_llm.max_tokens == 256  # type: ignore[attr-defined]
    assert smart_llm.max_tokens == 1024  # type: ignore[attr-defined]


def test_tier_reassignment_requires_no_node_code_changes() -> None:
    """Simulates moving a node to a different tier purely via the mapping."""
    original = NODE_TIER_MAP["router"]
    try:
        NODE_TIER_MAP["router"] = "smart"
        settings = Settings(llm_provider="fake")
        factory = LLMFactory(settings)
        assert factory.fallback_for_node("router") is not None
    finally:
        NODE_TIER_MAP["router"] = original

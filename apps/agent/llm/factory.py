"""Resolves, per agent node, a tier -> alias -> provider adapter LLM instance.

The per-node tier mapping is a plain data table, not conditional logic
scattered across nodes, so a node's tier can change without touching the
node's source code (see `design.md` — "LLM gateway strategy and provider
abstraction").
"""

from __future__ import annotations

from typing import Literal, cast

from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.google import build_google_llm
from apps.agent.llm.openai_compatible import build_openai_compatible_llm
from apps.agent.llm.port import LLMPort
from apps.agent.llm.settings import Settings
from apps.agent.llm.traced import TracedLLM

Tier = Literal["fast", "smart"]

NODE_TIER_MAP: dict[str, Tier] = {
    "router": "fast",
    "financial_analyst": "smart",
    "offer_simulator": "smart",
    "responder": "smart",
    "compliance_guard": "fast",
    "knowledge_agent": "smart",
}
"""Per-node model tier. `consent_check` makes no LLM call at all (see
design.md — "Synthetic consent flow") and is intentionally absent here."""


class LLMFactory:
    """Builds `LLMPort` instances for agent nodes, from settings alone.

    `enable_tracing=True` (used by the real app, never by tests) wraps
    every `for_node`/`fallback_for_node` result in `TracedLLM`, so each
    call opens an LLM span on the active turn trace (see
    `apps.agent.observability.tracing`).
    """

    def __init__(self, settings: Settings, enable_tracing: bool = False) -> None:
        self._settings = settings
        self._enable_tracing = enable_tracing

    def _build(self, model: str, max_tokens: int, timeout_seconds: float) -> LLMPort:
        # Concrete LangChain chat models expose a wider surface than
        # `LLMPort` (different parameter names on `ainvoke`/`bind_tools`);
        # this cast is the one place that narrows them down to the
        # provider-agnostic interface nodes are allowed to depend on.
        if self._settings.llm_provider == "fake":
            return FakeLLM()
        if self._settings.llm_provider == "google":
            return cast(LLMPort, build_google_llm(self._settings, model, max_tokens))
        return cast(
            LLMPort,
            build_openai_compatible_llm(self._settings, model, max_tokens, timeout_seconds),
        )

    def for_alias(self, tier: Tier) -> LLMPort:
        """Build the `LLMPort` for a tier directly (`fast` or `smart`)."""
        if tier == "fast":
            return self._build(
                self._settings.llm_model_fast,
                self._settings.llm_max_tokens_fast,
                self._settings.llm_timeout_seconds_fast,
            )
        return self._build(
            self._settings.llm_model_smart,
            self._settings.llm_max_tokens_smart,
            self._settings.llm_timeout_seconds_smart,
        )

    def for_node(self, node_name: str) -> LLMPort:
        """Resolve the LLM for `node_name` per `NODE_TIER_MAP`.

        Raises `KeyError` for a node with no configured tier (e.g.
        `consent_check`, which never calls an LLM) — callers should not
        call this for such nodes.
        """
        llm = self.for_alias(NODE_TIER_MAP[node_name])
        if self._enable_tracing:
            return TracedLLM(llm, call_name=node_name)
        return llm

    def fallback_for_node(self, node_name: str) -> LLMPort | None:
        """The degraded alias for `node_name`, if any.

        Only `smart`-tier nodes degrade, to `fast`; `fast`-tier nodes
        have no further fallback (see `design.md` — degradation).
        """
        if NODE_TIER_MAP[node_name] != "smart":
            return None
        llm = self.for_alias("fast")
        if self._enable_tracing:
            return TracedLLM(llm, call_name=f"{node_name}_fallback")
        return llm

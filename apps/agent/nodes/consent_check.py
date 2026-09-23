"""consent_check node: reads synthetic consent, drives the request-and-
confirm flow — no LLM call anywhere in this node.

See `design.md` — "Synthetic consent flow" and
`specs/conversation-graph/spec.md` — "Deterministic consent confirmation
detection".
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import cast

from apps.agent.nodes.messages import last_human_text
from apps.agent.nodes.text_utils import normalize
from apps.agent.repositories.customers import CustomerRepository
from apps.agent.state import ConsentStatusValue, ConversationState
from apps.agent.synthetic_data.models import ConsentStatus

ConsentCheckNode = Callable[[ConversationState], Awaitable[ConversationState]]

_AUTHORIZE_PATTERN = re.compile(r"\bautorizo\b|\bconfirmo\s+a\s+autorizacao\b|\bpode\s+acessar\b")
_REFUSE_PATTERN = re.compile(r"\bnao\s+autorizo\b|\bnao\s+quero\b|\brecuso\b")


def make_consent_check_node(customer_repository: CustomerRepository) -> ConsentCheckNode:
    async def consent_check_node(state: ConversationState) -> ConversationState:
        if state.get("active_flow") == "consent_confirmation":
            return _handle_confirmation_reply(state)
        return _handle_initial_gate(state, customer_repository)

    return consent_check_node


def _handle_confirmation_reply(state: ConversationState) -> ConversationState:
    text = normalize(last_human_text(state))

    # Refusal is checked first: "não autorizo" would otherwise also match
    # a naive "autorizo" pattern.
    if _REFUSE_PATTERN.search(text):
        return ConversationState(
            active_flow="none",
            pending_intent=None,
            consent_prompt=None,
            consent_refused=True,
        )

    if _AUTHORIZE_PATTERN.search(text):
        return ConversationState(
            consent_status="just_granted",
            active_flow="none",
            intent=state.get("pending_intent"),
            consent_prompt=None,
            consent_refused=False,
        )

    # Ambiguous reply: stay in the flow and let responder re-ask.
    return ConversationState(consent_prompt="reask")


def _handle_initial_gate(
    state: ConversationState, customer_repository: CustomerRepository
) -> ConversationState:
    intent = state.get("intent")
    if intent not in ("loan_simulation", "profile_analysis"):
        # Should not normally be routed here for other intents, but stay
        # safe rather than raise.
        return ConversationState()

    customer = customer_repository.get(state["customer_id"])
    consent_status: ConsentStatus = customer.consent.status if customer else ConsentStatus.MISSING

    if consent_status == ConsentStatus.VALID:
        return ConversationState(consent_status="valid")

    return ConversationState(
        consent_status=cast(ConsentStatusValue, consent_status.value),
        pending_intent=intent,
        active_flow="consent_confirmation",
        consent_prompt="ask",
        consent_refused=False,
    )

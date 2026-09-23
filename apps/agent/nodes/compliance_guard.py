"""compliance_guard node: deterministic PII masking + disclaimer, and
approval-promise detection backed by both an LLM check and a
deterministic keyword/regex safety net.

See `design.md` — "Deterministic compliance enforcement" and
`specs/conversation-graph/spec.md` — "Compliance guardrails".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import structlog
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.structured import ainvoke_structured
from apps.agent.nodes.compliance import (
    BLOCKED_PROMISE_REPLY,
    INFORMATIONAL_DISCLAIMER,
    inject_disclaimer,
    keyword_flags_approval_promise,
    keyword_flags_unhedged_approval_mention,
    mask_pii,
)
from apps.agent.state import ConversationState

logger = structlog.get_logger(__name__)

APPROVAL_CHECK_DEGRADED_FLAG = "approval_check_degraded"

ComplianceGuardNode = Callable[[ConversationState], Awaitable[ConversationState]]

_DISCLAIMER_INTENTS = {"loan_simulation", "profile_analysis"}
_INFORMATIONAL_DISCLAIMER_INTENTS = {"regulatory_question"}


class ApprovalPromiseCheck(BaseModel):
    promises_approval: bool


async def _llm_flags_approval_promise(llm_factory: LLMFactory, text: str) -> bool | None:
    """The LLM's verdict, or `None` when the check could not produce a
    valid result (structured-output failure, timeout, or the turn
    deadline) — the caller must fail closed on `None`, never treat it
    as "no promise"."""
    llm = llm_factory.for_node("compliance_guard")
    prompt = [
        HumanMessage(
            content=(
                "Este texto promete ou garante a aprovação de um crédito de forma "
                f"que não deveria (sem ressalvas)? Responda apenas com base no texto.\n\n"
                f"Texto: {text}"
            )
        )
    ]
    try:
        check = await ainvoke_structured(llm, prompt, ApprovalPromiseCheck, max_retries=2)
    except Exception as exc:  # noqa: BLE001 - every failure degrades to the strict screen
        # Only the exception type is logged — never the reply text.
        logger.warning(
            "approval_promise_check_unavailable",
            exception_type=type(exc).__name__,
            fallback="strict_keyword_screen",
        )
        return None
    return check.promises_approval


def make_compliance_guard_node(llm_factory: LLMFactory) -> ComplianceGuardNode:
    async def compliance_guard_node(state: ConversationState) -> ConversationState:
        text = state.get("draft_reply") or ""
        masked = mask_pii(text)

        compliance_flags: list[str] = []
        blocked = keyword_flags_approval_promise(masked)
        if not blocked:
            llm_verdict = await _llm_flags_approval_promise(llm_factory, masked)
            if llm_verdict is None:
                # Fail closed: without a valid LLM verdict, decide with
                # the stricter deterministic screen and say so.
                compliance_flags.append(APPROVAL_CHECK_DEGRADED_FLAG)
                blocked = keyword_flags_unhedged_approval_mention(masked)
            else:
                blocked = llm_verdict
        if blocked:
            masked = BLOCKED_PROMISE_REPLY
            compliance_flags.append("approval_promise_blocked")

        intent = state.get("intent")
        needs_disclaimer = intent in _DISCLAIMER_INTENTS
        needs_informational_disclaimer = intent in _INFORMATIONAL_DISCLAIMER_INTENTS
        final_reply = inject_disclaimer(masked, needs_disclaimer)
        final_reply = inject_disclaimer(
            final_reply, needs_informational_disclaimer, INFORMATIONAL_DISCLAIMER
        )

        return ConversationState(draft_reply=final_reply, compliance_flags=compliance_flags)

    return compliance_guard_node

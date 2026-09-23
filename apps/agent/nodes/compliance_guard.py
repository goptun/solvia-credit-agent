"""compliance_guard node: deterministic PII masking + disclaimer, and
approval-promise detection backed by both an LLM check and a
deterministic keyword/regex safety net.

See `design.md` — "Deterministic compliance enforcement" and
`specs/conversation-graph/spec.md` — "Compliance guardrails".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.resilience import call_with_retries
from apps.agent.nodes.compliance import (
    BLOCKED_PROMISE_REPLY,
    inject_disclaimer,
    keyword_flags_approval_promise,
    mask_pii,
)
from apps.agent.state import ConversationState

ComplianceGuardNode = Callable[[ConversationState], Awaitable[ConversationState]]

_DISCLAIMER_INTENTS = {"loan_simulation", "profile_analysis"}


class ApprovalPromiseCheck(BaseModel):
    promises_approval: bool


async def _llm_flags_approval_promise(llm_factory: LLMFactory, text: str) -> bool:
    llm = llm_factory.for_node("compliance_guard")
    structured = llm.with_structured_output(ApprovalPromiseCheck)
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
        check: ApprovalPromiseCheck = await call_with_retries(structured, prompt, max_retries=2)
        return check.promises_approval
    except Exception:
        # The keyword check already ran and is the safety net; an LLM
        # hiccup here should not block the turn.
        return False


def make_compliance_guard_node(llm_factory: LLMFactory) -> ComplianceGuardNode:
    async def compliance_guard_node(state: ConversationState) -> ConversationState:
        text = state.get("draft_reply") or ""
        masked = mask_pii(text)

        keyword_flag = keyword_flags_approval_promise(masked)
        llm_flag = False
        if not keyword_flag:
            llm_flag = await _llm_flags_approval_promise(llm_factory, masked)

        compliance_flags: list[str] = []
        if keyword_flag or llm_flag:
            masked = BLOCKED_PROMISE_REPLY
            compliance_flags.append("approval_promise_blocked")

        needs_disclaimer = state.get("intent") in _DISCLAIMER_INTENTS
        final_reply = inject_disclaimer(masked, needs_disclaimer)

        return ConversationState(draft_reply=final_reply, compliance_flags=compliance_flags)

    return compliance_guard_node

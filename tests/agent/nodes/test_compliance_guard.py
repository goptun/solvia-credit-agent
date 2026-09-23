"""compliance_guard: deterministic masking/disclaimer, and dual-mechanism
approval-promise detection."""

from __future__ import annotations

from apps.agent.llm.fake import FakeLLM, FakeStructuredLLM
from apps.agent.nodes.compliance_guard import ApprovalPromiseCheck, make_compliance_guard_node
from apps.agent.state import ConversationState, initial_state
from tests.agent.nodes.fakes import ScriptedLLMFactory


def _reply(state: ConversationState) -> str:
    draft_reply = state.get("draft_reply")
    assert draft_reply is not None
    return draft_reply


def _factory_with_llm_verdict(
    promises_approval: bool,
) -> tuple[ScriptedLLMFactory, FakeStructuredLLM]:
    fast_llm = FakeLLM(responses=[ApprovalPromiseCheck(promises_approval=promises_approval)])
    # `with_structured_output` is memoized on `FakeLLM`, so this is the
    # same object `compliance_guard` will call internally — its `.calls`
    # log is a reliable way to assert whether the LLM check ran.
    structured = fast_llm.with_structured_output(ApprovalPromiseCheck)
    return ScriptedLLMFactory(fast=fast_llm), structured


async def test_pii_is_masked_deterministically() -> None:
    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Seu CPF 123.456.789-00 foi verificado com sucesso."

    updates = await node(state)
    reply = _reply(updates)

    assert "123.456.789-00" not in reply
    assert "[DADO PROTEGIDO]" in reply


async def test_disclaimer_is_added_for_simulation_and_analysis_intents() -> None:
    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Aqui está sua simulação."
    state["intent"] = "loan_simulation"

    updates = await node(state)

    assert "demonstração" in _reply(updates)


async def test_disclaimer_is_not_added_for_other_intents() -> None:
    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Aqui está a resposta."
    state["intent"] = "product_question"

    updates = await node(state)

    assert "demonstração" not in _reply(updates)


async def test_keyword_check_blocks_and_short_circuits_the_llm_call() -> None:
    factory, structured = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Sua aprovação está garantida!"

    updates = await node(state)

    assert _reply(updates).startswith("Não posso garantir")
    assert "approval_promise_blocked" in updates["compliance_flags"]
    assert structured.calls == []  # the keyword check alone was enough to block


async def test_llm_check_blocks_a_promise_the_keyword_check_misses() -> None:
    factory, structured = _factory_with_llm_verdict(True)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Você provavelmente vai conseguir esse crédito sem problemas."

    updates = await node(state)

    assert "approval_promise_blocked" in updates["compliance_flags"]
    assert structured.calls  # the LLM check did run this time


async def test_no_promise_language_passes_through_unblocked() -> None:
    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Aqui está a simulação solicitada."

    updates = await node(state)

    assert updates["compliance_flags"] == []
    assert "Não posso garantir" not in _reply(updates)

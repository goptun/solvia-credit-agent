"""compliance_guard: deterministic masking/disclaimer, and dual-mechanism
approval-promise detection."""

from __future__ import annotations

import pytest
import structlog
from langchain_core.messages import AIMessage

from apps.agent.llm.fake import FakeLLM, FakeStructuredLLM
from apps.agent.nodes.compliance import keyword_flags_unhedged_approval_mention
from apps.agent.nodes.compliance_guard import (
    APPROVAL_CHECK_DEGRADED_FLAG,
    ApprovalPromiseCheck,
    make_compliance_guard_node,
)
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


async def test_informational_disclaimer_is_added_for_regulatory_questions() -> None:
    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Aqui está a resposta sobre a regulação."
    state["intent"] = "regulatory_question"

    updates = await node(state)

    assert "não constitui aconselhamento jurídico" in _reply(updates)


async def test_informational_disclaimer_is_absent_for_other_intents() -> None:
    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Aqui está sua simulação."
    state["intent"] = "loan_simulation"

    updates = await node(state)

    assert "aconselhamento jurídico" not in _reply(updates)


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


def _factory_where_the_llm_check_cannot_produce_a_result() -> ScriptedLLMFactory:
    return ScriptedLLMFactory(
        fast=FakeLLM(responses=[None, AIMessage(content="não consigo responder em JSON")])
    )


async def test_no_tool_call_uses_the_json_fallback_and_the_verdict_is_honored() -> None:
    verdict = ApprovalPromiseCheck(promises_approval=True)
    fast_llm = FakeLLM(responses=[None, AIMessage(content=verdict.model_dump_json())])
    node = make_compliance_guard_node(ScriptedLLMFactory(fast=fast_llm))
    state = initial_state("cust-1")
    state["draft_reply"] = "Você provavelmente vai conseguir esse crédito sem problemas."

    updates = await node(state)

    assert "approval_promise_blocked" in updates["compliance_flags"]
    assert APPROVAL_CHECK_DEGRADED_FLAG not in updates["compliance_flags"]


async def test_check_that_cannot_produce_a_result_fails_closed_on_an_unhedged_approval() -> None:
    node = make_compliance_guard_node(_factory_where_the_llm_check_cannot_produce_a_result())
    state = initial_state("cust-1")
    state["draft_reply"] = "Boas notícias: você será aprovado rapidamente."

    with structlog.testing.capture_logs() as logs:
        updates = await node(state)

    assert _reply(updates).startswith("Não posso garantir")
    assert "approval_promise_blocked" in updates["compliance_flags"]
    assert APPROVAL_CHECK_DEGRADED_FLAG in updates["compliance_flags"]
    (warning,) = [entry for entry in logs if entry["log_level"] == "warning"]
    assert warning["event"] == "approval_promise_check_unavailable"
    assert "aprovado" not in str(warning)  # never the reply text


async def test_check_that_cannot_produce_a_result_lets_a_hedged_reply_through_but_flags_it() -> (
    None
):
    node = make_compliance_guard_node(_factory_where_the_llm_check_cannot_produce_a_result())
    state = initial_state("cust-1")
    state["draft_reply"] = "A aprovação depende de análise de crédito."

    with structlog.testing.capture_logs() as logs:
        updates = await node(state)

    assert "Não posso garantir" not in _reply(updates)
    assert updates["compliance_flags"] == [APPROVAL_CHECK_DEGRADED_FLAG]
    assert any(entry["event"] == "approval_promise_check_unavailable" for entry in logs)


async def test_check_that_cannot_produce_a_result_never_blocks_a_reply_without_approval() -> None:
    node = make_compliance_guard_node(_factory_where_the_llm_check_cannot_produce_a_result())
    state = initial_state("cust-1")
    state["draft_reply"] = "Aqui está a simulação solicitada."

    updates = await node(state)

    assert _reply(updates) == "Aqui está a simulação solicitada."
    assert updates["compliance_flags"] == [APPROVAL_CHECK_DEGRADED_FLAG]


async def test_an_expired_turn_deadline_degrades_the_check_instead_of_skipping_it() -> None:
    from apps.agent.llm.deadline import turn_deadline

    factory, _ = _factory_with_llm_verdict(False)
    node = make_compliance_guard_node(factory)
    state = initial_state("cust-1")
    state["draft_reply"] = "Você será aprovado rapidamente."

    with turn_deadline(0.0):
        updates = await node(state)

    assert "approval_promise_blocked" in updates["compliance_flags"]
    assert APPROVAL_CHECK_DEGRADED_FLAG in updates["compliance_flags"]


@pytest.mark.parametrize(
    "promise",
    [
        "Seu crédito está aprovado, veja a simulação.",
        "Após análise, seu empréstimo foi aprovado.",
        "Você será aprovado rapidamente.",
        "Aprovação estimada para hoje: seu crédito está aprovado.",
    ],
)
def test_strict_screen_flags_promises_that_reuse_words_like_analise_and_simulacao(
    promise: str,
) -> None:
    assert keyword_flags_unhedged_approval_mention(promise) is True


@pytest.mark.parametrize(
    "hedged",
    [
        "Sua aprovação está sujeita à análise de crédito.",
        "A proposta sujeita a análise pode ser recusada; aprovação não é automática.",
        "A aprovação depende de análise de crédito.",
        "Não posso garantir a aprovação do seu crédito.",
        "Não garantimos a aprovação.",
        "Não há garantia de aprovação.",
        "A oferta é condicionada à aprovação de crédito.",
    ],
)
def test_strict_screen_lets_explicitly_hedged_sentences_through(hedged: str) -> None:
    assert keyword_flags_unhedged_approval_mention(hedged) is False


def test_strict_screen_ignores_text_that_never_mentions_approval() -> None:
    assert keyword_flags_unhedged_approval_mention("Aqui está a simulação estimada.") is False


async def test_a_degraded_check_blocks_the_promise_examples_that_hedge_words_used_to_pass() -> None:
    node = make_compliance_guard_node(_factory_where_the_llm_check_cannot_produce_a_result())
    for promise in (
        "Seu crédito está aprovado, veja a simulação.",
        "Após análise, seu empréstimo foi aprovado.",
    ):
        state = initial_state("cust-1")
        state["draft_reply"] = promise

        updates = await node(state)

        assert "approval_promise_blocked" in updates["compliance_flags"], promise

"""The live suites' per-item functions and summaries, with the scripted fake LLM."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.compliance_guard import ApprovalPromiseCheck
from apps.agent.nodes.knowledge_agent import Claim, KnowledgeAnswer, RetrieveFn
from apps.agent.nodes.offer_simulator import SlotExtraction
from apps.agent.nodes.router import RouterDecision
from evals.core.grounding import (
    CAUSE_GOLD_NOT_RETRIEVED,
    CAUSE_LLM_REFUSED,
    CAUSE_THRESHOLD,
    GroundingItem,
)
from evals.core.schemas import (
    ApprovalItem,
    RetrievalItem,
    RouterItem,
    SlotItem,
    SlotValues,
)
from evals.suites.live import (
    COMPLIANCE_LLM,
    GROUNDING,
    ROUTER,
    SLOTS,
    GroundingDeps,
    LiveContext,
)
from rag.corpus.manifest import SourceType
from rag.embeddings.fake import FakeEmbeddings
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings
from tests.agent.nodes.fakes import ScriptedLLMFactory


def _context(
    fast: list[Any] | None = None, smart: list[Any] | None = None, deps: GroundingDeps | None = None
) -> LiveContext:
    factory = ScriptedLLMFactory(
        fast=FakeLLM(responses=fast) if fast else FakeLLM(),
        smart=FakeLLM(responses=smart) if smart else FakeLLM(),
    )
    return LiveContext(factory, deps)


# --- router -----------------------------------------------------------------


def _router_item(
    item_id: str,
    expected: str,
    flow: Literal["none", "consent_confirmation", "slot_filling"] = "none",
    category: Literal["clear", "ambiguous", "continuation"] = "clear",
) -> RouterItem:
    return RouterItem(
        id=item_id, message="mensagem", active_flow=flow, category=category, expected=expected
    )


async def test_router_reports_a_correct_and_an_incorrect_item() -> None:
    right = _router_item("RT-1", "complaint")
    wrong = _router_item("RT-2", "complaint")
    ctx = _context(fast=[RouterDecision(intent="complaint"), RouterDecision(intent="out_of_scope")])

    results = [
        (right, await ROUTER.run_item(right, ctx)),
        (wrong, await ROUTER.run_item(wrong, ctx)),
    ]
    summary = ROUTER.summarize([(i, p) for i, p in results if p is not None], [], 42)

    assert [p for _, p in results] == ["complaint", "out_of_scope"]
    assert summary.metrics["router.accuracy"].value == 0.5
    assert summary.details["misclassified"] == [
        {"id": "RT-2", "expected": "complaint", "predicted": "out_of_scope"}
    ]
    assert summary.details["confusion_matrix"]["complaint"]["out_of_scope"] == 1


async def test_a_continuation_is_predicted_when_the_router_does_not_reclassify() -> None:
    item = _router_item("RT-3", "continue", flow="slot_filling", category="continuation")
    ctx = _context(fast=[RouterDecision(intent="loan_simulation", is_new_request=False)])

    assert await ROUTER.run_item(item, ctx) == "continue"


async def test_a_router_failure_makes_the_item_unavailable() -> None:
    assert await ROUTER.run_item(_router_item("RT-4", "complaint"), _context()) is None


# --- slots ------------------------------------------------------------------


def _slot_item(item_id: str, expected: SlotValues, known: SlotValues | None = None) -> SlotItem:
    filled = expected.filled()
    tag: Literal["complete", "partial", "missing"] = (
        "complete" if filled == 3 else "missing" if filled == 0 else "partial"
    )
    return SlotItem(
        id=item_id,
        message="quero cinco mil em 12 meses",
        known=known or SlotValues(),
        expected=expected,
        tags=[tag],
    )


async def test_slots_report_a_correct_and_an_incorrect_extraction() -> None:
    expected = SlotValues(amount="5000", term_months=12)
    right = _slot_item("SL-1", expected)
    wrong = _slot_item("SL-2", expected)
    ctx = _context(
        smart=[
            SlotExtraction(amount=Decimal("5000.00"), term_months=12),
            SlotExtraction(amount=Decimal("500"), term_months=12),
        ]
    )

    results = [(right, await SLOTS.run_item(right, ctx)), (wrong, await SLOTS.run_item(wrong, ctx))]
    summary = SLOTS.summarize([(i, p) for i, p in results if p is not None], [], 42)

    assert results[0][1] == {"amount": "5000", "term_months": "12", "amortization_type": None}
    assert summary.metrics["slots.exact_match.amount"].value == 0.5
    assert summary.metrics["slots.exact_match.term_months"].value == 1.0
    assert summary.metrics["slots.exact_match.all_fields"].value == 0.5
    assert [m["id"] for m in summary.details["mismatches"]] == ["SL-2"]


async def test_known_slots_are_carried_and_an_invented_value_is_counted() -> None:
    item = _slot_item("SL-3", SlotValues(amount="1000"), known=SlotValues(amount="1000"))
    ctx = _context(smart=[SlotExtraction(term_months=24)])

    predicted = await SLOTS.run_item(item, ctx)
    assert predicted is not None
    summary = SLOTS.summarize([(item, predicted)], [], 42)

    assert predicted["amount"] == "1000" and predicted["term_months"] == "24"
    assert summary.metrics["slots.no_invented_value"].value == 0.5
    assert summary.metrics["slots.exact_match.term_months"].value == 0.0


# --- compliance LLM ---------------------------------------------------------


def _approval(item_id: str, label: Literal["promise", "hedge", "neutral"]) -> ApprovalItem:
    return ApprovalItem(id=item_id, text="Seu crédito está garantido.", label=label)


async def test_the_llm_check_reports_precision_and_recall() -> None:
    items = [
        _approval("AP-1", "promise"),
        _approval("AP-2", "promise"),
        _approval("AP-3", "neutral"),
    ]
    ctx = _context(
        fast=[
            ApprovalPromiseCheck(promises_approval=True),
            ApprovalPromiseCheck(promises_approval=False),
            ApprovalPromiseCheck(promises_approval=True),
        ]
    )

    results = [(item, await COMPLIANCE_LLM.run_item(item, ctx)) for item in items]
    summary = COMPLIANCE_LLM.summarize([(i, v) for i, v in results if v is not None], [], 42)

    assert summary.metrics["llm.precision"].value == 0.5
    assert summary.metrics["llm.recall"].value == 0.5
    assert summary.details["missed_promises"] == ["AP-2"]
    assert summary.details["false_positives"] == ["AP-3"]


async def test_an_unavailable_llm_check_yields_no_verdict() -> None:
    assert await COMPLIANCE_LLM.run_item(_approval("AP-4", "promise"), _context()) is None


# --- grounding --------------------------------------------------------------

_GOLD = ("cdc-consolidada", "art. 6º")


def _chunk(article: str, similarity: float, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="cdc-consolidada",
        norm="Lei nº 8.078/1990",
        source_type="regulation",
        article_ref=article,
        hierarchy_path=article,
        source_url="https://www.planalto.gov.br/test.htm",
        version_date=None,
        amendment_note=None,
        content="Texto do artigo.",
        fused_score=1.0,
        vector_similarity=similarity,
        matched_fts=True,
    )


def _deps(chunks: list[RetrievedChunk]) -> GroundingDeps:
    async def retrieve(
        question: str, vector: list[float], source_type: SourceType | None
    ) -> list[RetrievedChunk]:
        return chunks

    retrieve_fn: RetrieveFn = retrieve
    return GroundingDeps(FakeEmbeddings(), retrieve_fn, RagSettings(rag_min_relevance_score=0.5))


def _answerable() -> RetrievalItem:
    return RetrievalItem(
        id="R-1",
        question="Quais são os direitos básicos do consumidor?",
        kind="answerable",
        difficulty="easy",
        document="cdc-consolidada",
        expected_refs=["art. 6º"],
        evidence="São direitos básicos do consumidor",
        style="lexical",
    )


def _unanswerable() -> RetrievalItem:
    return RetrievalItem(
        id="R-2",
        question="Qual a previsão do tempo?",
        kind="unanswerable",
        difficulty="easy",
        distance="far",
    )


def _answer(chunk_id: str = "c1") -> KnowledgeAnswer:
    return KnowledgeAnswer(claims=[Claim(text="Há direitos básicos.", chunk_id=chunk_id)])


async def _grounded(
    item: RetrievalItem, chunks: list[RetrievedChunk], answer: KnowledgeAnswer | None
) -> GroundingItem:
    ctx = _context(smart=[answer] if answer else None, deps=_deps(chunks))
    outcome = await GROUNDING.run_item(item, ctx)
    assert outcome is not None
    return outcome


async def test_a_grounded_answer_cites_the_expected_reference_from_the_retrieved_set() -> None:
    outcome = await _grounded(_answerable(), [_chunk("art. 6º", 0.8)], _answer())

    assert not outcome.refused
    assert [(c.document_id, c.article_ref, c.in_retrieved) for c in outcome.cited] == [
        ("cdc-consolidada", "art. 6º", True)
    ]
    assert outcome.acceptable == (_GOLD,)


async def test_false_refusals_are_decomposed_into_exactly_one_cause_each() -> None:
    empty = KnowledgeAnswer(claims=[])
    by_threshold = await _grounded(_answerable(), [_chunk("art. 6º", 0.3)], None)
    gold_missing = await _grounded(_answerable(), [_chunk("art. 7º", 0.8)], empty)
    llm_refused = await _grounded(_answerable(), [_chunk("art. 6º", 0.8)], empty)
    far = await _grounded(_unanswerable(), [_chunk("art. 7º", 0.2)], None)

    summary = GROUNDING.summarize(
        [
            (_answerable(), by_threshold),
            (_answerable(), gold_missing),
            (_answerable(), llm_refused),
            (_unanswerable(), far),
        ],
        [],
        42,
    )

    assert by_threshold.refused_by_threshold and not gold_missing.refused_by_threshold
    assert summary.details["false_refusal_causes"] == {
        CAUSE_THRESHOLD: 1,
        CAUSE_GOLD_NOT_RETRIEVED: 1,
        CAUSE_LLM_REFUSED: 1,
    }
    assert summary.metrics["grounding.false_refusal_rate"].value == 1.0
    assert summary.metrics["grounding.refusal_accuracy.far"].value == 1.0


async def test_an_unavailable_answer_is_left_out_of_the_metrics() -> None:
    ctx = _context(deps=_deps([_chunk("art. 6º", 0.8)]))

    assert await GROUNDING.run_item(_answerable(), ctx) is None

"""Live suites: the router, slot extraction, the LLM approval-promise check and
end-to-end grounding, each as a per-item function plus a summary.

Items run strictly one at a time through one instrumented factory (see
`evals.live_runner`). An item whose node raised is *unavailable*: it is left out
of the metrics and listed, so an infrastructure failure never masquerades as a
wrong answer (contamination detection reports it separately)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable, Hashable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.messages import HumanMessage

from apps.agent.llm.errors import LLMDeadlineExceeded, StructuredOutputError
from apps.agent.llm.factory import LLMFactory
from apps.agent.nodes.compliance import mask_pii
from apps.agent.nodes.compliance_guard import llm_flags_approval_promise
from apps.agent.nodes.knowledge_agent import RetrieveFn, ground_answer
from apps.agent.nodes.offer_simulator import make_offer_simulator_node
from apps.agent.nodes.router import make_router_node
from apps.agent.state import ConversationState, SimulationSlots
from evals.adapters.retrieval import source_type_for
from evals.core.baseline import Direction
from evals.core.classification import (
    ClassificationCase,
    SlotCase,
    accuracy,
    confusion_matrix,
    no_invented_value,
    precision_recall,
    slot_exact_match,
)
from evals.core.grounding import CitedRef, GroundingItem, cites_expected, grounding_metrics
from evals.core.retrieval import RankedChunk
from evals.core.run import MetricValue, proportion_metric
from evals.core.schemas import (
    CONTINUE,
    INTENTS,
    ApprovalItem,
    ComplianceDataset,
    RetrievalDataset,
    RetrievalItem,
    RouterDataset,
    RouterItem,
    SlotItem,
    SlotsDataset,
    SlotValues,
)
from evals.core.stats import Proportion, proportion
from evals.datasets import LoadedDataset
from evals.review import strata_by_id
from rag.embeddings.port import EmbeddingsPort
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings

ERROR_LABEL = "error"
SLOT_FIELDS = ("amount", "term_months", "amortization_type")


@dataclass(frozen=True)
class GroundingDeps:
    """What `ground_answer` needs besides the model: retrieval and embeddings."""

    embeddings: EmbeddingsPort
    retrieve: RetrieveFn
    rag_settings: RagSettings


@dataclass(frozen=True)
class LiveContext:
    factory: LLMFactory
    grounding: GroundingDeps | None = None


@dataclass(frozen=True)
class Summary:
    metrics: dict[str, MetricValue]
    details: dict[str, Any]


@dataclass(frozen=True)
class ItemReport:
    """What one scored item contributes to a published run: its output and
    per-item scores (1.0 = success)."""

    output: dict[str, Any]
    scores: dict[str, float]


@dataclass(frozen=True)
class LiveSuite[I, O]:
    """One live suite: how to pick its items, run one, and summarize all.

    `run_item` returns `None` when the item was unavailable."""

    name: str
    dataset: str
    items: Callable[[LoadedDataset], Sequence[I]]
    item_id: Callable[[I], str]
    run_item: Callable[[I, LiveContext], Awaitable[O | None]]
    summarize: Callable[[Sequence[tuple[I, O]], Sequence[str], int], Summary]
    describe: Callable[[I, O], ItemReport] | None = None

    def stratum(self, loaded: LoadedDataset) -> Callable[[I], Hashable]:
        strata = strata_by_id(loaded)
        return lambda item: strata[self.item_id(item)]


def _share(name: str, value: Proportion, direction: Direction = "higher") -> dict[str, MetricValue]:
    return {name: proportion_metric(value, direction)}


# --- router -----------------------------------------------------------------


async def _run_router(item: RouterItem, ctx: LiveContext) -> str | None:
    node = make_router_node(ctx.factory)
    state = ConversationState(
        messages=[HumanMessage(content=item.message)], active_flow=item.active_flow
    )
    try:
        update = await node(state)
    except Exception:  # noqa: BLE001 - any node failure marks the item unavailable
        return None
    return update.get("intent") or CONTINUE


def _summarize_router(
    results: Sequence[tuple[RouterItem, str]], unavailable: Sequence[str], seed: int
) -> Summary:
    cases = [
        ClassificationCase(i.expected, predicted, tuple(i.acceptable)) for i, predicted in results
    ]
    metrics = _share("router.accuracy", accuracy(cases))
    by_category: dict[str, list[ClassificationCase]] = defaultdict(list)
    for (item, _), case in zip(results, cases, strict=True):
        by_category[item.category].append(case)
    for category, subset in sorted(by_category.items()):
        metrics.update(_share(f"router.accuracy.category.{category}", accuracy(subset)))
    labels = (*INTENTS, CONTINUE)
    return Summary(
        metrics,
        {
            "confusion_matrix": confusion_matrix(labels, cases),
            "misclassified": [
                {"id": item.id, "expected": item.expected, "predicted": predicted}
                for item, predicted in results
                if predicted != item.expected and predicted not in item.acceptable
            ],
            "unavailable_ids": list(unavailable),
        },
    )


def _describe_router(item: RouterItem, predicted: str) -> ItemReport:
    correct = predicted == item.expected or predicted in item.acceptable
    return ItemReport({"predicted": predicted}, {"correct": float(correct)})


ROUTER = LiveSuite[RouterItem, str](
    name="router",
    dataset="router",
    items=lambda loaded: _dataset(loaded, RouterDataset).items,
    item_id=lambda item: item.id,
    run_item=_run_router,
    summarize=_summarize_router,
    describe=_describe_router,
)


# --- slots ------------------------------------------------------------------


def _amount_text(value: Decimal | str | None) -> str | None:
    if value is None:
        return None
    return format(Decimal(str(value)).normalize(), "f")


def _slot_dict(
    amount: Decimal | str | None, term: int | None, kind: str | None
) -> dict[str, str | None]:
    return {
        "amount": _amount_text(amount),
        "term_months": None if term is None else str(term),
        "amortization_type": kind,
    }


def _known_slots(values: SlotValues) -> SimulationSlots:
    return SimulationSlots(
        amount=None if values.amount is None else Decimal(values.amount),
        term_months=values.term_months,
        amortization_type=values.amortization_type,
    )


async def _run_slots(item: SlotItem, ctx: LiveContext) -> dict[str, str | None] | None:
    node = make_offer_simulator_node(ctx.factory)
    state = ConversationState(
        messages=[HumanMessage(content=item.message)],
        simulation_slots=_known_slots(item.known),
    )
    try:
        update = await node(state)
    except Exception:  # noqa: BLE001 - any node failure marks the item unavailable
        return None
    slots = update.get("simulation_slots") or SimulationSlots()
    return _slot_dict(slots.amount, slots.term_months, slots.amortization_type)


def _summarize_slots(
    results: Sequence[tuple[SlotItem, dict[str, str | None]]], unavailable: Sequence[str], seed: int
) -> Summary:
    cases = [
        SlotCase(
            _slot_dict(i.expected.amount, i.expected.term_months, i.expected.amortization_type),
            predicted,
        )
        for i, predicted in results
    ]
    metrics: dict[str, MetricValue] = {}
    for name, value in slot_exact_match(cases, SLOT_FIELDS).items():
        metrics.update(_share(f"slots.exact_match.{name}", value))
    metrics.update(
        _share(
            "slots.exact_match.all_fields",
            proportion(sum(c.expected == c.predicted for c in cases), len(cases)),
        )
    )
    metrics.update(_share("slots.no_invented_value", no_invented_value(cases, SLOT_FIELDS)))
    return Summary(
        metrics,
        {
            "mismatches": [
                {"id": item.id, "expected": case.expected, "predicted": case.predicted}
                for (item, _), case in zip(results, cases, strict=True)
                if case.expected != case.predicted
            ],
            "unavailable_ids": list(unavailable),
        },
    )


def _describe_slots(item: SlotItem, predicted: dict[str, str | None]) -> ItemReport:
    expected = _slot_dict(
        item.expected.amount, item.expected.term_months, item.expected.amortization_type
    )
    return ItemReport({"predicted": predicted}, {"exact_match": float(expected == predicted)})


SLOTS = LiveSuite[SlotItem, dict[str, str | None]](
    name="slots",
    dataset="slots",
    items=lambda loaded: _dataset(loaded, SlotsDataset).items,
    item_id=lambda item: item.id,
    run_item=_run_slots,
    summarize=_summarize_slots,
    describe=_describe_slots,
)


# --- compliance (LLM approval-promise check) ---------------------------------


async def _run_compliance(item: ApprovalItem, ctx: LiveContext) -> bool | None:
    return await llm_flags_approval_promise(ctx.factory, mask_pii(item.text))


def _summarize_compliance(
    results: Sequence[tuple[ApprovalItem, bool]], unavailable: Sequence[str], seed: int
) -> Summary:
    outcome = precision_recall([(i.label == "promise", flagged) for i, flagged in results])
    metrics = {
        **_share("llm.precision", outcome.precision),
        **_share("llm.recall", outcome.recall),
        **_share("llm.false_positive_rate", outcome.false_positive_rate, "lower"),
    }
    return Summary(
        metrics,
        {
            "missed_promises": [
                i.id for i, flagged in results if i.label == "promise" and not flagged
            ],
            "false_positives": [i.id for i, flagged in results if i.label != "promise" and flagged],
            "unavailable_ids": list(unavailable),
        },
    )


def _describe_compliance(item: ApprovalItem, flagged: bool) -> ItemReport:
    return ItemReport(
        {"flags_promise": flagged}, {"correct": float(flagged == (item.label == "promise"))}
    )


COMPLIANCE_LLM = LiveSuite[ApprovalItem, bool](
    name="compliance-llm",
    dataset="compliance",
    items=lambda loaded: _dataset(loaded, ComplianceDataset).approval,
    item_id=lambda item: item.id,
    run_item=_run_compliance,
    summarize=_summarize_compliance,
    describe=_describe_compliance,
)


# --- grounding --------------------------------------------------------------


async def _run_grounding(item: RetrievalItem, ctx: LiveContext) -> GroundingItem | None:
    deps = ctx.grounding
    assert deps is not None, "the grounding suite needs retrieval dependencies"
    captured: list[RetrievedChunk] = []

    async def recording_retrieve(
        question: str, vector: list[float], source_type: Any
    ) -> list[RetrievedChunk]:
        chunks = await deps.retrieve(question, vector, source_type)
        captured[:] = chunks
        return chunks

    llm = ctx.factory.for_node("knowledge_agent")
    fallback = ctx.factory.fallback_for_node("knowledge_agent")
    try:
        answer = await ground_answer(
            item.question,
            source_type_for(item.document),
            llm,
            fallback,
            deps.embeddings,
            recording_retrieve,
            deps.rag_settings,
        )
    except (LLMDeadlineExceeded, StructuredOutputError):
        return None

    best = max((chunk.vector_similarity for chunk in captured), default=0.0)
    by_reference = {(c.norm, c.article_ref, c.source_url): c for c in captured}
    cited = []
    for citation in answer.citations:
        chunk = by_reference.get((citation.norm, citation.article_ref, citation.source_url))
        cited.append(
            CitedRef(
                chunk.document_id if chunk else "unknown",
                citation.article_ref,
                in_retrieved=chunk is not None,
            )
        )
    return GroundingItem(
        question_id=item.id,
        kind=item.kind,
        refused=answer.refused,
        refused_by_threshold=answer.refused
        and (not captured or best < deps.rag_settings.rag_min_relevance_score),
        retrieved=tuple(RankedChunk(c.document_id, c.article_ref) for c in captured),
        cited=tuple(cited),
        distance=item.distance,
        acceptable=tuple(item.acceptable()) if item.kind == "answerable" else (),
    )


def _summarize_grounding(
    results: Sequence[tuple[RetrievalItem, GroundingItem]], unavailable: Sequence[str], seed: int
) -> Summary:
    outcome = grounding_metrics([grounding for _, grounding in results])
    metrics = {
        **_share("grounding.false_refusal_rate", outcome.false_refusal, "lower"),
        **_share("grounding.refusal_accuracy", outcome.refusal_accuracy),
        **_share("grounding.refusal_accuracy.far", outcome.refusal_accuracy_far),
        **_share("grounding.refusal_accuracy.near_miss", outcome.refusal_accuracy_near_miss),
        **_share("grounding.citation_validity", outcome.citation_validity),
        **_share("grounding.expected_ref_hit", outcome.expected_ref_hit),
    }
    return Summary(
        metrics, {"false_refusal_causes": outcome.causes, "unavailable_ids": list(unavailable)}
    )


def _describe_grounding(item: RetrievalItem, grounding: GroundingItem) -> ItemReport:
    if item.kind == "answerable":
        scores = {
            "answered": float(not grounding.refused),
            "cites_expected": float(cites_expected(grounding)),
        }
    else:
        scores = {"refused": float(grounding.refused)}
    return ItemReport(
        {
            "refused": grounding.refused,
            "cited": [f"{c.document_id}:{c.article_ref}" for c in grounding.cited],
        },
        scores,
    )


GROUNDING = LiveSuite[RetrievalItem, GroundingItem](
    name="grounding",
    dataset="retrieval",
    items=lambda loaded: _dataset(loaded, RetrievalDataset).items,
    item_id=lambda item: item.id,
    run_item=_run_grounding,
    summarize=_summarize_grounding,
    describe=_describe_grounding,
)


def _dataset[D](loaded: LoadedDataset, kind: type[D]) -> D:
    assert isinstance(loaded.dataset, kind)
    return loaded.dataset


LIVE_SUITES: dict[str, LiveSuite[Any, Any]] = {
    suite.name: suite for suite in (ROUTER, SLOTS, COMPLIANCE_LLM, GROUNDING)
}

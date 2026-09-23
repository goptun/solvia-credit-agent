"""End-to-end grounding metrics: refusals, citation validity, expected-
reference hits and the cause of every false refusal."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from evals.core.retrieval import RankedChunk
from evals.core.stats import Proportion, proportion

CAUSE_THRESHOLD = "threshold"
CAUSE_GOLD_NOT_RETRIEVED = "gold_not_retrieved"
CAUSE_LLM_REFUSED = "llm_refused_gold_in_context"
CAUSES = (CAUSE_THRESHOLD, CAUSE_GOLD_NOT_RETRIEVED, CAUSE_LLM_REFUSED)


@dataclass(frozen=True)
class CitedRef:
    document_id: str
    article_ref: str | None
    in_retrieved: bool
    """The citation matches a chunk in that same turn's retrieved set."""


@dataclass(frozen=True)
class GroundingItem:
    question_id: str
    kind: str  # "answerable" | "unanswerable"
    refused: bool
    refused_by_threshold: bool
    retrieved: tuple[RankedChunk, ...]
    cited: tuple[CitedRef, ...] = ()
    distance: str | None = None  # unanswerable: "far" | "near_miss"
    document: str | None = None  # answerable: expected document
    expected_refs: tuple[str, ...] | None = None


def _matches(document: str | None, refs: tuple[str, ...] | None, doc: str, ref: str | None) -> bool:
    acceptable = set(refs) if refs else None
    return doc == document and (acceptable is None or ref in acceptable)


def gold_in_context(item: GroundingItem) -> bool:
    return any(
        _matches(item.document, item.expected_refs, chunk.document_id, chunk.article_ref)
        for chunk in item.retrieved
    )


def refusal_cause(item: GroundingItem) -> str | None:
    """Exactly one cause for a refused answerable question, checked in
    order: threshold, gold not retrieved, LLM refused with gold in context."""
    if item.kind != "answerable" or not item.refused:
        return None
    if item.refused_by_threshold:
        return CAUSE_THRESHOLD
    return CAUSE_LLM_REFUSED if gold_in_context(item) else CAUSE_GOLD_NOT_RETRIEVED


def cites_expected(item: GroundingItem) -> bool:
    return any(
        _matches(item.document, item.expected_refs, ref.document_id, ref.article_ref)
        for ref in item.cited
    )


@dataclass(frozen=True)
class GroundingMetrics:
    false_refusal: Proportion
    refusal_accuracy: Proportion
    refusal_accuracy_far: Proportion
    refusal_accuracy_near_miss: Proportion
    citation_validity: Proportion
    """Answers with at least one citation whose every citation is in the retrieved set."""
    expected_ref_hit: Proportion
    """Answered answerable questions citing the expected document and an acceptable ref."""
    causes: dict[str, int]


def grounding_metrics(items: Sequence[GroundingItem]) -> GroundingMetrics:
    answerable = [item for item in items if item.kind == "answerable"]
    unanswerable = [item for item in items if item.kind == "unanswerable"]

    def refused_share(subset: Sequence[GroundingItem]) -> Proportion:
        return proportion(sum(1 for item in subset if item.refused), len(subset))

    answered = [item for item in answerable if not item.refused]
    cited_answers = [item for item in items if not item.refused and item.cited]
    valid = sum(1 for item in cited_answers if all(ref.in_retrieved for ref in item.cited))

    causes = dict.fromkeys(CAUSES, 0)
    for item in answerable:
        cause = refusal_cause(item)
        if cause is not None:
            causes[cause] += 1

    return GroundingMetrics(
        false_refusal=refused_share(answerable),
        refusal_accuracy=refused_share(unanswerable),
        refusal_accuracy_far=refused_share([i for i in unanswerable if i.distance == "far"]),
        refusal_accuracy_near_miss=refused_share(
            [i for i in unanswerable if i.distance == "near_miss"]
        ),
        citation_validity=proportion(valid, len(cited_answers)),
        expected_ref_hit=proportion(sum(1 for i in answered if cites_expected(i)), len(answered)),
        causes=causes,
    )

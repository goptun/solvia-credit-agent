"""Classification, slot, precision/recall, grounding and operational metrics."""

from __future__ import annotations

import pytest

from evals.core.classification import (
    ClassificationCase,
    SlotCase,
    accuracy,
    confusion_matrix,
    exact_match_rate,
    no_invented_value,
    precision_recall,
    slot_exact_match,
)
from evals.core.grounding import (
    CAUSE_GOLD_NOT_RETRIEVED,
    CAUSE_LLM_REFUSED,
    CAUSE_THRESHOLD,
    CitedRef,
    GroundingItem,
    grounding_metrics,
    refusal_cause,
)
from evals.core.operational import CallRecord, model_mix, node_metrics
from evals.core.retrieval import RankedChunk

INTENTS = ["loan_simulation", "complaint", "out_of_scope", "continue"]


def test_accuracy_counts_acceptable_labels_for_ambiguous_items() -> None:
    cases = [
        ClassificationCase("loan_simulation", "loan_simulation"),
        ClassificationCase("complaint", "out_of_scope", acceptable=("out_of_scope",)),
        ClassificationCase("continue", "loan_simulation"),
    ]

    result = accuracy(cases)

    assert (result.successes, result.n) == (2, 3)


def test_confusion_matrix_has_a_row_per_label_and_keeps_unexpected_predictions() -> None:
    cases = [
        ClassificationCase("continue", "continue"),
        ClassificationCase("continue", "loan_simulation"),
        ClassificationCase("complaint", "billing"),
    ]

    matrix = confusion_matrix(INTENTS, cases)

    assert list(matrix) == INTENTS
    assert matrix["continue"]["continue"] == 1
    assert matrix["continue"]["loan_simulation"] == 1
    assert matrix["complaint"]["billing"] == 1
    assert matrix["out_of_scope"]["out_of_scope"] == 0


def test_slot_exact_match_per_field_and_no_invented_value() -> None:
    cases = [
        SlotCase({"amount": "5000", "term_months": None}, {"amount": "5000", "term_months": None}),
        SlotCase({"amount": "5000", "term_months": None}, {"amount": "500", "term_months": "12"}),
    ]
    fields = ["amount", "term_months"]

    exact = slot_exact_match(cases, fields)
    invented = no_invented_value(cases, fields)

    assert (exact["amount"].successes, exact["amount"].n) == (1, 2)
    assert (exact["term_months"].successes, exact["term_months"].n) == (1, 2)
    assert (invented.successes, invented.n) == (1, 2)


def test_precision_recall_and_false_positive_rate() -> None:
    outcomes = [
        (True, True),
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (False, False),
        (False, False),
    ]

    result = precision_recall(outcomes)

    assert (result.tp, result.fp, result.fn, result.tn) == (2, 1, 1, 3)
    assert result.precision.value == pytest.approx(2 / 3)
    assert result.recall.value == pytest.approx(2 / 3)
    assert result.false_positive_rate.value == pytest.approx(1 / 4)


def test_exact_match_rate() -> None:
    result = exact_match_rate([("a", "a"), ("b", "B"), ("c", "c")])

    assert (result.successes, result.n) == (2, 3)


_GOLD = RankedChunk("cdc-consolidada", "art. 54-A")
_OTHER = RankedChunk("cdc-consolidada", "art. 6º")


def _answerable(
    qid: str,
    *,
    refused: bool,
    by_threshold: bool = False,
    retrieved: tuple[RankedChunk, ...] = (_GOLD,),
    cited: tuple[CitedRef, ...] = (),
) -> GroundingItem:
    return GroundingItem(
        question_id=qid,
        kind="answerable",
        refused=refused,
        refused_by_threshold=by_threshold,
        retrieved=retrieved,
        cited=cited,
        document="cdc-consolidada",
        expected_refs=("art. 54-A",),
    )


def _unanswerable(qid: str, distance: str, *, refused: bool) -> GroundingItem:
    return GroundingItem(
        question_id=qid,
        kind="unanswerable",
        refused=refused,
        refused_by_threshold=refused,
        retrieved=(),
        distance=distance,
    )


def test_every_false_refusal_lands_in_exactly_one_cause() -> None:
    threshold = _answerable("t", refused=True, by_threshold=True)
    missing = _answerable("m", refused=True, retrieved=(_OTHER,))
    llm = _answerable("l", refused=True)
    answered = _answerable("a", refused=False)

    assert refusal_cause(threshold) == CAUSE_THRESHOLD
    assert refusal_cause(missing) == CAUSE_GOLD_NOT_RETRIEVED
    assert refusal_cause(llm) == CAUSE_LLM_REFUSED
    assert refusal_cause(answered) is None
    assert refusal_cause(_unanswerable("u", "far", refused=True)) is None


def test_threshold_cause_wins_even_when_gold_was_in_context() -> None:
    item = _answerable("t", refused=True, by_threshold=True, retrieved=(_GOLD,))

    assert refusal_cause(item) == CAUSE_THRESHOLD


def test_grounding_metrics() -> None:
    good = CitedRef("cdc-consolidada", "art. 54-A", in_retrieved=True)
    ghost = CitedRef("cdc-consolidada", "art. 99", in_retrieved=False)
    items = [
        _answerable("a1", refused=False, cited=(good,)),
        _answerable("a2", refused=False, cited=(good, ghost)),
        _answerable("a3", refused=True, by_threshold=True),
        _answerable("a4", refused=True, retrieved=(_OTHER,)),
        _unanswerable("u1", "far", refused=True),
        _unanswerable("u2", "near_miss", refused=True),
        _unanswerable("u3", "near_miss", refused=False),
    ]

    metrics = grounding_metrics(items)

    assert (metrics.false_refusal.successes, metrics.false_refusal.n) == (2, 4)
    assert (metrics.refusal_accuracy.successes, metrics.refusal_accuracy.n) == (2, 3)
    assert metrics.refusal_accuracy_far.value == 1.0
    assert metrics.refusal_accuracy_near_miss.value == 0.5
    assert (metrics.citation_validity.successes, metrics.citation_validity.n) == (1, 2)
    assert (metrics.expected_ref_hit.successes, metrics.expected_ref_hit.n) == (2, 2)
    assert metrics.causes == {
        CAUSE_THRESHOLD: 1,
        CAUSE_GOLD_NOT_RETRIEVED: 1,
        CAUSE_LLM_REFUSED: 0,
    }


def _call(
    op: str,
    node: str,
    kind: str,
    *,
    latency: float = 1.0,
    status: str = "ok",
    structured: bool = True,
    alias: str = "solvia-fast",
    model: str | None = "model-a",
) -> CallRecord:
    return CallRecord(op, node, alias, model, status, kind, latency, structured)


def test_operational_rates_and_latency_are_computed_per_node() -> None:
    records = [
        # op1: native ok
        _call("op1", "router", "tool_call", latency=1.0),
        # op2: native answered without a tool call, then a JSON-mode call
        _call("op2", "router", "no_tool_call", latency=2.0),
        _call("op2", "router", "no_tool_call", latency=3.0),
        # op3: native failed with 429, then a JSON-mode call
        _call("op3", "router", "error", latency=0.5, status="429"),
        _call("op3", "router", "no_tool_call", latency=4.0),
        # a plain (non-structured) node
        _call("op4", "responder", "plain", latency=7.0, structured=False),
    ]

    metrics = node_metrics(records)

    router = metrics["router"]
    assert router.calls == 5
    assert router.attempts_max == 2
    assert router.attempts_mean == pytest.approx(5 / 3)
    assert router.structured_operations == 3
    assert (router.no_tool_call_rate.successes, router.no_tool_call_rate.n) == (1, 3)
    assert (router.json_fallback_rate.successes, router.json_fallback_rate.n) == (2, 3)
    assert router.latency_p50 == pytest.approx(2.5)  # ok calls only: 1, 2, 3, 4
    responder = metrics["responder"]
    assert responder.structured_operations == 0
    assert responder.latency_p95 == 7.0


def test_model_mix_counts_every_call_per_alias() -> None:
    records = [
        _call("a", "router", "tool_call", model="m1"),
        _call("b", "router", "tool_call", model="m1"),
        _call("c", "router", "tool_call", model="m2"),
        _call("d", "offer_simulator", "tool_call", alias="solvia-smart", model=None),
    ]

    mix = model_mix(records)

    assert mix == {"solvia-fast": {"m1": 2, "m2": 1}, "solvia-smart": {"unknown": 1}}

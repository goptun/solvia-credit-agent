"""Baseline comparison, call budget and contamination detection."""

from __future__ import annotations

import pytest

from evals.core.baseline import Direction, MetricBaseline, compare, has_regression, render_table
from evals.core.budget import (
    BudgetExceeded,
    CallBudget,
    Estimate,
    check_estimate,
    estimate_calls,
)
from evals.core.contamination import ModelSets, assess_contamination
from evals.core.operational import CallRecord, status_breakdown


def _metric(value: float, tolerance: float, direction: Direction = "higher") -> MetricBaseline:
    return MetricBaseline(value=value, n=75, direction=direction, tolerance=tolerance)


def test_regression_beyond_tolerance_fails() -> None:
    rows = compare({"recall": _metric(0.68, 0.03)}, {"recall": 0.64})

    assert rows[0].status == "regression"
    assert has_regression(rows)
    assert rows[0].diff == pytest.approx(-0.04)


def test_change_within_tolerance_passes() -> None:
    rows = compare({"recall": _metric(0.68, 0.03)}, {"recall": 0.66})

    assert rows[0].status == "ok"
    assert not has_regression(rows)


def test_exactly_at_tolerance_is_not_a_regression() -> None:
    rows = compare({"recall": _metric(0.50, 0.03)}, {"recall": 0.47})

    assert rows[0].status == "ok"


def test_improvement_beyond_tolerance_passes_with_a_note() -> None:
    rows = compare({"recall": _metric(0.68, 0.03)}, {"recall": 0.75})

    assert rows[0].status == "improved"
    assert not has_regression(rows)
    assert "consider updating the baseline" in render_table(rows)


def test_lower_is_better_metrics_regress_upwards() -> None:
    rows = compare({"latency": _metric(10.0, 1.0, "lower")}, {"latency": 12.0})

    assert rows[0].status == "regression"


def test_zero_tolerance_fails_on_any_change_including_an_improvement() -> None:
    same = compare({"precision": _metric(1.0, 0.0)}, {"precision": 1.0})
    better = compare({"fpr": _metric(0.2, 0.0, "lower")}, {"fpr": 0.1})

    assert same[0].status == "ok"
    assert better[0].status == "changed"
    assert has_regression(better)


def test_a_metric_missing_from_the_new_run_fails_and_a_new_one_is_noted() -> None:
    rows = compare({"recall": _metric(0.68, 0.03)}, {"mrr": 0.5})

    assert [row.status for row in rows] == ["missing", "new"]
    assert has_regression(rows)


def test_table_has_the_documented_columns() -> None:
    table = render_table(compare({"recall": _metric(0.68, 0.03)}, {"recall": 0.64}))

    assert table.splitlines()[0] == "| metric | baseline | new | tolerance | diff | status |"
    assert "| recall | 0.6800 | 0.6400 | 0.0300 | -0.0400 | regression |" in table


def test_estimates_match_the_design_for_the_full_suites_and_the_sample() -> None:
    full = estimate_calls(
        router=60, slots=30, compliance_llm=40, grounding_answerable=75, grounding_unanswerable=25
    )
    sample = estimate_calls(
        router=18, slots=9, compliance_llm=12, grounding_answerable=22, grounding_unanswerable=8
    )

    assert full == Estimate(typical=201, pessimistic=268)
    assert sample == Estimate(typical=60, pessimistic=80)


def test_a_run_whose_estimate_exceeds_the_budget_refuses_to_start() -> None:
    check_estimate(Estimate(typical=200, pessimistic=268), max_calls=300)

    with pytest.raises(BudgetExceeded):
        check_estimate(Estimate(typical=200, pessimistic=268), max_calls=250)


def test_the_budget_stops_before_an_item_it_cannot_cover_and_marks_the_run() -> None:
    budget = CallBudget(max_calls=20)
    for _ in range(12):
        budget.record_call()

    assert budget.can_start_item() is True  # 8 left = one worst-case item
    budget.record_call()
    assert budget.can_start_item() is False
    assert budget.stopped_by_budget is True
    assert budget.remaining == 7


def _call(
    status: str = "ok", model: str | None = "primary", alias: str = "solvia-fast"
) -> CallRecord:
    return CallRecord("op", "router", alias, model, status, "tool_call", 1.0, True)


_SETS = {
    "solvia-fast": ModelSets(
        expected=frozenset({"primary", "backup"}), primary=frozenset({"primary"})
    )
}


def test_a_clean_run_is_not_flagged() -> None:
    verdict = assess_contamination(
        [_call() for _ in range(20)], _SETS, max_error_share=0.05, max_fallback_share=0.10
    )

    assert verdict.contaminated is False
    assert verdict.reasons == ()
    assert verdict.model_sets_unset is False


def test_rate_limiting_contaminates_a_run() -> None:
    records = [_call() for _ in range(17)] + [_call("429"), _call("503"), _call("ReadTimeout")]

    verdict = assess_contamination(records, _SETS, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is True
    assert "quota/unavailability" in verdict.reasons[0]


def test_an_unexpected_model_contaminates_and_is_named() -> None:
    records = [_call() for _ in range(19)] + [_call(model="gpt-oss-unexpected")]

    verdict = assess_contamination(records, _SETS, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is True
    assert any("gpt-oss-unexpected" in reason for reason in verdict.reasons)


def test_too_many_fallback_models_contaminates() -> None:
    records = [_call() for _ in range(16)] + [_call(model="backup") for _ in range(4)]

    verdict = assess_contamination(records, _SETS, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is True
    assert any("fallback models" in reason for reason in verdict.reasons)


def test_a_few_fallback_calls_within_the_threshold_are_tolerated() -> None:
    records = [_call() for _ in range(19)] + [_call(model="backup")]

    verdict = assess_contamination(records, _SETS, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is False


def test_incomplete_or_interrupted_runs_are_contaminated() -> None:
    clean = [_call() for _ in range(10)]

    incomplete = assess_contamination(
        clean, _SETS, max_error_share=0.05, max_fallback_share=0.10, incomplete=True
    )
    interrupted = assess_contamination(
        clean, _SETS, max_error_share=0.05, max_fallback_share=0.10, interrupted=True
    )

    assert incomplete.contaminated and "incomplete" in incomplete.reasons[0]
    assert interrupted.contaminated and "interrupted" in interrupted.reasons[0]


def test_unset_model_sets_do_not_contaminate_but_block_baselining() -> None:
    verdict = assess_contamination(
        [_call(model="anything") for _ in range(10)],
        None,
        max_error_share=0.05,
        max_fallback_share=0.10,
    )

    assert verdict.contaminated is False
    assert verdict.model_sets_unset is True


def test_provider_prefixes_do_not_matter_when_matching_model_sets() -> None:
    sets = {
        "solvia-fast": ModelSets(
            expected=frozenset({"gemini/gemini-3.5-flash-lite", "cf/@cf/openai/gpt-oss-120b"}),
            primary=frozenset({"gemini/gemini-3.5-flash-lite"}),
        )
    }
    records = [_call(model="gemini-3.5-flash-lite") for _ in range(19)]
    records.append(_call(model="gpt-oss-120b"))

    verdict = assess_contamination(records, sets, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is False
    stray = assess_contamination(
        [_call(model="gemini-9-unknown")], sets, max_error_share=0.05, max_fallback_share=0.10
    )
    assert stray.contaminated is True


def test_a_transient_error_a_retry_resolved_is_not_held_against_the_run() -> None:
    """The app already retries 503/429/timeout (`apps.agent.llm.resilience`);
    a raw failed attempt followed by a successful retry of the same
    operation must not count against the error share."""
    records = [
        CallRecord("op-1", "router", "solvia-fast", None, "503", "error", 0.1, True),
        CallRecord("op-1", "router", "solvia-fast", "primary", "ok", "tool_call", 1.0, True),
    ] + [_call() for _ in range(18)]

    verdict = assess_contamination(records, _SETS, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is False


def test_an_operation_whose_retries_are_all_exhausted_still_contaminates() -> None:
    records = [
        CallRecord("op-1", "router", "solvia-fast", None, "503", "error", 0.1, True),
        CallRecord("op-1", "router", "solvia-fast", None, "503", "error", 0.1, True),
    ] + [_call() for _ in range(18)]

    verdict = assess_contamination(records, _SETS, max_error_share=0.05, max_fallback_share=0.10)

    assert verdict.contaminated is True
    assert any("even after the app's own retry" in reason for reason in verdict.reasons)


def test_status_breakdown_reports_429_separately_from_503() -> None:
    records = [
        CallRecord("op-1", "router", "solvia-fast", None, "429", "error", 0.1, True),
        CallRecord("op-2", "router", "solvia-fast", None, "503", "error", 0.1, True),
        CallRecord("op-3", "router", "solvia-fast", None, "OpenAITimeoutError", "error", 0.1, True),
        CallRecord("op-4", "router", "solvia-fast", "primary", "ok", "tool_call", 0.1, True),
    ]

    breakdown = status_breakdown(records)

    assert breakdown == {"ok": 1, "429": 1, "503": 1, "timeout": 1, "other": 0}

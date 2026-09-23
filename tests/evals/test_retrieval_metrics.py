"""Retrieval metrics on hand-built ranked results."""

from __future__ import annotations

import pytest

from evals.core.retrieval import (
    THRESHOLDS,
    AnswerableResult,
    RankedChunk,
    UnanswerableResult,
    breakdown,
    hit_rank,
    mrr,
    recall_at_k,
    similarity_quantiles,
    threshold_table,
)


def _result(
    question_id: str,
    ranked: list[tuple[str, str | None]],
    *,
    document: str = "cdc-consolidada",
    refs: tuple[str, ...] | None = ("art. 6º",),
    style: str = "lexical",
    difficulty: str = "easy",
    similarity: float = 0.8,
) -> AnswerableResult:
    return AnswerableResult(
        question_id=question_id,
        style=style,
        document=document,
        difficulty=difficulty,
        acceptable=(tuple((document, ref) for ref in refs) if refs else ((document, None),)),
        ranked=tuple(RankedChunk(doc, ref) for doc, ref in ranked),
        best_similarity=similarity,
    )


def test_hit_at_rank_one_and_two() -> None:
    first = _result("a", [("cdc-consolidada", "art. 6º"), ("lgpd", "art. 1º")])
    second = _result("b", [("lgpd", "art. 6º"), ("cdc-consolidada", "art. 6º")])

    assert hit_rank(first) == 1
    assert hit_rank(second) == 2


def test_wrong_article_or_wrong_document_is_a_miss() -> None:
    wrong_article = _result("a", [("cdc-consolidada", "art. 7º")])
    wrong_document = _result("b", [("lgpd", "art. 6º")])

    assert hit_rank(wrong_article) is None
    assert hit_rank(wrong_document) is None


def test_any_acceptable_reference_counts_for_a_multi_ref_question() -> None:
    result = _result(
        "a", [("cdc-consolidada", "art. 25")], refs=("art. 51", "art. 25"), difficulty="medium"
    )

    assert hit_rank(result) == 1


def test_a_reference_in_another_document_also_counts() -> None:
    """R-020 case: the CET is defined by the CMN resolution *and* by CDC art. 54-B."""
    result = AnswerableResult(
        question_id="a",
        style="colloquial",
        document="cet-disclosure",
        difficulty="hard",
        acceptable=(("cet-disclosure", "art. 2º"), ("cdc-consolidada", "art. 54-B")),
        ranked=(RankedChunk("lgpd", "art. 1º"), RankedChunk("cdc-consolidada", "art. 54-B")),
        best_similarity=0.7,
    )

    assert hit_rank(result) == 2


def test_catalog_question_has_no_article_refs() -> None:
    result = _result("a", [("product-catalog", None)], document="product-catalog", refs=None)

    assert hit_rank(result) == 1


def test_recall_and_mrr_over_a_mixed_set() -> None:
    results = [
        _result("a", [("cdc-consolidada", "art. 6º")]),
        _result("b", [("lgpd", "art. 1º"), ("cdc-consolidada", "art. 6º")]),
        _result("c", [("lgpd", "art. 1º")]),
        _result("d", [("cdc-consolidada", "art. 7º")]),
    ]

    recall = recall_at_k(results)
    reciprocal = mrr(results, seed=1)

    assert (recall.successes, recall.n) == (2, 4)
    assert reciprocal.value == pytest.approx((1.0 + 0.5 + 0.0 + 0.0) / 4)


def test_breakdown_by_stratum() -> None:
    results = [
        _result("a", [("cdc-consolidada", "art. 6º")], style="lexical"),
        _result("b", [("lgpd", "art. 1º")], style="colloquial"),
        _result("c", [("cdc-consolidada", "art. 6º")], style="colloquial"),
    ]

    by_style = breakdown(results, lambda r: r.style, seed=1)

    assert list(by_style) == ["colloquial", "lexical"]
    assert by_style["colloquial"].recall.successes == 1
    assert by_style["colloquial"].recall.n == 2
    assert by_style["lexical"].recall.value == 1.0


def test_similarity_quantiles() -> None:
    quantiles = similarity_quantiles([0.2, 0.4, 0.6, 0.8])

    assert quantiles.minimum == 0.2
    assert quantiles.median == pytest.approx(0.5)
    assert quantiles.maximum == 0.8


def test_threshold_table_is_monotonic_and_uses_strict_less_than() -> None:
    answerable = [_result("a", [], similarity=0.5), _result("b", [], similarity=0.9)]
    unanswerable = [
        UnanswerableResult("u1", "far", 0.4),
        UnanswerableResult("u2", "near_miss", 0.6),
    ]

    rows = threshold_table(answerable, unanswerable)

    assert [row.threshold for row in rows] == list(THRESHOLDS)
    assert THRESHOLDS[0] == 0.30 and THRESHOLDS[-1] == 0.90 and len(THRESHOLDS) == 13
    false_refusals = [row.false_refusal.successes for row in rows]
    correct_refusals = [row.correct_refusal.successes for row in rows]
    assert false_refusals == sorted(false_refusals)
    assert correct_refusals == sorted(correct_refusals)
    at_050 = next(row for row in rows if row.threshold == 0.50)
    assert at_050.false_refusal.successes == 0  # 0.5 is not < 0.5
    assert at_050.correct_refusal.successes == 1
    at_055 = next(row for row in rows if row.threshold == 0.55)
    assert at_055.false_refusal.successes == 1

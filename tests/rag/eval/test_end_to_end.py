"""`rag.eval.end_to_end` aggregation, with a scripted grounding function
(no LLM, no database)."""

from __future__ import annotations

from rag.corpus.manifest import SourceType
from rag.eval.end_to_end import GroundingOutcome, format_report, run_end_to_end
from rag.eval.questions import AnswerableQuestion, EvalQuestionSet, UnanswerableQuestion

_ANSWERED = GroundingOutcome(refused=False, refused_by_threshold=False)
_THRESHOLD_REFUSAL = GroundingOutcome(refused=True, refused_by_threshold=True)
_LLM_REFUSAL = GroundingOutcome(refused=True, refused_by_threshold=False)


def _answerable(
    question: str, style: str = "lexical", doc: str = "cdc-consolidada"
) -> AnswerableQuestion:
    return AnswerableQuestion(
        question=question,
        expected_document_id=doc,
        expected_refs=None,
        evidence="x",
        style=style,  # type: ignore[arg-type]
    )


def _scripted(outcomes: dict[str, GroundingOutcome], seen: list[SourceType | None]):  # type: ignore[no-untyped-def]
    async def ground(question: str, source_type: SourceType | None) -> GroundingOutcome:
        seen.append(source_type)
        return outcomes[question]

    return ground


async def test_reports_false_refusals_and_refusal_accuracy_with_stage_attribution() -> None:
    questions = EvalQuestionSet(
        answerable=[
            _answerable("a-ok"),
            _answerable("a-threshold", style="colloquial"),
            _answerable("a-llm", style="colloquial"),
            _answerable("a-ok-2"),
        ],
        unanswerable=[
            UnanswerableQuestion(question="u-far-threshold", distance="far"),
            UnanswerableQuestion(question="u-near-llm", distance="near_miss"),
            UnanswerableQuestion(question="u-near-answered", distance="near_miss"),
        ],
    )
    outcomes = {
        "a-ok": _ANSWERED,
        "a-threshold": _THRESHOLD_REFUSAL,
        "a-llm": _LLM_REFUSAL,
        "a-ok-2": _ANSWERED,
        "u-far-threshold": _THRESHOLD_REFUSAL,
        "u-near-llm": _LLM_REFUSAL,
        "u-near-answered": _ANSWERED,
    }

    report = await run_end_to_end(questions, _scripted(outcomes, []))

    assert report.false_refusals == 2
    assert report.false_refusal_rate == 0.5
    assert report.false_refusals_by_threshold == 1
    assert report.false_refusals_by_llm == 1
    assert report.false_refusal_rate_by_style == {"lexical": 0.0, "colloquial": 1.0}
    assert report.correct_refusals == 2
    assert report.refusal_accuracy == 2 / 3
    assert report.correct_refusals_by_threshold == 1
    assert report.correct_refusals_by_llm == 1
    assert report.refusal_accuracy_by_distance == {"far": 1.0, "near_miss": 0.5}


async def test_answerable_use_their_source_type_and_unanswerable_search_the_whole_corpus() -> None:
    questions = EvalQuestionSet(
        answerable=[
            _answerable("reg"),
            _answerable("prod", doc="product-catalog"),
        ],
        unanswerable=[UnanswerableQuestion(question="nope", distance="far")],
    )
    seen: list[SourceType | None] = []
    outcomes = {"reg": _ANSWERED, "prod": _ANSWERED, "nope": _THRESHOLD_REFUSAL}

    await run_end_to_end(questions, _scripted(outcomes, seen))

    assert seen == ["regulation", "product_catalog", None]


async def test_format_report_mentions_every_metric() -> None:
    questions = EvalQuestionSet(
        answerable=[_answerable("a")],
        unanswerable=[UnanswerableQuestion(question="u", distance="far")],
    )
    report = await run_end_to_end(questions, _scripted({"a": _ANSWERED, "u": _LLM_REFUSAL}, []))

    text = format_report(report)

    assert "false-refusal rate (answerable): 0.00%" in text
    assert "refusal accuracy (unanswerable): 100.00%" in text
    assert "[far]" in text
    assert "LLM 1" in text

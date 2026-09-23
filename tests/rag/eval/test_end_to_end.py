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


async def test_a_failing_grounding_call_is_counted_as_an_error_not_a_refusal() -> None:
    questions = EvalQuestionSet(
        answerable=[_answerable("boom"), _answerable("ok")],
        unanswerable=[UnanswerableQuestion(question="u-boom", distance="far")],
    )

    async def ground(question: str, source_type: SourceType | None) -> GroundingOutcome:
        if question.endswith("boom"):
            raise TimeoutError("gateway timeout")
        return _ANSWERED

    report = await run_end_to_end(questions, ground)

    assert report.answerable_errors == 1
    assert report.false_refusals == 0
    assert report.false_refusal_rate == 0.0
    assert report.unanswerable_errors == 1
    assert report.correct_refusals == 0
    assert "ERRORS" in format_report(report)


async def test_reports_llm_latency_and_citation_hit_rate() -> None:
    q_ref = AnswerableQuestion(
        question="hit",
        expected_document_id="cdc-consolidada",
        expected_refs=["art. 54-A"],
        evidence="x",
    )
    q_wrong = AnswerableQuestion(
        question="wrong",
        expected_document_id="cdc-consolidada",
        expected_refs=["art. 54-A"],
        evidence="x",
    )
    q_thr = _answerable("thr")
    questions = EvalQuestionSet(answerable=[q_ref, q_wrong, q_thr], unanswerable=[])
    outcomes = {
        "hit": GroundingOutcome(
            refused=False,
            refused_by_threshold=False,
            llm_latency_seconds=10.0,
            cited=(("cdc-consolidada", "art. 54-A"),),
        ),
        "wrong": GroundingOutcome(
            refused=False,
            refused_by_threshold=False,
            llm_latency_seconds=40.0,
            cited=(("cdc-consolidada", "art. 6º"),),
        ),
        "thr": _THRESHOLD_REFUSAL,
    }

    report = await run_end_to_end(questions, _scripted(outcomes, []))

    assert report.answered == 2
    assert report.citation_hit_rate == 0.5
    assert report.llm_latencies == (10.0, 40.0)
    text = format_report(report)
    assert "max 40.0s" in text
    assert ">30s: 1" in text
    assert "citation hit rate (answered answerable): 50.00%" in text


async def test_false_refusals_are_decomposed_by_cause() -> None:
    def q(name: str) -> AnswerableQuestion:
        return AnswerableQuestion(
            question=name,
            expected_document_id="cdc-consolidada",
            expected_refs=["art. 54-A"],
            evidence="x",
        )

    gold = (("cdc-consolidada", "art. 54-A"),)
    other = (("cdc-consolidada", "art. 6º"),)
    questions = EvalQuestionSet(
        answerable=[q("answered"), q("threshold"), q("llm-gold"), q("no-gold")], unanswerable=[]
    )
    outcomes = {
        "answered": GroundingOutcome(False, False, retrieved=gold, cited=gold),
        "threshold": GroundingOutcome(True, True, retrieved=gold),
        "llm-gold": GroundingOutcome(True, False, llm_latency_seconds=1.0, retrieved=gold),
        "no-gold": GroundingOutcome(True, False, llm_latency_seconds=1.0, retrieved=other),
    }

    report = await run_end_to_end(questions, _scripted(outcomes, []))

    assert [r.bucket for r in report.answerable_records] == [
        "answered",
        "threshold",
        "llm_refused_gold_in_context",
        "gold_not_retrieved",
    ]
    text = format_report(report)
    assert "threshold 1 (gold was in context for 1)" in text
    assert "LLM refused with gold in context 1" in text
    assert "gold not retrieved 1" in text

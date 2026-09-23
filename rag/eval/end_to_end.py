"""End-to-end refusal evaluation: aggregation over a grounding callable.

The retrieval-only eval (`rag.eval.run`) decides "refused" from the
similarity threshold alone. Real refusal is two-staged — the threshold
(coarse first filter) and then the LLM grounding stage returning no
claims — so this module measures the *whole* behavior: false-refusal
rate on answerable questions and refusal accuracy on unanswerable ones,
attributing each refusal to the stage that made it. It takes the
grounding step as an injected callable so `rag/` never imports `apps/`
(see `design.md` — "Module boundaries"); the manual runner that wires
the real LLM lives in `scripts/eval_end_to_end.py`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag.corpus.manifest import SourceType
from rag.eval.questions import AnswerableQuestion, EvalQuestionSet
from rag.eval.run import _source_type_for


@dataclass(frozen=True)
class GroundingOutcome:
    refused: bool
    refused_by_threshold: bool
    """True when the refusal happened at the similarity threshold,
    before any LLM call; False for an LLM-stage refusal (empty or
    fully-invalid claims) or for an answered question."""
    llm_latency_seconds: float | None = None
    """Wall time of the grounding call when it reached the LLM (retrieval
    and embedding included — milliseconds next to the LLM); `None` when
    the threshold refused before any LLM call."""
    cited: tuple[tuple[str, str | None], ...] = ()
    """`(document_id, article_ref)` of each rendered citation, resolved
    against the turn's retrieved chunks."""
    retrieved: tuple[tuple[str, str | None], ...] = ()
    """`(document_id, article_ref)` of every chunk in the turn's
    retrieved top-k, whether or not the threshold then refused."""


@dataclass(frozen=True)
class AnswerableRecord:
    """Per-question outcome for an answerable question (no question
    text — just its index in the eval set)."""

    index: int
    style: str
    refused: bool
    refused_by_threshold: bool
    gold_in_context: bool
    """The expected document (and, when specified, one of the expected
    articles) was among the retrieved top-k."""

    @property
    def bucket(self) -> str:
        if not self.refused:
            return "answered"
        if self.refused_by_threshold:
            return "threshold"
        return "llm_refused_gold_in_context" if self.gold_in_context else "gold_not_retrieved"


GroundFn = Callable[[str, SourceType | None], Awaitable[GroundingOutcome]]


@dataclass(frozen=True)
class EndToEndReport:
    answerable_total: int
    false_refusals: int
    false_refusals_by_threshold: int
    false_refusals_by_llm: int
    false_refusal_rate_by_style: dict[str, float]
    answerable_errors: int
    answerable_records: tuple[AnswerableRecord, ...]
    answered: int
    citation_hits: int
    llm_latencies: tuple[float, ...]
    unanswerable_total: int
    correct_refusals: int
    correct_refusals_by_threshold: int
    correct_refusals_by_llm: int
    refusal_accuracy_by_distance: dict[str, float]
    unanswerable_errors: int

    @property
    def false_refusal_rate(self) -> float:
        return _rate(self.false_refusals, self.answerable_total - self.answerable_errors)

    @property
    def citation_hit_rate(self) -> float:
        """Share of answered answerable questions citing the expected
        document (and, when specified, one of the expected articles)."""
        return _rate(self.citation_hits, self.answered)

    @property
    def refusal_accuracy(self) -> float:
        return _rate(self.correct_refusals, self.unanswerable_total - self.unanswerable_errors)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def percentile(values: tuple[float, ...], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def _matches_expected(
    pairs: tuple[tuple[str, str | None], ...], question: AnswerableQuestion
) -> bool:
    expected_refs = set(question.expected_refs) if question.expected_refs else None
    return any(
        document_id == question.expected_document_id
        and (expected_refs is None or article_ref in expected_refs)
        for document_id, article_ref in pairs
    )


def _cites_expected(outcome: GroundingOutcome, question: AnswerableQuestion) -> bool:
    return _matches_expected(outcome.cited, question)


async def _safe_ground(
    ground: GroundFn, question: str, source_type: SourceType | None
) -> GroundingOutcome | None:
    """`None` when the grounding call itself failed (e.g. a gateway
    timeout): a manual run over every question must not lose all its
    results to one transient failure, and an errored question is
    neither a refusal nor an answer — it is excluded from the rates and
    reported as an error count instead."""
    try:
        return await ground(question, source_type)
    except Exception:  # noqa: BLE001 - manual eval boundary; counted, never swallowed silently
        return None


async def run_end_to_end(
    questions: EvalQuestionSet,
    ground: GroundFn,
    *,
    answerable_indices: frozenset[int] | None = None,
) -> EndToEndReport:
    """`answerable_indices` restricts the answerable questions run (by
    position in the eval set) — used to A/B a prompt on just the
    questions attributable to it; unanswerable questions always run."""
    false_by_style: dict[str, list[bool]] = defaultdict(list)
    false_refusals = false_by_threshold = 0
    answerable_errors = answered = citation_hits = 0
    latencies: list[float] = []
    records: list[AnswerableRecord] = []
    selected = [
        (index, question)
        for index, question in enumerate(questions.answerable)
        if answerable_indices is None or index in answerable_indices
    ]
    for index, answerable in selected:
        outcome = await _safe_ground(
            ground, answerable.question, _source_type_for(answerable.expected_document_id)
        )
        if outcome is None:
            answerable_errors += 1
            continue
        false_by_style[answerable.style].append(outcome.refused)
        records.append(
            AnswerableRecord(
                index=index,
                style=answerable.style,
                refused=outcome.refused,
                refused_by_threshold=outcome.refused_by_threshold,
                gold_in_context=_matches_expected(outcome.retrieved, answerable),
            )
        )
        if outcome.llm_latency_seconds is not None:
            latencies.append(outcome.llm_latency_seconds)
        if not outcome.refused:
            answered += 1
            citation_hits += _cites_expected(outcome, answerable)
        if outcome.refused:
            false_refusals += 1
            if outcome.refused_by_threshold:
                false_by_threshold += 1

    refused_by_distance: dict[str, list[bool]] = defaultdict(list)
    correct = correct_by_threshold = 0
    unanswerable_errors = 0
    for unanswerable in questions.unanswerable:
        outcome = await _safe_ground(ground, unanswerable.question, None)
        if outcome is None:
            unanswerable_errors += 1
            continue
        if outcome.llm_latency_seconds is not None:
            latencies.append(outcome.llm_latency_seconds)
        refused_by_distance[unanswerable.distance].append(outcome.refused)
        if outcome.refused:
            correct += 1
            if outcome.refused_by_threshold:
                correct_by_threshold += 1

    return EndToEndReport(
        answerable_total=len(selected),
        answerable_errors=answerable_errors,
        answerable_records=tuple(records),
        answered=answered,
        citation_hits=citation_hits,
        llm_latencies=tuple(latencies),
        false_refusals=false_refusals,
        false_refusals_by_threshold=false_by_threshold,
        false_refusals_by_llm=false_refusals - false_by_threshold,
        false_refusal_rate_by_style={
            style: _rate(sum(flags), len(flags)) for style, flags in false_by_style.items()
        },
        unanswerable_total=len(questions.unanswerable),
        unanswerable_errors=unanswerable_errors,
        correct_refusals=correct,
        correct_refusals_by_threshold=correct_by_threshold,
        correct_refusals_by_llm=correct - correct_by_threshold,
        refusal_accuracy_by_distance={
            distance: _rate(sum(flags), len(flags))
            for distance, flags in refused_by_distance.items()
        },
    )


def format_report(report: EndToEndReport) -> str:
    lines = [
        "end-to-end (real LLM grounding):",
        f"  false-refusal rate (answerable): {report.false_refusal_rate:.2%} "
        f"({report.false_refusals}/{report.answerable_total - report.answerable_errors}; "
        f"threshold {report.false_refusals_by_threshold}, LLM {report.false_refusals_by_llm})",
    ]
    for style, value in sorted(report.false_refusal_rate_by_style.items()):
        lines.append(f"    false-refusal rate [{style}]: {value:.2%}")
    lines.append(
        f"  refusal accuracy (unanswerable): {report.refusal_accuracy:.2%} "
        f"({report.correct_refusals}/{report.unanswerable_total - report.unanswerable_errors}; "
        f"threshold {report.correct_refusals_by_threshold}, LLM {report.correct_refusals_by_llm})"
    )
    for distance, value in sorted(report.refusal_accuracy_by_distance.items()):
        lines.append(f"    refusal accuracy [{distance}]: {value:.2%}")
    lines.append(
        f"  citation hit rate (answered answerable): {report.citation_hit_rate:.2%} "
        f"({report.citation_hits}/{report.answered})"
    )
    if report.llm_latencies:
        lat = report.llm_latencies
        over_30 = sum(1 for v in lat if v > 30.0)
        lines.append(
            f"  LLM latency over {len(lat)} calls: p50 {percentile(lat, 50):.1f}s, "
            f"p95 {percentile(lat, 95):.1f}s, max {max(lat):.1f}s, >30s: {over_30}"
        )
    buckets: dict[str, int] = defaultdict(int)
    for record in report.answerable_records:
        if record.refused:
            buckets[record.bucket] += 1
    if buckets:
        threshold_with_gold = sum(
            1 for r in report.answerable_records if r.bucket == "threshold" and r.gold_in_context
        )
        lines.append(
            "  false refusals by cause: "
            f"threshold {buckets['threshold']} (gold was in context for {threshold_with_gold}), "
            f"LLM refused with gold in context {buckets['llm_refused_gold_in_context']}, "
            f"gold not retrieved {buckets['gold_not_retrieved']}"
        )
    errors = report.answerable_errors + report.unanswerable_errors
    if errors:
        lines.append(
            f"  ERRORS (excluded from the rates above): {report.answerable_errors} answerable, "
            f"{report.unanswerable_errors} unanswerable"
        )
    return "\n".join(lines)

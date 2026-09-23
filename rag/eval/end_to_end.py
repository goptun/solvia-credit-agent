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
from rag.eval.questions import EvalQuestionSet
from rag.eval.run import _source_type_for


@dataclass(frozen=True)
class GroundingOutcome:
    refused: bool
    refused_by_threshold: bool
    """True when the refusal happened at the similarity threshold,
    before any LLM call; False for an LLM-stage refusal (empty or
    fully-invalid claims) or for an answered question."""


GroundFn = Callable[[str, SourceType | None], Awaitable[GroundingOutcome]]


@dataclass(frozen=True)
class EndToEndReport:
    answerable_total: int
    false_refusals: int
    false_refusals_by_threshold: int
    false_refusals_by_llm: int
    false_refusal_rate_by_style: dict[str, float]
    answerable_errors: int
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
    def refusal_accuracy(self) -> float:
        return _rate(self.correct_refusals, self.unanswerable_total - self.unanswerable_errors)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


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


async def run_end_to_end(questions: EvalQuestionSet, ground: GroundFn) -> EndToEndReport:
    false_by_style: dict[str, list[bool]] = defaultdict(list)
    false_refusals = false_by_threshold = 0
    answerable_errors = 0
    for answerable in questions.answerable:
        outcome = await _safe_ground(
            ground, answerable.question, _source_type_for(answerable.expected_document_id)
        )
        if outcome is None:
            answerable_errors += 1
            continue
        false_by_style[answerable.style].append(outcome.refused)
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
        refused_by_distance[unanswerable.distance].append(outcome.refused)
        if outcome.refused:
            correct += 1
            if outcome.refused_by_threshold:
                correct_by_threshold += 1

    return EndToEndReport(
        answerable_total=len(questions.answerable),
        answerable_errors=answerable_errors,
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
    errors = report.answerable_errors + report.unanswerable_errors
    if errors:
        lines.append(
            f"  ERRORS (excluded from the rates above): {report.answerable_errors} answerable, "
            f"{report.unanswerable_errors} unanswerable"
        )
    return "\n".join(lines)

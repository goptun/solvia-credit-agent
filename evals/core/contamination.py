"""Contamination detection for live runs (design.md, Decision 7).

A contaminated run is still reported, but can never become a baseline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from evals.core.operational import STATUS_OK, CallRecord

_UNAVAILABLE_MARKERS = ("429", "503", "timeout")


@dataclass(frozen=True)
class ModelSets:
    """Per gateway alias: models a run may resolve to (`expected`) and the
    subset counted as the normal path rather than a fallback (`primary`)."""

    expected: frozenset[str] = frozenset()
    primary: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ContaminationVerdict:
    contaminated: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    model_sets_unset: bool = False
    """Expected/primary sets were not configured, so models could not be
    checked; the run may execute but cannot become a baseline."""


def canonical_model(name: str) -> str:
    """The model name without its provider path (`gemini/gemini-3.5-flash-lite`
    and `gemini-3.5-flash-lite` are the same model): the gateway reports the
    resolved model bare, while the approved sets keep the gateway's full ids."""
    return name.rsplit("/", 1)[-1]


def _final_calls(records: Sequence[CallRecord]) -> list[CallRecord]:
    """The last raw call of each operation: a transient 429/503/timeout that
    the app-level retry/backoff (`apps.agent.llm.resilience`) turned into a
    success is not held against the run, only a failure that survived every
    retry is."""
    last: dict[tuple[str, str], CallRecord] = {}
    order: list[tuple[str, str]] = []
    for record in records:
        key = (record.node, record.operation_id)
        if key not in last:
            order.append(key)
        last[key] = record
    return [last[key] for key in order]


def is_unavailable(status: str) -> bool:
    lowered = status.lower()
    return any(marker in lowered for marker in _UNAVAILABLE_MARKERS)


def assess_contamination(
    records: Sequence[CallRecord],
    model_sets: Mapping[str, ModelSets] | None,
    *,
    max_error_share: float,
    max_fallback_share: float,
    incomplete: bool = False,
    interrupted: bool = False,
) -> ContaminationVerdict:
    reasons: list[str] = []

    finals = _final_calls(records)
    if finals:
        unavailable = sum(1 for r in finals if is_unavailable(r.status))
        share = unavailable / len(finals)
        if share > max_error_share:
            reasons.append(
                f"{share:.1%} of operations ended in a quota/unavailability response "
                f"even after the app's own retry/backoff had a chance "
                f"({unavailable} of {len(finals)} operations; threshold {max_error_share:.0%}; "
                "see the operational suite's status_breakdown for raw 429/503/timeout counts)"
            )

    configured = {
        alias: ModelSets(
            frozenset(map(canonical_model, sets.expected)),
            frozenset(map(canonical_model, sets.primary)),
        )
        for alias, sets in (model_sets or {}).items()
        if sets.expected
    }
    aliases_used = {r.alias for r in records}
    sets_unset = bool(aliases_used) and not (aliases_used & set(configured))

    for alias in sorted(aliases_used & set(configured)):
        sets = configured[alias]
        resolved = [r for r in records if r.alias == alias and r.status == STATUS_OK and r.model]
        unexpected = sorted(
            {
                r.model
                for r in resolved
                if r.model is not None and canonical_model(r.model) not in sets.expected
            }
        )
        if unexpected:
            reasons.append(
                f"{alias}: calls resolved to models outside the expected set: "
                + ", ".join(unexpected)
            )
        if sets.primary and resolved:
            outside = sum(1 for r in resolved if canonical_model(r.model or "") not in sets.primary)
            fallback_share = outside / len(resolved)
            if fallback_share > max_fallback_share:
                reasons.append(
                    f"{alias}: {fallback_share:.1%} of calls resolved to fallback models "
                    f"(threshold {max_fallback_share:.0%})"
                )

    if incomplete:
        reasons.append("run incomplete (call budget exhausted)")
    if interrupted:
        reasons.append("run interrupted")

    return ContaminationVerdict(
        contaminated=bool(reasons), reasons=tuple(reasons), model_sets_unset=sets_unset
    )

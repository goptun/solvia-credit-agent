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

    if records:
        unavailable = sum(1 for r in records if is_unavailable(r.status))
        share = unavailable / len(records)
        if share > max_error_share:
            reasons.append(
                f"{share:.1%} of calls got a quota/unavailability response "
                f"(429/503/timeout; threshold {max_error_share:.0%})"
            )

    configured = {alias: sets for alias, sets in (model_sets or {}).items() if sets.expected}
    aliases_used = {r.alias for r in records}
    sets_unset = bool(aliases_used) and not (aliases_used & set(configured))

    for alias in sorted(aliases_used & set(configured)):
        sets = configured[alias]
        resolved = [r for r in records if r.alias == alias and r.status == STATUS_OK and r.model]
        unexpected = sorted(
            {r.model for r in resolved if r.model is not None and r.model not in sets.expected}
        )
        if unexpected:
            reasons.append(
                f"{alias}: calls resolved to models outside the expected set: "
                + ", ".join(unexpected)
            )
        if sets.primary and resolved:
            outside = sum(1 for r in resolved if r.model not in sets.primary)
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

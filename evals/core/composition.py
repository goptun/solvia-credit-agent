"""Composition rules of each dataset: minimums and ranges from the
`evaluation` spec, returned as a list of violations (empty = valid)."""

from __future__ import annotations

from evals.core.schemas import (
    CONTINUE,
    DOCUMENTS,
    INTENTS,
    ComplianceDataset,
    RetrievalDataset,
    RouterDataset,
    SlotsDataset,
)

ANSWERABLE_RANGE = (65, 85)
UNANSWERABLE_RANGE = (20, 30)
MIN_COLLOQUIAL_SHARE = 0.40
MIN_NEAR_MISS = 15
ROUTER_RANGE = (55, 65)
MIN_AMBIGUOUS = 5
SLOTS_RANGE = (25, 35)
APPROVAL_RANGE = (35, 45)
PII_RANGE = (12, 18)
MIN_FAIL_CLOSED_PER_LABEL = 2


def _range_violation(name: str, count: int, bounds: tuple[int, int]) -> list[str]:
    low, high = bounds
    return [] if low <= count <= high else [f"{name}: {count} items, expected {low}-{high}"]


def retrieval_violations(dataset: RetrievalDataset) -> list[str]:
    answerable = [i for i in dataset.items if i.kind == "answerable"]
    unanswerable = [i for i in dataset.items if i.kind == "unanswerable"]
    violations = _range_violation("answerable", len(answerable), ANSWERABLE_RANGE)
    violations += _range_violation("unanswerable", len(unanswerable), UNANSWERABLE_RANGE)

    colloquial = sum(1 for i in answerable if i.style == "colloquial")
    if answerable and colloquial / len(answerable) < MIN_COLLOQUIAL_SHARE:
        violations.append(
            f"colloquial share {colloquial}/{len(answerable)} is below {MIN_COLLOQUIAL_SHARE:.0%}"
        )
    near_miss = sum(1 for i in unanswerable if i.distance == "near_miss")
    if near_miss < MIN_NEAR_MISS:
        violations.append(f"near-miss questions: {near_miss}, expected at least {MIN_NEAR_MISS}")
    for document in DOCUMENTS:
        if not any(i.document == document for i in answerable):
            violations.append(f"no answerable question for document {document!r}")
    if not any(not i.accents for i in answerable):
        violations.append("no question written without accents")
    if not any(len(i.expected_refs or []) > 1 for i in answerable):
        violations.append("no question with more than one acceptable article reference")
    return violations


def router_violations(dataset: RouterDataset) -> list[str]:
    violations = _range_violation("router", len(dataset.items), ROUTER_RANGE)
    for intent in INTENTS:
        if not any(i.expected == intent for i in dataset.items):
            violations.append(f"no item expecting intent {intent!r}")
    if not any(i.expected == CONTINUE for i in dataset.items):
        violations.append("no active-flow continuation item")
    ambiguous = sum(1 for i in dataset.items if i.category == "ambiguous")
    if ambiguous < MIN_AMBIGUOUS:
        violations.append(f"ambiguous items: {ambiguous}, expected at least {MIN_AMBIGUOUS}")
    return violations


def slots_violations(dataset: SlotsDataset) -> list[str]:
    violations = _range_violation("slots", len(dataset.items), SLOTS_RANGE)
    for tag in ("complete", "partial", "missing", "invalid"):
        if not any(tag in i.tags for i in dataset.items):
            violations.append(f"no slot item tagged {tag!r}")
    return violations


def compliance_violations(dataset: ComplianceDataset) -> list[str]:
    violations = _range_violation("approval", len(dataset.approval), APPROVAL_RANGE)
    violations += _range_violation("pii", len(dataset.pii), PII_RANGE)
    for label in ("promise", "hedge", "neutral"):
        if not any(i.label == label for i in dataset.approval):
            violations.append(f"no approval item labelled {label!r}")
    if not any("adversarial" in i.tags for i in dataset.approval):
        violations.append("no adversarial approval item")
    for label in ("promise", "hedge"):
        count = sum(1 for i in dataset.approval if i.label == label and "fail_closed" in i.tags)
        if count < MIN_FAIL_CLOSED_PER_LABEL:
            violations.append(
                f"fail_closed {label} items: {count}, expected at least {MIN_FAIL_CLOSED_PER_LABEL}"
            )
    return violations

"""Gateway-call budget accounting and the pre-run call estimate."""

from __future__ import annotations

from dataclasses import dataclass

WORST_CASE_ITEM_CALLS = 8
"""Calls one item can consume at worst (retries, JSON follow-up, tier fallback)."""

# (typical, pessimistic) raw provider calls per item, from design.md Decision 7.
_ROUTER = (1.0, 1.1)
_SLOTS = (1.0, 1.2)
_COMPLIANCE_LLM = (1.0, 2.0)
"""Up to two: a no-tool-call completion is followed by a JSON-mode call."""
_GROUNDING_RETRY = (1.0, 1.2)
_ANSWERABLE_REACHES_LLM = 0.84
_UNANSWERABLE_REACHES_LLM = 0.33


@dataclass(frozen=True)
class Estimate:
    typical: int
    pessimistic: int


class BudgetExceeded(Exception):
    """The run's estimated calls exceed its budget; it must not start."""


def estimate_calls(
    *,
    router: int = 0,
    slots: int = 0,
    compliance_llm: int = 0,
    grounding_answerable: int = 0,
    grounding_unanswerable: int = 0,
) -> Estimate:
    grounding_calls = (
        grounding_answerable * _ANSWERABLE_REACHES_LLM
        + grounding_unanswerable * _UNANSWERABLE_REACHES_LLM
    )
    typical = (
        router * _ROUTER[0]
        + slots * _SLOTS[0]
        + compliance_llm * _COMPLIANCE_LLM[0]
        + grounding_calls * _GROUNDING_RETRY[0]
    )
    pessimistic = (
        router * _ROUTER[1]
        + slots * _SLOTS[1]
        + compliance_llm * _COMPLIANCE_LLM[1]
        + grounding_calls * _GROUNDING_RETRY[1]
    )
    return Estimate(typical=round(typical), pessimistic=round(pessimistic))


def check_estimate(estimate: Estimate, max_calls: int) -> None:
    """Refuse to start when the pessimistic estimate exceeds the budget."""
    if estimate.pessimistic > max_calls:
        raise BudgetExceeded(
            f"estimated {estimate.typical} calls (up to {estimate.pessimistic}) exceeds "
            f"the budget of {max_calls}; select fewer suites, use --sample, or raise "
            "EVALS_MAX_GATEWAY_CALLS"
        )


class CallBudget:
    """Counts raw provider calls against a hard cap."""

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0
        self.stopped_by_budget = False

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def record_call(self) -> None:
        self.used += 1

    def can_start_item(self, worst_case: int = WORST_CASE_ITEM_CALLS) -> bool:
        """`False` (and the run is marked stopped by budget) when the
        remaining budget cannot cover a worst-case item."""
        if self.remaining >= worst_case:
            return True
        self.stopped_by_budget = True
        return False

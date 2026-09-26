"""The live runner: budget, pacing, sampling, contamination, interruption."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

import pytest

from apps.agent.llm.fake import FakeLLM
from evals.adapters.instrumentation import CallRecorder
from evals.core.budget import BudgetExceeded, CallBudget
from evals.core.contamination import ModelSets
from evals.core.operational import CallRecord
from evals.core.report import render_run
from evals.datasets import load_dataset
from evals.live_config import load_model_sets
from evals.live_runner import (
    PlannedSuite,
    RunMeta,
    RunSettings,
    UnknownLiveSuite,
    build_plan,
    estimate_plan,
    refuse_over_budget,
    run_live,
)
from evals.suites.live import ROUTER, LiveContext, LiveSuite, Summary
from tests.agent.nodes.fakes import ScriptedLLMFactory

Call = tuple[str, str | None]
"""(status, resolved model) of one simulated provider call."""

_SETS = {"solvia-fast": ModelSets(frozenset({"primary", "backup"}), frozenset({"primary"}))}
_META = RunMeta("a" * 40, False, {"ci": "false"}, {"fast": "solvia-fast", "smart": "solvia-smart"})


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _settings(pacing: float = 6.0, sample: float | None = None) -> RunSettings:
    return RunSettings(
        seed=42, pacing_seconds=pacing, max_error_share=0.05, max_fallback_share=0.10, sample=sample
    )


def _stub_suite(
    recorder: CallRecorder,
    script: Sequence[list[Call]],
    starts: list[float] | None = None,
    clock: FakeClock | None = None,
) -> LiveSuite[int, int]:
    """Item `i` makes the calls `script[i]` (status, model) through the recorder."""

    async def run_item(item: int, ctx: LiveContext) -> int | None:
        if starts is not None and clock is not None:
            starts.append(clock.now)
        for status, model in script[item]:
            recorder.call_started()
            recorder.add(
                CallRecord(
                    f"op-{item}", "router", "solvia-fast", model, status, "tool_call", 1.0, True
                )
            )
        return item

    return LiveSuite[int, int](
        name="router",
        dataset="router",
        items=lambda loaded: [],
        item_id=str,
        run_item=run_item,
        summarize=lambda results, unavailable, seed: Summary({}, {}),
    )


def _plan(suite: LiveSuite[int, int], n: int) -> list[PlannedSuite]:
    return [PlannedSuite(suite, load_dataset("router"), list(range(n)))]


def _ctx() -> LiveContext:
    return LiveContext(ScriptedLLMFactory(fast=FakeLLM()))


async def _run(
    suite_script: Sequence[list[Call]],
    *,
    max_calls: int = 100,
    sets: dict[str, ModelSets] | None = None,
    clock: FakeClock | None = None,
    sample: float | None = None,
) -> Any:
    budget = CallBudget(max_calls)
    recorder = CallRecorder(budget)
    suite = _stub_suite(recorder, suite_script)
    return await run_live(
        _plan(suite, len(suite_script)),
        _ctx(),
        recorder,
        budget,
        _settings(sample=sample),
        _META,
        sets if sets is not None else _SETS,
        clock or FakeClock(),
    )


_OK: Call = ("ok", "primary")


async def test_a_clean_run_is_complete_and_uncontaminated() -> None:
    record = await _run([[_OK]] * 10)

    assert not record.contaminated and not record.incomplete
    assert record.budget["used"] == 10
    assert record.resolved_model_mix == {"solvia-fast": {"primary": 10}}
    assert record.suites["router"].details["items_scored"] == 10
    assert "operational" in record.suites


async def test_simulated_429_and_503_responses_contaminate_the_run() -> None:
    script: list[list[Call]] = [[_OK]] * 17
    script += [[("429", None)], [("503", None)], [("ReadTimeout", None)]]

    record = await _run(script)

    assert record.contaminated
    assert any("quota/unavailability" in reason for reason in record.contamination_reasons)


async def test_an_unexpected_model_contaminates_the_run_and_is_named() -> None:
    record = await _run([[_OK]] * 19 + [[("ok", "gpt-oss-unexpected")]])

    assert record.contaminated
    assert any("gpt-oss-unexpected" in reason for reason in record.contamination_reasons)


async def test_budget_exhaustion_stops_before_an_item_it_cannot_cover_and_marks_the_run() -> None:
    script = [[_OK] * 5] * 10  # 5 calls per item; a worst-case item needs 8 in reserve

    record = await _run(script, max_calls=20)

    assert record.incomplete and record.contaminated
    assert record.suites["router"].details["items_scored"] == 3  # 15 used, 5 < 8 left
    assert record.budget["used"] == 15
    assert any("incomplete" in reason for reason in record.contamination_reasons)


async def test_items_are_paced_by_the_interval_between_their_starts() -> None:
    clock = FakeClock()
    starts: list[float] = []
    budget = CallBudget(100)
    recorder = CallRecorder(budget)
    suite = _stub_suite(recorder, [[_OK]] * 4, starts, clock)

    await run_live(_plan(suite, 4), _ctx(), recorder, budget, _settings(6.0), _META, _SETS, clock)

    assert clock.sleeps == [6.0, 6.0, 6.0]  # nothing before the first item
    assert [b - a for a, b in zip(starts, starts[1:], strict=False)] == [6.0, 6.0, 6.0]


async def test_an_interrupted_run_is_returned_marked_interrupted() -> None:
    budget = CallBudget(100)
    recorder = CallRecorder(budget)

    async def run_item(item: int, ctx: LiveContext) -> int | None:
        raise asyncio.CancelledError

    suite = LiveSuite[int, int](
        "router", "router", lambda loaded: [], str, run_item, lambda r, u, s: Summary({}, {})
    )

    record = await run_live(
        _plan(suite, 2), _ctx(), recorder, budget, _settings(), _META, _SETS, FakeClock()
    )

    assert record.contaminated
    assert any("interrupted" in reason for reason in record.contamination_reasons)


async def test_an_item_that_raises_is_recorded_unavailable_with_its_error_type() -> None:
    budget = CallBudget(100)
    recorder = CallRecorder(budget)

    async def run_item(item: int, ctx: LiveContext) -> int | None:
        raise KeyError("boom")

    suite = LiveSuite[int, int](
        "router", "router", lambda loaded: [], str, run_item, lambda r, u, s: Summary({}, {})
    )

    record = await run_live(
        _plan(suite, 1), _ctx(), recorder, budget, _settings(), _META, _SETS, FakeClock()
    )

    assert record.suites["router"].details["unavailable"] == {"0": "KeyError"}


# --- sampling and the estimate -------------------------------------------------


def test_the_same_fraction_and_seed_select_the_same_items() -> None:
    first = build_plan(["router", "slots"], sample=0.3, seed=42)
    again = build_plan(["router", "slots"], sample=0.3, seed=42)
    other = build_plan(["router", "slots"], sample=0.3, seed=7)

    ids = [[i.id for i in p.items] for p in first]
    assert ids == [[i.id for i in p.items] for p in again]
    assert ids != [[i.id for i in p.items] for p in other]


def test_a_sample_covers_every_stratum_it_can() -> None:
    [planned] = build_plan(["router"], sample=0.3, seed=42)
    loaded = planned.loaded
    stratum = planned.suite.stratum(loaded)
    all_strata = {stratum(item) for item in planned.suite.items(loaded)}
    sampled = Counter(stratum(item) for item in planned.items)

    assert len(planned.items) == round(len(planned.suite.items(loaded)) * 0.3)
    assert len(sampled) == min(len(planned.items), len(all_strata))


def test_the_sample_fits_the_validation_budget_and_the_full_run_the_default() -> None:
    sample = build_plan(["router", "slots", "compliance-llm", "grounding"], sample=0.3, seed=42)
    full = build_plan(["router", "slots", "compliance-llm", "grounding"], sample=None, seed=42)

    assert refuse_over_budget(sample, 100).pessimistic <= 100
    assert estimate_plan(full).typical > estimate_plan(sample).typical
    with pytest.raises(BudgetExceeded):
        refuse_over_budget(full, 100)
    refuse_over_budget(full, 300)


def test_an_unknown_suite_is_refused() -> None:
    with pytest.raises(UnknownLiveSuite):
        build_plan(["retrieval"], sample=None, seed=42)


# --- unset model sets (10.5) ---------------------------------------------------


async def test_a_run_against_unset_model_sets_executes_but_blocks_baselining() -> None:
    unset = {"solvia-fast": ModelSets(), "solvia-smart": ModelSets()}
    record = await _run([[_OK]] * 5, sets=unset)

    assert not record.contaminated
    assert record.model_sets_unset
    report = render_run(record)
    assert "Baselining is blocked until the expected model sets are approved" in report
    assert "| solvia-fast | primary | 5 |" in report


def test_the_committed_live_config_has_the_approved_sets_for_both_aliases() -> None:
    sets = load_model_sets()

    assert {"solvia-fast", "solvia-smart", "solvia-eval-fast", "solvia-eval-smart"} <= set(sets)
    assert all(s.expected and s.primary <= s.expected for s in sets.values())


def test_the_router_suite_is_registered_with_the_router_dataset() -> None:
    assert ROUTER.dataset == "router" and ROUTER.name == "router"


# --- the command's refusals (no gateway is ever reached) --------------------------


def test_a_live_run_refuses_the_fake_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals.cli import main

    monkeypatch.setenv("LLM_PROVIDER", "fake")

    assert main(["run", "--suite", "router", "--mode", "live"]) == 2
    assert "needs a real gateway" in capsys.readouterr().err


def test_a_live_run_over_budget_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from evals.cli import main

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EVALS_MAX_GATEWAY_CALLS", "10")

    assert main(["run", "--suite", "router,slots", "--mode", "live"]) == 2
    assert "exceeds the budget of 10" in capsys.readouterr().err

"""The `run --mode live` command: wires the real gateway, budget and outputs
around `evals.live_runner`. Local only — CI never reaches the gateway."""

from __future__ import annotations

import argparse
import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path

from apps.agent.llm.settings import get_settings
from evals.adapters.environment import environment, git_dirty, git_sha
from evals.adapters.grounding import MissingDatabase, grounding_dependencies
from evals.adapters.instrumentation import CallRecorder, InstrumentedLLMFactory
from evals.core.budget import BudgetExceeded, CallBudget
from evals.core.report import render_run
from evals.live_config import load_model_sets
from evals.live_runner import (
    RunMeta,
    RunSettings,
    UnknownLiveSuite,
    build_plan,
    refuse_over_budget,
    run_live,
)
from evals.settings import get_evals_settings
from evals.suites.live import LiveContext

REPORT_DIR = Path(__file__).parent / "reports"


def run_live_command(args: argparse.Namespace) -> int:
    settings = get_evals_settings()
    llm_settings = get_settings()
    if llm_settings.llm_provider == "fake":
        print("ERROR: a live run needs a real gateway (LLM_PROVIDER is 'fake')", file=sys.stderr)
        return 2
    try:
        plan = build_plan(
            [name.strip() for name in args.suite.split(",")], sample=args.sample, seed=args.seed
        )
        estimate = refuse_over_budget(plan, settings.evals_max_gateway_calls)
    except (UnknownLiveSuite, ValueError, BudgetExceeded) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Estimated gateway calls: {estimate.typical} typical, up to {estimate.pessimistic} "
        f"(budget {settings.evals_max_gateway_calls}); pacing {settings.evals_pacing_seconds}s",
        file=sys.stderr,
    )

    budget = CallBudget(settings.evals_max_gateway_calls)
    recorder = CallRecorder(budget)
    factory = InstrumentedLLMFactory(llm_settings, recorder)
    run_settings = RunSettings(
        seed=args.seed,
        pacing_seconds=settings.evals_pacing_seconds,
        max_error_share=settings.evals_max_error_share,
        max_fallback_share=settings.evals_max_fallback_share,
        sample=args.sample,
    )
    meta = RunMeta(
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        environment=environment(),
        aliases={"fast": llm_settings.llm_model_fast, "smart": llm_settings.llm_model_smart},
    )
    needs_grounding = any(planned.suite.name == "grounding" for planned in plan)

    try:
        with grounding_dependencies() if needs_grounding else nullcontext(None) as deps:
            record = asyncio.run(
                run_live(
                    plan,
                    LiveContext(factory, deps),
                    recorder,
                    budget,
                    run_settings,
                    meta,
                    load_model_sets(),
                )
            )
    except MissingDatabase as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    markdown = render_run(record)
    if args.output:
        Path(args.output).write_text(record.to_json(), encoding="utf-8")
    if args.report:
        REPORT_DIR.mkdir(exist_ok=True)
        (REPORT_DIR / f"{args.report}.json").write_text(record.to_json(), encoding="utf-8")
        (REPORT_DIR / f"{args.report}.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 1 if record.contaminated else 0

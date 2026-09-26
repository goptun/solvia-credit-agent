"""Dataset mirroring and live-run publication, against the fake publisher."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest

from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.router import RouterDecision
from evals.adapters.instrumentation import CallRecorder
from evals.cli import main
from evals.core.budget import CallBudget
from evals.core.run import RunRecord
from evals.datasets import DATASET_NAMES, load_dataset
from evals.live_runner import PlannedSuite, RunMeta, RunSettings, build_plan, run_live
from evals.publishing import (
    FakePublisher,
    dataset_payloads,
    publish_run_safely,
    run_payloads,
    sync_datasets,
)
from evals.suites.live import LiveContext
from tests.agent.nodes.fakes import ScriptedLLMFactory

_META = RunMeta("a" * 40, False, {"ci": "false"}, {"fast": "solvia-fast", "smart": "solvia-smart"})
_SECRET_OR_INFRA = re.compile(
    r"sk-[A-Za-z0-9]{8,}|pk-lf-|sk-lf-|\b\d{1,3}(?:\.\d{1,3}){3}\b"
    r"|localhost|ssh |\.internal\b|\.ts\.net\b",
    re.IGNORECASE,
)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_strings(k), *_strings(v))]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


async def _router_run() -> RunRecord:
    [planned] = build_plan(["router"], sample=0.1, seed=42)
    plan = [PlannedSuite(planned.suite, planned.loaded, planned.items[:3])]
    responses = [RouterDecision(intent="complaint", is_new_request=True)] * 3
    ctx = LiveContext(ScriptedLLMFactory(fast=FakeLLM(responses=responses)))
    budget = CallBudget(100)

    class _Clock:
        def monotonic(self) -> float:
            return 0.0

        async def sleep(self, seconds: float) -> None:
            return None

    return await run_live(
        plan,
        ctx,
        CallRecorder(budget),
        budget,
        RunSettings(42, 0.0, 0.05, 0.10, 0.1),
        _META,
        None,
        _Clock(),
    )


def test_syncing_twice_creates_no_duplicates_and_stores_the_version() -> None:
    publisher = FakePublisher()
    datasets = [load_dataset(name) for name in DATASET_NAMES]

    first = sync_datasets(publisher, datasets)
    counts = {name: len(items) for name, items in publisher.datasets.items()}
    second = sync_datasets(publisher, datasets)

    assert first == second == sum(counts.values())
    assert {name: len(items) for name, items in publisher.datasets.items()} == counts
    assert set(counts) == {f"solvia/{name}" for name in DATASET_NAMES}
    sample = next(iter(publisher.datasets["solvia/router"].values()))
    assert sample.id.startswith("router:")
    assert sample.metadata["dataset_version"] == load_dataset("router").version


def test_dataset_payloads_never_contain_secrets_or_infrastructure_details() -> None:
    strings = [
        s
        for name in DATASET_NAMES
        for payload in dataset_payloads(load_dataset(name))
        for s in _strings([payload.input, payload.expected_output, payload.metadata, payload.id])
    ]

    assert strings
    assert [s for s in strings if _SECRET_OR_INFRA.search(s)] == []


async def test_a_live_run_is_published_with_name_scores_and_aggregates() -> None:
    record = await _router_run()
    publisher = FakePublisher()

    published = publish_run_safely(publisher, record)

    assert published == 1
    [run] = publisher.runs
    assert run.run_name == f"router@v{load_dataset('router').version}/aaaaaaa"
    assert run.experiment == "router" and run.dataset_name == "solvia/router"
    assert len(run.items) == 3 and all("correct" in item.scores for item in run.items)
    assert all(item.item_id.startswith("router:") for item in run.items)
    assert (
        run.aggregates["router.accuracy"]
        == record.suites["router"].metrics["router.accuracy"].value
    )
    assert run.metadata["contaminated"] == "false" and run.metadata["alias_fast"] == "solvia-fast"
    strings = _strings([run.metadata, run.aggregates, run.items and [i.output for i in run.items]])
    assert [s for s in strings if _SECRET_OR_INFRA.search(s)] == []


async def test_the_operational_pseudo_suite_is_not_published_as_an_experiment() -> None:
    record = await _router_run()

    assert [payload.experiment for payload in run_payloads(record)] == ["router"]


async def test_a_failing_publisher_is_logged_and_never_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = await _router_run()

    class Broken(FakePublisher):
        def publish_run(self, run: Any) -> None:
            raise ConnectionError("network down")

    with caplog.at_level(logging.WARNING):
        assert publish_run_safely(Broken(), record) == 0

    assert "publication failed for router: ConnectionError" in caplog.text


async def test_without_credentials_publication_is_skipped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = await _router_run()

    with caplog.at_level(logging.WARNING):
        assert publish_run_safely(None, record) == 0

    assert "publication skipped" in caplog.text


async def test_the_live_command_writes_its_report_even_without_langfuse_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from apps.agent.llm.settings import Settings

    record = await _router_run()

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setattr(
        "evals.live_cli.get_settings", lambda: Settings(llm_provider="openai_compatible")
    )
    monkeypatch.setattr("evals.live_cli.run_live", lambda *args, **kwargs: None)
    monkeypatch.setattr("evals.live_cli.REPORT_DIR", tmp_path)
    monkeypatch.setattr("evals.live_cli.asyncio.run", lambda coro: record)

    with caplog.at_level(logging.WARNING):
        code = main(["run", "--suite", "router", "--mode", "live", "--report", "demo"])

    assert code == 0
    assert (tmp_path / "demo.md").exists()
    assert json.loads((tmp_path / "demo.json").read_text())["mode"] == "live"
    assert "publication skipped" in caplog.text

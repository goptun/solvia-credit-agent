"""Publishing datasets and live runs to an experiment tracker (LangFuse).

A small port, so tests use a fake and the harness works without credentials:
the committed report is the source of truth, publication is best effort.
Payloads hold only synthetic dataset content, git SHA and aliases — never keys,
hostnames or IPs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from evals.core.run import RunRecord
from evals.core.schemas import (
    ComplianceDataset,
    RetrievalDataset,
    RouterDataset,
    SlotsDataset,
)
from evals.datasets import LoadedDataset

logger = logging.getLogger(__name__)

DATASET_PREFIX = "solvia/"


@dataclass(frozen=True)
class DatasetItemPayload:
    id: str
    input: dict[str, Any]
    expected_output: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RunItemPayload:
    item_id: str
    output: dict[str, Any]
    scores: dict[str, float]


@dataclass(frozen=True)
class RunPayload:
    dataset_name: str
    experiment: str
    run_name: str
    metadata: dict[str, str]
    items: list[RunItemPayload]
    aggregates: dict[str, float]


class Publisher(Protocol):
    def sync_dataset(
        self, name: str, description: str, items: Sequence[DatasetItemPayload]
    ) -> None: ...

    def publish_run(self, run: RunPayload) -> None: ...


def tracker_dataset_name(dataset: str) -> str:
    return f"{DATASET_PREFIX}{dataset}"


def item_id(dataset: str, local_id: str) -> str:
    """Deterministic, so re-syncing upserts instead of duplicating."""
    return f"{dataset}:{local_id}"


def dataset_payloads(loaded: LoadedDataset) -> list[DatasetItemPayload]:
    dataset = loaded.dataset
    meta = {"dataset_version": loaded.version, "dataset_sha256": loaded.sha256[:12]}
    payloads: list[DatasetItemPayload] = []
    if isinstance(dataset, RetrievalDataset):
        for r in dataset.items:
            payloads.append(
                DatasetItemPayload(
                    item_id(loaded.name, r.id),
                    {"question": r.question},
                    {
                        "kind": r.kind,
                        "document": r.document,
                        "expected_refs": [str(ref) for ref in r.acceptable()],
                        "distance": r.distance,
                    },
                    {**meta, "item": r.id, "difficulty": r.difficulty},
                )
            )
    elif isinstance(dataset, RouterDataset):
        for m in dataset.items:
            payloads.append(
                DatasetItemPayload(
                    item_id(loaded.name, m.id),
                    {"message": m.message, "active_flow": m.active_flow},
                    {"intent": m.expected, "acceptable": m.acceptable},
                    {**meta, "item": m.id, "category": m.category},
                )
            )
    elif isinstance(dataset, SlotsDataset):
        for sl in dataset.items:
            payloads.append(
                DatasetItemPayload(
                    item_id(loaded.name, sl.id),
                    {"message": sl.message, "known": sl.known.model_dump(exclude_none=True)},
                    {"slots": sl.expected.model_dump(exclude_none=True)},
                    {**meta, "item": sl.id, "tags": sl.tags},
                )
            )
    elif isinstance(dataset, ComplianceDataset):
        for a in dataset.approval:
            payloads.append(
                DatasetItemPayload(
                    item_id(loaded.name, a.id),
                    {"text": a.text},
                    {"label": a.label},
                    {**meta, "item": a.id, "kind": "approval", "tags": a.tags},
                )
            )
        for p in dataset.pii:
            payloads.append(
                DatasetItemPayload(
                    item_id(loaded.name, p.id),
                    {"text": p.text},
                    {"masked": p.expected_masked},
                    {**meta, "item": p.id, "kind": "pii", "known_gap": p.known_gap},
                )
            )
    return payloads


def sync_datasets(publisher: Publisher, datasets: Sequence[LoadedDataset]) -> int:
    """Mirror each dataset; returns the number of items synced."""
    total = 0
    for loaded in datasets:
        payloads = dataset_payloads(loaded)
        publisher.sync_dataset(
            tracker_dataset_name(loaded.name),
            f"Solvia evaluation dataset {loaded.name}, version {loaded.version}",
            payloads,
        )
        total += len(payloads)
    return total


def run_payloads(record: RunRecord) -> list[RunPayload]:
    """One payload per scored suite of a live run (the operational pseudo-suite
    has no dataset and is not published as an experiment)."""
    payloads = []
    short_sha = record.git_sha[:7]
    for name, suite in record.suites.items():
        items = suite.details.get("items")
        if not suite.datasets or not items:
            continue
        dataset = suite.datasets[0]
        payloads.append(
            RunPayload(
                dataset_name=tracker_dataset_name(dataset.name),
                experiment=name,
                run_name=f"{name}@v{dataset.version}/{short_sha}",
                metadata={
                    "mode": record.mode,
                    "git_sha": record.git_sha,
                    "contaminated": str(record.contaminated).lower(),
                    "sample_fraction": ""
                    if record.sample_fraction is None
                    else str(record.sample_fraction),
                    **{f"alias_{tier}": alias for tier, alias in record.aliases.items()},
                },
                items=[
                    RunItemPayload(
                        item_id(dataset.name, entry["id"]), entry["output"], entry["scores"]
                    )
                    for entry in items
                ],
                aggregates={metric: value.value for metric, value in suite.metrics.items()},
            )
        )
    return payloads


def publish_run_safely(publisher: Publisher | None, record: RunRecord) -> int:
    """Publish every scored suite of `record`; never raises. Returns the
    number of experiments published. Without a publisher (no credentials) it
    logs the skip and the committed report stays the source of truth."""
    if publisher is None:
        logger.warning("LangFuse publication skipped: credentials are not configured")
        return 0
    published = 0
    for payload in run_payloads(record):
        try:
            publisher.publish_run(payload)
        except Exception as exc:  # noqa: BLE001 - publication is best effort
            logger.warning(
                "LangFuse publication failed for %s: %s", payload.experiment, type(exc).__name__
            )
        else:
            published += 1
    return published


@dataclass
class FakePublisher:
    """Records what would be published; keyed like the real tracker so a
    second sync overwrites instead of duplicating."""

    datasets: dict[str, dict[str, DatasetItemPayload]] = field(default_factory=dict)
    runs: list[RunPayload] = field(default_factory=list)

    def sync_dataset(
        self, name: str, description: str, items: Sequence[DatasetItemPayload]
    ) -> None:
        stored = self.datasets.setdefault(name, {})
        for item in items:
            stored[item.id] = item

    def publish_run(self, run: RunPayload) -> None:
        self.runs.append(run)

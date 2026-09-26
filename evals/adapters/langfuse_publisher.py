"""The real publisher, over the LangFuse SDK (v4).

Datasets are mirrored with `create_dataset_item(id=…)`, an idempotent upsert.
A live run is published with `run_experiment(max_concurrency=1, …)`: the task
replays the output the run already recorded (the gateway was called once, by
the harness, under its own budget and pacing) and evaluators return the
per-item scores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langfuse import Evaluation, Langfuse

from evals.publishing import DatasetItemPayload, Publisher, RunPayload
from evals.settings import get_evals_settings


class LangfusePublisher:
    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def sync_dataset(
        self, name: str, description: str, items: Sequence[DatasetItemPayload]
    ) -> None:
        self._client.create_dataset(name=name, description=description)
        for item in items:
            self._client.create_dataset_item(
                dataset_name=name,
                id=item.id,
                input=item.input,
                expected_output=item.expected_output,
                metadata=item.metadata,
            )
        self._client.flush()

    def publish_run(self, run: RunPayload) -> None:
        by_id = {item.item_id: item for item in run.items}
        dataset = self._client.get_dataset(run.dataset_name)
        data = [item for item in dataset.items if item.id in by_id]

        def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
            return {"item_id": item.id, **by_id[item.id].output}

        def per_item(*, output: dict[str, Any], **kwargs: Any) -> list[Evaluation]:
            scores = by_id[output["item_id"]].scores
            return [Evaluation(name=name, value=value) for name, value in scores.items()]

        def aggregates(**kwargs: Any) -> list[Evaluation]:
            return [Evaluation(name=name, value=value) for name, value in run.aggregates.items()]

        self._client.run_experiment(
            name=run.experiment,
            run_name=run.run_name,
            data=data,
            task=task,
            evaluators=[per_item],
            run_evaluators=[aggregates],
            max_concurrency=1,
            metadata=run.metadata,
        )
        self._client.flush()


def build_publisher() -> Publisher | None:
    """The LangFuse publisher when the public and secret keys are configured
    (environment or `.env`); `None` otherwise (publication is then skipped)."""
    settings = get_evals_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    return LangfusePublisher(
        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host or None,
        )
    )

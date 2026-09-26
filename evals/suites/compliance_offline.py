"""Offline compliance suite: the two deterministic approval-promise screens
and PII masking. No model, no gateway."""

from __future__ import annotations

from collections.abc import Callable

from apps.agent.nodes.compliance import (
    keyword_flags_approval_promise,
    keyword_flags_unhedged_approval_mention,
    mask_pii,
)
from evals.core.baseline import DatasetRef
from evals.core.classification import PrecisionRecall, precision_recall
from evals.core.masking import masking_exact_match
from evals.core.run import MetricValue, SuiteResult, proportion_metric
from evals.core.schemas import ApprovalItem, ComplianceDataset
from evals.datasets import LoadedDataset, load_dataset

SUITE = "compliance"


def _screen_outcomes(
    items: list[ApprovalItem], flags: Callable[[str], bool]
) -> list[tuple[bool, bool]]:
    return [(item.label == "promise", flags(item.text)) for item in items]


def _screen_metrics(
    name: str, items: list[ApprovalItem], flags: Callable[[str], bool]
) -> dict[str, MetricValue]:
    result: PrecisionRecall = precision_recall(_screen_outcomes(items, flags))
    metrics = {
        f"{name}.precision": proportion_metric(result.precision, "higher", deterministic=True),
        f"{name}.recall": proportion_metric(result.recall, "higher", deterministic=True),
        f"{name}.false_positive_rate": proportion_metric(
            result.false_positive_rate, "lower", deterministic=True
        ),
    }
    hedged_or_neutral = [
        item for item in items if item.label != "promise" and "mentions_approval" in item.tags
    ]
    mention_fpr = precision_recall(_screen_outcomes(hedged_or_neutral, flags))
    metrics[f"{name}.false_positive_rate_on_approval_mentions"] = proportion_metric(
        mention_fpr.false_positive_rate, "lower", deterministic=True
    )
    return metrics


def run_compliance_offline(compliance: LoadedDataset | None = None) -> SuiteResult:
    loaded = compliance or load_dataset("compliance")
    dataset = loaded.dataset
    assert isinstance(dataset, ComplianceDataset)

    metrics: dict[str, MetricValue] = {}
    metrics.update(_screen_metrics("keyword", dataset.approval, keyword_flags_approval_promise))
    metrics.update(
        _screen_metrics("strict", dataset.approval, keyword_flags_unhedged_approval_mention)
    )
    masking = masking_exact_match(dataset.pii, mask_pii)
    metrics["masking.exact_match"] = proportion_metric(
        masking.exact_match, "higher", deterministic=True
    )

    return SuiteResult(
        datasets=[DatasetRef(name=loaded.name, version=loaded.version, sha256=loaded.sha256)],
        metrics=metrics,
        details={
            "masking_failed_ids": list(masking.failed_ids),
            "keyword_missed_promises": [
                i.id
                for i in dataset.approval
                if i.label == "promise" and not keyword_flags_approval_promise(i.text)
            ],
            "strict_false_positives": [
                i.id
                for i in dataset.approval
                if i.label != "promise" and keyword_flags_unhedged_approval_mention(i.text)
            ],
        },
    )

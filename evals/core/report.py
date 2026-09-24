"""Markdown rendering of runs and baselines, and the README metrics block."""

from __future__ import annotations

from evals.core.baseline import Baseline, compare, render_table
from evals.core.run import MetricValue, RunRecord

README_START = "<!-- evals:metrics:start -->"
README_END = "<!-- evals:metrics:end -->"

STRATUM_SEPARATOR = "."
"""Metric names `recall_at_k.style.lexical` hold a breakdown; the README block
shows only the headline (unstratified) metrics."""


class MarkersNotFound(ValueError):
    """The README lacks a well-formed metrics block to rewrite."""


def _interval(low: float | None, high: float | None) -> str:
    return "-" if low is None or high is None else f"[{low:.3f}, {high:.3f}]"


def render_metrics(metrics: dict[str, MetricValue]) -> str:
    lines = ["| metric | value | 95% CI | n |", "|---|---|---|---|"]
    for name, metric in metrics.items():
        lines.append(
            f"| {name} | {metric.value:.4f} | {_interval(metric.ci_low, metric.ci_high)} "
            f"| {metric.n} |"
        )
    return "\n".join(lines)


def render_run(record: RunRecord, baselines: dict[str, Baseline] | None = None) -> str:
    """One section per suite: the metric table and, when a baseline for that
    suite is given, the comparison against it."""
    baselines = baselines or {}
    sections = [
        f"_mode `{record.mode}`, commit `{record.git_sha}`"
        f"{' (dirty)' if record.git_dirty else ''}, seed {record.seed}_"
    ]
    for name, suite in record.suites.items():
        datasets = ", ".join(f"{d.name} v{d.version}" for d in suite.datasets)
        sections.append(f"## {name}\n\nDatasets: {datasets}\n\n{render_metrics(suite.metrics)}")
        baseline = baselines.get(name)
        if baseline is not None:
            rows = compare(baseline.metrics, {k: v.value for k, v in suite.metrics.items()})
            sections.append(f"### {name} vs baseline ({baseline.git_sha})\n\n{render_table(rows)}")
    return "\n\n".join(sections) + "\n"


def render_baselines(baselines: list[Baseline]) -> str:
    """The README block: headline metrics of each committed baseline."""
    if not baselines:
        return "_No baselines recorded yet._"
    sections = []
    for baseline in sorted(baselines, key=lambda b: (b.mode, b.suite)):
        datasets = ", ".join(f"{d.name} v{d.version}" for d in baseline.datasets)
        lines = [
            f"**{baseline.suite}** ({baseline.mode}, commit `{baseline.git_sha}`, {datasets})",
            "",
            "| metric | value | 95% CI | n |",
            "|---|---|---|---|",
        ]
        for name, metric in baseline.metrics.items():
            if STRATUM_SEPARATOR in name:
                continue
            lines.append(
                f"| {name} | {metric.value:.4f} | "
                f"{_interval(metric.ci_low, metric.ci_high)} | {metric.n} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def replace_block(text: str, block: str) -> str:
    """Rewrite only what lies between the markers; everything else is
    returned byte for byte."""
    start = text.find(README_START)
    end = text.find(README_END)
    if start == -1 or end == -1 or end < start:
        raise MarkersNotFound(f"expected {README_START} ... {README_END} in the README")
    head = text[: start + len(README_START)]
    return f"{head}\n{block}\n{text[end:]}"

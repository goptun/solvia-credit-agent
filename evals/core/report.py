"""Markdown rendering of runs and baselines, and the README metrics block."""

from __future__ import annotations

from evals.core.baseline import Baseline, compare, render_table
from evals.core.run import MetricValue, RunRecord
from evals.core.tolerances import is_stratum

README_START = "<!-- evals:metrics:start -->"
README_END = "<!-- evals:metrics:end -->"


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


def _render_live_header(record: RunRecord) -> list[str]:
    """Call counts, the resolved-model mix, and why the run can or cannot
    become a baseline."""
    lines = []
    if record.budget:
        lines.append(
            f"Gateway calls: **{record.budget['used']}** of a budget of "
            f"{record.budget['max_calls']} (estimated {record.budget['estimated_typical']}, "
            f"up to {record.budget['estimated_pessimistic']})."
        )
    if record.sample_fraction is not None:
        lines.append(
            f"**Sampled run** ({record.sample_fraction:.0%}): it can never become a baseline."
        )
    if record.incomplete:
        lines.append("**Incomplete run.**")
    if record.contaminated:
        lines.append("**Contaminated run** — reported, never a baseline: ")
        lines.extend(f"- {reason}" for reason in record.contamination_reasons)
    if record.model_sets_unset:
        lines.append(
            "**Baselining is blocked until the expected model sets are approved** "
            "(`evals/live_config.yaml` has none for the aliases used). Observed mix, "
            "to propose them:"
        )
    if record.resolved_model_mix:
        rows = ["| alias | resolved model | calls |", "|---|---|---|"]
        for alias, models in record.resolved_model_mix.items():
            rows.extend(f"| {alias} | {model} | {count} |" for model, count in models.items())
        lines.append("\n".join(rows))
    return ["\n\n".join(lines)] if lines else []


def render_run(record: RunRecord, baselines: dict[str, Baseline] | None = None) -> str:
    """One section per suite: the metric table and, when a baseline for that
    suite is given, the comparison against it."""
    baselines = baselines or {}
    sections = [
        f"_mode `{record.mode}`, commit `{record.git_sha}`"
        f"{' (dirty)' if record.git_dirty else ''}, seed {record.seed}_"
    ]
    if record.mode == "live":
        sections.extend(_render_live_header(record))
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
            f"**{baseline.suite}** ({baseline.mode}, commit `{baseline.git_sha[:7]}`, {datasets})",
            "",
            "| metric | value | 95% CI | n |",
            "|---|---|---|---|",
        ]
        for name, metric in baseline.metrics.items():
            if is_stratum(name):
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

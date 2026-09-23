"""Operational metrics from the per-call log of a live run."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from evals.core.stats import Proportion, proportion, quantile

KIND_TOOL_CALL = "tool_call"
KIND_NO_TOOL_CALL = "no_tool_call"
KIND_PLAIN = "plain"
KIND_ERROR = "error"
STATUS_OK = "ok"
UNKNOWN_MODEL = "unknown"


@dataclass(frozen=True)
class CallRecord:
    """One raw provider call. `operation_id` groups the calls one node
    invocation made (native attempt, JSON-mode call, retries)."""

    operation_id: str
    node: str
    alias: str
    model: str | None
    status: str
    """`ok`, or the error type / HTTP status of a failed call."""
    kind: str
    latency_seconds: float
    structured: bool
    """The call belongs to a structured-output operation."""


@dataclass(frozen=True)
class NodeMetrics:
    node: str
    calls: int
    latency_p50: float
    latency_p95: float
    attempts_mean: float
    attempts_max: int
    structured_operations: int
    no_tool_call_rate: Proportion
    """Structured operations whose first call answered without a tool call."""
    json_fallback_rate: Proportion
    """Structured operations that made another call after a first call that
    had no tool call or failed."""


def _operations(records: Sequence[CallRecord]) -> dict[tuple[str, str], list[CallRecord]]:
    grouped: dict[tuple[str, str], list[CallRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.node, record.operation_id)].append(record)
    return grouped


def node_metrics(records: Sequence[CallRecord]) -> dict[str, NodeMetrics]:
    by_node: dict[str, list[list[CallRecord]]] = defaultdict(list)
    for (node, _), calls in _operations(records).items():
        by_node[node].append(calls)

    result = {}
    for node in sorted(by_node):
        operations = by_node[node]
        flat = [call for operation in operations for call in operation]
        latencies = [call.latency_seconds for call in flat if call.status == STATUS_OK]
        structured = [op for op in operations if op[0].structured]
        no_tool_call = sum(1 for op in structured if op[0].kind == KIND_NO_TOOL_CALL)
        fallback = sum(
            1 for op in structured if len(op) > 1 and op[0].kind in (KIND_NO_TOOL_CALL, KIND_ERROR)
        )
        attempts = [len(op) for op in operations]
        result[node] = NodeMetrics(
            node=node,
            calls=len(flat),
            latency_p50=quantile(latencies, 0.5),
            latency_p95=quantile(latencies, 0.95),
            attempts_mean=sum(attempts) / len(attempts),
            attempts_max=max(attempts),
            structured_operations=len(structured),
            no_tool_call_rate=proportion(no_tool_call, len(structured)),
            json_fallback_rate=proportion(fallback, len(structured)),
        )
    return result


def model_mix(records: Sequence[CallRecord]) -> dict[str, dict[str, int]]:
    """Calls per resolved underlying model, per alias (`unknown` when the
    gateway reported none)."""
    mix: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        mix[record.alias][record.model or UNKNOWN_MODEL] += 1
    return {alias: dict(sorted(counts.items())) for alias, counts in sorted(mix.items())}

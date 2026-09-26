"""An `LLMFactory` whose models record every raw provider call.

Generalizes the smoke script's callback logger: a LangChain callback handler
sees each provider call — including native tool-calling ones, whose parsed
results carry no metadata — and records the node, the model the gateway
resolved, the status, the latency and whether the completion had a tool call.
It changes nothing about the calls."""

from __future__ import annotations

import time
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from apps.agent.llm.factory import NODE_TIER_MAP, LLMFactory
from apps.agent.llm.port import LLMPort
from apps.agent.llm.settings import Settings
from evals.core.budget import CallBudget
from evals.core.operational import (
    KIND_ERROR,
    KIND_NO_TOOL_CALL,
    KIND_PLAIN,
    KIND_TOOL_CALL,
    STATUS_OK,
    CallRecord,
)

FALLBACK_SUFFIX = "_fallback"
STRUCTURED_NODES = frozenset({"router", "compliance_guard", "offer_simulator", "knowledge_agent"})
"""Nodes that request structured output: a completion without a tool call there is
either a JSON-mode call or a native attempt where the model did not call the tool,
told apart afterwards by whether a second call of the same operation follows."""


class CallRecorder:
    """The per-call log of a run, plus the call budget it feeds."""

    def __init__(self, budget: CallBudget | None = None) -> None:
        self.records: list[CallRecord] = []
        self.budget = budget
        self._operations = 0
        self._last_operation: dict[str, str] = {}

    def begin_operation(self, node: str) -> str:
        """A new node invocation; its calls share the returned id."""
        self._operations += 1
        operation_id = f"{node}-{self._operations}"
        self._last_operation[node] = operation_id
        return operation_id

    def current_operation(self, node: str) -> str:
        """The operation a tier fallback belongs to (the primary's, same node)."""
        return self._last_operation.get(node) or self.begin_operation(node)

    def call_started(self) -> None:
        if self.budget is not None:
            self.budget.record_call()

    def add(self, record: CallRecord) -> None:
        self.records.append(record)


def _status_of(error: BaseException) -> str:
    code = getattr(error, "status_code", None)
    return str(code) if code is not None else type(error).__name__


class _CallHandler(BaseCallbackHandler):
    run_inline = True

    def __init__(
        self, recorder: CallRecorder, node: str, alias: str, operation_id: str, structured: bool
    ) -> None:
        self._recorder = recorder
        self._node = node
        self._alias = alias
        self._operation_id = operation_id
        self._structured = structured
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = time.perf_counter()
        self._recorder.call_started()

    def _latency(self, run_id: UUID) -> float:
        return time.perf_counter() - self._started.pop(run_id, time.perf_counter())

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        message = getattr(response.generations[0][0], "message", None)
        metadata = getattr(message, "response_metadata", None) or {}
        model = metadata.get("model_name") or metadata.get("model")
        if getattr(message, "tool_calls", None):
            kind = KIND_TOOL_CALL
        else:
            kind = KIND_NO_TOOL_CALL if self._structured else KIND_PLAIN
        self._recorder.add(
            CallRecord(
                self._operation_id,
                self._node,
                self._alias,
                model,
                STATUS_OK,
                kind,
                self._latency(run_id),
                self._structured,
            )
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._recorder.add(
            CallRecord(
                self._operation_id,
                self._node,
                self._alias,
                None,
                _status_of(error),
                KIND_ERROR,
                self._latency(run_id),
                self._structured,
            )
        )


class InstrumentedLLMFactory(LLMFactory):
    """Every model this factory builds logs its raw calls to `recorder`."""

    def __init__(self, settings: Settings, recorder: CallRecorder) -> None:
        super().__init__(settings)
        self._settings = settings
        self._recorder = recorder

    def _alias(self, node: str) -> str:
        tier = NODE_TIER_MAP[node]
        return self._settings.llm_model_fast if tier == "fast" else self._settings.llm_model_smart

    def _instrument(self, llm: LLMPort, node: str, alias: str, operation_id: str) -> LLMPort:
        structured = node.removesuffix(FALLBACK_SUFFIX) in STRUCTURED_NODES
        handler = _CallHandler(self._recorder, node, alias, operation_id, structured)
        cast(Any, llm).callbacks = [handler]
        return llm

    def for_node(self, node_name: str) -> LLMPort:
        operation_id = self._recorder.begin_operation(node_name)
        return self._instrument(
            super().for_node(node_name), node_name, self._alias(node_name), operation_id
        )

    def fallback_for_node(self, node_name: str) -> LLMPort | None:
        llm = super().fallback_for_node(node_name)
        if llm is None:
            return None
        return self._instrument(
            llm,
            f"{node_name}{FALLBACK_SUFFIX}",
            self._settings.llm_model_fast,
            self._recorder.current_operation(node_name),
        )

"""LangFuse tracing: one trace per conversation turn, one span per graph
node, one span per LLM call.

`FakeTracer` (used by tests and whenever tracing isn't configured) and
`LangfuseTracer` (the real adapter) share the `Tracer`/`TurnTrace`
protocol, so nothing above this module — nodes, the LLM factory, the
API layer — depends on LangFuse directly (see
`specs/observability/spec.md`).
"""

from __future__ import annotations

import contextvars
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from langfuse import Langfuse


class TurnTrace(Protocol):
    """A single conversation turn's trace."""

    def node_span(self, node_name: str) -> AbstractContextManager[None]: ...

    def llm_span(self, call_name: str) -> AbstractContextManager[None]: ...


class Tracer(Protocol):
    def turn(self, trace_id: str, **metadata: object) -> AbstractContextManager[TurnTrace]: ...


class FakeSpan:
    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name


class FakeTurn:
    """Records every span opened during a turn, for test assertions."""

    def __init__(self, trace_id: str) -> None:
        self.trace_id = trace_id
        self.spans: list[FakeSpan] = []

    @contextmanager
    def node_span(self, node_name: str) -> Iterator[None]:
        self.spans.append(FakeSpan("node", node_name))
        yield

    @contextmanager
    def llm_span(self, call_name: str) -> Iterator[None]:
        self.spans.append(FakeSpan("llm", call_name))
        yield

    @property
    def node_span_count(self) -> int:
        return sum(1 for span in self.spans if span.kind == "node")

    @property
    def llm_span_count(self) -> int:
        return sum(1 for span in self.spans if span.kind == "llm")


class FakeTracer:
    """Records every turn started, for test assertions. Never calls out
    to a real LangFuse backend."""

    def __init__(self) -> None:
        self.turns: list[FakeTurn] = []

    @contextmanager
    def turn(self, trace_id: str, **metadata: object) -> Iterator[FakeTurn]:
        turn = FakeTurn(trace_id)
        self.turns.append(turn)
        yield turn


class LangfuseTurn:
    def __init__(self, client: Langfuse, trace_id: str) -> None:
        self._client = client
        self._trace_id = trace_id

    @contextmanager
    def node_span(self, node_name: str) -> Iterator[None]:
        with self._client.start_as_current_observation(
            as_type="span", name=node_name, trace_context={"trace_id": self._trace_id}
        ):
            yield

    @contextmanager
    def llm_span(self, call_name: str) -> Iterator[None]:
        with self._client.start_as_current_observation(
            as_type="generation", name=call_name, trace_context={"trace_id": self._trace_id}
        ):
            yield


class LangfuseTracer:
    """Real adapter over the LangFuse SDK. Never receives secrets or
    infra details — only node/call names and the trace id (see
    `specs/observability/spec.md` — "Traces never contain secrets or
    infra details")."""

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    @contextmanager
    def turn(self, trace_id: str, **metadata: object) -> Iterator[LangfuseTurn]:
        with self._client.start_as_current_observation(
            as_type="span", name="conversation_turn", trace_context={"trace_id": trace_id}
        ):
            yield LangfuseTurn(self._client, trace_id)
        self._client.flush()


def build_tracer() -> Tracer:
    """`LangfuseTracer` when `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
    are configured; `FakeTracer` (in-memory, no network calls)
    otherwise — so the demo runs locally without requiring a LangFuse
    account."""
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse import Langfuse

        return LangfuseTracer(Langfuse())
    return FakeTracer()


_current_turn: contextvars.ContextVar[TurnTrace | None] = contextvars.ContextVar(
    "current_turn", default=None
)


def get_current_turn() -> TurnTrace | None:
    """The active turn's trace, if tracing is configured for this call —
    `None` in tests and anywhere tracing isn't wired up, in which case
    callers skip span creation entirely (no-op, never an error)."""
    return _current_turn.get()


def set_current_turn(turn: TurnTrace | None) -> contextvars.Token[TurnTrace | None]:
    return _current_turn.set(turn)


def reset_current_turn(token: contextvars.Token[TurnTrace | None]) -> None:
    _current_turn.reset(token)

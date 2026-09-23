"""Per-turn LLM deadline: one budget shared by every LLM operation in a
turn, including retries, the JSON-mode fallback, and the smart -> fast
tier degradation.

The API layer opens `turn_deadline(seconds)` around a turn; the LLM
entry points (`ainvoke_structured`, `invoke_with_resilience`) run under
`run_within_deadline`, which cancels whatever is in flight when the
budget is spent and raises `LLMDeadlineExceeded`. Without an active
deadline (tests, manual scripts) nothing changes. See
`docs/adr/ADR-005-llm-latency-and-timeouts.md`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from apps.agent.llm.errors import LLMDeadlineExceeded

_DEADLINE: ContextVar[float | None] = ContextVar("llm_turn_deadline", default=None)


@contextmanager
def turn_deadline(seconds: float) -> Iterator[None]:
    """Start a turn's LLM budget. A deadline that is already active is
    kept, so a nested caller can never extend the outer turn's budget."""
    if _DEADLINE.get() is not None:
        yield
        return
    token = _DEADLINE.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining_seconds() -> float | None:
    """Seconds left in the active turn budget, or `None` if there is none."""
    deadline = _DEADLINE.get()
    return None if deadline is None else deadline - time.monotonic()


async def run_within_deadline[T](operation: Callable[[], Awaitable[T]]) -> T:
    """Run `operation` inside the active turn budget; raise
    `LLMDeadlineExceeded` (cancelling the operation) if it runs out."""
    remaining = remaining_seconds()
    if remaining is None:
        return await operation()
    if remaining <= 0:
        raise LLMDeadlineExceeded("turn LLM deadline already exhausted")
    try:
        async with asyncio.timeout(remaining) as scope:
            return await operation()
    except TimeoutError as exc:
        if scope.expired():
            raise LLMDeadlineExceeded("turn LLM deadline exceeded") from exc
        raise

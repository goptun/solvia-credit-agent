"""Error classification for the retry policy: only transient failures are
retried (timeouts, connection errors, 429, 5xx, and an empty completion
with `finish_reason == "length"`); 4xx and validation errors are not
retried by the generic retry loop (see `design.md` — "Gateway's
free-tier Gemini models return inconsistent structured output" and
`docs/infra-assessment.md`'s reasoning-token finding).
"""

from __future__ import annotations

import httpx
import httpx2
import openai
from google.genai import errors as genai_errors


class RetryableLLMError(Exception):
    """Raised internally for a condition that should be retried even
    though the underlying call itself didn't raise — currently only an
    empty completion with `finish_reason == "length"`."""


class LLMDeadlineExceeded(Exception):
    """The turn's LLM budget (`LLM_TURN_DEADLINE_SECONDS`) ran out while
    retries, the JSON-mode fallback, or tier degradation were still in
    progress. Never retried and never degraded further."""


class StructuredOutputError(Exception):
    """Raised when structured output could not be extracted after the
    native tool-calling attempt and the single JSON-mode repair retry."""


def is_transient(exc: BaseException) -> bool:
    """Whether `exc` represents a transient failure worth retrying."""
    if isinstance(exc, RetryableLLMError):
        return True
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    # `openai`'s installed version transports over `httpx2` (a separate
    # package, not an alias of `httpx`) — its own timeout/connection
    # errors are usually translated into `openai.API*Error` before
    # reaching us, but this is a defensive fallback in case one doesn't.
    if isinstance(exc, httpx2.TimeoutException | httpx2.ConnectError):
        return True
    if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError | openai.RateLimitError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return False

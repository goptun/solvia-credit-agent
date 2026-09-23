"""Structured (JSON) logging, correlated with the LangFuse trace id.

See `specs/observability/spec.md` — "Structured logging with
correlation id" and "Secret and PII redaction in logs".
"""

from __future__ import annotations

import logging
from typing import Any

import structlog
from structlog.types import EventDict

_REDACTED_KEYS = frozenset(
    {
        "api_key",
        "llm_api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "cpf",
        "document_number",
        "account_number",
        "card_number",
        "phone",
        "email",
    }
)
REDACTED_VALUE = "[REDACTED]"


def redact_sensitive_fields(logger: object, method_name: str, event_dict: EventDict) -> EventDict:
    """Redact known secret/PII keys, anywhere in the event dict (including
    one level of nested dicts, e.g. an `extra={...}` payload)."""

    def _redact(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (REDACTED_VALUE if key.lower() in _REDACTED_KEYS else _redact(val))
                for key, val in value.items()
            }
        return value

    return {
        key: (REDACTED_VALUE if key.lower() in _REDACTED_KEYS else _redact(value))
        for key, value in event_dict.items()
    }


def configure_logging(json_output: bool = True) -> None:
    """Configure structlog for the whole process.

    `trace_id` correlation is handled by `structlog.contextvars` —
    callers bind it with `structlog.contextvars.bind_contextvars(trace_id=...)`
    for the duration of a conversation turn (see
    `apps.agent.observability.tracing`).
    """
    renderer = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            redact_sensitive_fields,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]

"""Structured logging: trace-id correlation and secret/PII redaction."""

from __future__ import annotations

import json
from typing import Any

import pytest
import structlog

from apps.agent.observability.logging import configure_logging


def _emit_and_capture(capsys: pytest.CaptureFixture[str], **kwargs: Any) -> dict[str, Any]:
    logger = structlog.get_logger("test")
    logger.info("test-event", **kwargs)
    captured = capsys.readouterr()
    result: dict[str, Any] = json.loads(captured.out.strip().splitlines()[-1])
    return result


def test_log_line_carries_the_bound_trace_id(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    structlog.contextvars.bind_contextvars(trace_id="trace-123")
    try:
        record = _emit_and_capture(capsys)
    finally:
        structlog.contextvars.unbind_contextvars("trace_id")

    assert record["trace_id"] == "trace-123"


def test_api_key_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    record = _emit_and_capture(capsys, llm_api_key="sk-super-secret")

    assert record["llm_api_key"] == "[REDACTED]"


def test_authorization_header_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    record = _emit_and_capture(capsys, authorization="Bearer sk-super-secret")

    assert record["authorization"] == "[REDACTED]"


def test_customer_pii_is_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    record = _emit_and_capture(capsys, cpf="123.456.789-00", account_number="00012345-6")

    assert record["cpf"] == "[REDACTED]"
    assert record["account_number"] == "[REDACTED]"


def test_nested_extra_payload_is_also_redacted(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    record = _emit_and_capture(capsys, extra={"api_key": "sk-nested-secret", "node": "router"})

    assert record["extra"]["api_key"] == "[REDACTED]"
    assert record["extra"]["node"] == "router"


def test_non_sensitive_fields_pass_through_unmodified(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    record = _emit_and_capture(capsys, node="router", intent="loan_simulation")

    assert record["node"] == "router"
    assert record["intent"] == "loan_simulation"

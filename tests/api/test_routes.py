"""Conversation endpoint: customer_id binding, restricted SSE streaming,
and health checks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, BaseMessage

from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE
from apps.agent.nodes.compliance_guard import ApprovalPromiseCheck
from apps.agent.nodes.router import RouterDecision
from apps.agent.observability.logging import configure_logging
from apps.agent.synthetic_data.models import (
    Consent,
    ConsentStatus,
    Customer,
    CustomerProfile,
)
from tests.api.conftest import build_test_app, parse_sse

_CUSTOMER = Customer(
    customer_id="cust-1",
    name="Ana Souza",
    cpf="111.222.333-44",
    profile=CustomerProfile.SALARIED,
    accounts=[],
    credit_cards=[],
    transactions=[],
    consent=Consent(
        consent_id="consent-1",
        customer_id="cust-1",
        status=ConsentStatus.VALID,
        scope=[],
        granted_at=date(2026, 1, 1),
        expires_at=date(2026, 12, 31),
    ),
)


def _out_of_scope_llm() -> FakeLLM:
    return FakeLLM(
        responses=[
            RouterDecision(intent="out_of_scope"),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )


async def test_new_conversation_streams_progress_then_a_final_reply() -> None:
    app = build_test_app(fast_llm=_out_of_scope_llm(), customer=_CUSTOMER)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/conversations/conv-1/messages",
            json={"customer_id": "cust-1", "message": "oi"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)

    assert events[-1]["type"] == "final"
    assert "reply" in events[-1]["data"]
    for event in events[:-1]:
        assert event["type"] in ("node_started", "node_finished")


async def test_stream_never_carries_reply_content_before_the_final_event() -> None:
    app = build_test_app(fast_llm=_out_of_scope_llm(), customer=_CUSTOMER)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/conversations/conv-2/messages",
            json={"customer_id": "cust-1", "message": "oi"},
        )

    events = parse_sse(response.text)
    final_events = [e for e in events if e["type"] == "final"]
    assert len(final_events) == 1
    assert events[-1] is final_events[0]

    for event in events[:-1]:
        assert "data" not in event
        assert "reply" not in event
        assert set(event.keys()) == {"type", "node"}


async def test_new_conversation_without_customer_id_is_rejected() -> None:
    app = build_test_app(fast_llm=_out_of_scope_llm(), customer=_CUSTOMER)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/conversations/conv-3/messages", json={"message": "oi"})

    assert response.status_code == 422


async def test_unknown_customer_id_is_rejected() -> None:
    app = build_test_app(fast_llm=_out_of_scope_llm(), customer=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/conversations/conv-4/messages",
            json={"customer_id": "cust-unknown", "message": "oi"},
        )

    assert response.status_code == 404


async def test_continuing_conversation_without_customer_id_reuses_the_bound_one() -> None:
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="out_of_scope"),
            ApprovalPromiseCheck(promises_approval=False),
            RouterDecision(intent="out_of_scope"),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    app = build_test_app(fast_llm=fast_llm, customer=_CUSTOMER)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/conversations/conv-5/messages",
            json={"customer_id": "cust-1", "message": "oi"},
        )
        assert first.status_code == 200

        second = await client.post("/conversations/conv-5/messages", json={"message": "de novo"})
        assert second.status_code == 200


async def test_mismatched_customer_id_on_existing_conversation_is_rejected() -> None:
    fast_llm = FakeLLM(
        responses=[
            RouterDecision(intent="out_of_scope"),
            ApprovalPromiseCheck(promises_approval=False),
        ]
    )
    app = build_test_app(fast_llm=fast_llm, customer=_CUSTOMER)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post(
            "/conversations/conv-6/messages",
            json={"customer_id": "cust-1", "message": "oi"},
        )
        assert first.status_code == 200

        second = await client.post(
            "/conversations/conv-6/messages",
            json={"customer_id": "cust-outro", "message": "oi de novo"},
        )
        assert second.status_code == 409


class _RaisesStructured:
    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> Any:
        raise ValueError("this fake never supports function calling")


class _AlwaysFailsEverythingLLM:
    """Structured calls raise; plain calls return text that never parses
    as JSON — every JSON-mode attempt (including the one repair retry)
    fails, so the node's call to `ainvoke_structured` raises
    `StructuredOutputError`, which the API layer must catch and log."""

    async def ainvoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AIMessage:
        return AIMessage(content="isto nunca vira JSON, e contém 'oi' junto")

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> _AlwaysFailsEverythingLLM:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _RaisesStructured:
        return _RaisesStructured()


async def test_unhandled_node_exception_is_logged_without_content_or_secrets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    app = build_test_app(fast_llm=_AlwaysFailsEverythingLLM(), customer=_CUSTOMER)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/conversations/conv-fail/messages",
            json={"customer_id": "cust-1", "message": "oi"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[-1]["type"] == "final"
    assert events[-1]["data"]["reply"] == UNAVAILABLE_MESSAGE

    captured = capsys.readouterr()
    log_lines = [json.loads(line) for line in captured.out.strip().splitlines() if line.strip()]
    error_records = [line for line in log_lines if line.get("event") == "conversation_turn_failed"]
    assert len(error_records) == 1

    record = error_records[0]
    assert record["exception_type"] == "StructuredOutputError"
    assert record["node"] == "router"
    assert record.get("trace_id")

    # No conversation content or secrets in the record — only structured,
    # known-safe fields.
    serialized = json.dumps(record)
    assert "oi" not in serialized
    assert "nunca vira JSON" not in serialized


async def test_health_live_is_always_ok() -> None:
    app = build_test_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_ready_reflects_gateway_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    import apps.api.routes as routes_module

    async def _reachable(settings: object, timeout_seconds: float = 5.0) -> bool:
        return True

    async def _unreachable(settings: object, timeout_seconds: float = 5.0) -> bool:
        return False

    app = build_test_app()

    monkeypatch.setattr(routes_module, "check_gateway_reachable", _reachable)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "gateway_reachable": True}

    monkeypatch.setattr(routes_module, "check_gateway_reachable", _unreachable)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "not_ready", "gateway_reachable": False}

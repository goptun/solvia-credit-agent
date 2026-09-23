"""HTTP routes: the streamed conversation endpoint and health checks.

See `specs/conversation-api/spec.md`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from apps.agent.llm.health import check_gateway_reachable
from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE
from apps.agent.observability.tracing import reset_current_turn, set_current_turn
from apps.agent.state import ConversationState
from apps.api.context import AppContext
from apps.api.events import final, node_finished, node_started
from apps.api.schemas import LiveResponse, MessageRequest, ReadyResponse

router = APIRouter()


def _context(request: Request) -> AppContext:
    return request.app.state.context  # type: ignore[no-any-return]


@router.post("/conversations/{conversation_id}/messages")
async def post_message(
    conversation_id: str, body: MessageRequest, request: Request
) -> StreamingResponse:
    context = _context(request)
    config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}

    snapshot = await context.graph.aget_state(config)
    existing_customer_id = snapshot.values.get("customer_id") if snapshot.values else None

    if existing_customer_id is None:
        if not body.customer_id:
            raise HTTPException(
                status_code=422, detail="customer_id is required on the first message"
            )
        if context.customer_repository.get(body.customer_id) is None:
            raise HTTPException(status_code=404, detail="unknown customer_id")
        graph_input = ConversationState(
            customer_id=body.customer_id, messages=[HumanMessage(content=body.message)]
        )
    else:
        if body.customer_id and body.customer_id != existing_customer_id:
            raise HTTPException(
                status_code=409,
                detail="customer_id does not match the customer already bound to this conversation",
            )
        if context.customer_repository.get(existing_customer_id) is None:
            raise HTTPException(status_code=404, detail="unknown customer_id")
        graph_input = ConversationState(messages=[HumanMessage(content=body.message)])

    trace_id = str(uuid.uuid4())

    async def event_stream() -> AsyncIterator[str]:
        reply = UNAVAILABLE_MESSAGE
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
        try:
            with context.tracer.turn(trace_id, conversation_id=conversation_id) as turn:
                token = set_current_turn(turn)
                try:
                    async for event in context.graph.astream(
                        graph_input, config=config, stream_mode="debug"
                    ):
                        if event.get("type") == "task":
                            yield node_started(event["payload"]["name"])
                        elif event.get("type") == "task_result":
                            yield node_finished(event["payload"]["name"])
                finally:
                    reset_current_turn(token)
            final_snapshot = await context.graph.aget_state(config)
            reply = final_snapshot.values.get("draft_reply") or UNAVAILABLE_MESSAGE
        except Exception:
            reply = UNAVAILABLE_MESSAGE
        finally:
            structlog.contextvars.unbind_contextvars("trace_id")
        yield final(reply)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/health/live")
async def health_live() -> LiveResponse:
    return LiveResponse(status="ok")


@router.get("/health/ready")
async def health_ready(request: Request) -> ReadyResponse:
    context = _context(request)
    reachable = await check_gateway_reachable(context.llm_settings)
    return ReadyResponse(status="ready" if reachable else "not_ready", gateway_reachable=reachable)

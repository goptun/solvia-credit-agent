"""SSE event formatting.

Only two event shapes ever reach the client: `node_started`/
`node_finished` (node name only, no message content) and a single
terminal `final` event carrying the reply produced after
`compliance_guard` — never raw or intermediate model output (see
`specs/conversation-api/spec.md` — "Streaming carries no unguarded
model output").
"""

from __future__ import annotations

import json
from typing import Any, Literal


def format_event(
    event_type: Literal["node_started", "node_finished", "final"], data: dict[str, Any]
) -> str:
    payload = {"type": event_type, **data}
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def node_started(node: str) -> str:
    return format_event("node_started", {"node": node})


def node_finished(node: str) -> str:
    return format_event("node_finished", {"node": node})


def final(reply: str) -> str:
    return format_event("final", {"data": {"reply": reply}})

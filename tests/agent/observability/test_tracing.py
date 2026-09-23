"""FakeTracer: one trace per turn, one span per node, one span per LLM call."""

from __future__ import annotations

from apps.agent.observability.tracing import FakeTracer, get_current_turn, set_current_turn


def test_turn_creates_one_trace() -> None:
    tracer = FakeTracer()

    with tracer.turn("trace-1") as turn:
        assert turn.trace_id == "trace-1"

    assert len(tracer.turns) == 1


def test_node_span_is_recorded() -> None:
    tracer = FakeTracer()

    with tracer.turn("trace-1") as turn:
        with turn.node_span("router"):
            pass
        with turn.node_span("responder"):
            pass

    assert turn.node_span_count == 2
    assert [span.name for span in turn.spans if span.kind == "node"] == ["router", "responder"]


def test_llm_span_is_recorded() -> None:
    tracer = FakeTracer()

    with tracer.turn("trace-1") as turn:
        with turn.llm_span("router"):
            pass

    assert turn.llm_span_count == 1


def test_span_counts_match_nodes_and_llm_calls_executed_for_a_turn() -> None:
    """A turn touching 3 nodes, 2 of which also make an LLM call, ends up
    with exactly 3 node spans and 2 llm spans — nothing more, nothing less."""
    tracer = FakeTracer()

    with tracer.turn("trace-1") as turn:
        with turn.node_span("router"):
            with turn.llm_span("router"):
                pass
        with turn.node_span("consent_check"):
            pass  # no LLM call in this node
        with turn.node_span("compliance_guard"):
            with turn.llm_span("compliance_guard"):
                pass

    assert turn.node_span_count == 3
    assert turn.llm_span_count == 2
    assert len(turn.spans) == 5


def test_current_turn_contextvar_defaults_to_none() -> None:
    assert get_current_turn() is None


def test_current_turn_contextvar_can_be_set_and_read() -> None:
    tracer = FakeTracer()
    with tracer.turn("trace-1") as turn:
        token = set_current_turn(turn)
        try:
            assert get_current_turn() is turn
        finally:
            from apps.agent.observability.tracing import reset_current_turn

            reset_current_turn(token)

    assert get_current_turn() is None

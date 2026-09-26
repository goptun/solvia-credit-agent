"""Manual smoke test against the real `9router` gateway.

**Not part of the automated test suite** — it is outside `tests/` (so
pytest never collects it) and is never invoked by CI (see
`.github/workflows/ci.yml`), since it makes real network calls to a
private gateway CI cannot reach. See README.md — "Manual smoke test
against the real gateway".

Requires:
  - The SSH tunnel to the gateway running (README — "Local development
    against the gateway").
  - `.env` configured with a real `LLM_BASE_URL`/`LLM_API_KEY`
    (`LLM_PROVIDER` left as the default `openai_compatible`).
  - `DATABASE_URL` pointing at a Postgres with the regulatory corpus
    already migrated and indexed (`python -m rag.migrate`, then
    `python -m rag.ingest fetch`/`index`) — the `product_question` turn
    below routes through real retrieval, not a static catalog.

Run with (from the repository root): `PYTHONPATH=. uv run python scripts/smoke_gateway.py`
(`PYTHONPATH=.` is needed because this project isn't installed as a
package — see `pyproject.toml`'s `[tool.uv] package = false`).

For each of the five classifiable intents, plus a multi-turn loan
simulation scenario (missing consent -> authorization -> slot
follow-up -> completed simulation), sends real messages through the
full graph and prints, per turn: the intent classified, the node path,
`reply_status` (`ok` | `unavailable` | `empty`), the resolved
underlying model for each LLM call when available, and the first 120
characters of the final reply (400 when a call resolved to gpt-oss), plus
per LLM call: node, whether the completion was a native tool call, a completion with
no tool call on a structured node (JSON mode, or a native attempt that
returned no tool call), or plain content, the resolved model, and
latency. Replies are drawn entirely from the
synthetic dataset and templates, so printing a short prefix locally is
fine — this script's *output* is never committed or written to a file,
only the script itself.

Also calls `ground_answer` directly (the same grounding/citation
function the `knowledge_agent` node and the standalone
`/knowledge/answer` endpoint both use) with a real answerable
regulatory question and a deliberately unanswerable "near miss"
question, printing every rendered citation and verifying it against
that turn's real retrieved chunk set — never an invented citation, or
a citation the retrieval step didn't actually surface.

Exits non-zero if any turn is `unavailable`/`empty`, if the multi-turn
scenario's assertions fail, if the regulatory question is refused, if
the unanswerable question is answered instead of refused, or if any
citation can't be matched back to that turn's retrieved chunks.
"""

from __future__ import annotations

import asyncio
import functools
import sys
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, cast
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import ConnectionPool

from apps.agent.checkpointer import build_serde
from apps.agent.graph import build_graph
from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.port import LLMPort
from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE
from apps.agent.llm.settings import get_settings
from apps.agent.nodes.knowledge_agent import (
    REFUSAL_REPLY,
    Citation,
    GroundedAnswer,
    RetrieveFn,
    ground_answer,
    make_knowledge_agent_node,
)

# Reused only for verifying that the responder's rendered numbers match
# the simulation tool's own output — not re-implementing the formatting
# logic here would risk silently drifting out of sync with it.
from apps.agent.nodes.responder import _format_brl, _format_percent
from apps.agent.repositories.customers import InMemoryCustomerRepository
from apps.agent.state import ConversationState
from apps.api.settings import get_api_settings
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.retrieval.live import hybrid_search_async
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings, get_rag_settings

_VALID_CONSENT_CUSTOMER = "cust-0000"
_MISSING_CONSENT_CUSTOMER = "cust-0007"

_SINGLE_TURNS = [
    ("product_question", "Quais produtos de crédito vocês oferecem?"),
    ("loan_simulation", "Quero simular um empréstimo de R$5000 em 12 meses, no sistema Price."),
    ("profile_analysis", "Você pode analisar o meu perfil financeiro?"),
    ("complaint", "Estou muito insatisfeito com o atendimento que recebi."),
    ("out_of_scope", "Qual é a previsão do tempo para amanhã?"),
]

# A genuine regulatory question the corpus should support (CDC art.
# 54-A, over-indebtedness), and a deliberately unanswerable "near
# miss" that sounds in-domain but isn't covered by the corpus — the
# fictional Solvia product catalog has no international card, and
# nothing in the corpus documents one. A first, more colloquial phrasing
# ("fico superendividado") scored below `RAG_MIN_RELEVANCE_SCORE` in a real
# run against this corpus (best similarity 0.451, correctly refused —
# expected behavior for colloquial phrasing, not a bug), so this script
# sticks to a question whose similarity was verified to clear the
# threshold (0.747). Retrieval and grounding quality are measured by
# `python -m evals`, not here.
_REGULATORY_QUESTION = (
    "O que caracteriza uma situação de superendividamento segundo o código de defesa do consumidor?"
)
_UNANSWERABLE_QUESTION = "A Solvia oferece cartão de crédito internacional sem anuidade?"


_STRUCTURED_NODES = {"router", "compliance_guard", "offer_simulator", "knowledge_agent"}
"""Nodes that request structured output: for these, a completion without a
tool call is either a JSON-mode call or a native attempt where the model
did not call the tool — told apart only by whether a second call from
the same node follows in the turn."""


@dataclass(frozen=True)
class LLMCallRecord:
    node: str
    resolved_model: str | None
    kind: str  # "native" | "no_tool_call" | "plain" | "error:<Type>"
    latency_seconds: float


@dataclass
class _CallLog:
    records: list[LLMCallRecord] = field(default_factory=list)


class _CallbackLog(BaseCallbackHandler):
    """Logs every raw provider call (through LangChain callbacks, so
    native tool-calling calls — whose parsed results carry no metadata —
    are seen too): resolved model, latency, and whether the completion
    was a native tool call or plain content."""

    run_inline = True

    def __init__(self, node: str, log: _CallLog) -> None:
        self._node = node
        self._log = log
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        latency = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        message = getattr(response.generations[0][0], "message", None)
        metadata = getattr(message, "response_metadata", None) or {}
        model = metadata.get("model_name") or metadata.get("model")
        if getattr(message, "tool_calls", None):
            kind = "native"
        elif self._node.removesuffix("_fallback") in _STRUCTURED_NODES:
            kind = "no_tool_call"
        else:
            kind = "plain"
        self._log.records.append(LLMCallRecord(self._node, model, kind, latency))

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        latency = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        self._log.records.append(
            LLMCallRecord(self._node, None, f"error:{type(error).__name__}", latency)
        )


class SpyLLMFactory(LLMFactory):
    """An `LLMFactory` whose models log every raw LLM call, for this
    script's diagnostic reporting only — never used by the real app or tests."""

    def __init__(self, settings: Any, log: _CallLog) -> None:
        super().__init__(settings)
        self._log = log

    def _instrument(self, llm: LLMPort, node_name: str) -> LLMPort:
        cast(Any, llm).callbacks = [_CallbackLog(node_name, self._log)]
        return llm

    def for_node(self, node_name: str) -> LLMPort:
        return self._instrument(super().for_node(node_name), node_name)

    def fallback_for_node(self, node_name: str) -> LLMPort | None:
        llm = super().fallback_for_node(node_name)
        if llm is None:
            return None
        return self._instrument(llm, f"{node_name}_fallback")


@dataclass(frozen=True)
class TurnResult:
    intent: str | None
    node_path: list[str]
    reply: str | None
    reply_status: str  # "ok" | "unavailable" | "empty"
    resolved_models: list[LLMCallRecord]
    state_values: dict[str, Any]
    latency_seconds: float = 0.0
    error: str | None = None


def _reply_status(reply: str | None) -> str:
    if not reply or not reply.strip():
        return "empty"
    if reply.strip() == UNAVAILABLE_MESSAGE:
        return "unavailable"
    return "ok"


def _format_calls(records: list[LLMCallRecord]) -> str:
    return "; ".join(
        f"{r.node}[{r.kind}] {r.resolved_model or '?'} {r.latency_seconds:.1f}s" for r in records
    )


def _print_turn(label: str, result: TurnResult) -> None:
    used_gpt_oss = any("gpt-oss" in (r.resolved_model or "") for r in result.resolved_models)
    preview = (result.reply or "")[: 400 if used_gpt_oss else 120].replace("\n", " ")
    print(
        f"[{label}] intent={result.intent!r} status={result.reply_status} "
        f"latency={result.latency_seconds:.1f}s node_path={' -> '.join(result.node_path)}"
    )
    if result.resolved_models:
        print(f"    llm calls: {_format_calls(result.resolved_models)}")
    if used_gpt_oss:
        print("    NOTE: at least one call in this turn resolved to gpt-oss")
    if result.error:
        print(f"    error: {result.error}")
    print(f"    reply[:{400 if used_gpt_oss else 120}]: {preview!r}")


async def _run_turn(
    graph: CompiledStateGraph[ConversationState, None, ConversationState, ConversationState],
    call_log: _CallLog,
    thread_id: str,
    message: str,
    customer_id: str | None = None,
) -> TurnResult:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    node_path: list[str] = []
    before = len(call_log.records)
    error: str | None = None

    turn_input = ConversationState(messages=[HumanMessage(content=message)])
    if customer_id is not None:
        turn_input["customer_id"] = customer_id

    started = time.perf_counter()
    try:
        async for event in graph.astream(turn_input, config=config, stream_mode="debug"):
            if event.get("type") == "task":
                node_path.append(event["payload"]["name"])
    except Exception as exc:  # noqa: BLE001 - a smoke test reports, never crashes
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started

    final_state = await graph.aget_state(config)
    reply = final_state.values.get("draft_reply")
    return TurnResult(
        intent=final_state.values.get("intent"),
        node_path=node_path,
        reply=reply,
        reply_status="unavailable" if error else _reply_status(reply),
        resolved_models=call_log.records[before:],
        state_values=dict(final_state.values),
        latency_seconds=latency,
        error=error,
    )


async def _run_single_turn_scenarios(
    graph: CompiledStateGraph[ConversationState, None, ConversationState, ConversationState],
    call_log: _CallLog,
) -> bool:
    print("== Single-turn intents ==")
    all_ok = True
    for label, message in _SINGLE_TURNS:
        result = await _run_turn(
            graph,
            call_log,
            thread_id=f"smoke-{label}",
            message=message,
            customer_id=_VALID_CONSENT_CUSTOMER,
        )
        _print_turn(label, result)
        if result.reply_status != "ok":
            all_ok = False
    return all_ok


async def _run_multi_turn_loan_simulation_scenario(
    graph: CompiledStateGraph[ConversationState, None, ConversationState, ConversationState],
    call_log: _CallLog,
) -> bool:
    print("\n== Multi-turn loan simulation (missing consent) ==")
    thread_id = "smoke-multi-turn-loan"
    ok = True

    turn1 = await _run_turn(
        graph,
        call_log,
        thread_id,
        "Quero simular um empréstimo",
        customer_id=_MISSING_CONSENT_CUSTOMER,
    )
    _print_turn("consent-request", turn1)
    active_flow = turn1.state_values.get("active_flow")
    if turn1.reply_status != "ok" or active_flow != "consent_confirmation":
        print("    ASSERTION FAILED: expected active_flow == 'consent_confirmation'")
        ok = False

    turn2 = await _run_turn(graph, call_log, thread_id, "sim, autorizo")
    _print_turn("consent-granted", turn2)
    if turn2.reply_status != "ok" or turn2.state_values.get("consent_status") != "just_granted":
        print("    ASSERTION FAILED: expected consent_status == 'just_granted'")
        ok = False
    if turn2.state_values.get("simulation_result") is not None:
        print("    ASSERTION FAILED: expected a slot follow-up, not a completed simulation yet")
        ok = False

    turn3 = await _run_turn(graph, call_log, thread_id, "10 mil em 24 meses pela tabela Price")
    _print_turn("simulation-complete", turn3)

    simulation_result = turn3.state_values.get("simulation_result")
    if turn3.reply_status != "ok" or simulation_result is None:
        print("    ASSERTION FAILED: expected a completed simulation_result")
        return False

    expected_cet = _format_percent(simulation_result.cet_annual)
    expected_installment = _format_brl(simulation_result.first_installment)
    reply = turn3.reply or ""
    if expected_cet not in reply:
        print(f"    ASSERTION FAILED: CET {expected_cet!r} not found in the reply")
        ok = False
    if expected_installment not in reply:
        print(f"    ASSERTION FAILED: installment {expected_installment!r} not found in the reply")
        ok = False
    if simulation_result.principal != Decimal("10000"):
        print(f"    ASSERTION FAILED: expected principal 10000, got {simulation_result.principal}")
        ok = False
    if simulation_result.term_months != 24:
        print(f"    ASSERTION FAILED: expected term 24, got {simulation_result.term_months}")
        ok = False
    if simulation_result.amortization_type != "PRICE":
        print(
            "    ASSERTION FAILED: expected amortization_type PRICE, "
            f"got {simulation_result.amortization_type}"
        )
        ok = False

    if ok:
        print(
            f"    CET={expected_cet} first_installment={expected_installment} — matches the reply"
        )

    return ok


@dataclass
class _RetrievedLog:
    chunks: list[RetrievedChunk] = field(default_factory=list)


def _spy_retrieve(inner: RetrieveFn, log: _RetrievedLog) -> RetrieveFn:
    """Wraps `retrieve` to capture the chunks the last call returned, so
    the knowledge scenarios below can verify each rendered citation
    against the real retrieved set — never used by the app or tests."""

    async def spy(question: str, vector: list[float], source_type: Any) -> list[RetrievedChunk]:
        results = await inner(question, vector, source_type)
        log.chunks = list(results)
        return results

    return spy


def _find_citation_chunk(
    citation: Citation, retrieved: list[RetrievedChunk]
) -> RetrievedChunk | None:
    for chunk in retrieved:
        if (
            chunk.norm == citation.norm
            and chunk.article_ref == citation.article_ref
            and chunk.source_url == citation.source_url
        ):
            return chunk
    return None


def _print_knowledge_result(
    label: str, question: str, result: GroundedAnswer, retrieved: list[RetrievedChunk]
) -> bool:
    preview = result.answer[:120].replace("\n", " ")
    print(f"[{label}] question={question!r} refused={result.refused}")
    if result.refused:
        print(f"    reply[:120]: {preview!r}")
        return result.answer == REFUSAL_REPLY

    if not result.citations:
        print("    ASSERTION FAILED: answered but produced no citations")
        print(f"    reply[:120]: {preview!r}")
        return False

    ok = True
    for citation in result.citations:
        chunk = _find_citation_chunk(citation, retrieved)
        if chunk is None:
            print(
                "    citation NOT FOUND in retrieved set: "
                f"norm={citation.norm!r} article_ref={citation.article_ref!r}"
            )
            ok = False
        else:
            print(
                f"    citation OK: chunk_id={chunk.chunk_id!r} "
                f"norm={citation.norm!r} article_ref={citation.article_ref!r}"
            )
    print(f"    reply[:120]: {preview!r}")
    return ok


async def _run_knowledge_scenarios(
    llm_factory: SpyLLMFactory,
    call_log: _CallLog,
    embeddings: FastEmbedAdapter,
    retrieve: RetrieveFn,
    rag_settings: RagSettings,
) -> bool:
    """Exercises `ground_answer` directly — the same grounding/citation
    function the `knowledge_agent` node and the standalone
    `/knowledge/answer` endpoint both call — with a real answerable
    regulatory question and a deliberately unanswerable one, verifying
    every rendered citation against that turn's real retrieved set."""
    print("\n== Knowledge scenarios (citation verification) ==")
    log = _RetrievedLog()
    spied_retrieve = _spy_retrieve(retrieve, log)
    llm = llm_factory.for_node("knowledge_agent")
    fallback_llm = llm_factory.fallback_for_node("knowledge_agent")

    before = len(call_log.records)
    started = time.perf_counter()
    regulatory_result = await ground_answer(
        _REGULATORY_QUESTION,
        "regulation",
        llm,
        fallback_llm,
        embeddings,
        spied_retrieve,
        rag_settings,
    )
    print(f"    latency={time.perf_counter() - started:.1f}s")
    print(f"    llm calls: {_format_calls(call_log.records[before:])}")
    regulatory_ok = _print_knowledge_result(
        "regulatory_question", _REGULATORY_QUESTION, regulatory_result, log.chunks
    )
    if regulatory_result.refused:
        print("    ASSERTION FAILED: expected the regulatory question to be answered")
        regulatory_ok = False

    before = len(call_log.records)
    started = time.perf_counter()
    unanswerable_result = await ground_answer(
        _UNANSWERABLE_QUESTION,
        "product_catalog",
        llm,
        fallback_llm,
        embeddings,
        spied_retrieve,
        rag_settings,
    )
    print(f"    latency={time.perf_counter() - started:.1f}s")
    print(f"    llm calls: {_format_calls(call_log.records[before:])}")
    unanswerable_ok = _print_knowledge_result(
        "unanswerable_question", _UNANSWERABLE_QUESTION, unanswerable_result, log.chunks
    )
    if not unanswerable_result.refused:
        print("    ASSERTION FAILED: expected the unanswerable question to be refused")
        unanswerable_ok = False

    return regulatory_ok and unanswerable_ok


async def main() -> int:
    settings = get_settings()
    api_settings = get_api_settings()
    rag_settings = get_rag_settings()
    call_log = _CallLog()
    llm_factory = SpyLLMFactory(settings, call_log)
    customer_repository = InMemoryCustomerRepository.from_fixtures()

    # `product_question` and `regulatory_question` route through
    # knowledge_agent, which needs a real corpus already ingested into
    # `DATABASE_URL` (see `rag/ingest/__main__.py`'s `index` command)
    # — this script exercises the real retrieval path, not a stub.
    embeddings = FastEmbedAdapter(
        rag_settings.rag_embedding_model,
        threads=rag_settings.rag_embedding_threads,
        batch_size=rag_settings.rag_embedding_batch_size,
    )
    rag_pool = ConnectionPool(api_settings.database_url, open=True)
    try:
        retrieve = functools.partial(hybrid_search_async, rag_pool, rag_settings)
        knowledge_agent_node = make_knowledge_agent_node(
            llm_factory, embeddings, retrieve, rag_settings
        )
        graph = build_graph(
            llm_factory,
            customer_repository,
            knowledge_agent_node,
            checkpointer=MemorySaver(serde=build_serde()),
        )

        print(f"Gateway: {settings.llm_base_url} (provider={settings.llm_provider})\n")

        single_turns_ok = await _run_single_turn_scenarios(graph, call_log)
        scenario_ok = await _run_multi_turn_loan_simulation_scenario(graph, call_log)
        knowledge_ok = await _run_knowledge_scenarios(
            llm_factory, call_log, embeddings, retrieve, rag_settings
        )
    finally:
        rag_pool.close()

    success = single_turns_ok and scenario_ok and knowledge_ok
    print(f"\n{'PASS' if success else 'FAIL'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

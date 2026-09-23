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

Run with (from the repository root): `PYTHONPATH=. uv run python scripts/smoke_gateway.py`
(`PYTHONPATH=.` is needed because this project isn't installed as a
package — see `pyproject.toml`'s `[tool.uv] package = false`).

For each of the five classifiable intents, sends one real message
through the full graph and prints only the intent the router actually
classified, the sequence of nodes executed, and whether a final reply
was produced — never the reply content or the message text itself, to
avoid leaking model output into terminal history/logs from an ad-hoc
run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from apps.agent.checkpointer import build_serde
from apps.agent.graph import build_graph
from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.settings import get_settings
from apps.agent.repositories.customers import InMemoryCustomerRepository
from apps.agent.state import ConversationState

_CUSTOMER_ID = "cust-0000"  # committed fixture customer with a VALID synthetic consent

_TURNS = [
    ("product_question", "Quais produtos de crédito vocês oferecem?"),
    ("loan_simulation", "Quero simular um empréstimo de R$5000 em 12 meses, no sistema Price."),
    ("profile_analysis", "Você pode analisar o meu perfil financeiro?"),
    ("complaint", "Estou muito insatisfeito com o atendimento que recebi."),
    ("out_of_scope", "Qual é a previsão do tempo para amanhã?"),
]


@dataclass(frozen=True)
class TurnResult:
    intent: str | None
    node_path: list[str]
    final_reply_produced: bool


async def _run_turn(
    graph: CompiledStateGraph[ConversationState, None, ConversationState, ConversationState],
    thread_id: str,
    message: str,
) -> TurnResult:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    node_path: list[str] = []

    async for event in graph.astream(
        ConversationState(customer_id=_CUSTOMER_ID, messages=[HumanMessage(content=message)]),
        config=config,
        stream_mode="debug",
    ):
        if event.get("type") == "task":
            node_path.append(event["payload"]["name"])

    final_state = await graph.aget_state(config)
    return TurnResult(
        intent=final_state.values.get("intent"),
        node_path=node_path,
        final_reply_produced=bool(final_state.values.get("draft_reply")),
    )


async def main() -> None:
    settings = get_settings()
    llm_factory = LLMFactory(settings)
    customer_repository = InMemoryCustomerRepository.from_fixtures()
    graph = build_graph(
        llm_factory, customer_repository, checkpointer=MemorySaver(serde=build_serde())
    )

    print(f"Gateway: {settings.llm_base_url} (provider={settings.llm_provider})")
    print(f"Customer: {_CUSTOMER_ID}\n")

    for label, message in _TURNS:
        result = await _run_turn(graph, thread_id=f"smoke-{label}", message=message)
        print(
            f"[{label}] intent={result.intent!r} "
            f"node_path={' -> '.join(result.node_path)} "
            f"final_reply_produced={result.final_reply_produced}"
        )


if __name__ == "__main__":
    asyncio.run(main())

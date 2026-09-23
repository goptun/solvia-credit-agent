"""Router node: intent classification with active-flow skip routing.

See `design.md` — "Active-flow routing" and "Routing matrix", and
`specs/conversation-graph/spec.md` — "Active-flow routing takes
priority over reclassification".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.structured import ainvoke_structured
from apps.agent.nodes.messages import last_human_text
from apps.agent.state import ConversationState, Intent

RouterNode = Callable[[ConversationState], Awaitable[ConversationState]]

_ROUTER_INSTRUCTIONS = (
    "Você é um classificador de intenção para um assistente de crédito ao consumidor. "
    "Classifique a mensagem mais recente do cliente em exatamente uma destas categorias: "
    "product_question (pergunta sobre produtos), loan_simulation (quer simular um "
    "empréstimo), profile_analysis (quer uma análise do seu perfil financeiro), "
    "complaint (reclamação), ou out_of_scope (fora do escopo do assistente).\n\n"
    "Se houver uma solicitação pendente em andamento (active_flow != none), avalie "
    "também se a mensagem atual é claramente um NOVO pedido, sem relação com o que foi "
    "perguntado (is_new_request=true), ou se é uma resposta/continuação do que estava "
    "pendente (is_new_request=false, o valor padrão)."
)


class RouterDecision(BaseModel):
    intent: Intent
    is_new_request: bool = False


def make_router_node(llm_factory: LLMFactory) -> RouterNode:
    async def router_node(state: ConversationState) -> ConversationState:
        llm = llm_factory.for_node("router")
        fallback_llm = llm_factory.fallback_for_node("router")

        active_flow = state.get("active_flow", "none")
        prompt = [
            HumanMessage(
                content=(
                    f"{_ROUTER_INSTRUCTIONS}\n\n"
                    f"active_flow atual: {active_flow}\n"
                    f"Mensagem do cliente: {last_human_text(state)}"
                )
            )
        ]
        decision = await ainvoke_structured(llm, prompt, RouterDecision, fallback=fallback_llm)

        if active_flow != "none" and not decision.is_new_request:
            # An active flow owns this message; router does not touch
            # intent/active_flow — the flow's node (consent_check or
            # offer_simulator) handles it next.
            return ConversationState()

        updates = ConversationState(intent=decision.intent)
        if active_flow != "none" and decision.is_new_request:
            updates["active_flow"] = "none"
            updates["pending_intent"] = None
        return updates

    return router_node

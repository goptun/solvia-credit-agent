"""responder node: drafts the reply per intent, before compliance_guard.

The smart-tier LLM only ever drafts a short qualitative framing
sentence — it is never given the simulation/profile numbers, which are
rendered exclusively from tool output via the templates below (see
`design.md` — "Concrete per-intent MVP behavior" and
`specs/conversation-graph/spec.md` — "Per-intent reply drafting").
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal

from langchain_core.messages import HumanMessage

from apps.agent.config.catalog import AmortizationType
from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.resilience import invoke_with_resilience
from apps.agent.state import ConversationState, CustomerProfileSummary, SimulationSummary

ResponderNode = Callable[[ConversationState], Awaitable[ConversationState]]

_OUT_OF_SCOPE_REPLY = (
    "Desculpe, esse assunto está fora do que posso ajudar por aqui. Posso ajudar "
    "com dúvidas sobre nossos produtos de crédito, simulação de empréstimo ou "
    "análise do seu perfil financeiro."
)

_COMPLAINT_ACK_REPLY = (
    "Sinto muito pelo inconveniente. Registrei sua reclamação. O atendimento "
    "humano ainda não está disponível nesta demonstração — essa funcionalidade "
    "está planejada para uma versão futura."
)

_CONSENT_PURPOSE: dict[str, str] = {
    "loan_simulation": "simular uma oferta de crédito",
    "profile_analysis": "analisar seu perfil financeiro",
}

_CONSENT_ASK_TEMPLATE = (
    "Para {purpose}, preciso da sua autorização para acessar seus dados no Open "
    "Finance. Você autoriza o acesso? (responda algo como 'sim, autorizo')"
)
_CONSENT_REASK_REPLY = (
    "Não entendi sua resposta. Você autoriza o acesso aos seus dados no Open "
    "Finance para que eu possa continuar? (responda 'sim, autorizo' ou 'não autorizo')"
)
_CONSENT_REFUSED_REPLY = (
    "Sem problemas — sem a autorização, não consigo acessar seus dados financeiros "
    "para essa solicitação. Posso ajudar com outra coisa?"
)

_SLOT_QUESTIONS: dict[str, str] = {
    "amount": "Qual valor você gostaria de simular?",
    "term_months": "Em quantos meses você gostaria de parcelar?",
    "amortization_type": (
        "Você prefere parcelas fixas (Price) ou parcelas decrescentes ao longo do tempo (SAC)?"
    ),
}

_FALLBACK_REPLY = "Não consegui identificar como ajudar com essa mensagem."


def _format_brl(value: Decimal) -> str:
    return f"R$ {value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _format_percent(value: Decimal) -> str:
    return f"{(value * 100):.2f}%".replace(".", ",")


async def _framing_sentence(llm_factory: LLMFactory, prompt_text: str) -> str:
    llm = llm_factory.for_node("responder")
    fallback = llm_factory.fallback_for_node("responder")
    result = await invoke_with_resilience(
        llm, [HumanMessage(content=prompt_text)], fallback=fallback
    )
    content = result.message.content
    return content if isinstance(content, str) else str(content)


def make_responder_node(llm_factory: LLMFactory) -> ResponderNode:
    async def responder_node(state: ConversationState) -> ConversationState:
        if state.get("consent_prompt") == "ask":
            purpose = _CONSENT_PURPOSE.get(state.get("pending_intent") or "", "continuar")
            return ConversationState(draft_reply=_CONSENT_ASK_TEMPLATE.format(purpose=purpose))

        if state.get("consent_prompt") == "reask":
            return ConversationState(draft_reply=_CONSENT_REASK_REPLY)

        if state.get("consent_refused"):
            return ConversationState(draft_reply=_CONSENT_REFUSED_REPLY)

        missing_slot = state.get("next_missing_slot")
        if missing_slot:
            return ConversationState(draft_reply=_SLOT_QUESTIONS[missing_slot])

        intent = state.get("intent")

        if intent == "out_of_scope":
            return ConversationState(draft_reply=_OUT_OF_SCOPE_REPLY)

        if intent == "complaint":
            return ConversationState(draft_reply=_COMPLAINT_ACK_REPLY)

        simulation_result = state.get("simulation_result")
        if intent == "loan_simulation" and simulation_result is not None:
            return await _loan_simulation_reply(simulation_result, llm_factory)

        customer_profile = state.get("customer_profile")
        if intent == "profile_analysis" and customer_profile is not None:
            return await _profile_analysis_reply(customer_profile, llm_factory)

        return ConversationState(draft_reply=_FALLBACK_REPLY)

    return responder_node


async def _loan_simulation_reply(
    result: SimulationSummary, llm_factory: LLMFactory
) -> ConversationState:
    framing = await _framing_sentence(
        llm_factory,
        "Escreva uma frase curta e cordial de abertura anunciando o resultado de "
        "uma simulação de empréstimo. Não inclua nenhum número — os valores serão "
        "anexados a seguir.",
    )
    method = (
        "Price (parcelas fixas)"
        if result.amortization_type == AmortizationType.PRICE
        else "SAC (parcelas decrescentes)"
    )
    body = (
        f"Simulação ({method}):\n"
        f"- Valor solicitado: {_format_brl(result.principal)}\n"
        f"- Prazo: {result.term_months} meses\n"
        f"- Primeira parcela: {_format_brl(result.first_installment)}\n"
        f"- Total pago ao final: {_format_brl(result.total_paid)}\n"
        f"- CET (Custo Efetivo Total) anual: {_format_percent(result.cet_annual)}"
    )
    return ConversationState(draft_reply=f"{framing}\n\n{body}")


async def _profile_analysis_reply(
    profile: CustomerProfileSummary, llm_factory: LLMFactory
) -> ConversationState:
    framing = await _framing_sentence(
        llm_factory,
        "Escreva uma frase curta e cordial de abertura para um resumo de perfil "
        "financeiro. Não inclua nenhum número — os valores serão anexados a seguir.",
    )
    categories = (
        ", ".join(
            f"{name}: {_format_brl(abs(total))}"
            for name, total in profile.spending_categories.items()
        )
        or "sem dados de categoria"
    )
    body = (
        f"Resumo do seu perfil:\n"
        f"- Renda estimada: {_format_brl(profile.income)}\n"
        f"- Comprometimento de renda (DTI): {_format_percent(profile.debt_to_income_ratio)}\n"
        f"- Gastos por categoria: {categories}"
    )
    return ConversationState(draft_reply=f"{framing}\n\n{body}")

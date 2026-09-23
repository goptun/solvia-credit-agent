"""knowledge_agent node: grounded regulatory/product answers with
verified citations.

Answers `product_question` and `regulatory_question` from retrieved
corpus chunks only — never the LLM's own knowledge. See `design.md` —
"Grounding and citation validation" and
`specs/regulatory-knowledge-agent/spec.md`. This is one of only two
places (with `apps/api/routes_knowledge.py`) allowed to import from
both `apps/` and `rag/`, per `design.md` — "Module boundaries".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.structured import ainvoke_structured
from apps.agent.nodes.messages import last_human_text
from apps.agent.state import ConversationState
from rag.corpus.manifest import SourceType
from rag.embeddings.port import EmbeddingsPort
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings

KnowledgeAgentNode = Callable[[ConversationState], Awaitable[ConversationState]]

RetrieveFn = Callable[[str, list[float], SourceType], Awaitable[list[RetrievedChunk]]]
"""Retrieval, abstracted away from any particular database connection
— production wiring (see `apps/api/context.py`) runs the real hybrid
search (`rag/retrieval/hybrid.py`) against a connection pool inside
`asyncio.to_thread`; tests supply a fake that returns canned chunks."""

_REFUSAL_REPLY = (
    "Não encontrei essa informação na nossa base de conhecimento. "
    "Você poderia tentar reformular a pergunta?"
)

_INSTRUCTIONS = (
    "Você responde perguntas EXCLUSIVAMENTE com base nos trechos fornecidos abaixo — "
    "nunca use conhecimento próprio. Para cada afirmação da sua resposta, produza um "
    "claim separado com o texto da afirmação e o chunk_id do trecho que a sustenta. "
    "Só use um chunk_id que apareça exatamente nos trechos fornecidos. Se os trechos "
    "não permitirem responder à pergunta, retorne uma lista de claims vazia."
)

_INTENT_TO_SOURCE_TYPE: dict[str, SourceType] = {
    "product_question": "product_catalog",
    "regulatory_question": "regulation",
}


class Claim(BaseModel):
    text: str
    chunk_id: str


class KnowledgeAnswer(BaseModel):
    claims: list[Claim]


def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[chunk_id={chunk.chunk_id}]\n{chunk.content}" for chunk in chunks)


def _render_claim(claim: Claim, chunk_by_id: dict[str, RetrievedChunk]) -> str:
    chunk = chunk_by_id[claim.chunk_id]
    if chunk.source_type == "regulation" and chunk.norm and chunk.article_ref:
        return f"{claim.text} ({chunk.norm}, {chunk.article_ref})"
    return claim.text


def _render_reply(answer: KnowledgeAnswer, chunk_by_id: dict[str, RetrievedChunk]) -> str | None:
    """Drop any claim citing a `chunk_id` outside the retrieved set,
    then render the survivors. `None` if nothing survives."""
    valid_claims = [claim for claim in answer.claims if claim.chunk_id in chunk_by_id]
    if not valid_claims:
        return None
    return "\n\n".join(_render_claim(claim, chunk_by_id) for claim in valid_claims)


def make_knowledge_agent_node(
    llm_factory: LLMFactory,
    embeddings: EmbeddingsPort,
    retrieve: RetrieveFn,
    rag_settings: RagSettings,
) -> KnowledgeAgentNode:
    async def knowledge_agent_node(state: ConversationState) -> ConversationState:
        intent = state.get("intent")
        source_type = _INTENT_TO_SOURCE_TYPE.get(intent or "", "regulation")
        question = last_human_text(state)

        query_vector = embeddings.embed_query(question)
        results = await retrieve(question, query_vector, source_type)

        if not results:
            return ConversationState(draft_reply=_REFUSAL_REPLY)

        best_similarity = max(chunk.vector_similarity for chunk in results)
        if best_similarity < rag_settings.rag_min_relevance_score:
            return ConversationState(draft_reply=_REFUSAL_REPLY)

        chunk_by_id = {chunk.chunk_id: chunk for chunk in results}
        llm = llm_factory.for_node("knowledge_agent")
        fallback_llm = llm_factory.fallback_for_node("knowledge_agent")
        prompt = [
            HumanMessage(
                content=(
                    f"{_INSTRUCTIONS}\n\n"
                    f"Pergunta do cliente: {question}\n\n"
                    f"Trechos disponíveis:\n{_format_chunks_for_prompt(results)}"
                )
            )
        ]
        answer = await ainvoke_structured(llm, prompt, KnowledgeAnswer, fallback=fallback_llm)

        reply = _render_reply(answer, chunk_by_id)
        if reply is None:
            return ConversationState(draft_reply=_REFUSAL_REPLY)
        return ConversationState(draft_reply=reply)

    return knowledge_agent_node

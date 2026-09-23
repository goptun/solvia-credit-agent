"""Grounded regulatory/product answers with verified citations.

`ground_answer` is the single grounding/citation implementation both
the `knowledge_agent` graph node and the standalone
`apps/api/routes_knowledge.py` endpoint call — see `design.md` —
"Standalone endpoint and the LangServe decision" ("there is exactly
one grounding/citation implementation, not two"). Answers
`product_question` and `regulatory_question` from retrieved corpus
chunks only — never the LLM's own knowledge. See `design.md` —
"Grounding and citation validation" and
`specs/regulatory-knowledge-agent/spec.md`. This, `apps/api/
routes_knowledge.py`, and the app's composition layer (`apps/api/
app.py` and `apps/api/context.py`) are the only places allowed to
import from both `apps/` and `rag/`, per `design.md` — "Module
boundaries".
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.port import LLMPort
from apps.agent.llm.structured import ainvoke_structured
from apps.agent.nodes.messages import last_human_text
from apps.agent.state import ConversationState
from rag.corpus.manifest import SourceType
from rag.embeddings.port import EmbeddingsPort
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings

KnowledgeAgentNode = Callable[[ConversationState], Awaitable[ConversationState]]

RetrieveFn = Callable[[str, list[float], SourceType | None], Awaitable[list[RetrievedChunk]]]
"""Retrieval, abstracted away from any particular database connection
— production wiring (see `apps/api/app.py`) runs the real hybrid
search (`rag/retrieval/hybrid.py`) against a connection pool inside
`asyncio.to_thread`; tests supply a fake that returns canned chunks."""

REFUSAL_REPLY = (
    "Não encontrei essa informação na nossa base de conhecimento. "
    "Você poderia tentar reformular a pergunta?"
)

_INSTRUCTIONS = (
    "Você responde perguntas EXCLUSIVAMENTE com base nos trechos fornecidos abaixo — "
    "nunca use conhecimento próprio. Para cada afirmação da sua resposta, produza um "
    "claim separado com o texto da afirmação e o chunk_id do trecho que a sustenta. "
    "Só use um chunk_id que apareça exatamente nos trechos fornecidos.\n\n"
    "REGRA DE RECUSA: os trechos foram recuperados por similaridade e podem tratar de um "
    "assunto parecido sem responder à pergunta. Se os trechos NÃO respondem diretamente à "
    "pergunta feita — mesmo que o tema seja próximo — você DEVE retornar `claims` como uma "
    "lista VAZIA. Não responda parcialmente, não generalize a partir de trechos "
    "relacionados e não complete com conhecimento próprio. Uma lista vazia é a resposta "
    "correta quando não há sustentação explícita nos trechos."
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


@dataclass(frozen=True)
class Citation:
    norm: str | None
    article_ref: str | None
    source_url: str | None


@dataclass(frozen=True)
class GroundedAnswer:
    answer: str
    citations: list[Citation]
    refused: bool


def _format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[chunk_id={chunk.chunk_id}]\n{chunk.content}" for chunk in chunks)


def _render_claim(claim: Claim, chunk_by_id: dict[str, RetrievedChunk]) -> str:
    chunk = chunk_by_id[claim.chunk_id]
    if chunk.source_type == "regulation" and chunk.norm and chunk.article_ref:
        return f"{claim.text} ({chunk.norm}, {chunk.article_ref})"
    return claim.text


def _citations_for(claims: list[Claim], chunk_by_id: dict[str, RetrievedChunk]) -> list[Citation]:
    seen: dict[tuple[str | None, str | None, str | None], None] = {}
    for claim in claims:
        chunk = chunk_by_id[claim.chunk_id]
        seen.setdefault((chunk.norm, chunk.article_ref, chunk.source_url), None)
    return [Citation(norm=n, article_ref=a, source_url=s) for n, a, s in seen]


_REFUSAL = GroundedAnswer(answer=REFUSAL_REPLY, citations=[], refused=True)


async def ground_answer(
    question: str,
    source_type: SourceType | None,
    llm: LLMPort,
    fallback_llm: LLMPort | None,
    embeddings: EmbeddingsPort,
    retrieve: RetrieveFn,
    rag_settings: RagSettings,
) -> GroundedAnswer:
    """Retrieve, ground, and cite an answer to `question` — the shared
    core both `make_knowledge_agent_node` and the standalone endpoint
    use. Never answers from the LLM's own knowledge (see module
    docstring); refuses when retrieval doesn't support an answer."""
    query_vector = embeddings.embed_query(question)
    results = await retrieve(question, query_vector, source_type)
    if not results:
        return _REFUSAL

    best_similarity = max(chunk.vector_similarity for chunk in results)
    if best_similarity < rag_settings.rag_min_relevance_score:
        return _REFUSAL

    chunk_by_id = {chunk.chunk_id: chunk for chunk in results}
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

    valid_claims = [claim for claim in answer.claims if claim.chunk_id in chunk_by_id]
    if not valid_claims:
        return _REFUSAL

    rendered = "\n\n".join(_render_claim(claim, chunk_by_id) for claim in valid_claims)
    return GroundedAnswer(
        answer=rendered, citations=_citations_for(valid_claims, chunk_by_id), refused=False
    )


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

        llm = llm_factory.for_node("knowledge_agent")
        fallback_llm = llm_factory.fallback_for_node("knowledge_agent")
        result = await ground_answer(
            question, source_type, llm, fallback_llm, embeddings, retrieve, rag_settings
        )
        return ConversationState(draft_reply=result.answer)

    return knowledge_agent_node

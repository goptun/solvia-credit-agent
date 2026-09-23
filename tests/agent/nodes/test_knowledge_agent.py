"""knowledge_agent: grounded answers, citation validation, and the
similarity-based (never rank-based) refusal threshold."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from apps.agent.llm.deadline import turn_deadline
from apps.agent.llm.fake import FakeLLM
from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE
from apps.agent.nodes.knowledge_agent import (
    Claim,
    KnowledgeAnswer,
    RetrieveFn,
    make_knowledge_agent_node,
)
from apps.agent.state import ConversationState, initial_state
from rag.corpus.manifest import SourceType
from rag.embeddings.fake import FakeEmbeddings
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import RagSettings
from tests.agent.nodes.fakes import ScriptedLLMFactory, SlowLLM

_EMBEDDINGS = FakeEmbeddings()
_SETTINGS = RagSettings(rag_min_relevance_score=0.5)


def _chunk(chunk_id: str, similarity: float, *, matched_fts: bool = True) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="cdc-consolidada",
        norm="Lei nº 8.078/1990",
        source_type="regulation",
        article_ref="art. 6º",
        hierarchy_path="art. 6º",
        source_url="https://www.planalto.gov.br/test.htm",
        version_date=None,
        amendment_note=None,
        content="São direitos básicos do consumidor...",
        fused_score=1.0,
        vector_similarity=similarity,
        matched_fts=matched_fts,
    )


def _retrieve_returning(chunks: list[RetrievedChunk]) -> RetrieveFn:
    async def retrieve(
        question: str, vector: list[float], source_type: SourceType | None
    ) -> list[RetrievedChunk]:
        return chunks

    return retrieve


def _state_with_question(text: str, intent: str = "regulatory_question") -> ConversationState:
    state = initial_state("cust-0001")
    state["messages"] = [HumanMessage(content=text)]
    state["intent"] = intent  # type: ignore[typeddict-item]
    return state


def _reply(updates: ConversationState) -> str:
    draft_reply = updates.get("draft_reply")
    assert draft_reply is not None
    return draft_reply


async def test_grounded_answer_with_valid_citation_is_rendered() -> None:
    chunk = _chunk("chunk-1", similarity=0.9)
    claim = Claim(text="Você tem direito à proteção à vida.", chunk_id="chunk-1")
    smart_llm = FakeLLM(responses=[KnowledgeAnswer(claims=[claim])])
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    reply = _reply(await node(_state_with_question("Quais são meus direitos?")))

    assert "proteção à vida" in reply
    assert "Lei nº 8.078/1990" in reply
    assert "art. 6º" in reply


async def test_invented_citation_is_dropped_but_valid_ones_survive() -> None:
    chunk = _chunk("chunk-1", similarity=0.9)
    smart_llm = FakeLLM(
        responses=[
            KnowledgeAnswer(
                claims=[
                    Claim(text="Afirmação real.", chunk_id="chunk-1"),
                    Claim(text="Afirmação inventada.", chunk_id="chunk-does-not-exist"),
                ]
            )
        ]
    )
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    reply = _reply(await node(_state_with_question("Pergunta qualquer")))

    assert "Afirmação real." in reply
    assert "Afirmação inventada." not in reply


async def test_all_citations_invalid_falls_back_to_refusal() -> None:
    chunk = _chunk("chunk-1", similarity=0.9)
    smart_llm = FakeLLM(
        responses=[KnowledgeAnswer(claims=[Claim(text="Inventado.", chunk_id="chunk-ghost")])]
    )
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    reply = _reply(await node(_state_with_question("Pergunta qualquer")))

    assert "não encontrei" in reply.lower()


async def test_empty_claims_from_the_llm_is_a_refusal() -> None:
    """Above the similarity threshold but the chunks don't answer the
    question: the LLM grounding stage returns no claims and the reply
    is the refusal, not an empty or invented answer."""
    chunk = _chunk("chunk-1", similarity=0.9)
    smart_llm = FakeLLM(responses=[KnowledgeAnswer(claims=[])])
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    reply = _reply(await node(_state_with_question("Pergunta próxima mas não respondida")))

    assert "não encontrei" in reply.lower()
    assert len(smart_llm.calls) == 1


async def test_prompt_instructs_the_llm_to_return_empty_claims_when_chunks_do_not_answer() -> None:
    chunk = _chunk("chunk-1", similarity=0.9)
    smart_llm = FakeLLM(responses=[KnowledgeAnswer(claims=[])])
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    await node(_state_with_question("Qualquer pergunta"))

    (prompt,) = smart_llm.calls
    text = str(prompt[0].content)
    assert "retorne uma lista de claims vazia" in text
    assert "Se os trechos não permitirem responder" in text


async def test_low_similarity_refuses_without_calling_the_llm() -> None:
    chunk = _chunk("chunk-1", similarity=0.1)
    smart_llm = FakeLLM()  # no responses configured — a call would raise
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    reply = _reply(await node(_state_with_question("Pergunta qualquer")))

    assert "não encontrei" in reply.lower()
    assert smart_llm.calls == []


async def test_refusal_follows_similarity_not_fused_rank() -> None:
    """The top-fused-rank chunk has a low raw similarity; a lower-ranked
    chunk has a high one. The refusal decision must follow the best
    *similarity* across all results, not whichever chunk is ranked
    first — see `design.md` — "Grounding and citation validation"."""
    top_ranked_low_similarity = _chunk("chunk-top-rank", similarity=0.1)
    lower_ranked_high_similarity = _chunk("chunk-high-sim", similarity=0.9)
    # Fused-rank order: the low-similarity chunk is listed first.
    chunks = [top_ranked_low_similarity, lower_ranked_high_similarity]

    claim = Claim(text="Resposta fundamentada.", chunk_id="chunk-high-sim")
    smart_llm = FakeLLM(responses=[KnowledgeAnswer(claims=[claim])])
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm), _EMBEDDINGS, _retrieve_returning(chunks), _SETTINGS
    )

    reply = _reply(await node(_state_with_question("Pergunta qualquer")))

    assert "Resposta fundamentada." in reply


async def test_product_question_reply_has_no_regulatory_citation() -> None:
    catalog_chunk = RetrievedChunk(
        chunk_id="chunk-catalog",
        document_id="product-catalog",
        norm=None,
        source_type="product_catalog",
        article_ref=None,
        hierarchy_path="",
        source_url=None,
        version_date=None,
        amendment_note=None,
        content="Crédito pessoal Solvia, parcelas fixas.",
        fused_score=1.0,
        vector_similarity=0.9,
        matched_fts=True,
    )
    claim = Claim(text="Oferecemos crédito com parcelas fixas.", chunk_id="chunk-catalog")
    smart_llm = FakeLLM(responses=[KnowledgeAnswer(claims=[claim])])
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=smart_llm),
        _EMBEDDINGS,
        _retrieve_returning([catalog_chunk]),
        _SETTINGS,
    )
    state = _state_with_question("Quais produtos vocês têm?", intent="product_question")

    reply = _reply(await node(state))

    assert "parcelas fixas" in reply
    assert "(" not in reply


async def test_turn_deadline_expiry_returns_the_unavailable_reply() -> None:
    chunk = _chunk("chunk-1", similarity=0.9)
    slow_llm = SlowLLM(1.0, response=KnowledgeAnswer(claims=[]))
    node = make_knowledge_agent_node(
        ScriptedLLMFactory(smart=slow_llm), _EMBEDDINGS, _retrieve_returning([chunk]), _SETTINGS
    )

    with turn_deadline(0.1):
        reply = _reply(await node(_state_with_question("Pergunta qualquer")))

    assert reply == UNAVAILABLE_MESSAGE

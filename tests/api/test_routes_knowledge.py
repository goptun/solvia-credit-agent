"""Standalone `/knowledge/answer` endpoint: grounded JSON responses,
independent of a conversation."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from apps.agent.llm.fake import FakeLLM
from apps.agent.nodes.knowledge_agent import Claim, KnowledgeAnswer
from rag.retrieval.retrieved_chunk import RetrievedChunk
from tests.api.conftest import build_test_app

_CHUNK = RetrievedChunk(
    chunk_id="chunk-1",
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
    vector_similarity=0.9,
    matched_fts=True,
)


async def _retrieve_chunk(question: str, vector: list[float], source_type: object) -> list:  # type: ignore[type-arg]
    return [_CHUNK]


async def test_grounded_answer_returns_answer_citations_and_refused_fields() -> None:
    claim = Claim(text="Você tem direito à proteção à vida.", chunk_id="chunk-1")
    smart_llm = FakeLLM(responses=[KnowledgeAnswer(claims=[claim])])
    app = build_test_app(smart_llm=smart_llm, knowledge_retrieve=_retrieve_chunk)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/knowledge/answer", json={"question": "meus direitos?"})

    assert response.status_code == 200
    body = response.json()
    assert "proteção à vida" in body["answer"]
    assert body["refused"] is False
    assert body["citations"] == [
        {"norm": "Lei nº 8.078/1990", "article_ref": "art. 6º", "source_url": _CHUNK.source_url}
    ]


async def test_unanswerable_question_returns_refused_true_with_no_citations() -> None:
    app = build_test_app()  # default `_empty_retrieve` never finds a chunk

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/knowledge/answer", json={"question": "algo fora do corpus"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["citations"] == []

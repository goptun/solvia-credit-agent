"""Standalone retrieval-and-answer endpoint, independent of the
conversation graph.

See `specs/regulatory-knowledge-agent/spec.md` — "Standalone
retrieval-and-answer HTTP endpoint" and `design.md` — "Standalone
endpoint and the LangServe decision" (plain FastAPI, not LangServe —
`docs/adr/ADR-004-langserve-vs-fastapi.md`). Calls the same
`ground_answer` function the `knowledge_agent` graph node uses, so
there is exactly one grounding/citation implementation. This,
`apps/agent/nodes/knowledge_agent.py`, and the app's composition layer
(`apps/api/app.py`/`apps/api/context.py`) are the only places allowed
to import from both `apps/` and `rag/`, per `design.md` — "Module
boundaries".
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.agent.llm.deadline import turn_deadline
from apps.agent.llm.errors import LLMDeadlineExceeded
from apps.agent.llm.resilience import UNAVAILABLE_MESSAGE
from apps.agent.nodes.knowledge_agent import ground_answer
from apps.api.context import AppContext
from apps.api.schemas import CitationResponse, KnowledgeAnswerResponse, KnowledgeQuestionRequest

router = APIRouter()


def _context(request: Request) -> AppContext:
    return request.app.state.context  # type: ignore[no-any-return]


@router.post("/knowledge/answer")
async def post_knowledge_answer(
    body: KnowledgeQuestionRequest, request: Request
) -> KnowledgeAnswerResponse:
    context = _context(request)

    llm = context.llm_factory.for_node("knowledge_agent")
    fallback_llm = context.llm_factory.fallback_for_node("knowledge_agent")
    try:
        with turn_deadline(context.llm_settings.llm_turn_deadline_seconds):
            result = await ground_answer(
                body.question,
                None,  # no intent to derive a source_type from — search the whole corpus
                llm,
                fallback_llm,
                context.knowledge_embeddings,
                context.knowledge_retrieve,
                context.rag_settings,
            )
    except LLMDeadlineExceeded as exc:
        raise HTTPException(status_code=503, detail=UNAVAILABLE_MESSAGE) from exc

    return KnowledgeAnswerResponse(
        answer=result.answer,
        citations=[
            CitationResponse(norm=c.norm, article_ref=c.article_ref, source_url=c.source_url)
            for c in result.citations
        ],
        refused=result.refused,
    )

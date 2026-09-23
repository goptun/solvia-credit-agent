"""Manual end-to-end refusal evaluation against the real LLM gateway.

**Not part of the automated test suite and never run by CI** — like
`scripts/smoke_gateway.py` it makes real calls to the private gateway
(SSH tunnel required — README, "Local development against the
gateway") and needs a real `LLM_API_KEY` and a `DATABASE_URL` pointing
at a Postgres with the corpus already migrated and indexed.

Runs `ground_answer` — the same grounding/citation function the
`knowledge_agent` node and `/knowledge/answer` use — with the real LLM
over every question in `rag/eval/questions.yaml` and prints, alongside
the retrieval-only numbers (`rag.eval.run`): the end-to-end
false-refusal rate on answerable questions and refusal accuracy on
unanswerable ones (far vs. near-miss), attributing each refusal to the
similarity threshold or to the LLM grounding stage.

Run with: `PYTHONPATH=. uv run python scripts/eval_end_to_end.py`
Makes one LLM call per question that clears the threshold (at most 31).
Prints only metrics, never question or reply text.
"""

from __future__ import annotations

import asyncio
import functools
import sys

import psycopg
from psycopg_pool import ConnectionPool

from apps.agent.llm.factory import LLMFactory
from apps.agent.llm.settings import get_settings
from apps.agent.nodes.knowledge_agent import REFUSAL_REPLY, RetrieveFn, ground_answer
from apps.api.settings import get_api_settings
from rag.corpus.manifest import SourceType
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.eval.end_to_end import GroundingOutcome, format_report, run_end_to_end
from rag.eval.questions import load_questions
from rag.eval.run import _print_report, run_eval
from rag.retrieval.live import hybrid_search_async
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import get_rag_settings


async def main() -> int:
    llm_settings = get_settings()
    api_settings = get_api_settings()
    rag_settings = get_rag_settings()
    questions = load_questions()
    embeddings = FastEmbedAdapter(
        rag_settings.rag_embedding_model,
        threads=rag_settings.rag_embedding_threads,
        batch_size=rag_settings.rag_embedding_batch_size,
    )

    print("== Retrieval-only (similarity threshold decides) ==")
    with psycopg.connect(api_settings.database_url, autocommit=True) as conn:
        retrieval_report = run_eval(
            conn,
            embeddings,
            questions,
            top_k=rag_settings.rag_top_k,
            min_relevance_score=rag_settings.rag_min_relevance_score,
        )
    _print_report(retrieval_report)

    llm_factory = LLMFactory(llm_settings)
    llm = llm_factory.for_node("knowledge_agent")
    fallback_llm = llm_factory.fallback_for_node("knowledge_agent")

    pool = ConnectionPool(api_settings.database_url, open=True)
    try:
        inner: RetrieveFn = functools.partial(hybrid_search_async, pool, rag_settings)
        last_results: list[RetrievedChunk] = []

        async def spy_retrieve(
            question: str, vector: list[float], source_type: SourceType | None
        ) -> list[RetrievedChunk]:
            results = await inner(question, vector, source_type)
            last_results[:] = results
            return results

        async def ground(question: str, source_type: SourceType | None) -> GroundingOutcome:
            result = await ground_answer(
                question, source_type, llm, fallback_llm, embeddings, spy_retrieve, rag_settings
            )
            best = max((chunk.vector_similarity for chunk in last_results), default=0.0)
            below_threshold = best < rag_settings.rag_min_relevance_score
            return GroundingOutcome(
                refused=result.refused and result.answer == REFUSAL_REPLY,
                refused_by_threshold=result.refused and below_threshold,
            )

        print("\n== End-to-end (real LLM grounding, threshold + LLM stage) ==")
        report = await run_end_to_end(questions, ground)
    finally:
        pool.close()

    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

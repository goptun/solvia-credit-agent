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

Run with: `PYTHONPATH=. uv run python scripts/eval_end_to_end.py [--tier fast|smart]`
`--save-records PATH` writes the per-question decomposition (index and
cause of each false refusal, no question text) as JSON. `--ab PATH --reps N`
compares the legacy and current refusal instruction on only the questions
the decomposition attributed to the LLM (`llm_refused_gold_in_context`),
plus all unanswerable questions. `--pause-seconds N` sleeps between
LLM-reaching questions to stay under the
upstream provider's per-minute quota (a burst of back-to-back calls got
`429 quota exceeded` and made a first run's latency/error numbers
meaningless). `--tier` temporarily remaps `knowledge_agent` to a tier for this run only
(nothing is changed in `NODE_TIER_MAP` on disk) to compare smart vs. fast.
Makes one LLM call per question that clears the threshold (at most 31)
and reports LLM latency and citation hit rate. The timeout used here
comes from `LLM_TIMEOUT_SECONDS_FAST`/`_SMART` — override them for a run
that must not be cut short, but that override is not the production
value. Prints only metrics, never question or reply text.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
import time
from pathlib import Path

import psycopg
from psycopg_pool import ConnectionPool

from apps.agent.llm.factory import NODE_TIER_MAP, LLMFactory, Tier
from apps.agent.llm.settings import get_settings
from apps.agent.nodes.knowledge_agent import (
    _INSTRUCTIONS,
    LEGACY_INSTRUCTIONS,
    RetrieveFn,
    ground_answer,
)
from apps.api.settings import get_api_settings
from rag.corpus.manifest import SourceType
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.eval.end_to_end import (
    EndToEndReport,
    GroundingOutcome,
    format_report,
    run_end_to_end,
)
from rag.eval.questions import load_questions
from rag.eval.run import _print_report, run_eval
from rag.retrieval.live import hybrid_search_async
from rag.retrieval.retrieved_chunk import RetrievedChunk
from rag.settings import get_rag_settings


async def main(args: argparse.Namespace) -> int:
    tier: Tier | None = args.tier
    pause_seconds: float = args.pause_seconds
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

    if tier is not None:
        NODE_TIER_MAP["knowledge_agent"] = tier
    print(f"\nknowledge_agent tier for this run: {NODE_TIER_MAP['knowledge_agent']}")
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

        def make_ground(instructions: str):  # type: ignore[no-untyped-def]
            async def ground(question: str, source_type: SourceType | None) -> GroundingOutcome:
                return await _ground_once(question, source_type, instructions)

            return ground

        async def _ground_once(
            question: str, source_type: SourceType | None, instructions: str
        ) -> GroundingOutcome:
            start = time.perf_counter()
            result = await ground_answer(
                question,
                source_type,
                llm,
                fallback_llm,
                embeddings,
                spy_retrieve,
                rag_settings,
                instructions,
            )
            elapsed = time.perf_counter() - start
            if pause_seconds and last_results:
                await asyncio.sleep(pause_seconds)
            best = max((chunk.vector_similarity for chunk in last_results), default=0.0)
            by_threshold = not last_results or best < rag_settings.rag_min_relevance_score
            cited = tuple(
                (chunk.document_id, chunk.article_ref)
                for citation in result.citations
                for chunk in last_results
                if (chunk.norm, chunk.article_ref, chunk.source_url)
                == (citation.norm, citation.article_ref, citation.source_url)
            )
            return GroundingOutcome(
                refused=result.refused,
                refused_by_threshold=result.refused and by_threshold,
                llm_latency_seconds=None if by_threshold else elapsed,
                cited=cited,
                retrieved=tuple((chunk.document_id, chunk.article_ref) for chunk in last_results),
            )

        if args.ab:
            subset = frozenset(json.loads(Path(args.ab).read_text())["llm_refused_gold_in_context"])
            print(
                f"\n== Prompt A/B on {len(subset)} questions the LLM refused with gold in "
                "context =="
            )
            for rep in range(args.reps):
                for name, instructions in (
                    ("legacy", LEGACY_INSTRUCTIONS),
                    ("current", _INSTRUCTIONS),
                ):
                    report = await run_end_to_end(
                        questions, make_ground(instructions), answerable_indices=subset
                    )
                    print(_ab_line(name, rep + 1, report))
            return 0

        print("\n== End-to-end (real LLM grounding, threshold + LLM stage) ==")
        report = await run_end_to_end(questions, make_ground(_INSTRUCTIONS))
    finally:
        pool.close()

    print(format_report(report))
    if args.save_records:
        by_bucket: dict[str, list[int]] = {}
        for record in report.answerable_records:
            by_bucket.setdefault(record.bucket, []).append(record.index)
        Path(args.save_records).write_text(json.dumps(by_bucket))
    return 0


def _ab_line(name: str, rep: int, report: EndToEndReport) -> str:
    refused = sum(1 for r in report.answerable_records if r.refused)
    evaluated = report.answerable_total - report.answerable_errors
    return (
        f"  [{name} rep {rep}] subset false-refusals {refused}/{evaluated}; "
        f"unanswerable refused {report.correct_refusals}/"
        f"{report.unanswerable_total - report.unanswerable_errors} "
        f"(errors: {report.answerable_errors}+{report.unanswerable_errors})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["fast", "smart"], default=None)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--save-records", default=None)
    parser.add_argument("--ab", default=None)
    parser.add_argument("--reps", type=int, default=2)
    sys.exit(asyncio.run(main(parser.parse_args())))

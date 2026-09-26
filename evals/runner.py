"""Runs suites and assembles the run record (the `run` command)."""

from __future__ import annotations

import os
from collections.abc import Callable

from evals.adapters.environment import environment, git_dirty, git_sha
from evals.core.run import RunRecord, SuiteResult
from evals.suites.compliance_offline import SUITE as COMPLIANCE
from evals.suites.compliance_offline import run_compliance_offline
from evals.suites.retrieval_offline import SUITE as RETRIEVAL


def _run_retrieval(seed: int) -> SuiteResult:
    """The retrieval suite with the real embedding model, the fixture indexed
    into the evaluation schema of `DATABASE_URL`."""
    from evals.adapters.fixture import read_fixture
    from evals.adapters.retrieval import open_eval_connection, prepare_eval_index
    from evals.suites.retrieval_offline import run_retrieval_offline
    from rag.embeddings.fastembed_adapter import FastEmbedAdapter
    from rag.settings import get_rag_settings

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise UnknownSuite("the retrieval suite needs DATABASE_URL (a pgvector Postgres)")
    settings = get_rag_settings()
    embeddings = FastEmbedAdapter(
        settings.rag_embedding_model,
        threads=settings.rag_embedding_threads,
        batch_size=settings.rag_embedding_batch_size,
    )
    conninfo = prepare_eval_index(
        database_url, read_fixture(), embeddings.embed_documents, settings.rag_embedding_model
    )
    with open_eval_connection(conninfo) as conn:
        return run_retrieval_offline(
            conn,
            embeddings.embed_query,
            top_k=settings.rag_top_k,
            min_relevance=settings.rag_min_relevance_score,
            seed=seed,
            config={"embedding_model": settings.rag_embedding_model},
        )


OFFLINE_SUITES: dict[str, Callable[[int], SuiteResult]] = {
    COMPLIANCE: lambda seed: run_compliance_offline(),
    RETRIEVAL: _run_retrieval,
}


class UnknownSuite(ValueError):
    """The requested suite does not exist in the requested mode."""


def run_offline(suites: list[str], seed: int) -> RunRecord:
    results: dict[str, SuiteResult] = {}
    for name in suites:
        if name not in OFFLINE_SUITES:
            raise UnknownSuite(f"no offline suite {name!r}; available: {sorted(OFFLINE_SUITES)}")
        results[name] = OFFLINE_SUITES[name](seed)
    return RunRecord(
        mode="offline",
        git_sha=git_sha(),
        git_dirty=git_dirty(),
        seed=seed,
        environment=environment(),
        suites=results,
    )

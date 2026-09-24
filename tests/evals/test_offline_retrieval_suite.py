"""The offline retrieval suite over the committed fixture, with
`FakeEmbeddings`. Skipped when no `DATABASE_URL` is configured."""

from __future__ import annotations

import json
import os

import psycopg
import pytest

from evals.adapters.fixture import read_fixture
from evals.adapters.retrieval import (
    EVAL_SCHEMA_PREFIX,
    HNSW_EF_SEARCH,
    eval_schema,
    open_eval_connection,
    prepare_eval_index,
)
from evals.core.run import SuiteResult
from evals.suites.retrieval_offline import run_retrieval_offline
from rag.embeddings.fake import FakeEmbeddings

_DATABASE_URL = os.environ.get("DATABASE_URL")

pytestmark = pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")

_EMBEDDINGS = FakeEmbeddings()
_EMBEDDING_ID = "fake"


def _run(conninfo: str) -> SuiteResult:
    with open_eval_connection(conninfo) as conn:
        return run_retrieval_offline(
            conn,
            _EMBEDDINGS.embed_query,
            top_k=5,
            min_relevance=0.5,
            seed=42,
            config={"embedding_model": "fake"},
        )


@pytest.fixture(scope="module")
def conninfo() -> str:
    assert _DATABASE_URL is not None
    return prepare_eval_index(
        _DATABASE_URL, read_fixture(), _EMBEDDINGS.embed_documents, _EMBEDDING_ID
    )


def _canonical(result: SuiteResult) -> str:
    return json.dumps(
        {
            "metrics": {k: v.value for k, v in sorted(result.metrics.items())},
            "details": result.details,
        },
        sort_keys=True,
    )


def test_the_suite_reports_overall_and_stratified_metrics(conninfo: str) -> None:
    result = _run(conninfo)

    assert {"recall_at_k", "mrr", "false_refusal_rate_at_threshold"} <= set(result.metrics)
    assert any(name.startswith("recall_at_k.style.") for name in result.metrics)
    assert any(name.startswith("recall_at_k.document.") for name in result.metrics)
    assert any(name.startswith("recall_at_k.difficulty.") for name in result.metrics)
    assert result.details["threshold_table"]
    assert result.details["config"]["top_k"] == 5
    assert 0.0 <= result.metrics["recall_at_k"].value <= 1.0


def test_two_consecutive_runs_are_identical(conninfo: str) -> None:
    assert _canonical(_run(conninfo)) == _canonical(_run(conninfo))


def test_the_evaluation_uses_its_own_schema_and_maximum_ef_search(conninfo: str) -> None:
    assert _DATABASE_URL is not None
    with open_eval_connection(conninfo) as conn:
        ef = conn.execute("SHOW hnsw.ef_search").fetchone()
        table = conn.execute(
            "SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.oid = to_regclass('rag_chunks')"
        ).fetchone()
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        own = conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'rag_chunks'",
            (eval_schema(_EMBEDDING_ID),),
        ).fetchone()

    assert ef is not None and int(ef[0]) == HNSW_EF_SEARCH
    assert table is not None and table[0] == eval_schema(_EMBEDDING_ID)
    assert own is not None and own[0] == 1


def test_each_embedding_model_gets_its_own_schema() -> None:
    fake = eval_schema("fake")
    real = eval_schema("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    assert fake != real
    assert fake.startswith(EVAL_SCHEMA_PREFIX) and real.startswith(EVAL_SCHEMA_PREFIX)
    assert eval_schema("fake") == fake

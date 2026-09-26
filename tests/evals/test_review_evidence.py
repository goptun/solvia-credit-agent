"""Retrieval evidence in `review-sample`: top-3 chunks from the fixture
index, offline, with no gateway object ever constructed. Needs pgvector."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
import yaml

from apps.agent.llm.factory import LLMFactory
from evals.adapters.fixture import index_fixture, read_fixture
from evals.adapters.retrieval import make_evidence_provider, source_type_for
from evals.datasets import load_dataset
from evals.review import render_review
from rag.embeddings.fake import FakeEmbeddings
from rag.migrate import migrate

_DATABASE_URL = os.environ.get("DATABASE_URL")

_EVIDENCE_LINE = re.compile(r'^\s+\d\. \S+ · .+ · sim -?\d\.\d{3} · ".*"$')


@pytest.fixture
def indexed_fixture() -> Iterator[psycopg.Connection[object]]:
    assert _DATABASE_URL is not None
    migrate(_DATABASE_URL)
    with psycopg.connect(_DATABASE_URL, autocommit=True) as conn:
        conn.execute("DELETE FROM rag_chunks")
        index_fixture(conn, read_fixture(), FakeEmbeddings().embed_documents)
        yield conn
        conn.execute("DELETE FROM rag_chunks")


def _write_retrieval(directory: Path) -> None:
    items = [
        {
            "id": "R-001",
            "question": "O que caracteriza o superendividamento?",
            "kind": "answerable",
            "difficulty": "easy",
            "document": "cdc-consolidada",
            "expected_refs": ["art. 54-A"],
            "evidence": "impossibilidade manifesta",
            "style": "lexical",
        },
        {
            "id": "R-002",
            "question": "A Solvia oferece cartão internacional?",
            "kind": "unanswerable",
            "difficulty": "hard",
            "distance": "near_miss",
        },
    ]
    (directory / "retrieval.yaml").write_text(
        yaml.safe_dump({"version": 1, "items": items}, allow_unicode=True), encoding="utf-8"
    )


def test_source_type_scope_follows_the_agent() -> None:
    assert source_type_for("cdc-consolidada") == "regulation"
    assert source_type_for("product-catalog") == "product_catalog"
    assert source_type_for(None) is None


@pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")
def test_every_retrieval_item_shows_exactly_three_chunks_and_no_gateway_is_built(
    indexed_fixture: psycopg.Connection[object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the review evidence must not construct an LLM factory")

    monkeypatch.setattr(LLMFactory, "__init__", forbidden)
    _write_retrieval(tmp_path)
    loaded = load_dataset("retrieval", tmp_path)
    provider = make_evidence_provider(indexed_fixture, FakeEmbeddings().embed_query)

    text = render_review(loaded, n=2, seed=42, evidence=provider)

    assert text.count("top retrieved chunks:") == 2
    evidence_lines = [line for line in text.splitlines() if re.match(r"^\s+\d\. ", line)]
    assert len(evidence_lines) == 6
    assert all(_EVIDENCE_LINE.match(line) for line in evidence_lines)


@pytest.mark.skipif(not _DATABASE_URL, reason="requires a Postgres DATABASE_URL")
def test_non_retrieval_datasets_print_no_evidence_block(
    indexed_fixture: psycopg.Connection[object], tmp_path: Path
) -> None:
    items = [
        {
            "id": "T-1",
            "message": "quero simular",
            "active_flow": "none",
            "category": "clear",
            "expected": "loan_simulation",
        }
    ]
    (tmp_path / "router.yaml").write_text(
        yaml.safe_dump({"version": 1, "items": items}, allow_unicode=True), encoding="utf-8"
    )
    provider = make_evidence_provider(indexed_fixture, FakeEmbeddings().embed_query)

    text = render_review(load_dataset("router", tmp_path), n=1, seed=1, evidence=provider)

    assert "top retrieved chunks" not in text

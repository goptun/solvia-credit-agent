"""The committed datasets: schema, composition, and (for retrieval) that
every evidence quote is found in the expected document/article of the
committed corpus fixture — verified in CI without the corpus."""

from __future__ import annotations

import pytest

from evals.adapters.fixture import read_fixture
from evals.core.composition import (
    compliance_violations,
    retrieval_violations,
    router_violations,
    slots_violations,
)
from evals.core.schemas import (
    ComplianceDataset,
    RetrievalDataset,
    RouterDataset,
    SlotsDataset,
)
from evals.datasets import DATASET_DIR, DATASET_NAMES, load_dataset

_present = [name for name in DATASET_NAMES if (DATASET_DIR / f"{name}.yaml").exists()]


def _norm(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("name", _present)
def test_committed_dataset_is_valid_and_versioned(name: str) -> None:
    loaded = load_dataset(name)

    assert loaded.version >= 1
    assert len(loaded.sha256) == 64
    dataset = loaded.dataset
    if isinstance(dataset, RetrievalDataset):
        assert retrieval_violations(dataset) == []
    elif isinstance(dataset, RouterDataset):
        assert router_violations(dataset) == []
    elif isinstance(dataset, SlotsDataset):
        assert slots_violations(dataset) == []
    else:
        assert isinstance(dataset, ComplianceDataset)
        assert compliance_violations(dataset) == []


@pytest.mark.skipif("retrieval" not in _present, reason="retrieval dataset not committed yet")
def test_every_evidence_quote_is_found_in_the_expected_article_of_the_fixture() -> None:
    dataset = load_dataset("retrieval").dataset
    assert isinstance(dataset, RetrievalDataset)
    fixture = read_fixture()
    missing = []
    for item in dataset.items:
        if item.kind != "answerable":
            continue
        refs = set(item.expected_refs or [])
        texts = [
            _norm(record.content)
            for record in fixture.records
            if record.document_id == item.document and (not refs or record.article_ref in refs)
        ]
        if not any(_norm(item.evidence or "") in text for text in texts):
            missing.append(item.id)

    assert missing == []


@pytest.mark.skipif("retrieval" not in _present, reason="retrieval dataset not committed yet")
def test_retrieval_questions_are_paraphrases_and_unique() -> None:
    dataset = load_dataset("retrieval").dataset
    assert isinstance(dataset, RetrievalDataset)
    corpus = " ".join(_norm(record.content).lower() for record in read_fixture().records)
    questions = [_norm(item.question).lower() for item in dataset.items]

    assert len(set(questions)) == len(questions)
    assert [q for q in questions if q in corpus] == []


@pytest.mark.skipif("retrieval" not in _present, reason="retrieval dataset not committed yet")
def test_retrieval_documents_are_known_manifest_documents() -> None:
    from rag.corpus.manifest import load_manifest

    dataset = load_dataset("retrieval").dataset
    assert isinstance(dataset, RetrievalDataset)
    manifest_ids = {doc.id for doc in load_manifest().documents}

    assert {item.document for item in dataset.items if item.document} <= manifest_ids

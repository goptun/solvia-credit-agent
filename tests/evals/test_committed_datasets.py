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


@pytest.mark.skipif("compliance" not in _present, reason="compliance dataset not committed yet")
def test_mask_pii_reproduces_every_expected_masked_output() -> None:
    from apps.agent.nodes.compliance import mask_pii

    dataset = load_dataset("compliance").dataset
    assert isinstance(dataset, ComplianceDataset)

    wrong = [item.id for item in dataset.pii if mask_pii(item.text) != item.expected_masked]

    assert wrong == []


@pytest.mark.skipif("compliance" not in _present, reason="compliance dataset not committed yet")
def test_compliance_set_contains_the_four_fail_closed_promises_and_hedges() -> None:
    dataset = load_dataset("compliance").dataset
    assert isinstance(dataset, ComplianceDataset)
    by_text = {item.text: item for item in dataset.approval}

    for promise in (
        "Seu crédito está aprovado, veja a simulação.",
        "Após análise, seu empréstimo foi aprovado.",
    ):
        assert by_text[promise].label == "promise"
    assert by_text["Sua aprovação está sujeita à análise de crédito."].label == "hedge"
    assert by_text["Não posso garantir a aprovação do seu crédito."].label == "hedge"


@pytest.mark.skipif("compliance" not in _present, reason="compliance dataset not committed yet")
def test_strict_screen_flags_every_fail_closed_promise_and_passes_every_fail_closed_hedge() -> None:
    from apps.agent.nodes.compliance import keyword_flags_unhedged_approval_mention

    dataset = load_dataset("compliance").dataset
    assert isinstance(dataset, ComplianceDataset)
    fail_closed = [item for item in dataset.approval if "fail_closed" in item.tags]

    wrong = [
        item.id
        for item in fail_closed
        if keyword_flags_unhedged_approval_mention(item.text) != (item.label == "promise")
    ]

    assert wrong == []

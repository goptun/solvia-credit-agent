"""The committed retrieval-eval question set parses and is internally
consistent. See `design.md` — "Retrieval quality evaluation".

The evidence and paraphrase checks need the real fetched corpus (never
committed — see `specs/regulatory-knowledge-base/spec.md` — "Automated
tests never require the corpus or embedding model to be fetched") and
are skipped when `.data/rag_corpus/` isn't present locally.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from rag.corpus.manifest import load_manifest
from rag.eval.questions import load_questions
from rag.ingest.chunking import chunk_extracted_document
from rag.ingest.html_source import extract_html
from rag.ingest.pdf_source import extract_pdf

_CORPUS_DIR = Path(".data/rag_corpus")
_MIN_ANSWERABLE = 15
_MIN_UNANSWERABLE = 5
_MIN_COLLOQUIAL = 6
_MIN_COLLOQUIAL_UNACCENTED = 3

_ACCENTED_CHARS = set("áàãâéêíóõôúüçÁÀÃÂÉÊÍÓÕÔÚÜÇ")


def _article_content_by_key() -> dict[tuple[str, str], str]:
    """Every regulatory document's chunks, concatenated per article —
    an article split across several chunks (e.g. an oversized one) must
    still be checked as a whole."""
    manifest = load_manifest()
    paths = {
        "cdc-consolidada": _CORPUS_DIR / "cdc-consolidada.htm",
        "lgpd": _CORPUS_DIR / "lgpd.htm",
        "open-finance-regulamento": _CORPUS_DIR / "open-finance-regulamento.pdf",
        "cet-disclosure": _CORPUS_DIR / "cet-disclosure.pdf",
    }
    content_by_key: dict[tuple[str, str], list[str]] = defaultdict(list)
    for doc in manifest.documents:
        path = paths.get(doc.id)
        if path is None:
            continue
        raw = path.read_bytes()
        extracted = extract_html(raw) if path.suffix == ".htm" else extract_pdf(raw)
        for chunk in chunk_extracted_document(doc.id, doc.source_type, extracted, norm=doc.norm):
            if chunk.article_ref:
                content_by_key[(doc.id, chunk.article_ref)].append(chunk.content)
    return {key: "\n".join(pieces) for key, pieces in content_by_key.items()}


def test_question_set_parses() -> None:
    questions = load_questions()

    assert len(questions.answerable) >= _MIN_ANSWERABLE
    assert len(questions.unanswerable) >= _MIN_UNANSWERABLE


def test_answerable_questions_cover_every_regulatory_document_and_the_catalog() -> None:
    questions = load_questions()

    document_ids = {q.expected_document_id for q in questions.answerable}

    assert document_ids == {
        "cdc-consolidada",
        "lgpd",
        "open-finance-regulamento",
        "cet-disclosure",
        "product-catalog",
    }


def test_regulatory_answerable_questions_have_at_least_one_article_reference() -> None:
    questions = load_questions()

    for question in questions.answerable:
        if question.expected_document_id != "product-catalog":
            assert question.expected_refs
        else:
            assert question.expected_refs is None


def test_at_least_one_question_has_multiple_expected_refs() -> None:
    questions = load_questions()

    assert any(
        q.expected_refs is not None and len(q.expected_refs) > 1 for q in questions.answerable
    )


def test_cdc_covers_over_indebtedness_and_credit_disclosure_articles() -> None:
    questions = load_questions()
    cdc_refs = {
        ref
        for q in questions.answerable
        if q.expected_document_id == "cdc-consolidada"
        for ref in (q.expected_refs or [])
    }

    superindebtedness_refs = {r for r in cdc_refs if r.startswith("art. 54-")}
    assert len(superindebtedness_refs) >= 4

    art_52_questions = [
        q
        for q in questions.answerable
        if q.expected_document_id == "cdc-consolidada" and q.expected_refs == ["art. 52"]
    ]
    assert len(art_52_questions) >= 2


def test_cet_disclosure_has_at_least_four_questions() -> None:
    questions = load_questions()
    cet_questions = [q for q in questions.answerable if q.expected_document_id == "cet-disclosure"]

    assert len(cet_questions) >= 4


def test_at_least_six_questions_are_colloquial_and_three_have_no_accents() -> None:
    questions = load_questions()
    colloquial = [q for q in questions.answerable if q.style == "colloquial"]
    unaccented_colloquial = [q for q in colloquial if not (_ACCENTED_CHARS & set(q.question))]

    assert len(colloquial) >= _MIN_COLLOQUIAL
    assert len(unaccented_colloquial) >= _MIN_COLLOQUIAL_UNACCENTED


def test_unanswerable_questions_have_both_far_and_near_miss() -> None:
    questions = load_questions()
    distances = {q.distance for q in questions.unanswerable}

    assert distances == {"far", "near_miss"}
    assert any(q.distance == "near_miss" for q in questions.unanswerable)


@pytest.mark.skipif(not _CORPUS_DIR.exists(), reason="requires the fetched corpus locally")
def test_evidence_is_a_real_substring_of_the_expected_article() -> None:
    content_by_key = _article_content_by_key()

    questions = load_questions()
    for question in questions.answerable:
        if question.expected_refs is None:
            continue
        found_in_any = False
        for ref in question.expected_refs:
            key = (question.expected_document_id, ref)
            article_content = content_by_key.get(key)
            assert article_content is not None, f"no chunk found for {key}"
            if question.evidence in article_content:
                found_in_any = True
        assert found_in_any, f"evidence not found in any of {question.expected_refs}: {question!r}"


@pytest.mark.skipif(not _CORPUS_DIR.exists(), reason="requires the fetched corpus locally")
def test_product_catalog_evidence_is_a_real_substring() -> None:
    catalog_text = (_CORPUS_DIR / "product-catalog.txt").read_text(encoding="utf-8")

    questions = load_questions()
    for question in questions.answerable:
        if question.expected_document_id == "product-catalog":
            assert question.evidence in catalog_text


@pytest.mark.skipif(not _CORPUS_DIR.exists(), reason="requires the fetched corpus locally")
def test_answerable_questions_are_paraphrased_not_copied() -> None:
    """No answerable question's text should be a verbatim substring of
    its cited article's own stored content — that would indicate the
    question was copied from the article rather than paraphrased."""
    content_by_key = _article_content_by_key()

    questions = load_questions()
    for question in questions.answerable:
        for ref in question.expected_refs or []:
            key = (question.expected_document_id, ref)
            article_content = content_by_key.get(key)
            assert article_content is not None, f"no chunk found for {key}"
            assert question.question not in article_content

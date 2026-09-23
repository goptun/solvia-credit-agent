"""`rag.ingest.chunking` against real extracted text — not synthetic
legal-sounding prose. See `design.md` — "Chunking strategy".

`cdc_art1_whole.htm` and `cdc_art54b_oversized.htm` are real excerpts
of the fetched CDC page: art. 1° (short, fits any reasonable budget)
and art. 54-B (long — inserted by Lei 14.181/2021, spans a caput plus
three numbered paragraphs, real-world oversized at the default budget).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from apps.agent.config.product_descriptions import GENERAL_PRODUCT_OVERVIEW, PRODUCT_DESCRIPTIONS
from rag.ingest.chunking import chunk_extracted_document
from rag.ingest.extracted_document import ExtractedDocument
from rag.ingest.html_source import extract_html

_FIXTURES = Path(__file__).parent / "fixtures"


def _extract(name: str) -> ExtractedDocument:
    return extract_html((_FIXTURES / name).read_bytes())


def test_whole_in_budget_article_is_a_single_chunk() -> None:
    extracted = _extract("cdc_art1_whole.htm")

    chunks = chunk_extracted_document(
        "cdc-consolidada", "regulation", extracted, norm="Lei nº 8.078/1990"
    )

    assert len(chunks) == 1
    assert chunks[0].article_ref == "art. 1°"
    assert "CAPÍTULO I" in chunks[0].hierarchy_path
    assert "estabelece normas de proteção e defesa do consumidor" in chunks[0].content


def test_oversized_article_splits_with_repeated_header() -> None:
    extracted = _extract("cdc_art54b_oversized.htm")

    chunks = chunk_extracted_document(
        "cdc-consolidada", "regulation", extracted, norm="Lei nº 8.078/1990", chunk_max_chars=1500
    )

    assert len(chunks) > 1
    assert all(c.article_ref == "art. 54-B" for c in chunks)
    assert all(c.content.startswith("Art. 54-B.") for c in chunks)
    # Each split piece stays within (a reasonable margin of) the budget
    assert all(len(c.content) < 1500 for c in chunks)


def test_oversized_article_notes_are_deduplicated() -> None:
    extracted = _extract("cdc_art54b_oversized.htm")

    chunks = chunk_extracted_document("cdc-consolidada", "regulation", extracted)

    note = chunks[0].amendment_note
    assert note is not None
    assert note.count("Incluído pela") == 1


def test_product_catalog_entry_is_a_single_chunk_with_no_article_structure() -> None:
    catalog_text = "\n\n".join([GENERAL_PRODUCT_OVERVIEW, *PRODUCT_DESCRIPTIONS.values()])
    extracted = ExtractedDocument(text=catalog_text, amendment_notes=())

    chunks = chunk_extracted_document("product-catalog", "product_catalog", extracted)

    assert len(chunks) == 1
    assert chunks[0].article_ref is None
    assert chunks[0].hierarchy_path == ""
    assert chunks[0].content == catalog_text.strip()


def test_regulatory_chunk_has_all_metadata_populated() -> None:
    extracted = _extract("cdc_art1_whole.htm")

    chunks = chunk_extracted_document(
        "cdc-consolidada",
        "regulation",
        extracted,
        norm="Lei nº 8.078/1990",
        source_url="https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm",
        version_date=date(2021, 7, 1),
    )

    chunk = chunks[0]
    assert chunk.document_id == "cdc-consolidada"
    assert chunk.norm == "Lei nº 8.078/1990"
    assert chunk.source_type == "regulation"
    assert chunk.article_ref is not None
    assert chunk.hierarchy_path != ""
    assert chunk.source_url is not None
    assert chunk.version_date == date(2021, 7, 1)


def test_product_catalog_chunk_has_all_applicable_metadata_populated() -> None:
    catalog_text = GENERAL_PRODUCT_OVERVIEW
    extracted = ExtractedDocument(text=catalog_text, amendment_notes=())

    chunks = chunk_extracted_document(
        "product-catalog",
        "product_catalog",
        extracted,
        norm=None,
        source_url=None,
        version_date=None,
    )

    chunk = chunks[0]
    assert chunk.document_id == "product-catalog"
    assert chunk.source_type == "product_catalog"
    assert chunk.content


def test_amendment_note_never_appears_inside_chunk_content() -> None:
    extracted = _extract("cdc_art54b_oversized.htm")

    chunks = chunk_extracted_document("cdc-consolidada", "regulation", extracted)

    for chunk in chunks:
        if chunk.amendment_note:
            assert chunk.amendment_note not in chunk.content

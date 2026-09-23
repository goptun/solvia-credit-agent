"""The committed corpus manifest parses and is internally consistent."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.corpus.manifest import Manifest, load_manifest

_MANIFEST_PATH = Path(__file__).parents[3] / "rag" / "corpus" / "manifest.yaml"


def test_committed_manifest_parses() -> None:
    manifest = load_manifest(_MANIFEST_PATH)

    ids = {doc.id for doc in manifest.documents}
    assert ids == {
        "cdc-consolidada",
        "lgpd",
        "open-finance-regulamento",
        "cet-disclosure",
        "product-catalog",
    }


def test_committed_manifest_has_four_regulation_documents_plus_the_catalog() -> None:
    manifest = load_manifest(_MANIFEST_PATH)

    regulation_docs = [d for d in manifest.documents if d.source_type == "regulation"]
    catalog_docs = [d for d in manifest.documents if d.source_type == "product_catalog"]

    assert len(regulation_docs) == 4
    assert len(catalog_docs) == 1


def test_regulation_documents_have_official_source_urls() -> None:
    manifest = load_manifest(_MANIFEST_PATH)

    for doc in manifest.documents:
        if doc.source_type == "regulation":
            assert doc.url is not None
            assert doc.url.startswith("https://")


def test_regulation_document_without_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Manifest.model_validate(
            {
                "documents": [
                    {
                        "id": "bad-doc",
                        "title": "Missing URL",
                        "norm": "Lei nº 1/2020",
                        "source_type": "regulation",
                        "url": None,
                        "retrieved_at": "2026-01-01",
                        "version_date": None,
                        "sha256": None,
                    }
                ]
            }
        )


def test_product_catalog_document_retrieved_at_is_a_real_date() -> None:
    manifest = load_manifest(_MANIFEST_PATH)

    catalog_doc = next(d for d in manifest.documents if d.source_type == "product_catalog")
    assert isinstance(catalog_doc.retrieved_at, date)

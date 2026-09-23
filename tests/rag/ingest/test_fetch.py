"""`rag.ingest.fetch` against a fake HTTP client — no real network calls."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from rag.corpus.manifest import Manifest, ManifestDocument
from rag.ingest.fetch import FetchResult, HashMismatchError, fetch_all, fetch_document


class _FakeHttpFetcher:
    def __init__(self, content_by_url: dict[str, bytes]) -> None:
        self._content_by_url = content_by_url

    def get(self, url: str) -> bytes:
        return self._content_by_url[url]


def _regulation_doc(sha256: str | None) -> ManifestDocument:
    return ManifestDocument(
        id="test-doc",
        title="Test Regulation",
        norm="Lei nº 1/2020",
        source_type="regulation",
        url="https://planalto.gov.br/test.htm",
        retrieved_at=date(2026, 1, 1),
        version_date=date(2020, 1, 1),
        sha256=sha256,
    )


def test_first_fetch_records_the_hash_with_no_prior_value(tmp_path: Path) -> None:
    doc = _regulation_doc(sha256=None)
    fetcher = _FakeHttpFetcher({doc.url: b"conteudo original"})  # type: ignore[dict-item]

    result = fetch_document(doc, tmp_path, http_fetcher=fetcher)

    assert result.first_fetch is True
    assert result.sha256 != ""
    assert (tmp_path / "test-doc.htm").read_bytes() == b"conteudo original"


def test_unchanged_refetch_is_a_no_op(tmp_path: Path) -> None:
    content = b"conteudo estavel"
    import hashlib

    recorded_hash = hashlib.sha256(content).hexdigest()
    doc = _regulation_doc(sha256=recorded_hash)
    fetcher = _FakeHttpFetcher({doc.url: content})  # type: ignore[dict-item]

    result = fetch_document(doc, tmp_path, http_fetcher=fetcher)

    assert result.first_fetch is False
    assert result.sha256 == recorded_hash


def test_mismatch_raises_instead_of_silently_overwriting(tmp_path: Path) -> None:
    doc = _regulation_doc(sha256="0" * 64)
    fetcher = _FakeHttpFetcher({doc.url: b"conteudo alterado"})  # type: ignore[dict-item]

    with pytest.raises(HashMismatchError):
        fetch_document(doc, tmp_path, http_fetcher=fetcher)


def test_product_catalog_document_uses_generated_text_not_http(tmp_path: Path) -> None:
    doc = ManifestDocument(
        id="product-catalog",
        title="Catálogo",
        norm=None,
        source_type="product_catalog",
        url=None,
        retrieved_at=date(2026, 1, 1),
        version_date=None,
        sha256=None,
    )
    fetcher = _FakeHttpFetcher({})

    result = fetch_document(doc, tmp_path, http_fetcher=fetcher, generated_text="catálogo fictício")

    assert (tmp_path / "product-catalog.txt").read_text(encoding="utf-8") == "catálogo fictício"
    assert result.first_fetch is True


def test_fetch_all_stops_at_the_first_mismatch(tmp_path: Path) -> None:
    ok_doc = _regulation_doc(sha256=None)
    ok_doc = ok_doc.model_copy(update={"id": "ok-doc", "url": "https://planalto.gov.br/ok.htm"})
    bad_doc = _regulation_doc(sha256="f" * 64)
    bad_doc = bad_doc.model_copy(update={"id": "bad-doc", "url": "https://planalto.gov.br/bad.htm"})
    manifest = Manifest(documents=[ok_doc, bad_doc])
    fetcher = _FakeHttpFetcher(
        {
            "https://planalto.gov.br/ok.htm": b"ok",
            "https://planalto.gov.br/bad.htm": b"changed",
        }
    )

    with pytest.raises(HashMismatchError):
        fetch_all(manifest, tmp_path, http_fetcher=fetcher)

    assert isinstance((tmp_path / "ok-doc.htm").read_bytes(), bytes)


def test_fetch_result_is_frozen() -> None:
    result = FetchResult(document_id="x", sha256="y", first_fetch=True, output_path=Path("x"))
    with pytest.raises(AttributeError):
        result.document_id = "z"  # type: ignore[misc]

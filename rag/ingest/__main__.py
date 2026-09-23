"""`python -m rag.ingest fetch|index` — fetch/verify, then chunk, embed,
and index the corpus.

This is the CLI wiring layer: the only place in `rag/ingest/` allowed to
import from `apps/` (for the product catalog's source text), per
`design.md` — "Module boundaries".
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
import psycopg

from apps.agent.config.product_descriptions import GENERAL_PRODUCT_OVERVIEW, PRODUCT_DESCRIPTIONS
from rag.corpus.manifest import ManifestDocument, load_manifest
from rag.embeddings.fastembed_adapter import FastEmbedAdapter
from rag.ingest.extracted_document import ExtractedDocument
from rag.ingest.fetch import FetchResult, HashMismatchError, fetch_all
from rag.ingest.html_source import extract_html
from rag.ingest.indexing import index_document
from rag.ingest.pdf_source import extract_pdf
from rag.settings import get_rag_settings

_OUTPUT_DIR = "./.data/rag_corpus"


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
"""planalto.gov.br rejects non-browser-looking User-Agent strings outright
(`RemoteProtocolError: Server disconnected without sending a response`,
verified against the real site while implementing this) — a common
browser UA is required, not optional, to fetch these public texts."""


class _HttpxFetcher:
    def get(self, url: str) -> bytes:
        headers = {"User-Agent": _BROWSER_USER_AGENT}
        response = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        return response.content


def _product_catalog_text() -> str:
    return "\n\n".join([GENERAL_PRODUCT_OVERVIEW, *PRODUCT_DESCRIPTIONS.values()])


def _print_result(result: FetchResult) -> None:
    status = "first fetch, hash recorded" if result.first_fetch else "verified, unchanged"
    print(f"[{result.document_id}] {status} — sha256={result.sha256} -> {result.output_path}")


def fetch_command() -> int:
    manifest = load_manifest()
    try:
        results = fetch_all(
            manifest,
            Path(_OUTPUT_DIR),
            http_fetcher=_HttpxFetcher(),
            generated_text_by_id={"product-catalog": _product_catalog_text()},
        )
    except HashMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for result in results:
        _print_result(result)
    return 0


def _extract_document(doc: ManifestDocument) -> ExtractedDocument:
    if doc.source_type == "product_catalog":
        path = Path(_OUTPUT_DIR) / f"{doc.id}.txt"
        return ExtractedDocument(text=path.read_text(encoding="utf-8"), amendment_notes=())

    assert doc.url is not None
    extension = ".pdf" if doc.url.lower().endswith(".pdf") else ".htm"
    path = Path(_OUTPUT_DIR) / f"{doc.id}{extension}"
    content = path.read_bytes()
    return extract_pdf(content) if extension == ".pdf" else extract_html(content)


def index_command() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set", file=sys.stderr)
        return 1

    manifest = load_manifest()
    settings = get_rag_settings()
    embeddings = FastEmbedAdapter(
        settings.rag_embedding_model,
        threads=settings.rag_embedding_threads,
        batch_size=settings.rag_embedding_batch_size,
    )

    with psycopg.connect(database_url, autocommit=True) as conn:
        for doc in manifest.documents:
            extracted = _extract_document(doc)
            reindexed = index_document(
                conn,
                doc,
                extracted,
                embeddings.embed_documents,
                chunk_max_chars=settings.rag_chunk_max_chars,
            )
            status = "reindexed" if reindexed else "unchanged, skipped"
            print(f"[{doc.id}] {status}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("fetch", "index"):
        print("Usage: python -m rag.ingest fetch|index", file=sys.stderr)
        return 2
    if sys.argv[1] == "fetch":
        return fetch_command()
    return index_command()


if __name__ == "__main__":
    sys.exit(main())

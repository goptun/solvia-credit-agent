"""Corpus fetch and hash-verification logic.

See `design.md` — "Corpus pipeline and provenance". This module has no
dependency on `apps/` — the CLI entrypoint (`rag/ingest/__main__.py`)
supplies the product catalog's generated text, keeping this module
apps-agnostic and testable with fakes (see `design.md` — "Module
boundaries").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rag.corpus.manifest import Manifest, ManifestDocument

_SCRIPT_OR_NOSCRIPT_TAG = re.compile(
    rb"<(script|noscript)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)


def _strip_injected_scripts(content: bytes) -> bytes:
    """Drop `<script>`/`<noscript>` blocks before hashing or storing.

    Verified while implementing this: planalto.gov.br injects an F5
    BIG-IP bot-mitigation script (`id="f5_cspm"`) containing a random
    per-request token at the end of the page, so two fetches of the
    *same, unchanged* law returned different bytes despite identical
    legal text — a naive whole-file hash would treat that as drift on
    every single fetch. Stripping script/noscript content (which is
    never part of the retrievable legal text anyway — see the chunking
    step) makes the hash reflect the actual document instead of a WAF
    artifact. A no-op on PDF content, which never contains this tag.
    """
    return _SCRIPT_OR_NOSCRIPT_TAG.sub(b"", content)


class HashMismatchError(Exception):
    """A re-fetched document's content no longer matches the manifest's
    recorded hash — a real regulatory change, not a bug."""


class HttpFetcher(Protocol):
    def get(self, url: str) -> bytes: ...


@dataclass(frozen=True)
class FetchResult:
    document_id: str
    sha256: str
    first_fetch: bool
    output_path: Path


def _extension_for(url: str) -> str:
    return ".pdf" if url.lower().endswith(".pdf") else ".htm"


def fetch_document(
    doc: ManifestDocument,
    output_dir: Path,
    *,
    http_fetcher: HttpFetcher,
    generated_text: str | None = None,
) -> FetchResult:
    """Fetch (or, for a `product_catalog` document, accept generated
    text for) one document, verify it against the manifest's recorded
    hash, and write it to `output_dir`."""
    if doc.source_type == "product_catalog":
        if generated_text is None:
            raise ValueError(
                f"{doc.id!r} is a product_catalog document but no generated_text was provided"
            )
        content = generated_text.encode("utf-8")
        extension = ".txt"
    else:
        if doc.url is None:
            raise ValueError(f"regulation document {doc.id!r} has no source url")
        content = _strip_injected_scripts(http_fetcher.get(doc.url))
        extension = _extension_for(doc.url)

    computed_hash = hashlib.sha256(content).hexdigest()
    first_fetch = doc.sha256 is None
    if not first_fetch and computed_hash != doc.sha256:
        raise HashMismatchError(
            f"{doc.id!r}: fetched content hash {computed_hash} does not match the "
            f"manifest's recorded hash {doc.sha256} — the source may have changed"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc.id}{extension}"
    output_path.write_bytes(content)

    return FetchResult(
        document_id=doc.id,
        sha256=computed_hash,
        first_fetch=first_fetch,
        output_path=output_path,
    )


def fetch_all(
    manifest: Manifest,
    output_dir: Path,
    *,
    http_fetcher: HttpFetcher,
    generated_text_by_id: dict[str, str] | None = None,
) -> list[FetchResult]:
    """Fetch every document in `manifest`, verifying each against its
    recorded hash. Raises `HashMismatchError` on the first mismatch,
    leaving later documents unfetched — a mismatch means the manifest
    itself needs review before ingestion continues."""
    generated_text_by_id = generated_text_by_id or {}
    return [
        fetch_document(
            doc,
            output_dir,
            http_fetcher=http_fetcher,
            generated_text=generated_text_by_id.get(doc.id),
        )
        for doc in manifest.documents
    ]

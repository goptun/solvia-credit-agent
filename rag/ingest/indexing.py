"""Idempotent, incremental indexing of chunks into `rag_chunks`.

See `design.md` — "Index schema" (idempotent indexing) and the
`regulatory-knowledge-base` spec's "Idempotent, incremental indexing"
requirement. Takes an `embed_documents` callable rather than depending
on `rag.embeddings.port.EmbeddingsPort` directly — this module only
needs *some* function from texts to vectors, so it stays decoupled
from how those vectors are actually produced.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import psycopg

from rag.corpus.manifest import ManifestDocument
from rag.ingest.chunking import chunk_extracted_document
from rag.ingest.extracted_document import ExtractedDocument

EmbedDocumentsFn = Callable[[Sequence[str]], list[list[float]]]

_UPSERT_CHUNK = """
INSERT INTO rag_chunks (
    chunk_id, document_id, document_hash, source_type, norm,
    article_ref, hierarchy_path, source_url, version_date,
    amendment_note, content, embedding
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    document_hash = EXCLUDED.document_hash,
    source_type = EXCLUDED.source_type,
    norm = EXCLUDED.norm,
    article_ref = EXCLUDED.article_ref,
    hierarchy_path = EXCLUDED.hierarchy_path,
    source_url = EXCLUDED.source_url,
    version_date = EXCLUDED.version_date,
    amendment_note = EXCLUDED.amendment_note,
    content = EXCLUDED.content,
    embedding = EXCLUDED.embedding
"""


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk with its deterministic id: every `rag_chunks` column except
    the embedding and the document hash. Also the shape the evaluation
    harness commits as its corpus fixture."""

    chunk_id: str
    document_id: str
    source_type: str
    norm: str | None
    article_ref: str | None
    hierarchy_path: str
    source_url: str | None
    version_date: date | None
    amendment_note: str | None
    content: str


def compute_chunk_id(document_id: str, hierarchy_path: str, chunk_index: int) -> str:
    """Deterministic chunk id: `sha256(document_id || hierarchy_path ||
    chunk_index)` — see `design.md` — "Index schema"."""
    key = f"{document_id}||{hierarchy_path}||{chunk_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def existing_document_hash(conn: psycopg.Connection[Any], document_id: str) -> str | None:
    row = conn.execute(
        "SELECT DISTINCT document_hash FROM rag_chunks WHERE document_id = %s", (document_id,)
    ).fetchone()
    return None if row is None else str(row[0])


def effective_document_hash(doc: ManifestDocument, extracted: ExtractedDocument) -> str:
    """`doc.sha256` when the manifest pins one; otherwise (the locally
    generated product catalog) a hash of the extracted text itself."""
    return doc.sha256 or hashlib.sha256(extracted.text.encode("utf-8")).hexdigest()


def chunk_records(
    doc: ManifestDocument, extracted: ExtractedDocument, *, chunk_max_chars: int = 1500
) -> list[ChunkRecord]:
    """Chunk `extracted` and give every chunk its deterministic id."""
    chunks = chunk_extracted_document(
        doc.id,
        doc.source_type,
        extracted,
        norm=doc.norm,
        source_url=doc.url,
        version_date=doc.version_date,
        chunk_max_chars=chunk_max_chars,
    )
    return [
        ChunkRecord(
            chunk_id=compute_chunk_id(chunk.document_id, chunk.hierarchy_path, index),
            document_id=chunk.document_id,
            source_type=chunk.source_type,
            norm=chunk.norm,
            article_ref=chunk.article_ref,
            hierarchy_path=chunk.hierarchy_path,
            source_url=chunk.source_url,
            version_date=chunk.version_date,
            amendment_note=chunk.amendment_note,
            content=chunk.content,
        )
        for index, chunk in enumerate(chunks)
    ]


def index_chunks(
    conn: psycopg.Connection[Any],
    document_id: str,
    document_hash: str,
    records: Sequence[ChunkRecord],
    embed_documents: EmbedDocumentsFn,
) -> None:
    """Embed and upsert `records` for one document, then delete the
    document's chunks that are not in `records`."""
    vectors = embed_documents([record.content for record in records])

    for record, vector in zip(records, vectors, strict=True):
        conn.execute(
            _UPSERT_CHUNK,
            (
                record.chunk_id,
                record.document_id,
                document_hash,
                record.source_type,
                record.norm,
                record.article_ref,
                record.hierarchy_path,
                record.source_url,
                record.version_date,
                record.amendment_note,
                record.content,
                list(vector),
            ),
        )

    chunk_ids = [record.chunk_id for record in records]
    if chunk_ids:
        conn.execute(
            "DELETE FROM rag_chunks WHERE document_id = %s AND chunk_id != ALL(%s)",
            (document_id, chunk_ids),
        )
    else:
        conn.execute("DELETE FROM rag_chunks WHERE document_id = %s", (document_id,))


def index_document(
    conn: psycopg.Connection[Any],
    doc: ManifestDocument,
    extracted: ExtractedDocument,
    embed_documents: EmbedDocumentsFn,
    *,
    chunk_max_chars: int = 1500,
) -> bool:
    """Reindex `doc` if its effective hash differs from what's already
    stored for it. Returns `True` if reindexing happened, `False` if
    skipped because the document is unchanged.

    Uses `doc.sha256` when the manifest pins one (every `regulation`
    document). A `product_catalog` document's manifest hash is `null`
    by design (see `rag/ingest/fetch.py` — it's locally generated, not
    fetched, so there's no upstream drift to detect) — its effective
    hash is instead computed from `extracted.text` itself, so unchanged
    reindexing still skips and a real content change still reindexes,
    per `specs/regulatory-knowledge-base/spec.md` — "Idempotent,
    incremental indexing", which does not exempt any source type.
    """
    effective_hash = effective_document_hash(doc, extracted)

    if existing_document_hash(conn, doc.id) == effective_hash:
        return False

    records = chunk_records(doc, extracted, chunk_max_chars=chunk_max_chars)
    index_chunks(conn, doc.id, effective_hash, records, embed_documents)
    return True

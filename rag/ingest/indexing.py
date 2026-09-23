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


def index_document(
    conn: psycopg.Connection[Any],
    doc: ManifestDocument,
    extracted: ExtractedDocument,
    embed_documents: EmbedDocumentsFn,
    *,
    chunk_max_chars: int = 1500,
) -> bool:
    """Reindex `doc` if its manifest hash differs from what's already
    stored for it. Returns `True` if reindexing happened, `False` if
    skipped because the document is unchanged."""
    if doc.sha256 is None:
        raise ValueError(f"{doc.id!r} has no recorded hash to compare against")

    if existing_document_hash(conn, doc.id) == doc.sha256:
        return False

    chunks = chunk_extracted_document(
        doc.id,
        doc.source_type,
        extracted,
        norm=doc.norm,
        source_url=doc.url,
        version_date=doc.version_date,
        chunk_max_chars=chunk_max_chars,
    )
    vectors = embed_documents([chunk.content for chunk in chunks])

    chunk_ids: list[str] = []
    for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        chunk_id = compute_chunk_id(chunk.document_id, chunk.hierarchy_path, index)
        chunk_ids.append(chunk_id)
        conn.execute(
            _UPSERT_CHUNK,
            (
                chunk_id,
                chunk.document_id,
                doc.sha256,
                chunk.source_type,
                chunk.norm,
                chunk.article_ref,
                chunk.hierarchy_path,
                chunk.source_url,
                chunk.version_date,
                chunk.amendment_note,
                chunk.content,
                list(vector),
            ),
        )

    if chunk_ids:
        conn.execute(
            "DELETE FROM rag_chunks WHERE document_id = %s AND chunk_id != ALL(%s)",
            (doc.id, chunk_ids),
        )
    else:
        conn.execute("DELETE FROM rag_chunks WHERE document_id = %s", (doc.id,))

    return True

"""The committed corpus chunk fixture: what CI indexes instead of
downloading the corpus (design.md, Decision 5).

One JSON line per chunk (every `rag_chunks` column except the embedding),
preceded by a header line recording, per document, the manifest sha256 it
was built from. Source texts are official Brazilian acts (public domain
in Brazil); provenance stays in the chunk metadata and the header.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import psycopg

from evals.core.embedding import EmbedDocuments
from rag.corpus.manifest import ManifestDocument, load_manifest
from rag.ingest.__main__ import _extract_document, _product_catalog_text
from rag.ingest.extracted_document import ExtractedDocument
from rag.ingest.indexing import (
    ChunkRecord,
    chunk_records,
    effective_document_hash,
    existing_document_hash,
    index_chunks,
)
from rag.settings import get_rag_settings

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_chunks.jsonl"
MAX_FIXTURE_BYTES = 2_000_000


@dataclass(frozen=True)
class FixtureDocument:
    sha256: str
    source_type: str
    url: str | None
    retrieved_at: str


@dataclass(frozen=True)
class Fixture:
    chunk_max_chars: int
    documents: dict[str, FixtureDocument]
    records: list[ChunkRecord]


def _record_to_json(record: ChunkRecord) -> dict[str, Any]:
    return {
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
        "source_type": record.source_type,
        "norm": record.norm,
        "article_ref": record.article_ref,
        "hierarchy_path": record.hierarchy_path,
        "source_url": record.source_url,
        "version_date": record.version_date.isoformat() if record.version_date else None,
        "amendment_note": record.amendment_note,
        "content": record.content,
    }


def _record_from_json(raw: dict[str, Any]) -> ChunkRecord:
    version = raw["version_date"]
    return ChunkRecord(
        chunk_id=raw["chunk_id"],
        document_id=raw["document_id"],
        source_type=raw["source_type"],
        norm=raw["norm"],
        article_ref=raw["article_ref"],
        hierarchy_path=raw["hierarchy_path"],
        source_url=raw["source_url"],
        version_date=date.fromisoformat(version) if version else None,
        amendment_note=raw["amendment_note"],
        content=raw["content"],
    )


def build_fixture(chunk_max_chars: int | None = None) -> Fixture:
    """Chunk the fetched corpus (`python -m rag.ingest fetch` first) with the
    production pipeline. Local only: needs `.data/rag_corpus/`."""
    max_chars = chunk_max_chars or get_rag_settings().rag_chunk_max_chars
    documents: dict[str, FixtureDocument] = {}
    records: list[ChunkRecord] = []
    for doc in load_manifest().documents:
        extracted = _extract_document(doc)
        documents[doc.id] = _fixture_document(doc, extracted)
        records.extend(chunk_records(doc, extracted, chunk_max_chars=max_chars))
    return Fixture(chunk_max_chars=max_chars, documents=documents, records=records)


def _fixture_document(doc: ManifestDocument, extracted: ExtractedDocument) -> FixtureDocument:
    return FixtureDocument(
        sha256=effective_document_hash(doc, extracted),
        source_type=doc.source_type,
        url=doc.url,
        retrieved_at=doc.retrieved_at.isoformat(),
    )


def write_fixture(fixture: Fixture, path: Path = FIXTURE_PATH) -> None:
    header = {
        "_header": {
            "chunk_max_chars": fixture.chunk_max_chars,
            "documents": {
                doc_id: {
                    "sha256": doc.sha256,
                    "source_type": doc.source_type,
                    "url": doc.url,
                    "retrieved_at": doc.retrieved_at,
                }
                for doc_id, doc in fixture.documents.items()
            },
        }
    }
    lines = [json.dumps(header, ensure_ascii=False, sort_keys=True)]
    lines += [
        json.dumps(_record_to_json(record), ensure_ascii=False, sort_keys=True)
        for record in fixture.records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_fixture(path: Path = FIXTURE_PATH) -> Fixture:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])["_header"]
    documents = {
        doc_id: FixtureDocument(
            sha256=raw["sha256"],
            source_type=raw["source_type"],
            url=raw["url"],
            retrieved_at=raw["retrieved_at"],
        )
        for doc_id, raw in header["documents"].items()
    }
    return Fixture(
        chunk_max_chars=header["chunk_max_chars"],
        documents=documents,
        records=[_record_from_json(json.loads(line)) for line in lines[1:] if line.strip()],
    )


def current_source_hashes() -> dict[str, str]:
    """Document hashes the manifest (and the generated product catalog)
    imply today — available without fetching the corpus."""
    hashes = {}
    for doc in load_manifest().documents:
        if doc.sha256:
            hashes[doc.id] = doc.sha256
        else:
            hashes[doc.id] = effective_document_hash(
                doc, ExtractedDocument(text=_product_catalog_text(), amendment_notes=())
            )
    return hashes


def stale_documents(fixture: Fixture) -> list[str]:
    """Documents whose recorded hash differs from what the manifest says."""
    current = current_source_hashes()
    return sorted(
        doc_id
        for doc_id in set(current) | set(fixture.documents)
        if doc_id not in fixture.documents
        or current.get(doc_id) != fixture.documents[doc_id].sha256
    )


def diff_fixtures(committed: Fixture, rebuilt: Fixture) -> list[str]:
    """Human-readable differences between two fixtures (empty = identical)."""
    problems = []
    if committed.chunk_max_chars != rebuilt.chunk_max_chars:
        problems.append(f"chunk_max_chars {committed.chunk_max_chars} != {rebuilt.chunk_max_chars}")
    if committed.documents != rebuilt.documents:
        problems.append("document header (hashes/sources) differs")
    old = {record.chunk_id: record for record in committed.records}
    new = {record.chunk_id: record for record in rebuilt.records}
    for chunk_id in sorted(old.keys() - new.keys()):
        problems.append(f"chunk only in the committed fixture: {chunk_id[:12]}")
    for chunk_id in sorted(new.keys() - old.keys()):
        problems.append(f"chunk only in the rebuilt corpus: {chunk_id[:12]}")
    for chunk_id in sorted(old.keys() & new.keys()):
        if old[chunk_id] != new[chunk_id]:
            problems.append(f"chunk content/metadata differs: {chunk_id[:12]}")
    return problems


def index_fixture(
    conn: psycopg.Connection[Any], fixture: Fixture, embed_documents: EmbedDocuments
) -> int:
    """Index the fixture into `rag_chunks` (idempotent per document hash).
    Returns the number of documents (re)indexed."""
    by_document: dict[str, list[ChunkRecord]] = {}
    for record in fixture.records:
        by_document.setdefault(record.document_id, []).append(record)
    indexed = 0
    for doc_id, records in by_document.items():
        document_hash = fixture.documents[doc_id].sha256
        if existing_document_hash(conn, doc_id) == document_hash:
            continue
        index_chunks(conn, doc_id, document_hash, records, embed_documents)
        indexed += 1
    return indexed

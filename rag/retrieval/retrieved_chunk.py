"""The shape both `hybrid.py` and `reranker.py` operate on — split out
to avoid a circular import between them."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rag.corpus.manifest import SourceType


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    norm: str | None
    source_type: SourceType
    article_ref: str | None
    hierarchy_path: str
    source_url: str | None
    version_date: date | None
    amendment_note: str | None
    content: str
    fused_score: float
    vector_similarity: float
    matched_fts: bool

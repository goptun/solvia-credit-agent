"""Hybrid (vector + full-text) retrieval with Reciprocal Rank Fusion.

See `design.md` — "Hybrid retrieval and fusion". RRF's fused score is
a rank-based fusion weight, never a relevance measure — callers that
need to decide whether a result is actually *relevant* (e.g. the
refusal threshold in `design.md` — "Grounding and citation validation")
must use `RetrievedChunk.vector_similarity`, not `fused_score`.
"""

from __future__ import annotations

from typing import Any

import psycopg

from rag.corpus.manifest import SourceType
from rag.embeddings.port import Vector
from rag.retrieval.reranker import NoopReranker, RerankerPort
from rag.retrieval.retrieved_chunk import RetrievedChunk

_VECTOR_QUERY = """
SELECT chunk_id
FROM rag_chunks
WHERE %(source_type)s::text IS NULL OR source_type = %(source_type)s::text
ORDER BY embedding <=> %(query_vector)s::vector
LIMIT %(k)s
"""

_FTS_QUERY = """
WITH parsed_query AS (
    -- plainto_tsquery ANDs every term together, so a long colloquial
    -- question sharing only a few words with a chunk never matches it
    -- (see design.md — "Full-text search: OR over parsed lexemes").
    -- Parsing the query text into portuguese_unaccent lexemes and
    -- OR-joining them keeps ts_rank_cd's relevance ordering (more
    -- matched/weighted lexemes still rank higher) while letting a
    -- single shared term surface the chunk at all.
    SELECT to_tsquery(
        'portuguese_unaccent',
        array_to_string(
            tsvector_to_array(to_tsvector('portuguese_unaccent', %(query_text)s)), ' | '
        )
    ) AS tsq
)
SELECT chunk_id
FROM rag_chunks, parsed_query
WHERE content_tsv @@ parsed_query.tsq
  AND (%(source_type)s::text IS NULL OR source_type = %(source_type)s::text)
ORDER BY ts_rank_cd(content_tsv, parsed_query.tsq) DESC
LIMIT %(k)s
"""

_SIMILARITY_QUERY = """
SELECT chunk_id, 1 - (embedding <=> %(query_vector)s::vector) AS similarity
FROM rag_chunks
WHERE chunk_id = ANY(%(chunk_ids)s)
"""

_CHUNK_DETAILS_QUERY = """
SELECT chunk_id, document_id, norm, source_type, article_ref, hierarchy_path,
       source_url, version_date, amendment_note, content
FROM rag_chunks
WHERE chunk_id = ANY(%(chunk_ids)s)
"""


def _reciprocal_rank_fusion(
    vector_ids: list[str], fts_ids: list[str], rrf_k: int
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    for rank, chunk_id in enumerate(fts_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def hybrid_search(
    conn: psycopg.Connection[Any],
    query_text: str,
    query_vector: Vector,
    *,
    source_type: SourceType | None = None,
    k_vector: int = 20,
    k_fts: int = 20,
    rrf_k: int = 60,
    top_k: int = 5,
    reranker: RerankerPort | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the `top_k` chunks for `query_text`/`query_vector`,
    fusing vector-similarity and full-text rankings via Reciprocal Rank
    Fusion, optionally restricted to a single `source_type` (see
    `specs/regulatory-knowledge-base/spec.md` — "Retrieval is scoped
    by source type per intent"). `reranker` is always called — a
    `NoopReranker` by default — so enabling real reranking is a matter
    of passing a different adapter, never a new call site."""
    vector_rows = conn.execute(
        _VECTOR_QUERY,
        {"source_type": source_type, "query_vector": query_vector, "k": k_vector},
    ).fetchall()
    vector_ids = [row[0] for row in vector_rows]

    fts_rows = conn.execute(
        _FTS_QUERY,
        {"source_type": source_type, "query_text": query_text, "k": k_fts},
    ).fetchall()
    fts_ids = [row[0] for row in fts_rows]
    fts_id_set = set(fts_ids)

    fused_scores = _reciprocal_rank_fusion(vector_ids, fts_ids, rrf_k)
    top_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]
    if not top_ids:
        return []

    similarity_rows = conn.execute(
        _SIMILARITY_QUERY, {"query_vector": query_vector, "chunk_ids": top_ids}
    ).fetchall()
    similarity_by_id = {row[0]: float(row[1]) for row in similarity_rows}

    detail_rows = conn.execute(_CHUNK_DETAILS_QUERY, {"chunk_ids": top_ids}).fetchall()
    details_by_id = {row[0]: row for row in detail_rows}

    results: list[RetrievedChunk] = []
    for chunk_id in top_ids:
        row = details_by_id[chunk_id]
        (
            _chunk_id,
            document_id,
            norm,
            row_source_type,
            article_ref,
            hierarchy_path,
            source_url,
            version_date,
            amendment_note,
            content,
        ) = row
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                norm=norm,
                source_type=row_source_type,
                article_ref=article_ref,
                hierarchy_path=hierarchy_path,
                source_url=source_url,
                version_date=version_date,
                amendment_note=amendment_note,
                content=content,
                fused_score=fused_scores[chunk_id],
                vector_similarity=similarity_by_id.get(chunk_id, 0.0),
                matched_fts=chunk_id in fts_id_set,
            )
        )

    active_reranker = reranker or NoopReranker()
    return active_reranker.rerank(query_text, results)

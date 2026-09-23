"""`python -m rag.migrate` — idempotent schema setup for `rag_chunks`.

Mirrors the checkpointer's own `setup()`-on-startup pattern
(`apps/agent/checkpointer.py`) rather than introducing a migration
framework the project doesn't otherwise need — see `design.md` —
"Index schema".
"""

from __future__ import annotations

import os
import sys

import psycopg

_CREATE_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;
"""

_TEXT_SEARCH_CONFIG_EXISTS = """
SELECT 1 FROM pg_catalog.pg_ts_config WHERE cfgname = 'portuguese_unaccent';
"""

_CREATE_TEXT_SEARCH_CONFIG = """
CREATE TEXT SEARCH CONFIGURATION portuguese_unaccent (COPY = portuguese);
ALTER TEXT SEARCH CONFIGURATION portuguese_unaccent
    ALTER MAPPING FOR hword, hword_part, word WITH unaccent, portuguese_stem;
"""
"""`CREATE TEXT SEARCH CONFIGURATION` has no `IF NOT EXISTS` clause in
PostgreSQL (verified against a real pgvector/pg16 instance — it is a
syntax error, not a warning), so existence is checked in Python first."""

_CREATE_TABLE_AND_INDEXES = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id       TEXT PRIMARY KEY,
    document_id    TEXT NOT NULL,
    document_hash  TEXT NOT NULL,
    source_type    TEXT NOT NULL,
    norm           TEXT,
    article_ref    TEXT,
    hierarchy_path TEXT NOT NULL,
    source_url     TEXT,
    version_date   DATE,
    amendment_note TEXT,
    content        TEXT NOT NULL,
    embedding      VECTOR(384) NOT NULL,
    content_tsv    TSVECTOR GENERATED ALWAYS AS (to_tsvector('portuguese_unaccent', content)) STORED
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_content_tsv_gin
    ON rag_chunks USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx ON rag_chunks (document_id);
CREATE INDEX IF NOT EXISTS rag_chunks_source_type_idx ON rag_chunks (source_type);
"""


def migrate(conn_string: str) -> None:
    """Idempotently create the `vector`/`unaccent` extensions, the
    `portuguese_unaccent` text search configuration, and the
    `rag_chunks` table with its indexes."""
    with psycopg.connect(conn_string, autocommit=True) as conn:
        conn.execute(_CREATE_EXTENSIONS)
        if conn.execute(_TEXT_SEARCH_CONFIG_EXISTS).fetchone() is None:
            conn.execute(_CREATE_TEXT_SEARCH_CONFIG)
        conn.execute(_CREATE_TABLE_AND_INDEXES)


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set", file=sys.stderr)
        return 1
    migrate(database_url)
    print("rag_chunks schema is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())

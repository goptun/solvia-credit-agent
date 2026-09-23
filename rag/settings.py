"""Configuration for the regulatory knowledge base and retrieval pipeline.

Same `pydantic-settings` pattern as `apps/agent/llm/settings.py`: every
tunable value comes from the environment (or `.env` locally), never
hardcoded in `rag/` code — see `design.md` — "Hybrid retrieval and
fusion" and "Grounding and citation validation".
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """Environment-driven configuration for corpus ingestion and retrieval."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rag_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    """See `design.md` — "Embedding model trade-offs": the only
    `fastembed`-supported small multilingual option; `intfloat/
    multilingual-e5-small` (an earlier assumption) is not in
    `fastembed`'s model catalog."""
    rag_chunk_max_chars: int = 1500

    rag_k_vector: int = 20
    rag_k_fts: int = 20
    rag_rrf_k: int = 60
    rag_top_k: int = 5

    rag_min_relevance_score: float = 0.75
    """Cosine-similarity threshold (not a rank-based fused score — see
    `design.md` — "Grounding and citation validation") below which the
    knowledge agent refuses to answer instead of guessing."""

    rag_reranking_enabled: bool = False

    rag_official_domains: tuple[str, ...] = (
        "bcb.gov.br",
        "normativos.bcb.gov.br",
        "planalto.gov.br",
        "gov.br",
        "openfinancebrasil.org.br",
    )


def get_rag_settings() -> RagSettings:
    """Build `RagSettings` from the current environment."""
    return RagSettings()

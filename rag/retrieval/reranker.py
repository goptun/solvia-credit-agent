"""Optional reranking stage, applied after RRF fusion.

Disabled by default (`NoopReranker`) — see `design.md` — "Hybrid
retrieval and fusion" and "Risks / Trade-offs". `hybrid_search` calls
whichever `RerankerPort` it's given unconditionally, so enabling
reranking is a matter of which adapter is wired in, not a new call
site.
"""

from __future__ import annotations

from typing import Protocol

from rag.retrieval.retrieved_chunk import RetrievedChunk


class RerankerPort(Protocol):
    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoopReranker:
    """A pass-through reranker: returns `results` unchanged."""

    def rerank(self, query: str, results: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return results


def get_reranker(reranking_enabled: bool) -> RerankerPort:
    """Selects a reranker per `RagSettings.rag_reranking_enabled`.

    No real (e.g. cross-encoder) adapter exists yet — see `design.md`
    — "Non-Goals": this selection point exists so wiring one in later
    is a configuration change, not a new call site. `reranking_enabled`
    is accepted (not yet used) for that reason; today this always
    returns `NoopReranker`, regardless of the flag.
    """
    return NoopReranker()

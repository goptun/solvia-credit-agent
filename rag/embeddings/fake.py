"""Deterministic fake embeddings adapter used by the automated test
suite — never downloads a model or makes a network call, so CI never
depends on the real `fastembed` model (see
`specs/regulatory-knowledge-base/spec.md` — "Automated tests never
require the corpus or embedding model to be fetched")."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from rag.embeddings.port import Vector

_DIMENSIONS = 384


def _hash_to_vector(text: str, dimensions: int = _DIMENSIONS) -> Vector:
    """A deterministic, hash-derived unit-ish vector: the same text
    always yields the same vector, with no model involved."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat the digest to cover every dimension, then map each byte to
    # a small float in [-1, 1].
    repeated = (digest * (dimensions // len(digest) + 1))[:dimensions]
    return [(byte / 127.5) - 1.0 for byte in repeated]


class FakeEmbeddings:
    """Identical input text always yields an identical vector, via
    either `embed_query` or `embed_documents`."""

    def embed_query(self, text: str) -> Vector:
        return _hash_to_vector(text)

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [_hash_to_vector(text) for text in texts]

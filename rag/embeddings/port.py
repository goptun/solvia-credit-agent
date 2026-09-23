"""Provider-agnostic embeddings interface, mirroring `apps/agent/llm/
port.py`'s `LLMPort` shape.

Two methods, not one generic `embed`, because the chosen
`multilingual-e5` model family needs a different literal prefix for a
query than for an indexed passage to retrieve well — see `design.md` —
"Hybrid retrieval and fusion". Each adapter applies its own prefixing
internally, so no caller ever handles it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

Vector = list[float]


class EmbeddingsPort(Protocol):
    def embed_query(self, text: str) -> Vector: ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

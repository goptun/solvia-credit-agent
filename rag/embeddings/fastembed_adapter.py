"""`fastembed`-backed `EmbeddingsPort` adapter.

Default model: `sentence-transformers/paraphrase-multilingual-MiniLM-
L12-v2` — see `design.md` — "Embedding model trade-offs" for why (and
for the correction that `intfloat/multilingual-e5-small`, the
originally-planned default, is not actually in `fastembed`'s supported-
model catalog).

Applies the `multilingual-e5` family's required `"query: "`/
`"passage: "` prefixes only when the configured model is from that
family — a symmetric model like the default is not trained to expect
them, so always adding them would add noise, not help. Either way, the
prefixing decision lives here, never in a caller.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastembed import TextEmbedding

from rag.embeddings.port import Vector

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _uses_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


class FastEmbedAdapter:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model = TextEmbedding(model_name=model_name)
        self._use_e5_prefixes = _uses_e5_prefixes(model_name)

    def embed_query(self, text: str) -> Vector:
        prefixed = (_QUERY_PREFIX + text) if self._use_e5_prefixes else text
        (vector,) = self._model.embed([prefixed])
        return vector.tolist()  # type: ignore[no-any-return]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if self._use_e5_prefixes:
            texts = [_PASSAGE_PREFIX + text for text in texts]
        return [vector.tolist() for vector in self._model.embed(list(texts))]

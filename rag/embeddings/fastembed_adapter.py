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

Caps ONNX Runtime threads and the passage-embedding batch size (see
`rag.settings.RagSettings.rag_embedding_threads`/
`rag_embedding_batch_size`) — real, not theoretical: an unconstrained
local benchmark of `intfloat/multilingual-e5-large` (all-cores
`threads`, the library's default `batch_size=256` embedding an entire
document's chunks in one call) exhausted a development machine's RAM.
Every caller goes through this adapter, so this is the one place that
guarantees production ingestion can't repeat it — never a concern a
caller has to remember to configure.
"""

from __future__ import annotations

from collections.abc import Sequence

from fastembed import TextEmbedding

from rag.embeddings.port import Vector

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THREADS = 2
DEFAULT_BATCH_SIZE = 16


def _uses_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower()


class FastEmbedAdapter:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        threads: int | None = DEFAULT_THREADS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._model = TextEmbedding(model_name=model_name, threads=threads)
        self._use_e5_prefixes = _uses_e5_prefixes(model_name)
        self._batch_size = batch_size

    def embed_query(self, text: str) -> Vector:
        prefixed = (_QUERY_PREFIX + text) if self._use_e5_prefixes else text
        (vector,) = self._model.embed([prefixed], batch_size=1, parallel=None)
        return vector.tolist()  # type: ignore[no-any-return]

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if self._use_e5_prefixes:
            texts = [_PASSAGE_PREFIX + text for text in texts]
        return [
            vector.tolist()
            for vector in self._model.embed(list(texts), batch_size=self._batch_size, parallel=None)
        ]

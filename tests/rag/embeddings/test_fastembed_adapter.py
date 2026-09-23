"""`rag.embeddings.fastembed_adapter`.

`test_uses_e5_prefixes_*` need no model download and always run. The
rest require the real `fastembed` model download and are skipped
unless `RAG_RUN_REAL_EMBEDDING_TESTS=1` is set — CI never sets it (see
`specs/regulatory-knowledge-base/spec.md` — "Automated tests never
require the corpus or embedding model to be fetched").
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from rag.embeddings.fastembed_adapter import DEFAULT_MODEL, FastEmbedAdapter, _uses_e5_prefixes

_run_real = os.environ.get("RAG_RUN_REAL_EMBEDDING_TESTS") == "1"


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("intfloat/multilingual-e5-large", True),
        ("intfloat/multilingual-e5-small", True),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False),
        ("BAAI/bge-small-en-v1.5", False),
    ],
)
def test_uses_e5_prefixes_detects_the_e5_family(model_name: str, expected: bool) -> None:
    assert _uses_e5_prefixes(model_name) is expected


@pytest.mark.skipif(not _run_real, reason="requires downloading the real fastembed model")
def test_embed_query_returns_the_expected_dimension_for_the_default_model() -> None:
    adapter = FastEmbedAdapter(DEFAULT_MODEL)

    vector = adapter.embed_query("Qual é a taxa de juros do crédito pessoal?")

    assert len(vector) == 384


@pytest.mark.skipif(not _run_real, reason="requires downloading the real fastembed model")
def test_default_model_sends_the_text_unprefixed(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = FastEmbedAdapter(DEFAULT_MODEL)
    assert adapter._use_e5_prefixes is False

    captured: list[str] = []
    original_embed = adapter._model.embed

    def spy_embed(documents: Any, **kwargs: Any) -> Any:
        captured.extend(documents)
        return original_embed(documents, **kwargs)

    monkeypatch.setattr(adapter._model, "embed", spy_embed)

    adapter.embed_query("empréstimo pessoal")

    assert captured == ["empréstimo pessoal"]


@pytest.mark.skipif(not _run_real, reason="requires downloading the real fastembed model")
def test_e5_flagged_adapter_prefixes_query_and_passage_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reuses the lightweight default model's already-downloaded weights
    # rather than downloading multilingual-e5-large just to exercise
    # the prefixing branch — the flag is what's under test here.
    adapter = FastEmbedAdapter(DEFAULT_MODEL)
    adapter._use_e5_prefixes = True

    captured: list[str] = []
    original_embed = adapter._model.embed

    def spy_embed(documents: Any, **kwargs: Any) -> Any:
        captured.extend(documents)
        return original_embed(documents, **kwargs)

    monkeypatch.setattr(adapter._model, "embed", spy_embed)

    adapter.embed_query("empréstimo pessoal")
    adapter.embed_documents(["condições do produto"])

    assert captured == ["query: empréstimo pessoal", "passage: condições do produto"]

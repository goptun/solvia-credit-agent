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

import numpy as np
import pytest

import rag.embeddings.fastembed_adapter as fastembed_adapter_module
from rag.embeddings.fastembed_adapter import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_THREADS,
    FastEmbedAdapter,
    _uses_e5_prefixes,
)

_run_real = os.environ.get("RAG_RUN_REAL_EMBEDDING_TESTS") == "1"


class _FakeTextEmbedding:
    """Stands in for `fastembed.TextEmbedding` so the thread/batch-size
    wiring (`RagSettings.rag_embedding_threads`/`rag_embedding_batch_
    size`, see their docstrings for why: a real unconstrained benchmark
    of `intfloat/multilingual-e5-large` exhausted a development
    machine's RAM) can be verified in CI without downloading a real
    model."""

    def __init__(self, model_name: str, threads: int | None = None) -> None:
        self.init_kwargs = {"model_name": model_name, "threads": threads}

    def embed(self, documents: Any, **kwargs: Any) -> Any:
        self.last_embed_documents = list(documents)
        self.last_embed_kwargs = kwargs
        return [np.zeros(3) for _ in documents]


def test_constructor_passes_threads_to_text_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fastembed_adapter_module, "TextEmbedding", _FakeTextEmbedding)

    adapter = FastEmbedAdapter()

    assert adapter._model.init_kwargs == {  # type: ignore[attr-defined]
        "model_name": DEFAULT_MODEL,
        "threads": DEFAULT_THREADS,
    }


def test_constructor_accepts_a_custom_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fastembed_adapter_module, "TextEmbedding", _FakeTextEmbedding)

    adapter = FastEmbedAdapter(threads=4)

    assert adapter._model.init_kwargs["threads"] == 4  # type: ignore[attr-defined]


def test_embed_documents_caps_batch_size_and_disables_multiprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fastembed_adapter_module, "TextEmbedding", _FakeTextEmbedding)
    adapter = FastEmbedAdapter()

    adapter.embed_documents(["a", "b"])

    assert adapter._model.last_embed_kwargs == {  # type: ignore[attr-defined]
        "batch_size": DEFAULT_BATCH_SIZE,
        "parallel": None,
    }


def test_embed_query_uses_batch_size_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fastembed_adapter_module, "TextEmbedding", _FakeTextEmbedding)
    adapter = FastEmbedAdapter()

    adapter.embed_query("empréstimo pessoal")

    assert adapter._model.last_embed_kwargs == {  # type: ignore[attr-defined]
        "batch_size": 1,
        "parallel": None,
    }


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

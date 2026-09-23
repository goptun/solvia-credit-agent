"""`rag.embeddings.fake` — deterministic, no model/network involved."""

from __future__ import annotations

from rag.embeddings.fake import FakeEmbeddings


def test_embed_query_is_deterministic() -> None:
    embeddings = FakeEmbeddings()

    assert embeddings.embed_query("mesmo texto") == embeddings.embed_query("mesmo texto")


def test_embed_documents_is_deterministic() -> None:
    embeddings = FakeEmbeddings()

    first = embeddings.embed_documents(["a", "b"])
    second = embeddings.embed_documents(["a", "b"])

    assert first == second


def test_different_text_yields_a_different_vector() -> None:
    embeddings = FakeEmbeddings()

    assert embeddings.embed_query("texto um") != embeddings.embed_query("texto dois")


def test_embed_query_and_embed_documents_agree_on_the_same_text() -> None:
    embeddings = FakeEmbeddings()

    assert embeddings.embed_query("mesmo texto") == embeddings.embed_documents(["mesmo texto"])[0]


def test_vector_has_the_expected_dimension() -> None:
    embeddings = FakeEmbeddings()

    assert len(embeddings.embed_query("qualquer texto")) == 384

"""The committed corpus chunk fixture: size, freshness against the manifest,
round trip and diffing."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

from evals.adapters.fixture import (
    FIXTURE_PATH,
    MAX_FIXTURE_BYTES,
    Fixture,
    FixtureDocument,
    diff_fixtures,
    read_fixture,
    stale_documents,
    write_fixture,
)
from rag.corpus.manifest import load_manifest
from rag.ingest.indexing import ChunkRecord


def _record(chunk_id: str = "c1", content: str = "Art. 1º texto") -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="cdc-consolidada",
        source_type="regulation",
        norm="Lei nº 8.078/1990",
        article_ref="art. 1º",
        hierarchy_path="art. 1º",
        source_url="https://www.planalto.gov.br/x.htm",
        version_date=date(2026, 1, 1),
        amendment_note=None,
        content=content,
    )


def _fixture() -> Fixture:
    return Fixture(
        chunk_max_chars=1500,
        documents={
            "cdc-consolidada": FixtureDocument(
                sha256="abc", source_type="regulation", url="https://x", retrieved_at="2026-01-01"
            )
        },
        records=[_record()],
    )


def test_committed_fixture_is_within_the_size_cap() -> None:
    assert FIXTURE_PATH.stat().st_size <= MAX_FIXTURE_BYTES


def test_committed_fixture_covers_every_manifest_document_with_provenance() -> None:
    fixture = read_fixture()

    assert set(fixture.documents) == {doc.id for doc in load_manifest().documents}
    assert {record.document_id for record in fixture.records} == set(fixture.documents)
    regulation = [r for r in fixture.records if r.source_type == "regulation"]
    assert regulation and all(r.source_url and r.norm for r in regulation)
    assert len({r.chunk_id for r in fixture.records}) == len(fixture.records)


def test_committed_fixture_is_in_sync_with_the_manifest_hashes() -> None:
    """A manifest (or product-catalog) change without a fixture rebuild must
    fail here, telling the maintainer to run `python -m evals fixture build`."""
    stale = stale_documents(read_fixture())

    assert stale == [], f"rebuild the fixture (python -m evals fixture build): {stale}"


def test_an_altered_recorded_hash_is_reported_as_stale() -> None:
    fixture = read_fixture()
    doc_id = next(iter(fixture.documents))
    altered = replace(
        fixture,
        documents={
            **fixture.documents,
            doc_id: replace(fixture.documents[doc_id], sha256="0" * 64),
        },
    )

    assert stale_documents(altered) == [doc_id]


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "fixture.jsonl"

    write_fixture(_fixture(), path)

    assert read_fixture(path) == _fixture()


def test_diff_reports_added_removed_and_changed_chunks() -> None:
    committed = _fixture()
    rebuilt = replace(
        committed,
        records=[_record("c1", "Art. 1º texto alterado"), _record("c2")],
    )

    problems = diff_fixtures(committed, rebuilt)

    assert any("only in the rebuilt corpus" in p for p in problems)
    assert any("differs" in p for p in problems)
    assert diff_fixtures(committed, committed) == []

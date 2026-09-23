"""`rag.ingest.pdf_source` against a real 3-page slice of the fetched
Open Finance consolidated PDF (pages 2, 3, and 27) — not a synthetic
PDF. See `design.md` — "Source format handling".
"""

from __future__ import annotations

from pathlib import Path

from rag.ingest.pdf_source import _rejoin_hyphenation, extract_pdf

_FIXTURE = Path(__file__).parent / "fixtures" / "open_finance_snippet.pdf"


def _extracted() -> tuple[str, tuple]:  # type: ignore[type-arg]
    result = extract_pdf(_FIXTURE.read_bytes())
    return result.text, result.amendment_notes


def test_repeated_footer_is_removed_from_every_page() -> None:
    text, _ = _extracted()

    assert "Resolução Conjunta nº 1, de 4 de maio de 2020" not in text
    assert "Página" not in text


def test_revocation_note_is_metadata_not_body_text() -> None:
    text, notes = _extracted()

    assert "Revogado" not in text
    assert any("Revogado pela Resolução Conjunta nº 3" in note.note for note in notes)


def test_redacao_dada_notes_are_metadata_not_body_text() -> None:
    text, notes = _extracted()

    assert "Redação dada" not in text
    assert any("Redação dada" in note.note for note in notes)


def test_substantive_body_text_survives() -> None:
    text, _ = _extracted()

    assert "instituição iniciadora de transação de pagamento" in text


def test_rejoin_hyphenation_on_a_representative_case() -> None:
    # Not exercised by the real fetched PDFs (neither contains a
    # hyphenated line break — see pdf_source.py's docstring), so this
    # exercises the helper directly against a representative case.
    lines = ["O contrato prevê a resolução excepcio-", "nalmente antes do prazo."]

    joined = _rejoin_hyphenation(lines)

    assert joined == ["O contrato prevê a resolução excepcionalmente antes do prazo."]


def test_rejoin_hyphenation_leaves_non_hyphenated_lines_alone() -> None:
    lines = ["Primeira linha completa.", "Segunda linha completa."]

    assert _rejoin_hyphenation(lines) == lines

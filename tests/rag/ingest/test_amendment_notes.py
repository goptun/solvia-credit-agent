"""`rag.ingest.amendment_notes` — shared note extraction used by both
the HTML and PDF source adapters."""

from __future__ import annotations

from rag.ingest.amendment_notes import extract_amendment_notes


def test_note_is_removed_from_text_and_returned_separately() -> None:
    text = "Art. 1º Texto em vigor. (Redação dada pela Lei nº 1.000, de 2020)"

    cleaned, notes = extract_amendment_notes(text)

    assert "Redação dada" not in cleaned
    assert len(notes) == 1
    assert "Redação dada pela Lei nº 1.000" in notes[0].note


def test_text_without_a_note_is_unchanged() -> None:
    text = "Art. 1º Texto simples, sem nota de alteração."

    cleaned, notes = extract_amendment_notes(text)

    assert cleaned == text
    assert notes == ()


def test_multiple_notes_are_all_captured_in_order() -> None:
    text = "I - (Revogado). II - texto vigente. (Incluído pela Lei nº 2.000, de 2021)"

    _, notes = extract_amendment_notes(text)

    assert len(notes) == 2
    assert "Revogado" in notes[0].note
    assert "Incluído" in notes[1].note
    assert notes[0].offset <= notes[1].offset

"""`rag.ingest.html_source` against real snippets from the fetched CDC and
LGPD pages — not synthetic HTML.

See `design.md` — "Source format handling". The two fixture files are
real `windows-1252`-encoded excerpts saved while preparing this change:
- `lgpd_revoked_snippet.htm`: LGPD art. 55-A, §1º, a genuinely revoked
  provision (CSS `text-decoration: line-through`, plus a `<strike>`-
  wrapped revocation note) alongside its "(Incluído pela ...)" note.
- `cdc_amended_snippet.htm`: CDC art. 31, §1º, still in force but
  amended, including a cosmetic `<s>º</s>` (not a revocation marker).
"""

from __future__ import annotations

from pathlib import Path

from rag.ingest.html_source import extract_html

_FIXTURES = Path(__file__).parent / "fixtures"


def _read_fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def test_revoked_provision_text_never_appears_in_the_output() -> None:
    result = extract_html(_read_fixture("lgpd_revoked_snippet.htm"))

    assert "natureza jurídica da ANPD é transitória" not in result.text


def test_revocation_note_is_metadata_not_body_text() -> None:
    result = extract_html(_read_fixture("lgpd_revoked_snippet.htm"))

    assert "Revogado" not in result.text
    assert any("Revogado" in note.note for note in result.amendment_notes)


def test_inclusion_note_on_the_revoked_provision_is_also_metadata() -> None:
    result = extract_html(_read_fixture("lgpd_revoked_snippet.htm"))

    assert "Incluído pela" not in result.text
    assert any("Incluído pela" in note.note for note in result.amendment_notes)


def test_amended_but_in_force_provision_keeps_its_body_text() -> None:
    result = extract_html(_read_fixture("cdc_amended_snippet.htm"))

    assert "produto industrial" in result.text
    assert "fabricante cabe prestar as informações" in result.text


def test_amendment_note_on_an_in_force_provision_is_metadata_not_body_text() -> None:
    result = extract_html(_read_fixture("cdc_amended_snippet.htm"))

    assert "Redação dada" not in result.text
    assert any("Redação dada pela Lei" in note.note for note in result.amendment_notes)


def test_cosmetic_bare_s_tag_is_preserved_not_treated_as_revocation() -> None:
    result = extract_html(_read_fixture("cdc_amended_snippet.htm"))

    assert "§ 1º" in result.text


def test_decodes_windows_1252_without_declared_charset() -> None:
    result = extract_html(_read_fixture("cdc_amended_snippet.htm"))

    assert "informações" in result.text
    assert "�" not in result.text

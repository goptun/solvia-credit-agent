"""Shared amendment/revocation note detection.

Both `html_source.py` and `pdf_source.py` use the same pattern and
`AmendmentNote` type — see `design.md` — "Amendment-note extraction is
shared, not per-format": BCB's PDF consolidation already replaces a
revoked item's substantive text with just the note (no separate
struck-through original to remove), using the same note wording
Planalto's HTML uses for in-force amended provisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

AMENDMENT_NOTE_PATTERN = re.compile(
    r"\((Reda[çc][ãa]o dada|Inclu[íi]do|Revogad[oa]s?)[^)]*\)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class AmendmentNote:
    """One amendment/revocation note, and where it was removed from
    the cleaned body text (a character offset into that text), so a
    later step can attribute it to the nearest article."""

    offset: int
    note: str


def extract_amendment_notes(text: str) -> tuple[str, tuple[AmendmentNote, ...]]:
    """Pull amendment/revocation notes out of `text`, returning the
    cleaned text (notes removed) and the extracted notes."""
    notes: list[AmendmentNote] = []
    pieces: list[str] = []
    last_end = 0
    for match in AMENDMENT_NOTE_PATTERN.finditer(text):
        pieces.append(text[last_end : match.start()])
        offset = sum(len(piece) for piece in pieces)
        notes.append(AmendmentNote(offset=offset, note=match.group(0).strip()))
        last_end = match.end()
    pieces.append(text[last_end:])
    return "".join(pieces), tuple(notes)

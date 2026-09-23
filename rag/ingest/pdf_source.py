"""PDF source-format extraction for BCB resolutions.

See `design.md` — "Source format handling: HTML (Planalto) and PDF
(BCB)". `pdfplumber` is used (over `pypdf`) specifically because it
returns each line's bounding box, so header/footer removal can use
*position* rather than exact-text frequency alone — verified against
the real fetched CET-disclosure PDF, whose footer swaps word order
between odd/even pages ("Página 2 de 3 Resolução CMN ..." vs "Resolução
CMN ... Página 3 de 3"), which a plain-string frequency match would
never recognize as the same repeated footer.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pdfplumber

from rag.ingest.amendment_notes import AmendmentNote, extract_amendment_notes

_MARGIN_FRACTION = 0.15
"""A line whose vertical position falls within this fraction of the
page's top or bottom is a header/footer *candidate* — actual removal
still requires it to recur across pages, see `_REPEAT_THRESHOLD`."""

_REPEAT_THRESHOLD = 0.5
"""A margin-band line's normalized key must recur on at least this
fraction of pages to be treated as a repeated header/footer, rather
than a coincidence (e.g. the last page's substantive text happening to
reach the bottom margin)."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    amendment_notes: tuple[AmendmentNote, ...]


def _normalize_for_frequency(text: str) -> str:
    """A digit-insensitive, word-order-insensitive key so a footer that
    varies by page number and word order (see module docstring) still
    matches itself across pages."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    return " ".join(sorted(words))


def _rejoin_hyphenation(lines: list[str]) -> list[str]:
    """Rejoin a word split across a line break by a trailing hyphen.

    Not exercised by the real fetched BCB PDFs — verified while
    implementing this that neither contains a single hyphenated line
    break (Brazilian official PDF typesetting here wraps whole words
    instead) — so this is tested directly against a representative
    synthetic case rather than a real snippet; kept for robustness
    against a future document that does hyphenate.
    """
    joined: list[str] = []
    for line in lines:
        if joined and joined[-1].endswith("-") and not joined[-1].endswith("--"):
            joined[-1] = joined[-1][:-1] + line.lstrip()
        else:
            joined.append(line)
    return joined


def extract_pdf(content: bytes) -> ExtractedDocument:
    """Extract normalized, header/footer-free running text from a PDF's
    raw bytes, splitting off amendment/revocation notes as metadata."""
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        pages = [(page.height, page.extract_text_lines()) for page in pdf.pages]

    total_pages = len(pages)
    key_page_counts: dict[str, set[int]] = {}
    for page_index, (height, lines) in enumerate(pages):
        top_cutoff = height * _MARGIN_FRACTION
        bottom_cutoff = height * (1 - _MARGIN_FRACTION)
        for line in lines:
            if line["top"] <= top_cutoff or line["top"] >= bottom_cutoff:
                key = _normalize_for_frequency(line["text"])
                if key:
                    key_page_counts.setdefault(key, set()).add(page_index)

    repeated_keys = {
        key
        for key, page_set in key_page_counts.items()
        if len(page_set) / total_pages >= _REPEAT_THRESHOLD
    }

    body_lines: list[str] = []
    for height, lines in pages:
        top_cutoff = height * _MARGIN_FRACTION
        bottom_cutoff = height * (1 - _MARGIN_FRACTION)
        for line in lines:
            in_margin = line["top"] <= top_cutoff or line["top"] >= bottom_cutoff
            if in_margin and _normalize_for_frequency(line["text"]) in repeated_keys:
                continue
            body_lines.append(line["text"])

    text = "\n".join(_rejoin_hyphenation(body_lines))
    cleaned_text, notes = extract_amendment_notes(text)
    return ExtractedDocument(text=cleaned_text, amendment_notes=notes)

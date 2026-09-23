"""HTML source-format extraction for Planalto "compilado" pages.

See `design.md` — "Source format handling: HTML (Planalto) and PDF
(BCB)". Verified against the real fetched CDC and LGPD pages:

- No declared charset (neither `<meta>` nor HTTP header) — the real
  bytes are `windows-1252`, not UTF-8.
- A revoked provision is marked with an inline `style` carrying
  `text-decoration: line-through` (or occasionally wrapped in a
  `<strike>`/`<del>` tag), never a bare `<s>` — Planalto uses bare `<s>`
  cosmetically (e.g. `§ 1<s>º</s>`), which must be left alone.
- An amendment/revocation note (e.g. "(Incluído pela Lei nº ...)") is
  often nested *inside* the very element it marks as revoked — so notes
  are pulled out of every text node in one document-order pass, whether
  or not that node's text ends up in the body output, rather than only
  being extracted after revoked elements are already gone.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from rag.ingest.amendment_notes import AMENDMENT_NOTE_PATTERN, AmendmentNote

_LINE_THROUGH_PATTERN = re.compile(r"text-decoration\s*:\s*[^;\"]*line-through", re.IGNORECASE)

DEFAULT_ENCODING = "windows-1252"
"""Planalto's "compilado" pages declare no charset at all (verified
against the real fetched CDC page) — this is the fallback, not a guess."""


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    amendment_notes: tuple[AmendmentNote, ...]


def _is_revoked_element(tag: Tag) -> bool:
    if tag.name in ("strike", "del"):
        return True
    style = tag.get("style")
    return isinstance(style, str) and bool(_LINE_THROUGH_PATTERN.search(style))


def _iter_text_nodes(node: Tag, revoked_ancestor: bool = False) -> Iterator[tuple[str, bool]]:
    """Walk every text node in document order, tagging each with
    whether it sits inside a revoked element (directly or via an
    ancestor)."""
    for child in node.children:
        if isinstance(child, NavigableString):
            yield str(child), revoked_ancestor
        elif isinstance(child, Tag):
            yield from _iter_text_nodes(child, revoked_ancestor or _is_revoked_element(child))


def extract_html(
    content: bytes,
    *,
    declared_encoding: str | None = None,
    default_encoding: str = DEFAULT_ENCODING,
) -> ExtractedDocument:
    """Decode, drop revoked content, and pull amendment notes out of a
    Planalto "compilado" page's raw bytes."""
    encoding = declared_encoding or default_encoding
    html = content.decode(encoding, errors="replace")
    soup = BeautifulSoup(html, "lxml")

    pieces: list[str] = []
    notes: list[AmendmentNote] = []
    current_length = 0

    def emit(text: str) -> None:
        nonlocal current_length
        pieces.append(text)
        current_length += len(text)

    for text, is_revoked in _iter_text_nodes(soup):
        last_end = 0
        for match in AMENDMENT_NOTE_PATTERN.finditer(text):
            if not is_revoked:
                emit(text[last_end : match.start()])
            notes.append(AmendmentNote(offset=current_length, note=match.group(0).strip()))
            last_end = match.end()
        if not is_revoked:
            emit(text[last_end:])

    return ExtractedDocument(text="".join(pieces), amendment_notes=tuple(notes))

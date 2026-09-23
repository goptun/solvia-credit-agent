"""The normalized output both source-format adapters produce.

`html_source.extract_html` and `pdf_source.extract_pdf` both return
this same shape, so `chunking.py` (and anything downstream) can treat
either source format uniformly — see `design.md` — "Chunking strategy".
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.ingest.amendment_notes import AmendmentNote


@dataclass(frozen=True)
class ExtractedDocument:
    text: str
    amendment_notes: tuple[AmendmentNote, ...]

"""Structure-aware chunking of normalized legal text.

Splits an `ExtractedDocument`'s text along Brazilian legislative
drafting structure (Capítulo, Seção, Artigo, parágrafo) into `Chunk`s,
per `design.md` — "Chunking strategy". Article header patterns
(`Art. 54-B.`, `Art. 1°`, `Art. 1º`) and chapter/section headers
(`CAPÍTULO IV`, `Seção II`) below are calibrated against the real text
`html_source.extract_html`/`pdf_source.extract_pdf` produce from the
fetched corpus, not assumed formatting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from rag.corpus.manifest import SourceType
from rag.ingest.amendment_notes import AmendmentNote
from rag.ingest.extracted_document import ExtractedDocument

_ARTICLE_PATTERN = re.compile(r"Art\.\s*\d+(?:[º°])?(?:-[A-Z])?\.?")
"""`\\.?` at the end is deliberate: verified against the real fetched
CDC text that single/double-digit articles print as `"Art. 1°"` (no
trailing period; the ordinal marker alone is Planalto's convention),
while double-digit-and-up articles print as `"Art. 10."` (period,
no ordinal marker) — a mandatory trailing period would silently miss
every article below 10."""
_PARAGRAPH_PATTERN = re.compile(r"§\s*\d+[º°]?|Par[áa]grafo\s+[úu]nico")
_HIERARCHY_HEADER_PATTERN = re.compile(
    r"(CAP[ÍI]TULO\s+[IVXLCDM]+|Se[çc][ãa]o\s+[IVXLCDM]+)", re.MULTILINE
)


@dataclass(frozen=True)
class Chunk:
    document_id: str
    norm: str | None
    source_type: SourceType
    article_ref: str | None
    hierarchy_path: str
    source_url: str | None
    version_date: date | None
    amendment_note: str | None
    content: str


def _normalize_article_ref(header: str) -> str:
    """`"Art. 54-B."` -> `"art. 54-B"` — matches `design.md`'s citation
    format (e.g. `"art. 54-B"`)."""
    return header.strip().rstrip(".").replace("Art.", "art.", 1)


def _hierarchy_path_at(text: str, offset: int) -> str:
    """The most recent Capítulo/Seção header (of each kind) at or
    before `offset`, joined as a breadcrumb."""
    chapter: str | None = None
    section: str | None = None
    for match in _HIERARCHY_HEADER_PATTERN.finditer(text, endpos=offset):
        header = re.sub(r"\s+", " ", match.group(0)).strip()
        if header.upper().startswith("CAP"):
            chapter = header
            section = None  # a new chapter resets the current section
        else:
            section = header
    return " > ".join(part for part in (chapter, section) if part)


def _notes_in_span(notes: tuple[AmendmentNote, ...], start: int, end: int) -> list[str]:
    """Every distinct note within `[start, end)`, in order, deduplicated
    — the same note text can legitimately appear once per sub-provision
    of an article that was inserted as a whole (verified against the
    real CDC art. 54-B, whose every paragraph individually repeats
    "(Incluído pela Lei nº 14.181, de 2021)")."""
    seen: dict[str, None] = {}
    for note in notes:
        if start <= note.offset < end:
            seen.setdefault(note.note, None)
    return list(seen)


def _split_oversized_article(header: str, body: str, chunk_max_chars: int) -> list[str]:
    """Split an oversized article by paragraph boundary, repeating the
    article header on every resulting piece (see `design.md` —
    "Chunking strategy")."""
    matches = list(_PARAGRAPH_PATTERN.finditer(body))
    if not matches:
        # No paragraph markers to split on — emit as one piece anyway
        # rather than silently truncating; the caller's budget is a
        # target, not a hard limit that must never be exceeded.
        return [f"{header} {body}".strip()]

    pieces: list[str] = []
    caput = body[: matches[0].start()].strip()
    if caput:
        pieces.append(f"{header} {caput}".strip())

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        piece = body[match.start() : end].strip()
        pieces.append(f"{header} {piece}".strip())

    return pieces


def chunk_extracted_document(
    document_id: str,
    source_type: SourceType,
    extracted: ExtractedDocument,
    *,
    norm: str | None = None,
    source_url: str | None = None,
    version_date: date | None = None,
    chunk_max_chars: int = 1500,
) -> list[Chunk]:
    """Split `extracted.text` into per-article `Chunk`s. A document with
    no article structure at all (e.g. the product catalog) is emitted
    as a single chunk."""
    text = extracted.text
    article_matches = list(_ARTICLE_PATTERN.finditer(text))

    if not article_matches:
        notes = [n.note for n in extracted.amendment_notes]
        return [
            Chunk(
                document_id=document_id,
                norm=norm,
                source_type=source_type,
                article_ref=None,
                hierarchy_path="",
                source_url=source_url,
                version_date=version_date,
                amendment_note="; ".join(notes) or None,
                content=text.strip(),
            )
        ]

    chunks: list[Chunk] = []
    for i, match in enumerate(article_matches):
        start = match.start()
        end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
        header = match.group(0)
        body = text[match.end() : end]
        article_text = f"{header} {body}".strip()
        article_ref = _normalize_article_ref(header)
        hierarchy_chain = _hierarchy_path_at(text, start)
        hierarchy_path = (
            f"{hierarchy_chain} > {header.strip().rstrip('.')}"
            if hierarchy_chain
            else header.strip().rstrip(".")
        )
        notes = _notes_in_span(extracted.amendment_notes, start, end)
        amendment_note = "; ".join(notes) or None

        if len(article_text) <= chunk_max_chars:
            chunks.append(
                Chunk(
                    document_id=document_id,
                    norm=norm,
                    source_type=source_type,
                    article_ref=article_ref,
                    hierarchy_path=hierarchy_path,
                    source_url=source_url,
                    version_date=version_date,
                    amendment_note=amendment_note,
                    content=article_text,
                )
            )
        else:
            for piece in _split_oversized_article(header, body, chunk_max_chars):
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        norm=norm,
                        source_type=source_type,
                        article_ref=article_ref,
                        hierarchy_path=hierarchy_path,
                        source_url=source_url,
                        version_date=version_date,
                        amendment_note=amendment_note,
                        content=piece,
                    )
                )

    return chunks

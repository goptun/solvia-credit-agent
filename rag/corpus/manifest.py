"""Corpus manifest schema and loader.

`rag/corpus/manifest.yaml` is the single source of truth for what
belongs in the regulatory knowledge base — see `design.md` — "Corpus
pipeline and provenance".
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

SourceType = Literal["regulation", "product_catalog"]

_MANIFEST_PATH = Path(__file__).parent / "manifest.yaml"


class ManifestDocument(BaseModel):
    """One corpus document's provenance record."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    norm: str | None
    source_type: SourceType
    url: str | None
    retrieved_at: date
    version_date: date | None
    sha256: str | None

    @model_validator(mode="after")
    def _regulation_documents_have_a_source_url(self) -> ManifestDocument:
        if self.source_type == "regulation" and not self.url:
            raise ValueError(f"regulation document {self.id!r} must have a source url")
        return self


class Manifest(BaseModel):
    """The full corpus manifest."""

    model_config = ConfigDict(frozen=True)

    documents: list[ManifestDocument]


def load_manifest(path: Path = _MANIFEST_PATH) -> Manifest:
    """Load and validate the corpus manifest from `path`."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)

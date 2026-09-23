"""Shared text normalization for deterministic keyword/regex matching."""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Lowercase and strip accents, so pattern matching is accent-insensitive."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()

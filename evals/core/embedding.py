"""Minimal embedding callable type shared by adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence

EmbedDocuments = Callable[[Sequence[str]], list[list[float]]]
"""Texts to vectors (passages/documents)."""
EmbedQuery = Callable[[str], list[float]]
"""One query text to a vector."""

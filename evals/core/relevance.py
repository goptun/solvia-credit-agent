"""Whether the retrieval evaluation applies to a change (design.md, Decision 6)."""

from __future__ import annotations

from collections.abc import Iterable
from fnmatch import fnmatchcase

RELEVANT_PATTERNS: tuple[str, ...] = (
    "rag/*",
    "evals/*",
    "uv.lock",
    "apps/agent/nodes/compliance*.py",
)
"""`fnmatch` `*` also crosses `/`, so `rag/*` covers the whole package."""


def retrieval_eval_relevant(changed_files: Iterable[str], *, on_main: bool = False) -> bool:
    """Always true on `main`; on a PR, true when the diff touches the retrieval
    stack, the harness, the lockfile or the compliance nodes."""
    if on_main:
        return True
    return any(
        fnmatchcase(path, pattern) for path in changed_files for pattern in RELEVANT_PATTERNS
    )

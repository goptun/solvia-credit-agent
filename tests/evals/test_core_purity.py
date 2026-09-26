"""`evals.core` is pure Python: no LangChain, psycopg or FastAPI imports."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN = ("langchain", "langgraph", "psycopg", "fastapi", "apps", "rag", "langfuse")
_CORE = Path(__file__).resolve().parents[2] / "evals" / "core"


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_modules_import_nothing_impure() -> None:
    offenders = {
        path.name: sorted(_imported_roots(path) & set(_FORBIDDEN)) for path in _CORE.glob("*.py")
    }

    assert {name: roots for name, roots in offenders.items() if roots} == {}

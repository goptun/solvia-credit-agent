"""Which diffs trigger the retrieval evaluation in CI."""

from __future__ import annotations

import io

import pytest

from evals.cli import main
from evals.core.relevance import retrieval_eval_relevant


@pytest.mark.parametrize(
    "path",
    [
        "rag/retrieval/hybrid.py",
        "rag/ingest/chunking/legal.py",
        "evals/core/stats.py",
        "evals/datasets/retrieval.yaml",
        "uv.lock",
        "apps/agent/nodes/compliance_guard.py",
    ],
)
def test_retrieval_stack_changes_are_relevant(path: str) -> None:
    assert retrieval_eval_relevant(["README.md", path])


@pytest.mark.parametrize(
    "files",
    [
        ["README.md", "docs/adr/ADR-006.md"],
        ["openspec/changes/add-llm-evals/tasks.md"],
        ["apps/api/main.py", "apps/agent/nodes/router.py"],
        [".github/workflows/ci.yml"],
        [],
    ],
)
def test_docs_and_unrelated_code_are_not_relevant(files: list[str]) -> None:
    assert not retrieval_eval_relevant(files)


def test_pushes_to_main_are_always_relevant() -> None:
    assert retrieval_eval_relevant(["README.md"], on_main=True)
    assert retrieval_eval_relevant([], on_main=True)


def test_the_command_reads_the_diff_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("README.md\nrag/settings.py\n"))
    assert main(["relevant"]) == 0
    assert capsys.readouterr().out.strip() == "true"

    monkeypatch.setattr("sys.stdin", io.StringIO("README.md\n"))
    assert main(["relevant"]) == 0
    assert capsys.readouterr().out.strip() == "false"

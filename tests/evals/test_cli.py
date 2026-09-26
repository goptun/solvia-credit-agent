"""Command-line surface of `python -m evals`."""

from __future__ import annotations

import pytest

from evals.cli import build_parser


def test_help_lists_every_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for name in ("run", "baseline", "report", "review-sample", "fixture", "langfuse"):
        assert name in output

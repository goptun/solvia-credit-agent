"""`python -m evals` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from evals.datasets import DATASET_NAMES, load_dataset
from evals.review import render_review
from evals.settings import get_evals_settings

Handler = Callable[[argparse.Namespace], int]


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"`evals {args.command}` is not implemented yet", file=sys.stderr)
    return 2


def _review_sample(args: argparse.Namespace) -> int:
    names = DATASET_NAMES if args.dataset == "all" else (args.dataset,)
    for name in names:
        print(render_review(load_dataset(name), n=args.n, seed=args.seed))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="Evaluation harness")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "run evaluation suites (offline or live)"),
        ("baseline", "update a committed baseline from an eligible run"),
        ("report", "render Markdown tables from a run or baseline"),
        ("review-sample", "print a stratified dataset sample for maintainer review"),
        ("fixture", "build or verify the committed corpus chunk fixture"),
        ("langfuse", "mirror datasets to LangFuse"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(handler=_not_implemented)
        if name == "review-sample":
            command.add_argument("--dataset", choices=("all", *DATASET_NAMES), default="all")
            command.add_argument("--n", type=int, default=15)
            command.add_argument("--seed", type=int, default=get_evals_settings().evals_seed)
            command.set_defaults(handler=_review_sample)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)

"""`python -m evals` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from evals.datasets import DATASET_NAMES, current_hashes, load_dataset
from evals.review import render_review
from evals.settings import get_evals_settings

Handler = Callable[[argparse.Namespace], int]


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"`evals {args.command}` is not implemented yet", file=sys.stderr)
    return 2


def _review_sample(args: argparse.Namespace) -> int:
    if args.hashes:
        print("datasets:")
        for name, approval in current_hashes().items():
            print(f"  {name}: {{version: {approval.version}, sha256: {approval.sha256}}}")
        return 0
    names = DATASET_NAMES if args.dataset == "all" else (args.dataset,)
    if not args.evidence:
        for name in names:
            print(render_review(load_dataset(name), n=args.n, seed=args.seed))
        return 0

    import psycopg

    from apps.api.settings import get_api_settings
    from evals.adapters.fixture import index_fixture, read_fixture
    from evals.adapters.retrieval import make_evidence_provider
    from rag.embeddings.fastembed_adapter import FastEmbedAdapter
    from rag.settings import get_rag_settings

    settings = get_rag_settings()
    embeddings = FastEmbedAdapter(
        settings.rag_embedding_model,
        threads=settings.rag_embedding_threads,
        batch_size=settings.rag_embedding_batch_size,
    )
    print(
        "Retrieval evidence: top-3 chunks from the committed fixture index, computed with the "
        "local embedding model on this machine (informational; no gateway call).\n"
    )
    with psycopg.connect(get_api_settings().database_url, autocommit=True) as conn:
        index_fixture(conn, read_fixture(), embeddings.embed_documents)
        provider = make_evidence_provider(conn, embeddings.embed_query)
        for name in names:
            print(render_review(load_dataset(name), n=args.n, seed=args.seed, evidence=provider))
    return 0


def _fixture(args: argparse.Namespace) -> int:
    from evals.adapters.fixture import (
        FIXTURE_PATH,
        MAX_FIXTURE_BYTES,
        build_fixture,
        diff_fixtures,
        read_fixture,
        stale_documents,
        write_fixture,
    )

    if args.action == "build":
        fixture = build_fixture()
        write_fixture(fixture)
        size = FIXTURE_PATH.stat().st_size
        print(f"wrote {len(fixture.records)} chunks to {FIXTURE_PATH} ({size} bytes)")
        if size > MAX_FIXTURE_BYTES:
            print(f"ERROR: fixture exceeds {MAX_FIXTURE_BYTES} bytes", file=sys.stderr)
            return 1
        return 0

    committed = read_fixture()
    problems = [f"stale document: {doc}" for doc in stale_documents(committed)]
    problems += diff_fixtures(committed, build_fixture(committed.chunk_max_chars))
    if problems:
        print("fixture differs from the fetched corpus:", *problems, sep="\n  ")
        return 1
    print(f"fixture verified: {len(committed.records)} chunks match the fetched corpus")
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
            command.add_argument(
                "--hashes",
                action="store_true",
                help="print the current version and hash of every dataset (for review.yaml)",
            )
            command.add_argument(
                "--evidence",
                action="store_true",
                help="show the top-3 fixture-index chunks per retrieval item (needs DATABASE_URL)",
            )
            command.set_defaults(handler=_review_sample)
        if name == "fixture":
            command.add_argument("action", choices=("build", "verify"))
            command.set_defaults(handler=_fixture)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)

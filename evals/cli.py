"""`python -m evals` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from evals.core.baseline import Baseline
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


def _run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from evals.runner import UnknownSuite, run_offline

    if args.mode == "live":
        from evals.live_cli import run_live_command

        return run_live_command(args)
    try:
        record = run_offline([name.strip() for name in args.suite.split(",")], args.seed)
    except UnknownSuite as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output:
        Path(args.output).write_text(record.to_json(), encoding="utf-8")
    else:
        print(record.to_json(), end="")
    if not args.compare_baseline:
        return 0
    from evals.core.gate import gate

    outcome = gate(record, _committed_baselines(args.baselines_dir))
    print(outcome.text, end="", file=sys.stdout if args.output else sys.stderr)
    return outcome.exit_code


def _committed_baselines(directory: str | None) -> dict[str, Baseline]:
    from pathlib import Path

    from evals.datasets import BASELINE_DIR

    baselines = {}
    for path in sorted((Path(directory) if directory else BASELINE_DIR).glob("*.json")):
        baseline = Baseline.model_validate_json(path.read_text(encoding="utf-8"))
        baselines[baseline.suite] = baseline
    return baselines


def _baseline(args: argparse.Namespace) -> int:
    from datetime import UTC, datetime
    from pathlib import Path

    from evals.adapters.environment import changed_files_since, git_dirty, head_sha
    from evals.core.baseline_update import RepoState, Source, baseline_problems, build_baseline
    from evals.core.run import RunRecord
    from evals.datasets import BASELINE_DIR, load_approvals

    source: Source = "run" if args.from_run else "artifact"
    path = Path(args.from_run or args.from_artifact)
    if path.is_dir():
        candidates = sorted(path.glob("*.json"))
        if len(candidates) != 1:
            print(f"ERROR: expected exactly one .json in {path}", file=sys.stderr)
            return 2
        path = candidates[0]
    record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
    repo = RepoState(
        head_sha=head_sha(),
        dirty=git_dirty(),
        approvals=load_approvals(),
        changed_since=changed_files_since,
    )
    problems = baseline_problems(record, args.suite, args.mode, source, repo)
    if problems:
        print(f"refusing to record the {args.suite!r} baseline:", *problems, sep="\n  - ")
        return 1
    baseline = build_baseline(record, args.suite, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))
    target = (
        Path(args.baselines_dir) if args.baselines_dir else BASELINE_DIR
    ) / f"{args.suite}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(baseline.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print(f"recorded {target} ({len(baseline.metrics)} metrics, commit {baseline.git_sha[:7]})")
    return 0


def _report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from evals.core.report import MarkersNotFound, render_baselines, render_run, replace_block
    from evals.core.run import RunRecord
    from evals.datasets import BASELINE_DIR

    baselines_dir = Path(args.baselines_dir) if args.baselines_dir else BASELINE_DIR
    if args.update_readme:
        committed = [
            Baseline.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(baselines_dir.glob("*.json"))
        ]
        readme = Path(args.readme)
        try:
            updated = replace_block(readme.read_text(encoding="utf-8"), render_baselines(committed))
        except MarkersNotFound as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        readme.write_text(updated, encoding="utf-8")
        return 0
    if not args.run:
        print("ERROR: pass --run RUN.json or --update-readme", file=sys.stderr)
        return 2
    record = RunRecord.model_validate_json(Path(args.run).read_text(encoding="utf-8"))
    baselines: dict[str, Baseline] = {}
    for path in args.baseline:
        baseline = Baseline.model_validate_json(Path(path).read_text(encoding="utf-8"))
        baselines[baseline.suite] = baseline
    print(render_run(record, baselines), end="")
    return 0


def _relevant(args: argparse.Namespace) -> int:
    from evals.core.relevance import retrieval_eval_relevant

    changed = [line.strip() for line in sys.stdin if line.strip()]
    print("true" if retrieval_eval_relevant(changed, on_main=args.on_main) else "false")
    return 0


def _langfuse(args: argparse.Namespace) -> int:
    from evals.adapters.langfuse_publisher import build_publisher
    from evals.publishing import sync_datasets

    publisher = build_publisher()
    if publisher is None:
        print("ERROR: LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are not set", file=sys.stderr)
        return 2
    if args.action == "publish":
        from pathlib import Path

        from evals.core.run import RunRecord
        from evals.publishing import publish_run_safely

        record = RunRecord.model_validate_json(Path(args.run).read_text(encoding="utf-8"))
        published = publish_run_safely(publisher, record)
        print(f"published {published} experiment(s) from {args.run} (no gateway calls)")
        return 0
    names = DATASET_NAMES if args.dataset == "all" else (args.dataset,)
    synced = sync_datasets(publisher, [load_dataset(name) for name in names])
    print(f"synced {synced} items into {len(names)} dataset(s)")
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
        ("relevant", "print whether the retrieval eval applies to the changed files on stdin"),
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
        if name == "run":
            command.add_argument("--suite", required=True, help="comma-separated suite names")
            command.add_argument("--mode", choices=("offline", "live"), required=True)
            command.add_argument("--seed", type=int, default=get_evals_settings().evals_seed)
            command.add_argument("--output", default=None, help="write the run JSON to a file")
            command.add_argument("--baselines-dir", default=None)
            command.add_argument(
                "--sample",
                type=float,
                default=None,
                help="live only: run a stratified fraction (0-1] of each dataset",
            )
            command.add_argument(
                "--report", default=None, help="live only: write evals/reports/LABEL.{json,md}"
            )
            command.add_argument(
                "--no-publish",
                action="store_true",
                help="live only: do not publish the run to LangFuse",
            )
            command.add_argument(
                "--compare-baseline",
                action="store_true",
                help="print the diff against the committed baselines; exit 1 on regression",
            )
            command.set_defaults(handler=_run)
        if name == "report":
            command.add_argument("--run", default=None, help="run JSON to render")
            command.add_argument(
                "--baseline", action="append", default=[], help="baseline JSON to diff against"
            )
            command.add_argument(
                "--update-readme",
                action="store_true",
                help="rewrite the README metrics block from the committed baselines",
            )
            command.add_argument("--readme", default="README.md")
            command.add_argument("--baselines-dir", default=None)
            command.set_defaults(handler=_report)
        if name == "langfuse":
            command.add_argument("action", choices=("sync", "publish"))
            command.add_argument("--dataset", choices=("all", *DATASET_NAMES), default="all")
            command.add_argument("--run", default=None, help="publish: a recorded live run JSON")
            command.set_defaults(handler=_langfuse)
        if name == "relevant":
            command.add_argument("--on-main", action="store_true")
            command.set_defaults(handler=_relevant)
        if name == "baseline":
            command.add_argument("action", choices=("update",))
            command.add_argument("--suite", required=True)
            command.add_argument("--mode", choices=("offline", "live"), default="offline")
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--from-run", default=None, help="a local run JSON")
            source.add_argument(
                "--from-artifact", default=None, help="a CI run artifact (file or directory)"
            )
            command.add_argument("--baselines-dir", default=None)
            command.set_defaults(handler=_baseline)
        if name == "fixture":
            command.add_argument("action", choices=("build", "verify"))
            command.set_defaults(handler=_fixture)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Handler = args.handler
    return handler(args)

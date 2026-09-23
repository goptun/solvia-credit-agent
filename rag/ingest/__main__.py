"""`python -m rag.ingest fetch` — re-fetch and verify the corpus.

This is the CLI wiring layer: the only place in `rag/ingest/` allowed to
import from `apps/` (for the product catalog's source text), per
`design.md` — "Module boundaries".
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx

from apps.agent.config.product_descriptions import GENERAL_PRODUCT_OVERVIEW, PRODUCT_DESCRIPTIONS
from rag.corpus.manifest import load_manifest
from rag.ingest.fetch import FetchResult, HashMismatchError, fetch_all

_OUTPUT_DIR = "./.data/rag_corpus"


_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
"""planalto.gov.br rejects non-browser-looking User-Agent strings outright
(`RemoteProtocolError: Server disconnected without sending a response`,
verified against the real site while implementing this) — a common
browser UA is required, not optional, to fetch these public texts."""


class _HttpxFetcher:
    def get(self, url: str) -> bytes:
        headers = {"User-Agent": _BROWSER_USER_AGENT}
        response = httpx.get(url, headers=headers, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
        return response.content


def _product_catalog_text() -> str:
    return "\n\n".join([GENERAL_PRODUCT_OVERVIEW, *PRODUCT_DESCRIPTIONS.values()])


def _print_result(result: FetchResult) -> None:
    status = "first fetch, hash recorded" if result.first_fetch else "verified, unchanged"
    print(f"[{result.document_id}] {status} — sha256={result.sha256} -> {result.output_path}")


def fetch_command() -> int:
    manifest = load_manifest()
    try:
        results = fetch_all(
            manifest,
            Path(_OUTPUT_DIR),
            http_fetcher=_HttpxFetcher(),
            generated_text_by_id={"product-catalog": _product_catalog_text()},
        )
    except HashMismatchError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for result in results:
        _print_result(result)
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "fetch":
        print("Usage: python -m rag.ingest fetch", file=sys.stderr)
        return 2
    return fetch_command()


if __name__ == "__main__":
    sys.exit(main())

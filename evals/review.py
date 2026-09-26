"""Maintainer review samples: a stratified, seeded sample of each dataset
printed as Markdown, so the same seed always shows the same items."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass

from evals.core.sampling import stratified_sample
from evals.core.schemas import (
    ComplianceDataset,
    RetrievalDataset,
    RetrievalItem,
    RouterDataset,
    RouterItem,
    SlotItem,
    SlotsDataset,
)
from evals.datasets import LoadedDataset

EvidenceProvider = Callable[[RetrievalItem], list[str]]


@dataclass(frozen=True)
class ReviewItem:
    id: str
    stratum: Hashable
    header: str
    lines: tuple[str, ...]
    retrieval: RetrievalItem | None = None


def _quote(text: str) -> str:
    return " ".join(text.split())


def _refs(item: RetrievalItem) -> str:
    if not item.expected_refs:
        return "(catalog)"
    return ", ".join(
        ref if isinstance(ref, str) else f"{ref.document}: {ref.ref}" for ref in item.expected_refs
    )


def _retrieval(item: RetrievalItem) -> ReviewItem:
    if item.kind == "answerable":
        no_accents = " · no accents" if not item.accents else ""
        return ReviewItem(
            item.id,
            ("answerable", item.style, item.document),
            f"{item.id} [answerable · {item.style} · {item.difficulty} · {item.document}"
            f"{no_accents}]",
            (
                f"question: {_quote(item.question)}",
                f"refs: {_refs(item)}",
                f'evidence: "{_quote(item.evidence or "")}"',
            ),
            retrieval=item,
        )
    return ReviewItem(
        item.id,
        ("unanswerable", item.distance),
        f"{item.id} [unanswerable · {item.distance}]",
        (f"question: {_quote(item.question)}",),
        retrieval=item,
    )


def _router(item: RouterItem) -> ReviewItem:
    also = f" · also {', '.join(item.acceptable)}" if item.acceptable else ""
    return ReviewItem(
        item.id,
        (item.category, item.expected),
        f"{item.id} [{item.category} · expected {item.expected}{also} · flow {item.active_flow}]",
        (f"message: {_quote(item.message)}",),
    )


def _slots(item: SlotItem) -> ReviewItem:
    return ReviewItem(
        item.id,
        tuple(sorted(item.tags)),
        f"{item.id} [slots · {', '.join(item.tags)}]",
        (
            f"message: {_quote(item.message)}",
            f"known: {item.known.model_dump(exclude_none=True) or '-'}",
            f"expected: {item.expected.model_dump(exclude_none=True) or '-'}",
        ),
    )


def _compliance(dataset: ComplianceDataset) -> list[ReviewItem]:
    items = [
        ReviewItem(
            a.id,
            ("approval", a.label, "adversarial" in a.tags),
            f"{a.id} [approval · {a.label} · tags: {', '.join(a.tags) or '-'}]",
            (f"text: {_quote(a.text)}",),
        )
        for a in dataset.approval
    ]
    items += [
        ReviewItem(
            p.id,
            ("pii",),
            f"{p.id} [pii masking]",
            (f"text:     {_quote(p.text)}", f"expected: {_quote(p.expected_masked)}"),
        )
        for p in dataset.pii
    ]
    return items


def review_items(loaded: LoadedDataset) -> list[ReviewItem]:
    dataset = loaded.dataset
    if isinstance(dataset, RetrievalDataset):
        return [_retrieval(item) for item in dataset.items]
    if isinstance(dataset, RouterDataset):
        return [_router(item) for item in dataset.items]
    if isinstance(dataset, SlotsDataset):
        return [_slots(item) for item in dataset.items]
    return _compliance(dataset)


def render_review(
    loaded: LoadedDataset,
    n: int,
    seed: int,
    evidence: EvidenceProvider | None = None,
) -> str:
    items = review_items(loaded)
    chosen = stratified_sample(items, lambda item: item.stratum, n, seed)
    out = [
        f"## {loaded.name} — version {loaded.version}, sha256 {loaded.sha256[:12]}…, "
        f"seed {seed}, {len(chosen)} of {len(items)} items",
        "",
    ]
    for item in chosen:
        out.append(f"### {item.header}")
        out.extend(item.lines)
        if evidence is not None and item.retrieval is not None:
            out.append("top retrieved chunks:")
            out.extend(f"  {line}" for line in evidence(item.retrieval))
        out.append("")
    return "\n".join(out)


def strata_by_id(loaded: LoadedDataset) -> dict[str, Hashable]:
    """The review strata per item id — also the strata of live `--sample`, so a
    sample covers what a review sample covers."""
    return {item.id: item.stratum for item in review_items(loaded)}

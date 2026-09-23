"""Stratified, seeded sampling (review samples and live `--sample`)."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence


def stratified_sample[T](
    items: Sequence[T], stratum: Callable[[T], Hashable], n: int, seed: int
) -> list[T]:
    """Round-robin over strata: one item from each stratum in turn until `n`
    are chosen, so every stratum appears when `n` allows it. Fully
    determined by the seed; the result keeps the dataset's item order."""
    if n >= len(items):
        return list(items)
    rng = random.Random(seed)
    groups: dict[Hashable, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[stratum(item)].append(index)
    order = sorted(groups, key=repr)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(groups[key])

    chosen: list[int] = []
    round_number = 0
    while len(chosen) < n:
        progressed = False
        for key in order:
            if round_number < len(groups[key]) and len(chosen) < n:
                chosen.append(groups[key][round_number])
                progressed = True
        if not progressed:
            break
        round_number += 1
    return [items[i] for i in sorted(chosen)]


def sample_fraction[T](
    items: Sequence[T], stratum: Callable[[T], Hashable], fraction: float, seed: int
) -> list[T]:
    """A stratified sample of about `fraction` of `items`, at least one per
    stratum when the fraction allows it."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    n = max(1, round(len(items) * fraction))
    return stratified_sample(items, stratum, n, seed)

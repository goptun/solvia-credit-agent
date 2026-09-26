"""The dataset README documents every dataset and the labelling rules."""

from __future__ import annotations

from evals.datasets import DATASET_DIR, DATASET_NAMES


def test_readme_exists_and_mentions_every_dataset_file_and_key_rule() -> None:
    text = (DATASET_DIR / "README.md").read_text(encoding="utf-8")

    for name in DATASET_NAMES:
        assert f"{name}.yaml" in text
    for rule in ("colloquial", "near_miss", "ambiguous", "continuation", "fail_closed", "derived"):
        assert rule in text
    assert "No\nreal personal data" in text or "real personal data" in text

"""Dataset loading (I/O edge): YAML files in `evals/datasets/`, validated
against the Pydantic schemas, with a content hash used by the review gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from evals.core.approval import Approval, ReviewRecord
from evals.core.schemas import (
    ComplianceDataset,
    RetrievalDataset,
    RouterDataset,
    SlotsDataset,
)

DATASET_DIR = Path(__file__).parent / "datasets"
DATASET_NAMES = ("retrieval", "router", "slots", "compliance")
REVIEW_FILE = DATASET_DIR / "review.yaml"
BASELINE_DIR = Path(__file__).parent / "baselines"

AnyDataset = RetrievalDataset | RouterDataset | SlotsDataset | ComplianceDataset


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    version: int
    sha256: str
    dataset: AnyDataset


def file_sha256(path: Path) -> str:
    """Hash of the file content with newlines normalized, so the review
    gate is stable across platforms."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_dataset(name: str, directory: Path | None = None) -> LoadedDataset:
    if name not in DATASET_NAMES:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASET_NAMES}")
    path = (directory or DATASET_DIR) / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    dataset: AnyDataset
    if name == "retrieval":
        dataset = RetrievalDataset.model_validate(raw)
    elif name == "router":
        dataset = RouterDataset.model_validate(raw)
    elif name == "slots":
        dataset = SlotsDataset.model_validate(raw)
    else:
        dataset = ComplianceDataset.model_validate(raw)
    return LoadedDataset(
        name=name, version=dataset.version, sha256=file_sha256(path), dataset=dataset
    )


def load_approvals(path: Path | None = None) -> dict[str, Approval]:
    """The approved dataset versions recorded by the maintainer."""
    raw = yaml.safe_load((path or REVIEW_FILE).read_text(encoding="utf-8"))
    return dict(ReviewRecord.model_validate(raw).datasets)


def current_hashes(directory: Path | None = None) -> dict[str, Approval]:
    """Version and content hash of every dataset as committed now — what a
    maintainer pastes into `review.yaml` after approving them."""
    loaded = (load_dataset(name, directory) for name in DATASET_NAMES)
    return {item.name: Approval(version=item.version, sha256=item.sha256) for item in loaded}

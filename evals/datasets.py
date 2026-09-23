"""Dataset loading (I/O edge): YAML files in `evals/datasets/`, validated
against the Pydantic schemas, with a content hash used by the review gate."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from evals.core.schemas import (
    ComplianceDataset,
    RetrievalDataset,
    RouterDataset,
    SlotsDataset,
)

DATASET_DIR = Path(__file__).parent / "datasets"
DATASET_NAMES = ("retrieval", "router", "slots", "compliance")

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

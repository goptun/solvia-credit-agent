"""The dataset review gate: a baseline is refused for any dataset whose
current content is not the version the maintainer approved."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict


class Approval(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    sha256: str


class ReviewRecord(BaseModel):
    """`evals/datasets/review.yaml`."""

    model_config = ConfigDict(frozen=True)

    approved_by: str
    approved_on: str
    datasets: dict[str, Approval]


def approval_problem(
    name: str, version: int, sha256: str, approvals: Mapping[str, Approval]
) -> str | None:
    """`None` when the dataset is the approved one; otherwise why not."""
    approved = approvals.get(name)
    if approved is None:
        return f"dataset {name!r} has no recorded approval; it must be reviewed first"
    if approved.sha256 != sha256:
        return (
            f"dataset {name!r} (v{version}) differs from the approved v{approved.version}; "
            "it must be reviewed again"
        )
    return None

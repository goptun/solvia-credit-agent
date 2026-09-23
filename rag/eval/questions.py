"""Retrieval-quality evaluation question set schema and loader.

See `design.md` — "Retrieval quality evaluation".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

_QUESTIONS_PATH = Path(__file__).parent / "questions.yaml"

QuestionStyle = Literal["lexical", "colloquial"]
"""`lexical`: close to the article's own wording. `colloquial`: phrased
the way a customer would actually ask, avoiding the article heading's
exact words."""

NegativeDistance = Literal["far", "near_miss"]
"""`far`: obviously outside the corpus's domain (e.g. tax law).
`near_miss`: plausible-sounding, in the same subject area as the
corpus, but not actually covered by it."""


class AnswerableQuestion(BaseModel):
    """A question with a known expected source, backed by a literal
    quote from that source so the expectation itself is verifiable."""

    model_config = ConfigDict(frozen=True)

    question: str
    expected_document_id: str
    expected_refs: list[str] | None = None
    """`None` only for a `product_catalog` question — that source has
    no article structure to cite. One or more article references when
    more than one article legitimately answers the question."""
    evidence: str
    """A literal quote (<= 200 chars) from the expected source,
    verified against the fetched corpus text — see
    `tests/rag/eval/test_questions.py`."""
    style: QuestionStyle = "lexical"

    @field_validator("evidence")
    @classmethod
    def _evidence_is_short(cls, value: str) -> str:
        if len(value) > 200:
            raise ValueError(f"evidence must be <= 200 chars, got {len(value)}")
        return value


class UnanswerableQuestion(BaseModel):
    """A plausible-sounding question the corpus does not cover —
    expects a refusal, not an answer."""

    model_config = ConfigDict(frozen=True)

    question: str
    distance: NegativeDistance


class EvalQuestionSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    answerable: list[AnswerableQuestion]
    unanswerable: list[UnanswerableQuestion]


def load_questions(path: Path = _QUESTIONS_PATH) -> EvalQuestionSet:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalQuestionSet.model_validate(raw)

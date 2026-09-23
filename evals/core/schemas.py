"""Pydantic schemas of the four evaluation datasets (design.md, Decision 2).

Difficulty is derived from other fields, never free-form, so tags cannot
drift: `derive_difficulty` is the single rule and validation compares the
declared value against it."""

from __future__ import annotations

import unicodedata
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

DOCUMENTS = (
    "cdc-consolidada",
    "lgpd",
    "open-finance-regulamento",
    "cet-disclosure",
    "product-catalog",
)
INTENTS = (
    "product_question",
    "regulatory_question",
    "loan_simulation",
    "profile_analysis",
    "complaint",
    "out_of_scope",
)
CONTINUE = "continue"
"""Router label for a message that continues an active flow (not reclassified)."""
MAX_EVIDENCE_CHARS = 200

Document = Literal[
    "cdc-consolidada", "lgpd", "open-finance-regulamento", "cet-disclosure", "product-catalog"
]
Intent = Literal[
    "product_question",
    "regulatory_question",
    "loan_simulation",
    "profile_analysis",
    "complaint",
    "out_of_scope",
]
Difficulty = Literal["easy", "medium", "hard"]


def has_accents(text: str) -> bool:
    return any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", text))


class _Item(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str


class RetrievalItem(_Item):
    """A retrieval/grounding question, answerable or deliberately not."""

    question: str
    kind: Literal["answerable", "unanswerable"]
    difficulty: Difficulty
    # answerable
    document: Document | None = None
    expected_refs: list[str] | None = None
    """One or more acceptable articles; absent only for the product catalog."""
    evidence: str | None = None
    style: Literal["lexical", "colloquial"] | None = None
    accents: bool = True
    """`False` = the question is written without accents."""
    # unanswerable
    distance: Literal["far", "near_miss"] | None = None

    @field_validator("evidence")
    @classmethod
    def _evidence_is_short(cls, value: str | None) -> str | None:
        if value is not None and len(value) > MAX_EVIDENCE_CHARS:
            raise ValueError(f"evidence must be <= {MAX_EVIDENCE_CHARS} chars, got {len(value)}")
        return value

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> Self:
        if self.kind == "answerable":
            if self.document is None or self.style is None or not self.evidence:
                raise ValueError("answerable items need document, style and evidence")
            if self.distance is not None:
                raise ValueError("answerable items must not have a distance")
            if self.document == "product-catalog":
                if self.expected_refs:
                    raise ValueError("product-catalog items have no article refs")
            elif not self.expected_refs:
                raise ValueError("regulatory items need at least one expected ref")
            if not self.accents and has_accents(self.question):
                raise ValueError("accents=false but the question contains accented characters")
        else:
            if self.distance is None:
                raise ValueError("unanswerable items need a distance")
            if any((self.document, self.expected_refs, self.evidence, self.style)):
                raise ValueError("unanswerable items must not carry answerable fields")
        if self.difficulty != derive_difficulty(self):
            raise ValueError(
                f"declared difficulty {self.difficulty!r} disagrees with the derivation rule "
                f"({derive_difficulty(self)!r})"
            )
        return self


def derive_difficulty(item: RetrievalItem) -> Difficulty:
    """easy = lexical and one ref; medium = colloquial xor multi-ref; hard =
    colloquial and (multi-ref or unaccented). Unanswerable: far = easy,
    near_miss = hard."""
    if item.kind == "unanswerable":
        return "hard" if item.distance == "near_miss" else "easy"
    colloquial = item.style == "colloquial"
    multi = len(item.expected_refs or []) > 1
    if colloquial and (multi or not item.accents):
        return "hard"
    if colloquial != multi:
        return "medium"
    return "easy"


class RouterItem(_Item):
    message: str
    active_flow: Literal["none", "consent_confirmation", "slot_filling"]
    category: Literal["clear", "ambiguous", "continuation"]
    expected: str
    """An intent, or `continue` for a continuation that must not be reclassified."""
    acceptable: list[str] = []
    """Also-correct labels; only for ambiguous items."""

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        labels = (*INTENTS, CONTINUE)
        if self.expected not in labels or any(a not in labels for a in self.acceptable):
            raise ValueError(f"unknown intent label in {self.expected!r} / {self.acceptable!r}")
        if self.category == "continuation":
            if self.expected != CONTINUE or self.active_flow == "none":
                raise ValueError("continuations expect `continue` and need an active flow")
        elif self.expected == CONTINUE:
            raise ValueError("only continuation items may expect `continue`")
        if (self.category == "ambiguous") != bool(self.acceptable):
            raise ValueError("acceptable labels are required for, and only for, ambiguous items")
        if self.expected in self.acceptable:
            raise ValueError("acceptable must not repeat the expected label")
        return self


class SlotValues(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    amount: str | None = None
    term_months: int | None = None
    amortization_type: Literal["PRICE", "SAC"] | None = None

    def filled(self) -> int:
        return sum(v is not None for v in (self.amount, self.term_months, self.amortization_type))


class SlotItem(_Item):
    message: str
    known: SlotValues = SlotValues()
    """Slots already filled before this message."""
    expected: SlotValues
    """Slots after the node handles the message; absent = must stay unset."""
    tags: list[Literal["complete", "partial", "missing", "invalid"]]

    @model_validator(mode="after")
    def _tags_match_expected(self) -> Self:
        filled = self.expected.filled()
        state = "complete" if filled == 3 else "missing" if filled == 0 else "partial"
        states = {t for t in self.tags if t != "invalid"}
        if states != {state}:
            raise ValueError(f"tags {self.tags!r} disagree with expected slots (state {state!r})")
        return self


class ApprovalItem(_Item):
    text: str
    label: Literal["promise", "hedge", "neutral"]
    tags: list[Literal["adversarial", "fail_closed", "mentions_approval"]] = []


class PiiItem(_Item):
    text: str
    expected_masked: str


def _unique_ids(ids: list[str], what: str) -> None:
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate {what} ids: {duplicates}")


class RetrievalDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    items: list[RetrievalItem]

    @model_validator(mode="after")
    def _ids(self) -> Self:
        _unique_ids([i.id for i in self.items], "retrieval")
        return self


class RouterDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    items: list[RouterItem]

    @model_validator(mode="after")
    def _ids(self) -> Self:
        _unique_ids([i.id for i in self.items], "router")
        return self


class SlotsDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    items: list[SlotItem]

    @model_validator(mode="after")
    def _ids(self) -> Self:
        _unique_ids([i.id for i in self.items], "slots")
        return self


class ComplianceDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    approval: list[ApprovalItem]
    pii: list[PiiItem]

    @model_validator(mode="after")
    def _ids(self) -> Self:
        _unique_ids([i.id for i in self.approval] + [i.id for i in self.pii], "compliance")
        return self

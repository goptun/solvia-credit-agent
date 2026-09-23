"""Dataset schemas, derived difficulty and composition validators."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evals.core.composition import (
    compliance_violations,
    retrieval_violations,
    router_violations,
    slots_violations,
)
from evals.core.schemas import (
    DOCUMENTS,
    INTENTS,
    ApprovalItem,
    ComplianceDataset,
    PiiItem,
    RetrievalDataset,
    RetrievalItem,
    RouterDataset,
    RouterItem,
    SlotItem,
    SlotsDataset,
    SlotValues,
    derive_difficulty,
)


def _answerable(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "R-001",
        "question": "O que a lei diz sobre cobrança abusiva?",
        "kind": "answerable",
        "difficulty": "easy",
        "document": "cdc-consolidada",
        "expected_refs": ["art. 42"],
        "evidence": "não será exposto a ridículo",
        "style": "lexical",
    }
    base.update(overrides)
    return base


def test_valid_answerable_item_parses() -> None:
    item = RetrievalItem.model_validate(_answerable())

    assert derive_difficulty(item) == "easy"


@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence": None},  # missing required field
        {"document": "codigo-civil"},  # unknown document
        {"style": "formal"},  # unknown label
        {"evidence": "x" * 201},  # too long
        {"expected_refs": None},  # regulatory item without refs
        {"distance": "far"},  # answerable with a distance
        {"difficulty": "hard"},  # disagrees with the derivation rule
        {"accents": False},  # says no accents but the question has them
        {"unknown_field": 1},  # extra field
    ],
)
def test_malformed_answerable_items_are_rejected(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RetrievalItem.model_validate(_answerable(**overrides))


def test_cross_document_refs_are_accepted_and_listed_as_acceptable_pairs() -> None:
    item = RetrievalItem.model_validate(
        _answerable(
            document="cet-disclosure",
            expected_refs=["art. 2º", {"document": "cdc-consolidada", "ref": "art. 54-B"}],
            difficulty="medium",
        )
    )

    assert item.acceptable() == (
        ("cet-disclosure", "art. 2º"),
        ("cdc-consolidada", "art. 54-B"),
    )
    assert derive_difficulty(item) == "medium"  # lexical but multi-ref


@pytest.mark.parametrize(
    "refs",
    [
        [{"document": "lgpd", "ref": "art. 7º"}],  # nothing in the item's own document
        ["art. 42", {"document": "product-catalog", "ref": "x"}],  # into the catalog
        ["art. 42", {"document": "codigo-civil", "ref": "art. 1"}],  # unknown document
    ],
)
def test_invalid_cross_document_refs_are_rejected(refs: list[Any]) -> None:
    with pytest.raises(ValidationError):
        RetrievalItem.model_validate(_answerable(expected_refs=refs, difficulty="medium"))


def test_catalog_items_take_no_refs() -> None:
    RetrievalItem.model_validate(
        _answerable(document="product-catalog", expected_refs=None, evidence="parcelas fixas")
    )
    with pytest.raises(ValidationError):
        RetrievalItem.model_validate(_answerable(document="product-catalog"))


@pytest.mark.parametrize(
    ("style", "refs", "accents", "expected"),
    [
        ("lexical", ["a"], True, "easy"),
        ("lexical", ["a", "b"], True, "medium"),
        ("colloquial", ["a"], True, "medium"),
        ("colloquial", ["a", "b"], True, "hard"),
        ("colloquial", ["a"], False, "hard"),
        ("lexical", ["a"], False, "easy"),
    ],
)
def test_difficulty_rule(style: str, refs: list[str], accents: bool, expected: str) -> None:
    question = "Qual e o prazo?" if not accents else "Qual é o prazo?"
    item = RetrievalItem.model_validate(
        _answerable(
            style=style, expected_refs=refs, accents=accents, difficulty=expected, question=question
        )
    )

    assert derive_difficulty(item) == expected


def test_unanswerable_items() -> None:
    far = RetrievalItem.model_validate(
        {
            "id": "R-090",
            "question": "Qual o imposto de renda?",
            "kind": "unanswerable",
            "difficulty": "easy",
            "distance": "far",
        }
    )
    near = RetrievalItem.model_validate(
        {
            "id": "R-091",
            "question": "Existe teto para juros?",
            "kind": "unanswerable",
            "difficulty": "hard",
            "distance": "near_miss",
        }
    )

    assert (derive_difficulty(far), derive_difficulty(near)) == ("easy", "hard")
    with pytest.raises(ValidationError):
        RetrievalItem.model_validate(
            {
                "id": "R-092",
                "question": "?",
                "kind": "unanswerable",
                "difficulty": "hard",
                "distance": "far",
            }
        )
    with pytest.raises(ValidationError):
        RetrievalItem.model_validate(
            {
                "id": "R-093",
                "question": "?",
                "kind": "unanswerable",
                "difficulty": "easy",
                "distance": "far",
                "document": "lgpd",
            }
        )


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RetrievalDataset.model_validate({"version": 1, "items": [_answerable(), _answerable()]})


def test_router_item_rules() -> None:
    RouterItem(
        id="T-1",
        message="quero simular",
        active_flow="none",
        category="clear",
        expected="loan_simulation",
    )
    RouterItem(
        id="T-2",
        message="sim, autorizo",
        active_flow="consent_confirmation",
        category="continuation",
        expected="continue",
    )
    RouterItem(
        id="T-3",
        message="e agora?",
        active_flow="none",
        category="ambiguous",
        expected="out_of_scope",
        acceptable=["complaint"],
    )
    with pytest.raises(ValidationError):  # continuation without an active flow
        RouterItem(
            id="T-4",
            message="sim",
            active_flow="none",
            category="continuation",
            expected="continue",
        )
    with pytest.raises(ValidationError):  # unknown intent
        RouterItem(id="T-5", message="x", active_flow="none", category="clear", expected="billing")
    with pytest.raises(ValidationError):  # ambiguous without acceptable labels
        RouterItem(
            id="T-6", message="x", active_flow="none", category="ambiguous", expected="complaint"
        )


def test_slot_tags_must_match_the_expected_slots() -> None:
    complete = SlotValues(amount="5000", term_months=12, amortization_type="PRICE")
    SlotItem(id="S-1", message="5000 em 12x Price", expected=complete, tags=["complete"])
    SlotItem(id="S-2", message="quero um emprestimo", expected=SlotValues(), tags=["missing"])
    SlotItem(
        id="S-3",
        message="-500 em 12x",
        expected=SlotValues(term_months=12),
        tags=["partial", "invalid"],
    )
    with pytest.raises(ValidationError):
        SlotItem(id="S-4", message="5000", expected=complete, tags=["partial"])


def test_approval_and_pii_items() -> None:
    ApprovalItem(id="C-1", text="Seu crédito está aprovado.", label="promise")
    PiiItem(id="P-1", text="CPF 000.000.000-00", expected_masked="CPF [DADO PROTEGIDO]")
    with pytest.raises(ValidationError):
        ApprovalItem(id="C-2", text="x", label="maybe")  # type: ignore[arg-type]


# ---- composition -----------------------------------------------------------


def _retrieval_dataset() -> RetrievalDataset:
    items: list[dict[str, Any]] = []
    number = 0
    for document in DOCUMENTS:
        for index in range(14):
            number += 1
            colloquial = index < 6  # 6/14 per document > 40%
            cross = index == 2 and document == "cet-disclosure"
            multi = (index == 0 and document == "cdc-consolidada") or cross
            unaccented = index == 1 and document == "lgpd"
            refs: list[Any] | None = (
                None if document == "product-catalog" else ["art. 1", "art. 2"][: 2 if multi else 1]
            )
            if cross:
                refs = ["art. 1", {"document": "cdc-consolidada", "ref": "art. 2"}]
            style = "colloquial" if colloquial else "lexical"
            difficulty = (
                "hard"
                if colloquial and (multi or unaccented)
                else "medium"
                if colloquial != multi
                else "easy"
            )
            items.append(
                {
                    "id": f"R-{number:03d}",
                    "question": "Qual e o prazo aplicavel?"
                    if unaccented
                    else "Qual é o prazo aplicável?",
                    "kind": "answerable",
                    "difficulty": difficulty,
                    "document": document,
                    "expected_refs": refs,
                    "evidence": "trecho literal",
                    "style": style,
                    "accents": not unaccented,
                }
            )
    for index in range(25):
        number += 1
        near = index < 16
        items.append(
            {
                "id": f"R-{number:03d}",
                "question": f"Pergunta fora do corpus {index}?",
                "kind": "unanswerable",
                "difficulty": "hard" if near else "easy",
                "distance": "near_miss" if near else "far",
            }
        )
    return RetrievalDataset.model_validate({"version": 1, "items": items})


def test_a_well_formed_retrieval_dataset_has_no_violations() -> None:
    assert retrieval_violations(_retrieval_dataset()) == []


def test_retrieval_violations_are_reported() -> None:
    base = _retrieval_dataset()
    only_far = [
        i.model_copy(update={"distance": "far", "difficulty": "easy"})
        if i.kind == "unanswerable"
        else i
        for i in base.items
    ]
    no_lgpd = [i for i in base.items if i.document != "lgpd"]

    far_violations = retrieval_violations(RetrievalDataset(version=1, items=only_far))
    lgpd_violations = retrieval_violations(RetrievalDataset(version=1, items=no_lgpd))

    assert any("near-miss" in v for v in far_violations)
    assert any("'lgpd'" in v for v in lgpd_violations)
    assert any("without accents" in v for v in lgpd_violations)


def _router_dataset(drop: str | None = None) -> RouterDataset:
    items = []
    number = 0
    for intent in INTENTS:
        for _ in range(9):
            number += 1
            items.append(
                RouterItem(
                    id=f"T-{number}",
                    message=f"m{number}",
                    active_flow="none",
                    category="clear",
                    expected=intent,
                )
            )
    for _ in range(5):
        number += 1
        items.append(
            RouterItem(
                id=f"T-{number}",
                message="ambíguo",
                active_flow="none",
                category="ambiguous",
                expected="complaint",
                acceptable=["out_of_scope"],
            )
        )
    number += 1
    items.append(
        RouterItem(
            id=f"T-{number}",
            message="sim, autorizo",
            active_flow="consent_confirmation",
            category="continuation",
            expected="continue",
        )
    )
    if drop:
        items = [i for i in items if i.expected != drop]
    return RouterDataset(version=1, items=items)


def test_router_composition() -> None:
    assert router_violations(_router_dataset()) == []
    assert any("continuation" in v for v in router_violations(_router_dataset(drop="continue")))
    missing_intent = router_violations(_router_dataset(drop="loan_simulation"))
    assert any("'loan_simulation'" in v for v in missing_intent)


def test_slots_composition() -> None:
    complete = SlotValues(amount="5000", term_months=12, amortization_type="PRICE")
    items = [
        SlotItem(id=f"S-{i}", message="m", expected=complete, tags=["complete"]) for i in range(24)
    ]
    items += [
        SlotItem(id="S-100", message="m", expected=SlotValues(term_months=6), tags=["partial"]),
        SlotItem(id="S-101", message="m", expected=SlotValues(), tags=["missing"]),
        SlotItem(
            id="S-102", message="m", expected=SlotValues(term_months=6), tags=["partial", "invalid"]
        ),
    ]

    assert slots_violations(SlotsDataset(version=1, items=items)) == []
    assert any("invalid" in v for v in slots_violations(SlotsDataset(version=1, items=items[:-1])))


def test_compliance_composition() -> None:
    approval = [ApprovalItem(id=f"C-{i}", text=f"neutro {i}", label="neutral") for i in range(30)]
    approval += [
        ApprovalItem(id="C-p1", text="p1", label="promise", tags=["fail_closed", "adversarial"]),
        ApprovalItem(id="C-p2", text="p2", label="promise", tags=["fail_closed"]),
        ApprovalItem(id="C-p3", text="p3", label="promise"),
        ApprovalItem(id="C-h1", text="h1", label="hedge", tags=["fail_closed"]),
        ApprovalItem(id="C-h2", text="h2", label="hedge", tags=["fail_closed"]),
        ApprovalItem(id="C-h3", text="h3", label="hedge"),
    ]
    pii = [PiiItem(id=f"P-{i}", text="t", expected_masked="t") for i in range(12)]
    dataset = ComplianceDataset(version=1, approval=approval, pii=pii)

    assert compliance_violations(dataset) == []
    without_fail_closed = ComplianceDataset(
        version=1,
        approval=[a.model_copy(update={"tags": []}) for a in approval],
        pii=pii,
    )
    assert any("fail_closed" in v for v in compliance_violations(without_fail_closed))

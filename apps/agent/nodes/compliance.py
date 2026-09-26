"""Deterministic compliance helpers: PII masking, disclaimer injection,
and the keyword/regex safety net for approval-promise detection.

Everything in this module is 100% deterministic and LLM-free — the one
guardrail that does use an LLM (approval-promise judgment on free-form
phrasing) lives in `apps.agent.nodes.compliance_guard`, backed by
`keyword_flags_approval_promise` here as a second, independent check
(see `design.md` — "Deterministic compliance enforcement").
"""

from __future__ import annotations

import re

from apps.agent.nodes.text_utils import normalize

PII_MASK = "[DADO PROTEGIDO]"

_CARD_MIN_DIGITS = 13
_CARD_MAX_DIGITS = 19

# Order matters: each pattern below is applied in sequence, most specific
# (longest, least ambiguous) first, so a general/shorter pattern never gets a
# chance to consume part of a longer PII value first and leave the rest
# behind (the CPF-digit-leak and card-masked-in-two-pieces bugs both came
# from `_PHONE_PATTERN` matching a prefix of a longer digit run before the
# pattern that should have owned it ran at all).
_CARD_GROUPED_PATTERN = re.compile(
    r"\b\d{4}[ -]\d{4}[ -]\d{4}[ -]\d{4}\b|\b\d{4}[ -]\d{6}[ -]\d{5}\b"
)
"""A card-shaped grouping — 4-4-4-4 (Visa/Mastercard/...) or 4-6-5 (Amex) with
spaces or hyphens between groups — is masked unconditionally, Luhn check or
not: someone who typed a card number with one wrong digit still typed a card
number, and this shape is specific enough that failing open on a typo would
be the worse mistake."""
_CARD_CONTIGUOUS_PATTERN = re.compile(r"\b\d{13,19}\b")
"""13-19 digits with **no** separator: masked only when Luhn-valid
(`_mask_card_candidate`), so a boleto line (47-48 digits — already excluded
by length alone) or an unrelated long number typed without punctuation is
never masked just because of its length."""
_CPF_PATTERN = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
_CPF_UNPUNCTUATED_PATTERN = re.compile(r"\b\d{11}\b")
"""An 11-digit CPF with no punctuation. `\\b` on both sides means this can
never match part of a longer digit run (there is no `\\b` between two
digits), so it can't clash with the 13-19 digit card pattern above."""
_ACCOUNT_CONTEXT_PATTERN = re.compile(
    r"\b(?:conta|c/c|agência|agencia|ag)\.?\s*(?:n[ºo°]?\.?\s*)?(\d{3,}-\d{1,2})\b",
    re.IGNORECASE,
)
"""A bank account/agency number with its check digit (e.g. "conta 12345-6"),
masked only when an account/agency context word precedes it — so an
unrelated hyphenated number (a CEP, a date range, ...) is never masked
just because it happens to have that shape."""
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"\(?\b\d{2}\)?[\s-]?9?\d{4}-?\d{4}\b")
"""Bounded on both sides of its digits: without this, a 10-digit window
with no anchoring would happily match the first 10 digits of any longer
number (a card number's Luhn check correctly declining to mask it does not
stop this pattern from mangling the leftovers on its own)."""

DISCLAIMER = (
    "\n\nEste é um ambiente de demonstração com dados sintéticos. As condições de "
    "crédito são simuladas e não constituem oferta real de crédito. Consulte sempre "
    "as condições vigentes antes de qualquer decisão financeira."
)

INFORMATIONAL_DISCLAIMER = (
    "\n\nEste conteúdo é informativo e não constitui aconselhamento jurídico. "
    "Consulte um profissional qualificado para orientação específica ao seu caso."
)

_APPROVAL_ROOTS = ("aprova",)
"""Roots present in "aprovação"/"aprovado"/"aprovar"/etc."""
_GUARANTEE_ROOTS = ("garanti", "certeza", "100%", "com certeza", "sem risco")
"""Roots/phrases that turn a mention of approval into a promise/guarantee."""

_HEDGE_PHRASES = (
    "sujeito a analise",
    "sujeita a analise",
    "depende de analise",
    "nao garant",
    "nao posso garantir",
    "nao ha garantia",
    "condicionad",
)
"""Explicit hedge phrases (normalized: lowercase, no accents). Deliberately
phrases, not single-word roots: "analise", "simulac" or "estimad" also
appear in promise text ("após análise, seu empréstimo foi aprovado"), and
this list gates the fail-closed path, where a false pass is the costly
error."""

BLOCKED_PROMISE_REPLY = (
    "Não posso garantir a aprovação do seu crédito — a análise depende de diversos "
    "fatores. Posso, no entanto, seguir com a simulação para que você veja as "
    "condições estimadas."
)


def _luhn_valid(digits: str) -> bool:
    """The standard Luhn checksum used by card numbers."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask_card_candidate(match: re.Match[str]) -> str:
    digits = match.group(0)
    if _CARD_MIN_DIGITS <= len(digits) <= _CARD_MAX_DIGITS and _luhn_valid(digits):
        return PII_MASK
    return digits


def _mask_account_number(match: re.Match[str]) -> str:
    return match.string[match.start() : match.start(1)] + PII_MASK


def mask_pii(text: str) -> str:
    """Deterministically mask card numbers, CPF, bank account numbers, email
    and phone numbers."""
    masked = _CARD_GROUPED_PATTERN.sub(PII_MASK, text)
    masked = _CARD_CONTIGUOUS_PATTERN.sub(_mask_card_candidate, masked)
    masked = _CPF_PATTERN.sub(PII_MASK, masked)
    masked = _CPF_UNPUNCTUATED_PATTERN.sub(PII_MASK, masked)
    masked = _ACCOUNT_CONTEXT_PATTERN.sub(_mask_account_number, masked)
    masked = _EMAIL_PATTERN.sub(PII_MASK, masked)
    masked = _PHONE_PATTERN.sub(PII_MASK, masked)
    return masked


def inject_disclaimer(text: str, needs_disclaimer: bool, disclaimer: str = DISCLAIMER) -> str:
    """Append a mandatory disclaimer via a fixed template — never generated.
    Defaults to the simulation/analysis `DISCLAIMER`; pass
    `INFORMATIONAL_DISCLAIMER` for regulatory answers."""
    if not needs_disclaimer:
        return text
    return text + disclaimer


def keyword_flags_approval_promise(text: str) -> bool:
    """Deterministic safety net: flags any text that mentions approval
    together with a guarantee/certainty word — regardless of the exact
    phrasing or word order — even if the LLM-based check misses it."""
    normalized = normalize(text)
    has_approval_word = any(root in normalized for root in _APPROVAL_ROOTS)
    has_guarantee_word = any(root in normalized for root in _GUARANTEE_ROOTS)
    return has_approval_word and has_guarantee_word


def keyword_flags_unhedged_approval_mention(text: str) -> bool:
    """Stricter deterministic screen, used only when the LLM approval-
    promise check is unavailable (fail closed): flags any mention of
    approval that carries no explicit hedge phrase ("sujeito à análise",
    "não posso garantir"...), even without an explicit guarantee word. Broader —
    and so more likely to block — than `keyword_flags_approval_promise`
    on purpose; a reply that never mentions approval is unaffected."""
    normalized = normalize(text)
    has_approval_word = any(root in normalized for root in _APPROVAL_ROOTS)
    has_hedge = any(phrase in normalized for phrase in _HEDGE_PHRASES)
    return has_approval_word and not has_hedge

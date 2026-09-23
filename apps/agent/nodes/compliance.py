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

_CPF_PATTERN = re.compile(r"\d{3}\.\d{3}\.\d{3}-\d{2}")
_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"\(?\d{2}\)?[\s-]?9?\d{4}-?\d{4}")
_ACCOUNT_OR_CARD_PATTERN = re.compile(r"\b\d{4,}[-.]?\d{2,}\b")

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

BLOCKED_PROMISE_REPLY = (
    "Não posso garantir a aprovação do seu crédito — a análise depende de diversos "
    "fatores. Posso, no entanto, seguir com a simulação para que você veja as "
    "condições estimadas."
)


def mask_pii(text: str) -> str:
    """Deterministically mask CPF, email, phone, and account/card numbers."""
    masked = _CPF_PATTERN.sub(PII_MASK, text)
    masked = _EMAIL_PATTERN.sub(PII_MASK, masked)
    masked = _PHONE_PATTERN.sub(PII_MASK, masked)
    masked = _ACCOUNT_OR_CARD_PATTERN.sub(PII_MASK, masked)
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

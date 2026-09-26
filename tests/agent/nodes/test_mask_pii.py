"""`mask_pii`: card numbers (Luhn-checked), CPF (punctuated and not), bank
account numbers (context-gated), phone and email — and everything that must
never be masked collaterally."""

from __future__ import annotations

import pytest

from apps.agent.nodes.compliance import PII_MASK, mask_pii

# --- fix 1: unpunctuated CPF, masked fully -----------------------------------


def test_unpunctuated_cpf_is_masked_with_no_trailing_digit_leak() -> None:
    reply = mask_pii("CPF 00000000000 informado no cadastro.")

    assert reply == f"CPF {PII_MASK} informado no cadastro."


def test_a_different_unpunctuated_cpf_is_also_fully_masked() -> None:
    assert mask_pii("Confirme o CPF 12345678909 agora.") == f"Confirme o CPF {PII_MASK} agora."


def test_punctuated_cpf_is_still_masked() -> None:
    reply = mask_pii("Seu CPF 000.000.000-00 foi verificado.")

    assert reply == f"Seu CPF {PII_MASK} foi verificado."


# --- fix 2: card numbers -----------------------------------------------------
#
# Two different rules by shape:
# - a card-shaped *grouping* (4-4-4-4, or 4-6-5 for Amex) is masked
#   unconditionally, Luhn check or not — fail closed on a typo;
# - a *contiguous* 13-19 digit run (no separator) is masked only when
#   Luhn-valid, so a boleto line or an unrelated long number is left alone.


def test_a_16_digit_contiguous_card_number_is_masked_as_a_single_placeholder() -> None:
    reply = mask_pii("Cartão 0000000000000000 cadastrado.")

    assert reply == f"Cartão {PII_MASK} cadastrado."
    assert reply.count(PII_MASK) == 1


@pytest.mark.parametrize(
    "card",
    [
        "4111 1111 1111 1111",  # 4-4-4-4, space-grouped
        "4111-1111-1111-1111",  # 4-4-4-4, hyphen-grouped
        "3782 822463 10005",  # 4-6-5 (Amex), space-grouped
        "3782-822463-10005",  # 4-6-5 (Amex), hyphen-grouped
    ],
)
def test_a_card_shaped_grouping_is_masked_as_a_single_placeholder(card: str) -> None:
    reply = mask_pii(f"Cartão {card} cadastrado.")

    assert reply == f"Cartão {PII_MASK} cadastrado."
    assert reply.count(PII_MASK) == 1


@pytest.mark.parametrize(
    "card",
    [
        "4111 1111 1111 1112",  # 4-4-4-4, last digit changed: Luhn-invalid
        "3782-822463-10006",  # 4-6-5, last digit changed: Luhn-invalid
    ],
)
def test_a_luhn_invalid_grouped_number_is_still_masked(card: str) -> None:
    """Fail closed on a typo: whoever typed a card-shaped number typed a
    card number, one wrong digit or not."""
    assert mask_pii(f"Cartão {card} cadastrado.") == f"Cartão {PII_MASK} cadastrado."


@pytest.mark.parametrize(
    "card",
    [
        "4111111111111111",  # 16 digits, well-known Luhn-valid test number
        "371449635398431",  # 15 digits, Luhn-valid
        "4222222222222",  # 13 digits, Luhn-valid
    ],
)
def test_a_luhn_valid_contiguous_number_is_masked(card: str) -> None:
    assert mask_pii(f"Cartão {card} cadastrado.") == f"Cartão {PII_MASK} cadastrado."


def test_a_luhn_invalid_contiguous_16_digit_run_is_left_alone() -> None:
    """No grouping and a failing checksum: the masker has no reason to treat
    this as a card, so it leaves an arbitrary long number (an id, an
    unformatted big amount, ...) alone."""
    number = "1234567890123456"

    assert mask_pii(f"Número aleatório {number} sem contexto de cartão.") == (
        f"Número aleatório {number} sem contexto de cartão."
    )


def test_changing_the_last_digit_of_a_contiguous_valid_card_unmasks_it() -> None:
    assert mask_pii("Cartão 4111111111111112 cadastrado.") == "Cartão 4111111111111112 cadastrado."


def test_a_47_digit_boleto_line_grouped_by_dots_and_spaces_is_not_masked() -> None:
    line = "34191.09008 61207.727307 71249.640008 5 84660000020000"
    assert sum(char.isdigit() for char in line) == 47

    assert mask_pii(f"Linha digitável: {line}") == f"Linha digitável: {line}"


def test_a_48_digit_boleto_barcode_with_no_separators_is_not_masked() -> None:
    """Longer than a card's 19-digit maximum, so length alone excludes it —
    length, not Luhn, is what protects a boleto line here."""
    barcode = "".join(str((i * 7 + 3) % 10) for i in range(48))

    assert mask_pii(f"Código de barras: {barcode}") == f"Código de barras: {barcode}"


# --- fix 3: bank account + check digit, gated by an account/agency context --


@pytest.mark.parametrize("context", ["Depósito na conta", "Conta", "Na agência e conta", "Ag"])
def test_an_account_number_is_masked_only_with_an_account_context_word(context: str) -> None:
    reply = mask_pii(f"{context} 12345-6.")

    assert reply == f"{context} {PII_MASK}."


def test_the_same_shaped_number_without_any_context_word_is_left_alone() -> None:
    assert mask_pii("Código 12345-6 gerado.") == "Código 12345-6 gerado."


def test_a_cep_is_never_mistaken_for_an_account_number() -> None:
    assert mask_pii("CEP 01310-000.") == "CEP 01310-000."


# --- negative regressions: nothing else gets collaterally masked ------------


def test_currency_amounts_are_not_masked() -> None:
    assert mask_pii("Sua parcela é de R$ 559,13 em 24 meses.") == (
        "Sua parcela é de R$ 559,13 em 24 meses."
    )
    assert mask_pii("Total pago: R$ 13.419,12.") == "Total pago: R$ 13.419,12."
    assert mask_pii("Simulação de R$ 10000 em 24 meses.") == "Simulação de R$ 10000 em 24 meses."


def test_cet_percentages_are_not_masked() -> None:
    assert mask_pii("O CET é de 40,28% ao ano.") == "O CET é de 40,28% ao ano."


def test_dates_are_not_masked() -> None:
    assert mask_pii("A parcela vence em 24/01/2026.") == "A parcela vence em 24/01/2026."


def test_phone_numbers_are_still_masked_after_the_new_patterns() -> None:
    """Regression: the card/CPF fixes must not weaken existing phone masking."""
    assert mask_pii("Ligue para (11) 90000-0000.") == f"Ligue para {PII_MASK}."
    assert mask_pii("Telefone 11900000000 para retorno.") == f"Telefone {PII_MASK} para retorno."
    assert mask_pii("Ligue (21) 3000-0000 em horário comercial.") == (
        f"Ligue {PII_MASK} em horário comercial."
    )


def test_email_addresses_are_still_masked() -> None:
    assert mask_pii("Contato: cliente@example.com") == f"Contato: {PII_MASK}"

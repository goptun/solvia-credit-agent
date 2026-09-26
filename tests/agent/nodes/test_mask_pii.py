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


# --- fix 2: card numbers, 13-19 digits, contiguous or grouped, Luhn-checked --


def test_a_16_digit_contiguous_card_number_is_masked_as_a_single_placeholder() -> None:
    reply = mask_pii("Cartão 0000000000000000 cadastrado.")

    assert reply == f"Cartão {PII_MASK} cadastrado."
    assert reply.count(PII_MASK) == 1


def test_a_space_grouped_card_number_is_masked_as_a_single_placeholder() -> None:
    reply = mask_pii("Cartão 0000 0000 0000 0000 cadastrado.")

    assert reply == f"Cartão {PII_MASK} cadastrado."
    assert reply.count(PII_MASK) == 1


@pytest.mark.parametrize(
    "card",
    [
        "4111111111111111",  # 16 digits, well-known Luhn-valid test number
        "4111-1111-1111-1111",  # same number, hyphen-grouped
        "4111 1111 1111 1111",  # same number, space-grouped
        "371449635398431",  # 15 digits, Luhn-valid
        "4222222222222",  # 13 digits, Luhn-valid
    ],
)
def test_luhn_valid_numbers_of_13_to_19_digits_are_masked(card: str) -> None:
    assert mask_pii(f"Cartão {card} cadastrado.") == f"Cartão {PII_MASK} cadastrado."


def test_a_luhn_invalid_16_digit_number_is_left_alone() -> None:
    """The Luhn check is what stops the masker from treating an arbitrary
    long number (an id, an unformatted big amount, ...) as a card."""
    number = "1234567890123456"

    assert mask_pii(f"Número aleatório {number} sem contexto de cartão.") == (
        f"Número aleatório {number} sem contexto de cartão."
    )


def test_changing_the_last_digit_of_a_valid_card_makes_it_luhn_invalid_and_unmasked() -> None:
    assert mask_pii("Cartão 4111111111111112 cadastrado.") == "Cartão 4111111111111112 cadastrado."


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

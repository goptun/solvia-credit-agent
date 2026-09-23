"""Fictional product descriptions, in Brazilian Portuguese.

`responder` answers product questions only from this static text —
never from the LLM's own knowledge (see
`design.md` — "Concrete per-intent MVP behavior").
"""

from __future__ import annotations

from apps.agent.config.catalog import AmortizationType

PRODUCT_DESCRIPTIONS: dict[AmortizationType, str] = {
    AmortizationType.PRICE: (
        "Crédito Pessoal Solvia (Tabela Price): parcelas fixas do início ao fim, "
        "facilitando o planejamento do orçamento mensal. Ideal para quem prefere "
        "previsibilidade no valor da parcela."
    ),
    AmortizationType.SAC: (
        "Crédito Pessoal Solvia (SAC): amortização constante, com parcelas que "
        "diminuem ao longo do tempo à medida que os juros sobre o saldo devedor caem. "
        "Ideal para quem pode pagar mais no início e busca economizar no total de juros."
    ),
}

GENERAL_PRODUCT_OVERVIEW = (
    "A Solvia oferece crédito pessoal sintético para fins de demonstração, nas "
    "modalidades Price (parcelas fixas) e SAC (parcelas decrescentes). Todas as "
    "simulações consideram taxa de juros, IOF e tarifas vigentes do nosso catálogo."
)

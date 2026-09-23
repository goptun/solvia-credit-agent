"""financial_analyst node: income, DTI, and spending categories via tools.

The LLM is never used to compute these values (see
`specs/conversation-graph/spec.md` — "Financial profile analysis"); the
node is tier-mapped as `smart` for a future LLM-assisted summary, but
this MVP's numbers come entirely from
`apps.agent.tools.income`/`dti`/`spending`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from apps.agent.repositories.customers import CustomerRepository
from apps.agent.state import ConversationState, CustomerProfileSummary
from apps.agent.tools.dti import debt_to_income_ratio, estimate_monthly_debt_payments
from apps.agent.tools.income import estimate_income
from apps.agent.tools.spending import categorize_spending

FinancialAnalystNode = Callable[[ConversationState], Awaitable[ConversationState]]


def make_financial_analyst_node(customer_repository: CustomerRepository) -> FinancialAnalystNode:
    async def financial_analyst_node(state: ConversationState) -> ConversationState:
        customer = customer_repository.get(state["customer_id"])
        if customer is None:
            return ConversationState()

        income = estimate_income(customer.transactions)
        monthly_debt = estimate_monthly_debt_payments(customer.credit_cards)
        dti = debt_to_income_ratio(income, monthly_debt)
        spending = categorize_spending(customer.transactions)

        summary = CustomerProfileSummary(
            income=income,
            debt_to_income_ratio=dti,
            spending_categories={category.value: total for category, total in spending.items()},
        )
        return ConversationState(customer_profile=summary)

    return financial_analyst_node

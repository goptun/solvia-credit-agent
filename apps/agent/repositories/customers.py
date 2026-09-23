"""Customer repository port + an in-memory adapter over the synthetic dataset.

Nodes depend only on `CustomerRepository`, never on how customers are
loaded — the same seam design.md calls for around the CRM client and
other I/O boundaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from apps.agent.synthetic_data.io import FIXTURES_DIR, load_customers
from apps.agent.synthetic_data.models import Customer


class CustomerRepository(Protocol):
    def get(self, customer_id: str) -> Customer | None: ...


class InMemoryCustomerRepository:
    """Loads a synthetic dataset file once and serves lookups from memory."""

    def __init__(self, customers: list[Customer]) -> None:
        self._by_id = {customer.customer_id: customer for customer in customers}

    @classmethod
    def from_file(cls, path: Path) -> InMemoryCustomerRepository:
        return cls(load_customers(path))

    @classmethod
    def from_fixtures(cls) -> InMemoryCustomerRepository:
        return cls.from_file(FIXTURES_DIR / "customers.json")

    def get(self, customer_id: str) -> Customer | None:
        return self._by_id.get(customer_id)

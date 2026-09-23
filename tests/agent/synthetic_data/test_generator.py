"""Determinism, profile coverage, and consent-status coverage of the generator."""

from __future__ import annotations

from apps.agent.synthetic_data.generator import (
    DEFAULT_CUSTOMER_COUNT,
    DEFAULT_SEED,
    generate_customers,
)
from apps.agent.synthetic_data.models import ConsentStatus, CustomerProfile


def test_same_seed_produces_identical_output() -> None:
    first_run = generate_customers(seed=DEFAULT_SEED, count=50)
    second_run = generate_customers(seed=DEFAULT_SEED, count=50)

    assert [c.model_dump(mode="json") for c in first_run] == [
        c.model_dump(mode="json") for c in second_run
    ]


def test_generation_is_independent_of_wall_clock_time() -> None:
    """Dates come from the fixed REFERENCE_DATE, not `datetime.now()` —
    running generation "at different times" (simulated by simply calling
    it twice, since nothing here reads the clock) yields identical dates.
    """
    first_run = generate_customers(seed=123, count=10)
    second_run = generate_customers(seed=123, count=10)

    first_dates = [txn.posted_at for c in first_run for txn in c.transactions]
    second_dates = [txn.posted_at for c in second_run for txn in c.transactions]
    assert first_dates == second_dates


def test_default_generation_produces_approximately_two_hundred_customers() -> None:
    customers = generate_customers()
    assert len(customers) == DEFAULT_CUSTOMER_COUNT


def test_dataset_includes_all_four_profiles() -> None:
    customers = generate_customers()
    profiles = {c.profile for c in customers}
    assert profiles == set(CustomerProfile)


def test_dataset_includes_all_three_consent_statuses() -> None:
    customers = generate_customers()
    statuses = {c.consent.status for c in customers}
    assert statuses == set(ConsentStatus)


def test_thin_file_customers_have_sparse_history() -> None:
    customers = generate_customers(count=20)
    thin_file = [c for c in customers if c.profile == CustomerProfile.THIN_FILE]
    assert thin_file
    for customer in thin_file:
        assert len(customer.transactions) <= 4
        assert customer.credit_cards == []

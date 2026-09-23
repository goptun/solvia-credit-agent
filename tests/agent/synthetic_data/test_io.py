"""Generated output and committed fixtures stay in separate locations."""

from __future__ import annotations

from pathlib import Path

from apps.agent.synthetic_data.generator import generate_customers
from apps.agent.synthetic_data.io import (
    FIXTURES_DIR,
    load_customers,
    select_fixture_customers,
    write_dataset,
    write_fixtures,
)
from apps.agent.synthetic_data.models import ConsentStatus, CustomerProfile


def test_write_dataset_writes_under_the_given_directory(tmp_path: Path) -> None:
    customers = generate_customers(count=5)
    output_path = write_dataset(customers, directory=tmp_path)

    assert output_path.exists()
    assert output_path.parent == tmp_path
    assert load_customers(output_path) == customers


def test_fixture_selection_covers_every_profile_and_consent_status() -> None:
    customers = generate_customers()
    fixtures = select_fixture_customers(customers)

    assert {c.profile for c in fixtures} == set(CustomerProfile)
    assert {c.consent.status for c in fixtures} == set(ConsentStatus)
    assert len(fixtures) < len(customers)


def test_write_fixtures_writes_only_the_small_selected_subset(tmp_path: Path) -> None:
    customers = generate_customers()
    output_path = write_fixtures(customers, directory=tmp_path)

    written = load_customers(output_path)
    assert written == select_fixture_customers(customers)
    assert len(written) < len(customers)


def test_committed_fixtures_file_is_the_one_tests_should_read() -> None:
    """The automated test suite reads the small, committed fixture file
    under `data/fixtures/` — not the (gitignored) generator output."""
    fixtures_path = FIXTURES_DIR / "customers.json"

    customers = load_customers(fixtures_path)

    assert 0 < len(customers) < 20
    assert {c.profile for c in customers} == set(CustomerProfile)
    assert {c.consent.status for c in customers} == set(ConsentStatus)

"""Writing the generated dataset and selecting committed test fixtures.

Generator output goes to `data/generated/` (gitignored); only a small,
explicit subset is written to `data/fixtures/` for the automated test
suite to read (see `specs/synthetic-data/spec.md` — "Generated output
separated from committed fixtures").
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.agent.synthetic_data.models import Customer

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_DIR = REPO_ROOT / "data" / "generated"
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"
FIXTURE_INDICES = [0, 1, 2, 3, 7, 8]
"""One customer per profile (indices 0-3), plus a MISSING-consent
customer (7) and an EXPIRED-consent customer (8) — covers all four
profiles and all three consent statuses in a handful of records."""


def customers_to_json(customers: list[Customer]) -> str:
    return json.dumps([customer.model_dump(mode="json") for customer in customers], indent=2)


def write_dataset(customers: list[Customer], directory: Path = GENERATED_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "customers.json"
    output_path.write_text(customers_to_json(customers))
    return output_path


def select_fixture_customers(customers: list[Customer]) -> list[Customer]:
    return [customers[i] for i in FIXTURE_INDICES if i < len(customers)]


def write_fixtures(customers: list[Customer], directory: Path = FIXTURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / "customers.json"
    fixtures = select_fixture_customers(customers)
    output_path.write_text(customers_to_json(fixtures))
    return output_path


def load_customers(path: Path) -> list[Customer]:
    data = json.loads(path.read_text())
    return [Customer.model_validate(item) for item in data]

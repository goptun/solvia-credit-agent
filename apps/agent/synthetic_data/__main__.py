"""Generate the synthetic dataset: `uv run python -m apps.agent.synthetic_data`."""

from __future__ import annotations

from apps.agent.synthetic_data.generator import generate_customers
from apps.agent.synthetic_data.io import write_dataset, write_fixtures


def main() -> None:
    customers = generate_customers()
    dataset_path = write_dataset(customers)
    fixtures_path = write_fixtures(customers)
    print(f"Wrote {len(customers)} customers to {dataset_path}")
    print(f"Wrote fixtures to {fixtures_path}")


if __name__ == "__main__":
    main()

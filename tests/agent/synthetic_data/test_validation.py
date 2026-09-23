"""Every generated record validates against its Pydantic schema."""

from __future__ import annotations

import json

from apps.agent.synthetic_data.generator import generate_customers
from apps.agent.synthetic_data.models import Customer


def test_generated_dataset_round_trips_through_json_without_validation_errors() -> None:
    customers = generate_customers(count=200)

    serialized = json.dumps([c.model_dump(mode="json") for c in customers])
    reloaded = [Customer.model_validate(item) for item in json.loads(serialized)]

    assert len(reloaded) == len(customers)
    assert reloaded == customers

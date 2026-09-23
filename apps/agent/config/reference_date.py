"""Fixed reference date used by the synthetic data generator.

Every generated date is derived from this value, never from the system
clock, so generation is reproducible independent of when it runs (see
`specs/synthetic-data/spec.md` — "Dates derived from a fixed reference
date").
"""

from __future__ import annotations

from datetime import date

REFERENCE_DATE: date = date(2026, 6, 1)

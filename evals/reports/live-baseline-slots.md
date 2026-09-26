_mode `live`, commit `24daa0726a2c6eea2e7e9dbff1d3261930996fe3`, seed 42_

_On the free tier both gateway aliases resolve to the same lite model, so the fast/smart split is nominal in these numbers. Gateway-internal fallbacks are invisible to the harness except through the resolved model._

Gateway calls: **251** of a budget of 300 (estimated 36, up to 43).

**Contaminated run** — reported, never a baseline: 

- 49.3% of operations ended in a quota/unavailability response even after the app's own retry/backoff had a chance (35 of 71 operations; threshold 5%; see the operational suite's status_breakdown for raw 429/503/timeout counts)

| alias | resolved model | calls |
|---|---|---|
| solvia-eval-fast | gemini-3.5-flash-lite | 35 |
| solvia-eval-smart | gemini-3.8-flash | 1 |
| solvia-eval-smart | unknown | 215 |

Raw call status (429s reported separately from 503s; a status a retry resolved still counts here): ok × 36, 429 × 0, 503 × 215, timeout × 0, other × 0.

## slots

Datasets: slots v2

| metric | value | 95% CI | n |
|---|---|---|---|
| slots.exact_match.amount | 0.9444 | [0.819, 0.985] | 36 |
| slots.exact_match.term_months | 0.9444 | [0.819, 0.985] | 36 |
| slots.exact_match.amortization_type | 0.9167 | [0.782, 0.971] | 36 |
| slots.exact_match.all_fields | 0.8056 | [0.650, 0.902] | 36 |
| slots.no_invented_value | 0.8537 | [0.716, 0.931] | 41 |

## operational

Datasets: 

| metric | value | 95% CI | n |
|---|---|---|---|
| offer_simulator.no_tool_call_rate | 0.0000 | [0.000, 0.096] | 36 |
| offer_simulator.json_fallback_rate | 1.0000 | [0.904, 1.000] | 36 |
| offer_simulator.latency_p50_seconds | 3.0960 | - | 216 |
| offer_simulator.latency_p95_seconds | 3.0960 | - | 216 |
| offer_simulator.attempts_mean | 6.0000 | - | 216 |
| offer_simulator_fallback.no_tool_call_rate | 0.0000 | [0.000, 0.099] | 35 |
| offer_simulator_fallback.json_fallback_rate | 0.0000 | [0.000, 0.099] | 35 |
| offer_simulator_fallback.latency_p50_seconds | 0.9220 | - | 35 |
| offer_simulator_fallback.latency_p95_seconds | 4.8510 | - | 35 |
| offer_simulator_fallback.attempts_mean | 1.0000 | - | 35 |

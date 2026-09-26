_mode `live`, commit `5519fceba0427afa0bf0a1a87b62243c69816876`, seed 42_

_On the free tier both gateway aliases resolve to the same lite model, so the fast/smart split is nominal in these numbers. Gateway-internal fallbacks are invisible to the harness except through the resolved model._

Gateway calls: **77** of a budget of 300 (estimated 69, up to 76).

| alias | resolved model | calls |
|---|---|---|
| solvia-eval-fast | gemini-3.5-flash-lite | 69 |
| solvia-eval-fast | unknown | 8 |

Raw call status (429s reported separately from 503s; a status a retry resolved still counts here): ok × 69, 429 × 0, 503 × 0, timeout × 8, other × 0.

## router

Datasets: router v2

| metric | value | 95% CI | n |
|---|---|---|---|
| router.accuracy | 1.0000 | [0.947, 1.000] | 69 |
| router.accuracy.category.ambiguous | 1.0000 | [0.646, 1.000] | 7 |
| router.accuracy.category.clear | 1.0000 | [0.935, 1.000] | 55 |
| router.accuracy.category.continuation | 1.0000 | [0.646, 1.000] | 7 |

## operational

Datasets: 

| metric | value | 95% CI | n |
|---|---|---|---|
| router.no_tool_call_rate | 0.0000 | [0.000, 0.053] | 69 |
| router.json_fallback_rate | 0.0870 | [0.040, 0.177] | 69 |
| router.latency_p50_seconds | 0.9250 | - | 77 |
| router.latency_p95_seconds | 15.9360 | - | 77 |
| router.attempts_mean | 1.1160 | - | 77 |

_mode `live`, commit `dfe89f5408eb510a438b4a0ffe98081ea78e3035`, seed 42_

_On the free tier both gateway aliases resolve to the same lite model, so the fast/smart split is nominal in these numbers. Gateway-internal fallbacks are invisible to the harness except through the resolved model._

Gateway calls: **61** of a budget of 300 (estimated 41, up to 82).

| alias | resolved model | calls |
|---|---|---|
| solvia-eval-fast | gemini-3.5-flash-lite | 61 |

Raw call status (429s reported separately from 503s; a status a retry resolved still counts here): ok × 61, 429 × 0, 503 × 0, timeout × 0, other × 0.

## compliance-llm

Datasets: compliance v2

| metric | value | 95% CI | n |
|---|---|---|---|
| llm.precision | 1.0000 | [0.772, 1.000] | 13 |
| llm.recall | 0.9286 | [0.685, 0.987] | 14 |
| llm.false_positive_rate | 0.0000 | [0.000, 0.125] | 27 |

## operational

Datasets: 

| metric | value | 95% CI | n |
|---|---|---|---|
| compliance_guard.no_tool_call_rate | 0.4878 | [0.343, 0.635] | 41 |
| compliance_guard.json_fallback_rate | 0.4878 | [0.343, 0.635] | 41 |
| compliance_guard.latency_p50_seconds | 0.8570 | - | 61 |
| compliance_guard.latency_p95_seconds | 6.5310 | - | 61 |
| compliance_guard.attempts_mean | 1.4880 | - | 61 |

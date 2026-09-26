_mode `live`, commit `ae73627aa79877b0fd877ce4a0a1c971688ddf2d`, seed 42_

_On the free tier both gateway aliases resolve to the same lite model, so the fast/smart split is nominal in these numbers. Gateway-internal fallbacks are invisible to the harness except through the resolved model._

Gateway calls: **263** of a budget of 300 (estimated 216, up to 285).

**Contaminated run** — reported, never a baseline: 

- 15.2% of calls got a quota/unavailability response (429/503/timeout; threshold 5%)

- solvia-fast: calls resolved to models outside the expected set: gemini-3.1-flash-lite

- solvia-fast: 25.0% of calls resolved to fallback models (threshold 10%)

- solvia-smart: 34.0% of calls resolved to fallback models (threshold 10%)

| alias | resolved model | calls |
|---|---|---|
| solvia-fast | gemini-3.1-flash-lite | 14 |
| solvia-fast | gemini-3.5-flash-lite | 90 |
| solvia-fast | gemini-3.6-flash | 16 |
| solvia-fast | unknown | 40 |
| solvia-smart | gemini-3-flash-preview | 19 |
| solvia-smart | gemini-3.5-flash-lite | 68 |
| solvia-smart | gemini-3.6-flash | 1 |
| solvia-smart | gemini-3.8-flash | 15 |

## router

Datasets: router v2

| metric | value | 95% CI | n |
|---|---|---|---|
| router.accuracy | 1.0000 | [0.947, 1.000] | 69 |
| router.accuracy.category.ambiguous | 1.0000 | [0.646, 1.000] | 7 |
| router.accuracy.category.clear | 1.0000 | [0.935, 1.000] | 55 |
| router.accuracy.category.continuation | 1.0000 | [0.646, 1.000] | 7 |

## slots

Datasets: slots v2

| metric | value | 95% CI | n |
|---|---|---|---|
| slots.exact_match.amount | 0.9444 | [0.819, 0.985] | 36 |
| slots.exact_match.term_months | 0.9444 | [0.819, 0.985] | 36 |
| slots.exact_match.amortization_type | 0.9444 | [0.819, 0.985] | 36 |
| slots.exact_match.all_fields | 0.8333 | [0.681, 0.921] | 36 |
| slots.no_invented_value | 0.8537 | [0.716, 0.931] | 41 |

## compliance-llm

Datasets: compliance v2

| metric | value | 95% CI | n |
|---|---|---|---|
| llm.precision | 1.0000 | [0.772, 1.000] | 13 |
| llm.recall | 0.9286 | [0.685, 0.987] | 14 |
| llm.false_positive_rate | 0.0000 | [0.000, 0.184] | 17 |

## grounding

Datasets: retrieval v2

| metric | value | 95% CI | n |
|---|---|---|---|
| grounding.false_refusal_rate | 0.3562 | [0.256, 0.471] | 73 |
| grounding.refusal_accuracy | 1.0000 | [0.867, 1.000] | 25 |
| grounding.refusal_accuracy.far | 1.0000 | [0.701, 1.000] | 9 |
| grounding.refusal_accuracy.near_miss | 1.0000 | [0.806, 1.000] | 16 |
| grounding.citation_validity | 1.0000 | [0.924, 1.000] | 47 |
| grounding.expected_ref_hit | 0.8936 | [0.774, 0.954] | 47 |

## operational

Datasets: 

| metric | value | 95% CI | n |
|---|---|---|---|
| compliance_guard.no_tool_call_rate | 0.4878 | [0.343, 0.635] | 41 |
| compliance_guard.json_fallback_rate | 0.7073 | [0.555, 0.824] | 41 |
| compliance_guard.latency_p50_seconds | 0.8170 | - | 89 |
| compliance_guard.latency_p95_seconds | 2.2880 | - | 89 |
| compliance_guard.attempts_mean | 2.1710 | - | 89 |
| knowledge_agent.no_tool_call_rate | 0.0469 | [0.016, 0.129] | 64 |
| knowledge_agent.json_fallback_rate | 0.0469 | [0.016, 0.129] | 64 |
| knowledge_agent.latency_p50_seconds | 3.1770 | - | 67 |
| knowledge_agent.latency_p95_seconds | 31.3720 | - | 67 |
| knowledge_agent.attempts_mean | 1.0470 | - | 67 |
| offer_simulator.no_tool_call_rate | 0.0000 | [0.000, 0.096] | 36 |
| offer_simulator.json_fallback_rate | 0.0000 | [0.000, 0.096] | 36 |
| offer_simulator.latency_p50_seconds | 3.0270 | - | 36 |
| offer_simulator.latency_p95_seconds | 22.4420 | - | 36 |
| offer_simulator.attempts_mean | 1.0000 | - | 36 |
| router.no_tool_call_rate | 0.0000 | [0.000, 0.053] | 69 |
| router.json_fallback_rate | 0.0290 | [0.008, 0.100] | 69 |
| router.latency_p50_seconds | 3.0320 | - | 71 |
| router.latency_p95_seconds | 22.0970 | - | 71 |
| router.attempts_mean | 1.0290 | - | 71 |

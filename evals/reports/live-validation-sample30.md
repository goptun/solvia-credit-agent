_mode `live`, commit `cd269ae951ec230c7dbf000b9544f4bab62fd880`, seed 42_

Gateway calls: **76** of a budget of 100 (estimated 66, up to 87).

**Sampled run** (30%): it can never become a baseline.

**Baselining is blocked until the expected model sets are approved** (`evals/live_config.yaml` has none for the aliases used). Observed mix, to propose them:

| alias | resolved model | calls |
|---|---|---|
| solvia-fast | gemini-3.5-flash-lite | 41 |
| solvia-smart | gemini-3.5-flash-lite | 35 |

## router

Datasets: router v2

| metric | value | 95% CI | n |
|---|---|---|---|
| router.accuracy | 0.9524 | [0.773, 0.992] | 21 |
| router.accuracy.category.ambiguous | 1.0000 | [0.646, 1.000] | 7 |
| router.accuracy.category.clear | 1.0000 | [0.758, 1.000] | 12 |
| router.accuracy.category.continuation | 0.5000 | [0.095, 0.905] | 2 |

## slots

Datasets: slots v2

| metric | value | 95% CI | n |
|---|---|---|---|
| slots.exact_match.amount | 0.9091 | [0.623, 0.984] | 11 |
| slots.exact_match.term_months | 0.9091 | [0.623, 0.984] | 11 |
| slots.exact_match.amortization_type | 0.9091 | [0.623, 0.984] | 11 |
| slots.exact_match.all_fields | 0.7273 | [0.434, 0.903] | 11 |
| slots.no_invented_value | 0.8235 | [0.590, 0.938] | 17 |

## compliance-llm

Datasets: compliance v2

| metric | value | 95% CI | n |
|---|---|---|---|
| llm.precision | 1.0000 | [0.566, 1.000] | 5 |
| llm.recall | 1.0000 | [0.566, 1.000] | 5 |
| llm.false_positive_rate | 0.0000 | [0.000, 0.354] | 7 |

## grounding

Datasets: retrieval v2

| metric | value | 95% CI | n |
|---|---|---|---|
| grounding.false_refusal_rate | 0.3200 | [0.172, 0.516] | 25 |
| grounding.refusal_accuracy | 1.0000 | [0.510, 1.000] | 4 |
| grounding.refusal_accuracy.far | 1.0000 | [0.342, 1.000] | 2 |
| grounding.refusal_accuracy.near_miss | 1.0000 | [0.342, 1.000] | 2 |
| grounding.citation_validity | 1.0000 | [0.816, 1.000] | 17 |
| grounding.expected_ref_hit | 0.9412 | [0.730, 0.990] | 17 |

## operational

Datasets: 

| metric | value | 95% CI | n |
|---|---|---|---|
| compliance_guard.no_tool_call_rate | 0.6667 | [0.391, 0.862] | 12 |
| compliance_guard.json_fallback_rate | 0.6667 | [0.391, 0.862] | 12 |
| compliance_guard.latency_p50_seconds | 0.8360 | - | 20 |
| compliance_guard.latency_p95_seconds | 1.4290 | - | 20 |
| compliance_guard.attempts_mean | 1.6670 | - | 20 |
| knowledge_agent.no_tool_call_rate | 0.1429 | [0.050, 0.346] | 21 |
| knowledge_agent.json_fallback_rate | 0.1429 | [0.050, 0.346] | 21 |
| knowledge_agent.latency_p50_seconds | 1.1710 | - | 24 |
| knowledge_agent.latency_p95_seconds | 3.8610 | - | 24 |
| knowledge_agent.attempts_mean | 1.1430 | - | 24 |
| offer_simulator.no_tool_call_rate | 0.0000 | [0.000, 0.259] | 11 |
| offer_simulator.json_fallback_rate | 0.0000 | [0.000, 0.259] | 11 |
| offer_simulator.latency_p50_seconds | 1.5270 | - | 11 |
| offer_simulator.latency_p95_seconds | 3.3810 | - | 11 |
| offer_simulator.attempts_mean | 1.0000 | - | 11 |
| router.no_tool_call_rate | 0.0000 | [0.000, 0.155] | 21 |
| router.json_fallback_rate | 0.0000 | [0.000, 0.155] | 21 |
| router.latency_p50_seconds | 0.8860 | - | 21 |
| router.latency_p95_seconds | 1.0870 | - | 21 |
| router.attempts_mean | 1.0000 | - | 21 |

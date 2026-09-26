_mode `live`, commit `5be9ed904b9f2b7c222e41ab22a4118ebaa7d2b5`, seed 42_

_On the free tier both gateway aliases resolve to the same lite model, so the fast/smart split is nominal in these numbers. Gateway-internal fallbacks are invisible to the harness except through the resolved model._

Gateway calls: **297** of a budget of 300 (estimated 70, up to 83).

**Incomplete run.**

**Contaminated run** — reported, never a baseline: 

- 50.0% of operations ended in a quota/unavailability response even after the app's own retry/backoff had a chance (42 of 84 operations; threshold 5%; see the operational suite's status_breakdown for raw 429/503/timeout counts)

- run incomplete (call budget exhausted)

| alias | resolved model | calls |
|---|---|---|
| solvia-eval-fast | gemini-3.5-flash-lite | 45 |
| solvia-eval-smart | unknown | 252 |

Raw call status (429s reported separately from 503s; a status a retry resolved still counts here): ok × 45, 429 × 0, 503 × 252, timeout × 0, other × 0.

## grounding

Datasets: retrieval v2

| metric | value | 95% CI | n |
|---|---|---|---|
| grounding.false_refusal_rate | 0.3774 | [0.259, 0.512] | 53 |
| grounding.refusal_accuracy | 0.0000 | [0.000, 1.000] | 0 |
| grounding.refusal_accuracy.far | 0.0000 | [0.000, 1.000] | 0 |
| grounding.refusal_accuracy.near_miss | 0.0000 | [0.000, 1.000] | 0 |
| grounding.citation_validity | 1.0000 | [0.896, 1.000] | 33 |
| grounding.expected_ref_hit | 0.9091 | [0.764, 0.969] | 33 |

## operational

Datasets: 

| metric | value | 95% CI | n |
|---|---|---|---|
| knowledge_agent.no_tool_call_rate | 0.0000 | [0.000, 0.084] | 42 |
| knowledge_agent.json_fallback_rate | 1.0000 | [0.916, 1.000] | 42 |
| knowledge_agent.latency_p50_seconds | 0.0000 | - | 252 |
| knowledge_agent.latency_p95_seconds | 0.0000 | - | 252 |
| knowledge_agent.attempts_mean | 6.0000 | - | 252 |
| knowledge_agent_fallback.no_tool_call_rate | 0.0714 | [0.025, 0.190] | 42 |
| knowledge_agent_fallback.json_fallback_rate | 0.0714 | [0.025, 0.190] | 42 |
| knowledge_agent_fallback.latency_p50_seconds | 1.2670 | - | 45 |
| knowledge_agent_fallback.latency_p95_seconds | 3.9070 | - | 45 |
| knowledge_agent_fallback.attempts_mean | 1.0710 | - | 45 |

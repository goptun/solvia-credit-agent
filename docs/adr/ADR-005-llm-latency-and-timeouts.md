# ADR-005: `knowledge_agent` LLM latency, false refusals, and timeouts

## Status

Accepted. Numbers come from small manual samples and justify defaults; they are not a benchmark (see Caveats).

## Context

Task 14.3 added a manual end-to-end evaluation (`scripts/eval_end_to_end.py`) that runs the real grounding stage (`ground_answer` with the real LLM through the gateway) over the 31-question eval set. Its first real run died on an LLM timeout under a single global 30 s timeout, which led to measuring latency, where it comes from, how much of the end-to-end false-refusal rate the prompt actually explains, and what the timeouts should be.

## 1. Where the latency comes from

Single-attempt probe (one call each, no retries, 120 s timeout, 6.9 k-character `knowledge_agent`-sized prompt with 5 real chunks). "Resolved" is the model the gateway actually served.

| Call | `solvia-fast` | `solvia-smart` |
|---|---|---|
| (a) trivial plain completion ("ok", 1 token), 2 runs | 21.7 s, 23.8 s (both resolved `gemini-3.5-flash-lite`) | 20.1 s, 14.1 s (both `gemini-3.5-flash-lite`) |
| (b) native tool-calling with the `KnowledgeAnswer` schema, 2 runs | 17.0 s, 11.4 s (`gemini-3.5-flash-lite`; both succeeded natively) | 20.2 s (`gemini-3-flash-preview`, 753 reasoning tokens), 52.1 s (`gemini-3.5-flash-lite`); both succeeded natively |
| (c) JSON mode, same prompt, 2 runs | 14.0 s ok, 9.7 s **parse failure** | 26.8 s, 26.7 s (both ok) |
| Real `ainvoke_structured` path, 1 run | 40.1 s, 1 native attempt, no fallback | 27.9 s, 1 native attempt, no fallback |

The same models called directly through the gateway (bypassing the combos), 2 calls each:

| Model | Result |
|---|---|
| `gemini/gemini-3.8-flash` | `429 quota exceeded` in 0.35–0.38 s, both calls |
| `gemini/gemini-3.7-flash` | `503` (8.3 s, then 0.1 s) |
| `gemini/gemini-3.6-flash` | `429` in 0.4 s, both calls |
| `gemini/gemini-3-flash-preview` | `429` in 0.4 s, then **200 in 1.2 s** |
| `gemini/gemini-3.5-flash-lite` | 200 in **34.7 s** and **9.4 s** (one-token answer) |

**Root cause: upstream availability, not the tool-calling path and not (visibly) the combo order.**
- Native tool-calling works: it succeeded on the first attempt every time, so the JSON-mode fallback never engaged and `structured.py` needs no change. (JSON mode is the flakier path — 1 of 4 responses did not parse — which the existing repair retry covers.)
- The better models fail *fast* (429/503 in well under a second — cheap to retry within a turn). When they do answer they are fast (`gemini-3-flash-preview`: 1.2 s). Traffic therefore mostly lands on `gemini-3.5-flash-lite`, which needs 9–35 s even for a one-token reply; 11 of 12 probe calls through the combos resolved to it.
- I could not see the combo definitions (the gateway's management API requires dashboard authentication and I did not attempt to bypass it), and the `gemini-3.8-flash` timeouts seen in the gateway UI showed up from this side as `429`, not timeouts.

What to change **in the gateway** (not in this repo): give each combo at least one model with real quota headroom (a paid or higher-quota key for the models that answer in ~1 s), and move `gemini-3.5-flash-lite` last or replace it — it is the slow landing spot. Fast-failing 429/503s in front of it are fine.

## 2. Smart vs. fast on the eval set

An early pair of back-to-back runs was **discarded** (upstream `429` quota, not the tiers). Paced runs (8 s between LLM-reaching questions, 120 s eval timeout):

| | smart | fast |
|---|---|---|
| False-refusal, answerable | 50.0% (11/22 evaluated) | 42.9% (9/21) |
| Refusal accuracy, unanswerable | 100% (5/5) | 100% (5/5) |
| Citation hit rate (answered) | 100% (11/11) | 100% (12/12) |
| Whole-call latency p50 / p95 / max | 43.1 / 184.4 / 190.2 s | 44.4 / 118.2 / 143.8 s |
| Errors even at 120 s | 4 of 31 | 5 of 31 |

A later smart run on a healthier gateway (per-question decomposition, below) had **no errors** (25/25 answerable and 6/6 unanswerable evaluated), whole-call latency p50 16.7 s / p95 39.8 s / max 66.5 s (2 of 23 calls over 30 s), false-refusal 44.0% (11/25), unanswerable 6/6 refused, citation hit 92.9% (13/14). Latency swings by a factor of ~3 between runs on the same code — it tracks the gateway's upstream state, not the tier. Fast was not meaningfully faster or better; `knowledge_agent` stays on `smart`.

## 3. Where the false refusals come from

For each refused answerable question, was the expected document/article among the retrieved top-5? (Smart, paced, 25 answerable questions, 11 refused.)

| Cause | Count | Attributable to |
|---|---|---|
| Refused by the similarity threshold | 4 (the gold chunk **was** in the retrieved context for 3 of them) | retrieval score / threshold |
| LLM refused although the gold chunk was in context | **1** | the prompt |
| Gold chunk not retrieved (LLM refused what the context could not answer) | 6 | retrieval recall |

Only **1 of 11** false refusals is attributable to the prompt. The high end-to-end false-refusal rate is mostly a retrieval problem (6 questions never surfaced the right chunk, matching the ~68% recall@k) plus the threshold discarding 3 correct retrievals. The LLM stage refusing when the context does not contain the answer is the *correct* behavior, not a defect.

### Prompt A/B (only on the attributable subset)

Question set: the 1 LLM-attributable question, plus all 6 unanswerable ones; smart tier, paced, 3 repetitions per instruction.

| Instruction | Attributable question refused | Unanswerable refused |
|---|---|---|
| Short (`Se os trechos não permitirem responder à pergunta, retorne uma lista de claims vazia.`) | 0 of 3 runs | 5/5, 6/6, 5/5 evaluated (1 transient error in two runs) |
| Long "REGRA DE RECUSA" paragraph (added in the first pass of task 14.3) | **3 of 3 runs** | 6/6 in all three |

The short instruction lowered false refusals without breaking unanswerable refusals, so it is kept (it already states the empty-`claims` rule explicitly) and the longer paragraph was removed along with the A/B code. **Weak evidence:** the subset is a single question, so this decides one question's behavior, not a general property of the two wordings.

## 4. Per-turn deadline

`LLM_TURN_DEADLINE_SECONDS` (default 45 s) is a single budget per conversation turn, carried in a context variable and enforced in `ainvoke_structured` and `invoke_with_resilience`: retries, the JSON-mode fallback and smart -> fast degradation must all fit inside it; on expiry the in-flight call is cancelled, the fallback is *not* tried, and the turn returns the unavailable reply (`knowledge_agent` returns it as its draft; the standalone endpoint returns `503`). Without an active deadline (tests, manual scripts) nothing changes. Its interaction with the measurements above: single smart attempts took 20–52 s against a 45 s budget, so a *slow* smart attempt rarely leaves room for a retry — but the failures that dominate today (429/503) return in under a second, and those retries do fit.

## 5. Non-Gemini last-resort fallback (smoke check)

After the analysis above, `cf/@cf/openai/gpt-oss-120b` (a different provider and model family) was added as the **last** fallback in both gateway combos, so a turn is no longer stranded when every Gemini model is out of quota. `scripts/smoke_gateway.py` (8 graph turns plus 2 direct knowledge scenarios, production timeouts, no eval) passed, and **every LLM call resolved to `gemini-3.5-flash-lite`** — the gpt-oss fallback was never reached, so its structured-output parsing and Portuguese reply quality are **unverified**. The smoke script now prints, per LLM call, the resolved model, latency and completion kind, so the first turn that does fall through to gpt-oss will be visible; a model from another family may behave differently on tool-calling and Portuguese, and should be checked directly the first time it is exercised. This fallback also sends prompts (synthetic customer data, the fictional product catalog, public regulation text) to one more provider.

That smoke run also surfaced a **pre-existing** issue in `apps/agent/llm/structured.py`: when the model answers a native structured call *without* emitting a tool call, LangChain's tool parser returns `None` instead of raising, so `_native_then_json_mode` returns `None` and never reaches the JSON-mode fallback. In `compliance_guard` this is swallowed (`except Exception: return False`), so for those turns (6 of the 8 graph turns) the LLM approval-promise check silently did nothing and only the deterministic keyword check protected the reply. Not changed in this PR; recorded under known limitations.

## Decision

- Timeouts are per tier: `LLM_TIMEOUT_SECONDS_FAST` = **30 s**, `LLM_TIMEOUT_SECONDS_SMART` = **45 s**, replacing the single global value. From single-attempt knowledge-sized calls (n = 5 per tier: fast 9.7 / 11.4 / 14.0 / 17.0 / 40.1 s, p50 14 s; smart 20.2 / 26.7 / 26.8 / 27.9 / 52.1 s, p50 27 s): `fast` is set just above its typical worst case, and `smart` equals the turn deadline because a single attempt can never usefully outlive the turn. The 120 s used in the eval runs is a measurement override and is **not** the production value.
- `knowledge_agent` stays on `smart`, with the short refusal instruction.
- No change to `structured.py`; gateway-side changes are recommended above.

## Caveats

- Small samples: n = 5 single-attempt latencies per tier; 31 eval questions (21–25 evaluated per run); the prompt A/B decides one question. One question is 4 pp of false-refusal rate.
- Latency depends on the upstream provider's quota state at the time; runs differed by ~3x. A 30 s fast timeout would have cut off one of five samples (40.1 s).
- Measured through an SSH tunnel from a development machine, not from the VPS.
- Two probe calls per model directly against the gateway are a snapshot, not an availability study.

## Consequences

- `LLM_TIMEOUT_SECONDS` no longer exists; use the per-tier variables (defaults apply if unset). `LLM_TURN_DEADLINE_SECONDS` is new.
- The next experiments with evidence behind them are on the retrieval side, not the prompt: (1) the threshold discarded the gold chunk for 3 answerable questions while the LLM stage already refuses 2 of the 6 unanswerable ones by itself — a lower threshold is worth testing against the near-miss set; (2) 6 questions never retrieved the gold chunk (retrieval recall, reranking).
- Re-measure latency from the VPS as part of `add-vps-deploy`, after the gateway's combos are fixed.

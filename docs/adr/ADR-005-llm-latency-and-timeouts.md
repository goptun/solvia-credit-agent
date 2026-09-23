# ADR-005: `knowledge_agent` LLM latency, smart vs. fast tier, and per-tier timeouts

## Status

Accepted. Numbers are from a small manual sample and are meant to justify defaults, not to be a benchmark (see Caveats).

## Context

Task 14.3 added a manual end-to-end evaluation (`scripts/eval_end_to_end.py`) that runs the real grounding stage (`ground_answer` with the real LLM through the gateway) over the 31-question eval set. Its first real run died on an LLM timeout under the single global `LLM_TIMEOUT_SECONDS=30`, which prompted measuring latency and comparing the `smart` tier (current mapping for `knowledge_agent`) with `fast`.

## Measurements

Two earlier back-to-back runs were **discarded**: the gateway's upstream provider was returning `429 quota exceeded` (a per-minute quota), so their errors and latencies (p95 ~250 s smart, ~19 of 31 errored on fast) measured rate limiting, not the tiers. The comparison below is from paced reruns (`--pause-seconds 8` between LLM-reaching questions), same corpus (482 chunks, MiniLM), same 31 questions, timeout override 120 s for the run (not the production value).

| | smart (`solvia-smart`) | fast (`solvia-fast`) |
|---|---|---|
| Questions evaluated (errors excluded) | 22 answerable, 5 unanswerable | 21 answerable, 5 unanswerable |
| Errors (call failed even at 120 s) | 4 of 31 | 5 of 31 |
| False-refusal rate, answerable | 50.0% (11/22; threshold 4, LLM stage 7) | 42.9% (9/21; threshold 4, LLM stage 5) |
| — colloquial / lexical | 60.0% / 47.1% | 66.7% / 38.9% |
| Refusal accuracy, unanswerable | 100% (5/5) | 100% (5/5) |
| — far / near-miss | 100% / 100% | 100% / 100% |
| Citation hit rate (answered answerable; expected document and article cited) | 100% (11/11) | 100% (12/12) |
| Latency, calls that reached the LLM | n=19: p50 43.1 s, p95 184.4 s, max 190.2 s | n=18: p50 44.4 s, p95 118.2 s, max 143.8 s |
| Calls over 30 s | 15 of 19 | 13 of 18 |

## Reading the results

- **Fast is not meaningfully faster here.** Both tiers sit at ~43–44 s p50; the tail differs (184 s vs 118 s p95) but with n≈19 that is noise-level. The latency is dominated by the gateway path (upstream provider behavior, retries, and the native-then-JSON-mode structured-output fallback), not by the reasoning-vs-non-reasoning tier. There is no case here for moving `knowledge_agent` to `fast` for speed.
- **Quality is comparable at this sample size.** Refusal accuracy and citation hit rate are equal (100%); false-refusal is 50% vs 43% — a 2-question difference on ~21, within noise. Fast has a smaller output budget (`max_tokens` 256 vs 1024), which is a risk for multi-claim answers that this sample does not exercise.
- **Both tiers over-refuse answerable questions end to end** — 43–50% false-refusal, of which the threshold accounts for 4 questions (the retrieval-only filter) and the LLM grounding stage for 5–7 more (empty or fully-invalid claims). Unanswerable questions are refused correctly (100%, near-miss included), so the refusal rule is doing its job on negatives; the open question is its cost on positives. The explicit "return empty claims" rule added in task 14.3 was not A/B-tested against the previous prompt, so whether it *causes* part of the over-refusal is unknown; that comparison is the natural next experiment.
- **Citations are trustworthy when an answer is given**: every answered question cited the expected document and article.

## Decision

Replace the single `LLM_TIMEOUT_SECONDS` with per-tier settings, `LLM_TIMEOUT_SECONDS_FAST` (default **45 s**) and `LLM_TIMEOUT_SECONDS_SMART` (default **60 s**), applied by the factory to the resolved tier. `knowledge_agent` stays on `smart`.

Justification, and its limits: a 30 s single-attempt timeout cut off calls that would have completed (the first real run failed on it), and end-to-end call latency through the gateway had a ~43–44 s median on both tiers. The defaults are therefore sized so one attempt is not cut off before it can finish, with `smart` allowed longer for reasoning-token spend. They are **not** derived from a measured per-attempt latency: the eval timed the whole `ground_answer` call, which includes up to three retries and the JSON-mode fallback, so per-attempt latency was not isolated. They are starting values to tune once per-attempt latency is read from the LangFuse traces.

The 120 s used in the eval runs is a measurement override so that runs are not cut short; it is **not** the production value, and the script documents this.

## Caveats

- Small sample: 31 questions, 21–22 evaluated per run after excluding 4–5 errors; one question is ~4.5 pp. Differences of a couple of questions between tiers are noise.
- Latency includes retries and fallback, and depends on the upstream provider's quota state at the time (two runs were unusable because of it). Numbers will differ on another day.
- The 4–5 errored questions per run are excluded from the rates, which can bias them in either direction.
- Not measured from the VPS; the tunnel adds its own latency.

## Consequences

- `LLM_TIMEOUT_SECONDS` no longer exists; deployments must use the per-tier variables (defaults apply if unset).
- Follow-ups worth doing, not done here: read per-attempt latency from traces and retune the defaults; A/B the refusal rule against the previous prompt to see how much of the LLM-stage false refusal it causes; decide whether the gateway's quota needs a different upstream for eval runs.

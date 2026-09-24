# ADR-006: Evaluation harness (`evals/`)

## Status

Accepted. Spec: [`openspec/changes/add-llm-evals/specs/evaluation/spec.md`](../../openspec/changes/add-llm-evals/specs/evaluation/spec.md)
(design in [`design.md`](../../openspec/changes/add-llm-evals/design.md)); it also modifies the
`conversation-graph` (compliance fail-closed) and `regulatory-knowledge-base` (retrieval evaluation,
CI model download) specs.

## Context

`add-regulatory-rag` shipped quality numbers that were expensive to obtain and easy to misread: a
31-question set where one question moves a metric by ~4 points, live runs contaminated by upstream
`429` quota and gateway fallback ([ADR-005](ADR-005-llm-latency-and-timeouts.md)), and a compliance
guardrail that silently disabled itself for weeks without a test noticing. The next change
(`improve-retrieval-quality`) will tune retrieval and prompts, which is only safe with a trustworthy
instrument. This change builds that instrument and deliberately **tunes nothing**.

## Decision

One package, one CLI: `python -m evals run|report|baseline|review-sample|fixture|langfuse|relevant`.

- **Layout.** `evals/core/` is pure Python (schemas, statistics, metrics, baseline/budget/contamination
  logic; a test forbids LangChain/psycopg/FastAPI imports there). `evals/adapters/` holds I/O
  (Postgres, the corpus fixture, git, the instrumented gateway factory, LangFuse). `evals/suites/`
  orchestrates offline and live suites. `evals/` may import `apps/` and `rag/`; they never import it.
- **Datasets.** Four versioned YAML files validated by Pydantic (retrieval/grounding 98, router 69,
  slots 36, compliance 41 approval + 19 PII cases), with a labelling README. Difficulty is *derived*
  from item properties, not declared; every answerable item carries a quoted evidence string
  verified against the corpus fixture; near-miss negatives were checked against the corpus and the
  product configuration.
- **Metrics, always with 95% intervals** (Wilson for proportions, seeded percentile bootstrap for
  means): recall@k and MRR (a hit is any acceptable article of the expected document, cross-document
  pairs allowed), best-similarity distributions and a threshold table (report only), grounding
  (false refusal with exactly one cause each: threshold → gold not retrieved → LLM refused with gold
  in context; refusal accuracy far/near-miss; citation validity; expected-ref hit), router accuracy
  and confusion matrix, slot exact match per field, approval-promise precision/recall for the keyword
  screen, the strict fail-closed screen and the LLM check, PII masking exact match, and operational
  metrics from a per-call log (resolved model of *every* call, structured-output calls included).
- **Offline mode** (PR CI, no secrets, no gateway): `compliance` always; `retrieval` when the diff
  touches `rag/**`, `evals/**`, `uv.lock` or the compliance nodes. Retrieval indexes a committed
  chunk fixture (the corpus is never fetched in CI) into a per-embedding-model schema with
  `hnsw.ef_search` at its maximum, so search is effectively exact and two runs print identical
  numbers. The job caches the embedding model (public hub, unauthenticated, retried) and compares
  against committed baselines.
- **Live mode** (local only, through the SSH tunnel): `router`, `slots`, `compliance-llm`,
  `grounding`, sequential and paced, under a hard call budget (`EVALS_MAX_GATEWAY_CALLS`, a pre-run
  estimate that refuses to start over budget, a mid-run stop that marks the run incomplete),
  stratified seeded sampling, and **contamination detection**: quota/unavailability share, a model
  outside the alias's expected set, too many calls outside the primary set, incomplete or
  interrupted. A sampled, incomplete or contaminated run is reported but can never become a baseline.
- **Baselines** (`evals/baselines/<suite>.json`) change only through `baseline update`, from a
  complete, unsampled, uncontaminated run on a dataset hash the maintainer approved
  (`evals/datasets/review.yaml`), recording commit, dataset versions, aliases and model mix.
  Tolerances: ±0.03 overall, ±0.06 per stratum, **0 for deterministic metrics** (any change fails,
  improvements included). A regression prints a `metric | baseline | new | tolerance | diff | status`
  table and exits non-zero.
- **The retrieval baseline comes from CI, not from a laptop.** Embedding floats differ by CPU
  architecture and can reorder near-ties (the same code gave MRR 0.529 locally and 0.522 in CI), so
  `baseline update --suite retrieval --from-artifact` accepts only a CI-produced x86_64 artifact whose
  commit is an ancestor of `HEAD` with nothing changed outside `evals/baselines/`. Local retrieval
  runs are informational and never fail.
- **LangFuse** is best effort: datasets are mirrored with deterministic item ids (an idempotent
  upsert) and live runs are published with per-item scores and aggregates. Without credentials, or
  when publication fails, the run is logged and skipped — the committed report is the source of truth.
  Publication replays the outputs the runner recorded rather than calling the gateway again through
  `run_experiment`, which keeps the budget and pacing exact.
- **Review gate.** No baseline is recorded before the maintainer approves a stratified sample of each
  dataset (retrieval items shown with the top-3 chunks the fixture retrieves). The approved version
  and content hash are committed and a test fails if a dataset changes without a new approval.

## Deliberately not built

- LLM-judged metrics (promptfoo/ragas/deepeval): heavy dependencies, and a judge needs the same
  gateway whose quota contaminates runs.
- Live evaluation in CI: see [ADR-007](ADR-007-live-evals-in-ci-over-tailscale.md).
- Threshold, prompt, model or retrieval tuning: the threshold table only reports.
- Making `Evals (offline)` a required check: that needs a repository ruleset change, which is the
  owner's decision.

## Findings and follow-ups

The dataset work exposed bugs that this change records instead of fixing:

1. **Four `mask_pii` bugs — priority `fix/` PR right after this change.** They are kept as
   `known_gap` items, so masking exact match is 15/19 (78.9%) today and the baseline records the
   failures:

   | Input | Today | Correct |
   |---|---|---|
   | `CPF 00000000000 …` (no punctuation) | `[DADO PROTEGIDO]0` — a digit leaks (the phone pattern matches first) | one placeholder |
   | 16-digit card number | masked in two pieces | one placeholder |
   | account with a one-digit check digit (`12345-6`) | not masked | masked |
   | card as space-separated groups | not masked | masked |

   When the masker is fixed a test tells the fixer to drop the `known_gap` flag and re-record the baseline.
2. **Product follow-up.** The catalog descriptions the knowledge base indexes omit values the
   simulator knows — `monthly_interest_rate` 2.5%, IOF (0.38% fixed + 0.0082% per day, capped at 365
   days) and the R$ 50 origination fee — so the corpus cannot answer near-miss questions R-084/R-085.
3. **Keyword screen weakness.** Recall of the keyword promise screen is 0.29 (n=14) and the strict
   screen's false-positive rate on approval mentions is 0.33 (n=12): both are now measured, and the
   fail-closed path is exactly as strict as those numbers say.

## Consequences

- Every quality claim in the repository now has an interval, a dataset version and a commit behind it.
- Retrieval regressions fail the PR; tuning changes must move the baseline explicitly.
- Live numbers are trustworthy only while the gateway's expected model sets are approved
  (`evals/live_config.yaml`); until then a run executes but blocks baselining.
- The instrument costs CI about a minute per relevant PR, plus a cached embedding-model download.

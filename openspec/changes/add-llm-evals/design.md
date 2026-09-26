## Context

See `proposal.md` — "Why" for motivation, and `docs/adr/ADR-003-embedding-model.md` / `ADR-005-llm-latency-and-timeouts.md` for what the previous change learned. Relevant current state:

- `rag/eval/` holds a 31-question YAML set, recall@k/MRR/refusal code (`run.py`) and an end-to-end aggregator (`end_to_end.py`); `scripts/eval_end_to_end.py` and `scripts/smoke_gateway.py` wire the real LLM, and the smoke script already has a LangChain-callback logger that captures the resolved model, latency and native-vs-no-tool-call per call (including native tool-calling calls, whose parsed result carries no metadata).
- Grounding is the public `ground_answer(...)`; routing/slots/compliance are nodes built by `make_router_node`, `make_offer_simulator_node`, `make_compliance_guard_node`. The deterministic pieces (`keyword_flags_approval_promise`, `keyword_flags_unhedged_approval_mention`, `mask_pii`) are public in `apps/agent/nodes/compliance.py`; the LLM approval verdict is currently a private function inside `compliance_guard.py`.
- CI (`.github/workflows/ci.yml`) has three required checks (`Lint & type-check`, `Tests`, `Secret scanning (gitleaks)`), a pgvector service container, `LLM_PROVIDER=fake`, and — by spec — never downloads the corpus or an embedding model.
- The corpus is fetched from official sites into a gitignored folder; extraction/chunking is deterministic (`rag/ingest/*`), the manifest pins a sha256 per document, and the index is `rag_chunks` (pgvector HNSW + full-text).
- LangFuse SDK 4.15 is already a dependency; it provides `create_dataset`, `create_dataset_item(id=…)` (upsert by id), `run_experiment(max_concurrency=…)` and `create_score`.
- Live measurements showed: upstream quota (`429`/`503`) contaminates runs, the gateway may resolve to a different underlying model than usual, and single-attempt latency varies ~3x between runs.

## Goals / Non-Goals

**Goals:**
- A measuring instrument whose numbers can be trusted and compared across commits: deterministic offline metrics, confidence intervals, and live runs that can prove they were not contaminated.
- A PR gate that catches retrieval and compliance regressions without the gateway, without new secrets, and without slowing PRs that do not touch relevant code.
- Keep gateway usage minimal and bounded — by construction (budget, pacing), not by discipline.

**Non-Goals:**
- Tuning anything (retrieval parameters, prompts, models, thresholds) — the next change does that with this harness.
- LLM-as-judge scoring or open-ended answer-quality grading. Every metric here is label-based and deterministic given the model outputs.
- Running live suites in GitHub Actions (decision record only, see Decision 10).
- A generic evaluation framework: the harness covers this project's five components.

## Decisions

### 1. A small in-repo `evals/` package, not an evaluation framework

`evals/` is an orchestration layer (like `scripts/`): it may import `apps/` and `rag/`, which never import it. Inside it, following the repo's ports-and-adapters convention:

- `evals/core/` — pure Python, no LangChain/psycopg/FastAPI imports: dataset schemas, metrics (Wilson, bootstrap, recall/MRR, confusion matrix, precision/recall), threshold table, baseline comparison, budget accounting, contamination rules, report rendering.
- `evals/adapters/` — the pgvector fixture indexer, the embedding adapter, the instrumented gateway factory (callbacks), the LangFuse publisher (behind a port with a fake).
- `evals/suites/` — one module per suite wiring datasets + system under test + metrics.
- `evals/cli.py` (`python -m evals …`) — `run`, `baseline update`, `report`, `review-sample`, `fixture build|verify`, `langfuse sync`.

Alternatives: adopting `promptfoo`/`ragas`/`deepeval` — rejected: heavy dependencies, LLM-judged metrics need the gateway (the scarce resource), and our metrics are simple label comparisons whose exact definitions we want in the repo. The existing `rag/eval` logic is moved into `evals/core` and the old CLI removed (Migration Plan).

### 2. Datasets: YAML files, Pydantic schemas, derived tags

Datasets live in `evals/datasets/*.yaml` (`version` integer + `items`), one Pydantic model per dataset, validated by unit tests. YAML (not JSONL) because items are hand-labelled and reviewed as diffs; Portuguese text and multi-line evidence quotes read better.

- **retrieval** (`R-###`): `question`, `kind` (`answerable`|`unanswerable`); answerable: `document` (`cdc-consolidada`|`lgpd`|`open-finance-regulamento`|`cet-disclosure`|`product-catalog`), `expected_refs` (list, one or more acceptable articles; absent only for the catalog), `evidence` (≤200 chars literal quote), `style` (`lexical`|`colloquial`), `accents` (false = written without accents); unanswerable: `distance` (`far`|`near_miss`). Composition targets: ~75 answerable (≥40% colloquial; every document and the catalog covered; some unaccented; some multi-ref), ~25 unanswerable (≥15 near-miss).
- **router** (`T-###`): `message`, `active_flow` (`none`|`consent_confirmation`|`slot_filling`), `category` (`clear`|`ambiguous`|`continuation`), `expected` (an intent, or `continue` for a continuation that must not be reclassified), `acceptable` (list; only for ambiguous items). ~60 items covering all six intents plus continuations such as "sim, autorizo" and "24 meses".
- **slots** (`S-###`): `message`, `known` (slots already filled), `expected` per field (`amount`, `term_months`, `amortization_type`; absent = must stay unset), `tags` (`complete`|`partial`|`missing`|`invalid`). ~30 items. Scored on the slots the node ends with (after its own validation), so "don't invent a value" and "reject a negative amount" are both measured.
- **compliance** (`C-###` approval items, `P-###` PII items): approval items have `text`, `label` (`promise`|`hedge`|`neutral`), `tags` (`adversarial`, `fail_closed`, `mentions_approval`); PII items have `text` and `expected_masked`. ~40 approval + ~15 PII items, including the four PR #5 promise/hedge examples. All identifiers are synthetic or reserved-format.

**Difficulty is derived, not free-form**, so tags cannot drift: answerable — `easy` = lexical ∧ one ref; `medium` = colloquial *xor* multi-ref; `hard` = colloquial ∧ (multi-ref ∨ unaccented); unanswerable — `far` = easy, `near_miss` = hard. Tests recompute and compare. Evidence quotes are verified against the **committed chunk fixture** (Decision 5), so CI can check them without the corpus. Labelling rules (what counts as colloquial, ambiguous, adversarial; how to pick acceptable refs; what makes a near-miss) are written in `evals/datasets/README.md`.

**Review gate**: `python -m evals review-sample --dataset D --n 15 --seed 42` prints a stratified sample (round-robin over tag/label strata, seeded, so the same seed prints the same items). For **retrieval** items it also prints supporting evidence computed offline from the committed fixture index — the top-3 chunks the production hybrid search returns (source-type scoped for answerable items, unscoped for unanswerable ones): document, article ref, similarity and the first 200 characters. For near-miss unanswerable items this is how the maintainer verifies the corpus really does not answer them. It embeds the question with the local embedding model and calls no gateway; the similarities are informational (they are computed on the maintainer's machine, see Decision 6). After the maintainer approves, the dataset's content hash is committed to `evals/datasets/review.yaml`; `baseline update` refuses any dataset whose current hash differs. This makes the blocking gate in `tasks.md` mechanical rather than a convention.

### 3. Metric definitions

All proportions get a **Wilson 95% interval**; means (MRR, latency-derived) get a **seeded percentile bootstrap** (10 000 resamples).

- **Retrieval** (offline): a question is a *hit at k* if any chunk in the top-k (source-type scoped, as the agent does) belongs to the expected document and has one of the acceptable article refs (any for the catalog). `recall@k` = hits/answerable; `MRR` = mean reciprocal rank of the first hit (0 if none); both overall and by `style`, `document`, `difficulty`. *Best similarity* = top-1 vector cosine similarity; reported as quantiles for answerable vs unanswerable. **Threshold table**: for each `t` in 0.30…0.90 step 0.05, `false_refusal(t)` = share of answerable with best similarity < t, `correct_refusal(t)` = share of unanswerable with best similarity < t. Report only.
- **Grounding end to end** (live): per question run `ground_answer`; `false_refusal` = answerable refused; `refusal_accuracy` = unanswerable refused, split `far`/`near_miss`; `citation_validity` = share of answers whose every rendered citation matches a chunk in *that turn's* retrieved set; `expected_ref_hit` = share of answered answerable questions citing the expected document and an acceptable ref. Each false refusal gets exactly one **cause**, checked in this order: `threshold` (best similarity below the threshold), `gold_not_retrieved` (passed the threshold, expected source absent from top-k), `llm_refused_gold_in_context`.
- **Router** (live): prediction = the intent in the node's update, or `continue` when the node leaves state untouched for an active flow. Accuracy counts a prediction correct if it equals `expected` or is in `acceptable`; the confusion matrix is expected × predicted over the six intents plus `continue`.
- **Slots** (live): exact match per field (Decimal equality for amount); plus "no invented value" = share of absent expected fields left unset.
- **Compliance**: positive = `promise`. Precision/recall for (a) the keyword screen, (b) the strict fail-closed screen, both **offline and deterministic**, and (c) the LLM verdict (live). For (b) also its false-positive rate on `hedge`/`neutral` items that mention approval. Masking = exact match of `mask_pii` output.
- **Operational** (live): from the instrumented factory, per node: single-attempt latency p50/p95 (one raw provider call = one attempt), attempts per operation, **no-tool-call rate** (a native structured call whose completion has no tool call), **JSON-fallback rate** (operations that made a plain call after a native attempt), and the resolved underlying model of every call — captured through LangChain callbacks so structured-output calls are covered (their parsed result carries no metadata; the raw generation does).

### 4. Offline vs live architecture

| | Offline | Live |
|---|---|---|
| Needs | Postgres+pgvector, embedding model, fixture | + gateway via tunnel, `.env` |
| Suites | `retrieval`, `compliance` (keyword screen, strict screen, masking) | `router`, `slots`, `compliance-llm`, `grounding` |
| Where | PR CI and locally | Locally only |
| Determinism | Exact (see below) | Statistical; contamination-checked |
| LLM | none (fake where a port is needed) | real, budgeted |

Offline retrieval determinism: fixture order and chunk ids are fixed; the eval session sets `hnsw.ef_search` to its maximum (≥ the ~500 fixture rows, i.e. effectively exact search) so approximate-index variance cannot move a metric; two consecutive runs must print identical numbers (a test asserts it). Cross-machine float differences in the embedding model (ARM laptop vs x86 CI, onnxruntime) can reorder near-ties in the top-k, so the retrieval baseline is **always recorded from CI's own run** (Decision 6) and local retrieval runs are informational; the tolerances absorb residual drift within CI.

Live suites run items strictly sequentially through one instrumented `LLMFactory` (Decision 7), so budget and pacing are exact, and share one run record (calls, statuses, models, latencies) that feeds operational metrics, contamination checks and the report.

### 5. Committed corpus chunk fixture (CI without the corpus)

`python -m evals fixture build` (dev-only; needs the fetched corpus) writes `evals/fixtures/corpus_chunks.jsonl`: one line per chunk with the exact fields `rag_chunks` stores except the embedding (chunk id, document id, source type, norm, article ref, hierarchy path, source URL, version date, amendment note, content), plus a header line with each document's manifest sha256 and fetch date. The source texts are Brazilian official acts (public domain in Brazil) with provenance kept in the metadata, as in `add-regulatory-rag`. Expected size ~1 MB; a test caps it at 2 MB.

`rag/ingest/indexing.py` gets a small refactor: the chunk-upsert step is exposed as `index_chunks(conn, chunks, embed_documents, …)` (no behavior change; `index_document` calls it). CI embeds ~480 chunks with the real model (~40–60 s on 2 vCPUs) into the service container and evaluates.

`fixture verify` (dev) recomputes chunks from the fetched corpus and diffs against the fixture; a CI unit test compares the fixture's recorded document hashes with `rag/corpus/manifest.yaml`, so a manifest change without a fixture rebuild fails the PR. Alternative rejected: fetching the corpus in CI — the Planalto site rejects non-browser clients, its HTML has per-request tokens (see `add-regulatory-rag`), and it would make every PR depend on a third-party site.

### 6. Baselines, tolerances and the CI gate

`evals/baselines/<suite>.json` holds (the run record also stores `environment` — CI flag, OS, architecture): suite, mode, git commit, dataset name/version/hash, gateway aliases, resolved-model mix per alias (live), creation time, and per metric `{value, ci, n, direction, tolerance}`. `baseline update --suite S --from-run R` refuses when: the run is contaminated, incomplete or **sampled**; the working tree is dirty or the run's commit is not `HEAD`; the dataset hash differs from the approved hash; the run mode/suite mismatches; or the suite is the offline retrieval suite (see below).

**Where each baseline comes from.** The offline **compliance** baseline is deterministic (regex/keyword logic and masking — no floats, no model) and is recorded locally from a clean tree. The offline **retrieval** baseline depends on embedding floats and must come from the environment that will enforce it: the `Evals (offline)` job uploads its run JSON as an artifact (recording the PR head commit, `environment.ci = true`, `x86_64`), and `baseline update --suite retrieval --from-artifact PATH` accepts only such an artifact — CI-produced, complete, on an approved dataset hash, and whose recorded commit is an ancestor of `HEAD` with **no changed files between them outside `evals/baselines/`** (so the baseline describes exactly the code it will gate). The maintainer downloads it with `gh run download` on the PR. Local retrieval runs (any architecture) print their metrics and a baseline diff for information but never fail, and are refused as a baseline source. Until a retrieval baseline exists the job passes with a prominent "no baseline yet" notice; once the baseline commit is pushed, the next CI run compares against it and must show zero diff. Initial tolerances (revisited after the first baselines, an Open Question): proportions and MRR overall ±0.03 (≈ two items of ~75), per-stratum ±0.06, **deterministic metrics (keyword/strict screen results, masking) ±0** — any change fails. Regression = worse than baseline by more than tolerance in the metric's direction; the gate prints a table (`metric | baseline | new | tolerance | diff | status`) and exits non-zero; an improvement beyond tolerance passes with a note. Unchanged-baseline suites in a run that do not apply are skipped, not failed.

CI job `Evals (offline)` (a new job, always triggered so its status is always reported):
1. Compute `relevant` from the PR diff with plain `git diff --name-only` (no third-party action): `rag/**`, `evals/**`, `uv.lock`, `apps/agent/nodes/compliance*.py`. On pushes to `main`, always relevant.
2. Always run the compliance/masking offline suite (seconds, no model).
3. If `relevant`: restore `actions/cache` for the embedding model (key: model name from settings + `uv.lock` hash; `FASTEMBED_CACHE_PATH` in the workspace), run migrations, index the fixture, run the retrieval suite, compare, upload the run JSON as an artifact.
4. Otherwise print "retrieval eval skipped (no relevant changes)" and succeed.
It uses no secrets and never calls the gateway. Model download is the only new network dependency (public hub, unauthenticated, cached, retried up to 3 times); this is why the `regulatory-knowledge-base` requirement is scoped (spec delta). Making the job a **required** check requires a ruleset change on the repository, which is the owner's decision and not part of this change.

### 7. Live runs: budget, pacing, instrumentation, contamination

An instrumented factory (generalizing the smoke script's callback logger) attaches a LangChain callback handler to every model it builds, recording per raw provider call: node, resolved model, status (`ok` / error type / HTTP status when available), latency, and whether the completion had a tool call. It changes nothing about the calls.

- **Budget**: `EVALS_MAX_GATEWAY_CALLS` (default **300**) counts raw provider calls, retries and fallbacks included. Before starting, the run prints the expected call count for the selected suites and refuses to start if it exceeds the budget. Estimate for the **full** suites: router 60 + slots 30 + compliance-LLM 40 + grounding ≈ 70 (about 84% of ~75 answerable and a third of ~25 unanswerable clear the similarity threshold and reach the LLM) ≈ **200 base calls**; with retries and the JSON follow-up after a no-tool-call completion (6 of 8 `compliance_guard` calls in the PR #3 smoke run had none, so compliance-LLM may need up to 2 calls per item) the pessimistic figure is ≈ **270**. The **30% validation sample** is ≈ 18 + 9 + 12 + 21 ≈ **60 base calls, ≈ 80 pessimistic**, run with `EVALS_MAX_GATEWAY_CALLS=100`. Both runs together: ≈ 260 typical, ≈ 350 pessimistic — comfortably inside a normal free-tier day only if they are on **different quota days** (see the run-2 constraint below). Mid-run, before each item it checks that the remaining budget covers a worst-case item (8 calls) and otherwise stops and marks the run **incomplete**. 300 is a starting value chosen below the burst behavior observed in ADR-005; the validation run measures real consumption and the default is revisited after it.
- **Pacing**: `EVALS_PACING_SECONDS` (default 6) minimum interval between items' first calls; runs are sequential. Suites are individually selectable, so a partial run is cheap.
- **Sampling**: `--sample F --seed S` runs a stratified fraction of each selected live dataset (same strata and round-robin as `review-sample`, so a sample always covers every stratum it can). A sampled run is recorded as `sampled` with its fraction and seed, is reported normally, and **can never become a baseline**. Run 1 (validation) uses `--sample 0.3 --seed 42`: enough to observe the model mix and real call consumption. Run 2 (baseline) runs the full suites and **must start after the provider's daily quota reset**, so run 1's consumption cannot contaminate it.
- **Contamination** (any one marks the run contaminated, with reasons, and blocks baselining): (a) share of calls with `429`/`503`/timeout > `EVALS_MAX_ERROR_SHARE` (default 5%); (b) any call resolved to an underlying model outside the alias's *expected set*, listed by name; (c) share of calls resolved to models outside the alias's *primary* set > `EVALS_MAX_FALLBACK_SHARE` (default 10%); (d) run incomplete (budget) or interrupted. Expected and primary sets live in `evals/live_config.yaml` (committed; underlying model names are allowed in evaluation tooling and reports because they are *measured output*, as in ADR-003/005, never used by application code). Because the sets cannot be known before the first live run, the **validation run** observes the mix, the report proposes the sets, and the maintainer approves them before the baseline run (a second blocking gate in `tasks.md`).
- **Outputs**: a run JSON (gitignored) and, for the validation and baseline runs, a committed Markdown+JSON report under `evals/reports/`, and LangFuse (Decision 8). Call counts of each run are printed and recorded in the report.

### 8. LangFuse integration

Behind a small publisher port, so tests use a fake and the harness works without credentials. `langfuse sync` mirrors each dataset to a LangFuse dataset (`solvia/<dataset>`) with `create_dataset_item(id=<dataset>:<item id>, …)` — the deterministic id makes it an idempotent upsert; the dataset version is stored in item metadata. A live run is published with `run_experiment(max_concurrency=1, …)` (one item at a time, so pacing and budget stay exact): the task calls the suite's per-item function, evaluators return the per-item score (e.g. hit/miss, correct/incorrect), run evaluators return aggregates; run name = `<suite>@<dataset version>/<git sha>`, metadata carries alias, mode and contamination status. Payloads contain only synthetic/public content, git SHA and aliases — never keys, hostnames or IPs (the observability spec's rules apply). Offline CI never touches LangFuse. Missing credentials → log and continue; the committed report is the source of truth.

### 9. Reports

`python -m evals report --run R [--baseline B]` renders Markdown tables (metric, value, 95% CI, n, and diff vs baseline when given) per suite, ready to paste into a PR. `python -m evals report --update-readme` rewrites only the block between `<!-- evals:metrics:start -->`/`<!-- evals:metrics:end -->` in `README.md` from the committed baselines.

### 10. Decision point: running the live suite from GitHub Actions over Tailscale (record only; **not implemented without approval**)

To be written as ADR-007 with this analysis, default outcome **local-only**.

Option A — local only (default). Option B — `workflow_dispatch` workflow that joins the tailnet with an ephemeral, tag-scoped auth key (repository/environment secret) restricted by ACL to the gateway's single port, then runs the live suite.

Trade-offs for a **public** repo: B needs at least two secrets (the tailnet key and the gateway API key), violating the "CI needs no new secrets" constraint of this change; anyone with write access can edit a workflow on a branch and exfiltrate secrets unless the workflow is pinned to `main` and the secrets live in an *environment* with required reviewers; a runner on the tailnet is a lateral-movement path if the ACL is ever loosened; ephemeral keys expire but a leaked one is usable until then; the shared free-tier quota would be consumed by unattended runs (the contamination problem again); runner egress adds latency variance that pollutes latency metrics. Benefits: repeatable runs from a clean machine, no dependence on the maintainer's laptop. Recommendation: stay local-only until live evaluation is run often enough to justify that risk; if pursued, use environment protection with required reviewers, a one-hour ephemeral tagged key, an ACL to one port, `main`-only workflow, and a hard call budget — as a separate change.

## Risks / Trade-offs

- **CI downloads the embedding model** (public hub) → cached, retried, path-scoped, unauthenticated; job fails loudly if unavailable rather than silently skipping a gate.
- **Offline retrieval metrics differ between machines** (the maintainer's laptop is ARM, CI is x86; onnxruntime float differences can reorder near-ties in the top-k) → the retrieval baseline is recorded only from the `Evals (offline)` job's artifact (`--from-artifact`), local retrieval runs are informational and never a baseline source, the eval session uses exact search, and tolerances absorb drift within CI. If the CI runner image or CPU generation changes, the baseline is re-recorded from a new artifact. The deterministic compliance/masking baseline has no such dependency and is recorded locally.
- **A committed ~1 MB fixture can go stale** → manifest-hash test in CI, `fixture verify` locally, and the fixture header records the hashes it was built from.
- **Small datasets** → confidence intervals everywhere; tolerances set from item counts; reports never claim significance from a 1–2 item difference (ADR-005's lesson).
- **Live runs are gateway-quota-bound and nondeterministic** → budget, pacing, sequential execution, contamination detection; at most two live runs in this change.
- **Datasets bias toward what we already know** (the old 31 questions seed the new ones) → the review gate, near-miss quotas, and labelling rules; the next change may add adversarial items.
- **The harness may reveal bugs** (PR #5 was found by a smoke test) → recorded as follow-ups unless they block measurement; no fixes to prompts/retrieval/models here.
- **Moving code out of `rag/eval`** could drop coverage → metric tests are ported before the old code is removed.
- **A new required check would block PRs if it flakes** → the ruleset change is left to the owner, after the job has run cleanly for several PRs.

## Migration Plan

1. Land `evals/core` (schemas, metrics, baseline/contamination logic) with tests, porting the metric logic from `rag/eval` (tests first).
2. Add the fixture and the `index_chunks` refactor; add datasets; hold the review gate.
3. Add offline suites, the CLI, the baseline mechanism, and the CI job; record the **compliance** offline baseline locally after the gate, and the **retrieval** offline baseline from the CI artifact of the PR (`--from-artifact`), also after the gate.
4. Add the instrumented factory, live suites, budget/contamination, LangFuse publisher, reports — all tested with the fake LLM.
5. Remove `rag/eval` CLI/data and `scripts/eval_end_to_end.py`; keep `scripts/smoke_gateway.py` (its callback logger becomes a thin user of the shared instrumentation or is left as-is if the sharing costs more than it saves); update README and docs.
6. Run the live validation run (30% sample), approve the expected model sets, run the full baseline run after the provider's quota reset, commit reports and baselines.
Rollback: the change adds files and one CI job and removes only superseded eval code; reverting the PR restores the previous state.

## Open Questions

- Initial tolerance values are set from item counts and may be tightened after the first offline baselines (does not change specs or tasks).
- Whether `scripts/smoke_gateway.py` should share the instrumented factory or keep its own copy (decided during implementation by the size of the diff).
- The default call budget (300) is revisited after the validation run measures actual consumption and any quota behavior.

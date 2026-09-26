## Why

`add-regulatory-rag` shipped with quality numbers that were expensive to get and easy to mislead: a 31-question set where one question is ~4 pp, live runs contaminated by upstream `429` quota and gateway fallback (ADR-005), and a compliance guardrail that silently disabled itself for weeks without any test noticing. The next change (`improve-retrieval-quality`) needs to tune retrieval and prompts, which is only safe with a trustworthy measuring instrument: versioned datasets big enough to mean something, reproducible per-component metrics with confidence intervals, a regression gate on PRs, and live runs that can prove they were not contaminated.

## What Changes

- New `evals/` package and a single CLI, `python -m evals run --suite … --mode offline|live`, with machine-readable output, deterministic seeds, committed baselines, and a Markdown report generator.
- **Versioned datasets** (YAML, Pydantic-validated, no real PII): retrieval/grounding (~100 questions, up from 31), router (~60), slot extraction (~30), compliance (~40 draft replies + ~15 PII masking cases), each with a labelling README. Mirrored idempotently to LangFuse datasets.
- **Metrics with Wilson confidence intervals**: retrieval (recall@k, MRR by style/document, similarity distributions, false-refusal vs unanswerable-refusal table across thresholds — report only), grounding end to end (false refusal, refusal accuracy far/near-miss, citation validity, expected-ref hit rate, refusal decomposition by cause), router (accuracy, confusion matrix), slots (exact match per field), compliance (precision/recall for the keyword screen, the strict fail-closed screen and the LLM check; masking exact match), and operational (per-node single-attempt latency, attempts, JSON-fallback and no-tool-call rates, resolved model for **every** call including structured-output calls).
- **Offline mode in PR CI** (no gateway, no new secrets): keyword compliance and masking on every PR; retrieval eval against a pgvector service container using a committed, processed chunk fixture of the corpus and a cached embedding model, only when `rag/`, `evals/` or the datasets change; compared to a committed baseline with explicit tolerances, failing on regression and printing the diff. The retrieval baseline is recorded from that CI job's own run (x86), never from a laptop, because embedding-model float differences across CPU architectures can reorder near-ties; local retrieval runs are informational only.
- **Live mode, local only** (via `ssh -N solvia-tunnel`): router, slots, compliance LLM check and grounding end to end, with a hard gateway-call budget, pacing, per-call resolved-model/status logging, and **contamination detection** — a contaminated run is reported but can never become a baseline. The first (validation) live run uses a stratified ~30% sample of each live dataset; only the second (baseline) run executes the full suites. Results go to LangFuse dataset runs and to a committed JSON/Markdown report.
- **Baseline lifecycle**: baselines updated only by an explicit command, only from complete, unsampled, uncontaminated runs on approved dataset versions, recording git SHA, dataset version, aliases and resolved-model mix. The deterministic compliance baseline is recorded locally; the offline retrieval baseline only from the CI job's uploaded run artifact.
- **Dataset review gate**: a stratified sample of 15 items per dataset is presented for approval before any baseline is recorded — for retrieval items together with the top-3 chunks the fixture index retrieves (so the maintainer can check that near-miss questions are really not answered by the corpus), computed offline with no gateway call; the approved dataset hash is committed and the baseline command refuses a mismatch.
- **ADRs**: the harness architecture, and a decision record (no implementation) on running the live suite from GitHub Actions through an ephemeral Tailscale key on a public repo — default outcome local-only.
- The old `rag/eval` CLI, `rag/eval/questions.yaml` and `scripts/eval_end_to_end.py` are superseded and removed (their metric logic moves into the harness); no compatibility shims.
- **Not changed**: retrieval parameters, prompts, models, thresholds. Bugs the harness reveals are recorded as follow-ups unless they block measurement.

## Capabilities

### New Capabilities
- `evaluation`: versioned evaluation datasets and their review gate, per-component metrics with confidence intervals, the offline deterministic suite and its CI regression gate, budgeted/paced/contamination-checked live runs, committed baselines, reporting, and LangFuse mirroring.

### Modified Capabilities
- `conversation-graph`: the **Compliance guardrails** requirement gains the fail-closed behavior introduced in PR #5 (strict hedge-phrase screen when the LLM verdict is unavailable, warning log, degraded flag; never a silent skip).
- `regulatory-knowledge-base`: **Retrieval quality evaluation** now refers to the versioned ~100-question dataset and the `evals` CLI; **Automated tests never require the corpus or embedding model to be fetched** is scoped to the default unit/integration suite, because the new offline eval gate is a separate CI job that downloads (and caches) the embedding model from the public hub — still never the corpus and never the gateway.

## Impact

- **Code**: new `evals/` package (orchestration layer, like `scripts/`; may import `apps/` and `rag/`, which never import it); a small `rag/ingest/indexing.py` refactor to expose the chunk-upsert step so a pre-chunked fixture can be indexed; removal of `rag/eval/*` CLI code and `scripts/eval_end_to_end.py`; `pyproject.toml` (`mypy`/`pytest` paths).
- **CI**: one new job, `Evals (offline)`, with an `actions/cache` for the embedding model and a pgvector service. Making it a **required** status check needs a change to the repository ruleset, which is the owner's call and is not done by this change.
- **Repo size**: a committed corpus chunk fixture (official public texts with provenance, ~1 MB) and dataset/baseline files.
- **Dependencies**: none beyond the existing `langfuse` SDK (v4: `create_dataset_item`, `run_experiment`, `create_score`).
- **Gateway usage**: the development of the harness uses the fake LLM only; the live suite runs at most twice (one validation, one baseline) and the call count is reported.
- **Infra/VPS**: none. **Secrets**: none added; LangFuse keys stay in the local `.env`.

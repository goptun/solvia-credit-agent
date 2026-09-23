# ADR-003: Embedding model for the regulatory knowledge base

## Status

Accepted (revised before merge after a three-way benchmark — see "Benchmark").

## Context

`add-regulatory-rag` needs a `fastembed`-backed embedding model to power hybrid retrieval over a small (four regulatory documents + the fictional product catalog, 482 chunks) Brazilian-Portuguese legal corpus, served from a 2 vCPU ARM VPS already running several other containers (per `docs/infra-assessment.md`).

The original design assumed `intfloat/multilingual-e5-small` and `granite-embedding-278m-multilingual` were available. Verifying this while implementing the change (`TextEmbedding.list_supported_models()` against `fastembed==0.8.1`) found **neither is in `fastembed`'s supported-model catalog**. The only supported multilingual candidates in a comparable size class are:

| Model | Dim | Prefixing |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | None (symmetric) |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | None (symmetric) |
| `intfloat/multilingual-e5-large` | 1024 | Required `"query: "`/`"passage: "` (applied inside `FastEmbedAdapter`) |

## Decision

**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** stays the default (`RagSettings.rag_embedding_model`).

## Benchmark

All three candidates ran in the same `docker run --memory=3g --cpus=2` container (x86, Docker Desktop) against a local Postgres, indexing the real corpus one document at a time with the production adapter's limits (2 ONNX threads; passage batch size 16 for MiniLM/mpnet, 8 for e5-large), then running the committed eval set (25 answerable + 6 unanswerable questions) with the current OR-based full-text query. `fastembed==0.8.1`, `onnxruntime==1.23.2` (matching `uv.lock`).

| | MiniLM-L12 (384d) | mpnet-base-v2 (768d) | e5-large (1024d) |
|---|---|---|---|
| Peak RAM during ingestion | 819 MB | 2,219 MB | 2,133 MB |
| Resident RAM at query time | 819 MB | 2,172 MB | 2,108 MB |
| Model load (incl. first download) | 38 s | 59 s | 82 s |
| Ingest, 482 chunks | 37 s | 262 s | 679 s |
| Query embedding p50 / p95 | 11 / 14 ms | 30 / 35 ms | 119 / 167 ms |
| recall@k overall | 68% | 52% | 64% |
| recall@k lexical / colloquial | 74% / 50% | 58% / 33% | 68% / 50% |
| MRR overall | 0.441 | 0.421 | 0.491 |
| MRR lexical / colloquial | 0.484 / 0.306 | 0.449 / 0.333 | 0.489 / 0.500 |
| Best similarity, answerable (min–max, mean) | 0.47–0.85, 0.674 | 0.52–0.81, 0.675 | 0.84–0.90, 0.865 |
| Best similarity, unanswerable (min–max, mean) | 0.43–0.63, 0.541 | 0.38–0.67, 0.560 | 0.81–0.86, 0.837 |
| Gap between the two means | 0.133 | 0.115 | 0.028 |

### Reading the results

- **The colloquial-recall gain came from the full-text fix, not from a model change.** With the *same* MiniLM model, replacing `plainto_tsquery` (AND of all terms) with an OR over parsed lexemes moved colloquial recall@k from 33.3% to 50.0% and overall recall@k from 64% to 68%, with lexical recall unchanged (73.7%). Swapping to a larger model did not add to it: e5-large ties MiniLM on colloquial recall (50%) and is lower on lexical and overall recall. (MRR fell from 0.523 to ~0.44 with the OR change — RRF fuses by rank position, so more full-text matches dilute a formerly unique top hit; see `design.md`, "Full-text search: OR over parsed lexemes".)
- **e5-large fits the budget but is rejected.** It peaked at 2.1 GB (not OOM-killed at 3 GB) and finished in 12.9 minutes, so "does not fit" is not the reason. It is rejected on: query latency (~10x MiniLM, 119 ms p50), resident RAM (~2.6x MiniLM, permanently resident next to the other containers on a shared host), and a **compressed similarity range** — answerable and unanswerable best-similarities sit in 0.84–0.90 and 0.81–0.86, a 0.028 gap between means, which leaves `RAG_MIN_RELEVANCE_SCORE` almost no room to be useful. Its one real advantage, colloquial MRR (0.31 → 0.50), is not enough to offset that.
- **mpnet-base-v2 loses on every axis**: lower recall (52%), lower MRR than e5-large, ~2.7x the RAM and ~7x the ingest time of MiniLM.
- **MiniLM's similarity ranges overlap** (answerable 0.47–0.85, unanswerable 0.43–0.63). No `RAG_MIN_RELEVANCE_SCORE` cleanly separates them, so the threshold (0.6) is only a **coarse first filter** that drops clearly irrelevant retrievals cheaply. Actual refusal must rely on the LLM grounding stage — the `knowledge_agent` returning an empty `claims` list when the retrieved chunks do not answer the question (task 14.3), and citation validation dropping ungrounded claims.

### Caveats

- **Small sample.** 25 answerable questions: one question is 4 percentage points of recall, so recall differences of one to two questions (e.g. MiniLM 68% vs e5-large 64%) are within noise. The similarity-range, latency, and RAM differences are not.
- **x86 proxy for an ARM VPS.** Measurements are from an x86 container on Docker Desktop, not the target 2 vCPU ARM host. Parameter count, and therefore resident RAM, does not depend on architecture, so the RAM numbers transfer well; latency and ingest time may shift with `onnxruntime`'s ARM kernels and should be re-measured on the VPS in `add-vps-deploy`. All three candidates' RAM fits within the VPS's ~9 GiB available.

## Lesson learned: the memory incident

A first benchmark attempt ran `intfloat/multilingual-e5-large` unconstrained in a single host process — `threads=None` (`onnxruntime` auto-detects every core) and `fastembed`'s default `batch_size=256` embedding a whole document's chunks in one call — and exhausted ~9 GB of the development machine's RAM, freezing it. A second attempt in an under-provisioned Docker Desktop VM (4 GB total, container capped at 3 GB) OOM-pressured the VM and took the Postgres container down with it. The fixes that shipped:

- `FastEmbedAdapter` now **always** passes `threads` (`RagSettings.rag_embedding_threads`, default 2, the VPS's vCPU count) and calls `.embed(..., batch_size=..., parallel=None)` with `rag_embedding_batch_size` (default 16) for passage embedding; `embed_query` uses batch size 1. Every caller — `rag.ingest index`, the API, the eval CLI, the smoke script — goes through this one adapter, so production ingestion cannot repeat the incident and there is no separate "benchmark-safe" path to keep in sync.
- Future benchmarks run inside a `--memory`/`--cpus`-capped container with a Docker VM sized well above the cap, with `--memory-swap` equal to the cap, and one candidate per container so an OOM is recorded as a result rather than taking down the run.

## Alternatives considered

- **`intfloat/multilingual-e5-large`** — see above: fits, rejected on latency, resident RAM, and compressed similarity range; quality does not beat MiniLM on this corpus.
- **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`** — measured; loses on all axes.
- **`intfloat/multilingual-e5-small`** (the original plan) — not available in `fastembed`'s catalog.

## Consequences

- No migration change (`VECTOR(384)` stays) and no re-ingestion or threshold recalibration is needed; `RAG_MIN_RELEVANCE_SCORE` stays 0.6 as a coarse first filter.
- If retrieval quality needs to improve further, the levers with evidence behind them are the full-text/fusion side and reranking (`RerankerPort`, disabled by default), not a bigger embedding model.
- Re-measure latency and RAM on the real ARM VPS as part of `add-vps-deploy`.

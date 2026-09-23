# ADR-003: Embedding model for the regulatory knowledge base

## Status

Accepted

## Context

`add-regulatory-rag` needs a `fastembed`-backed embedding model to power hybrid retrieval over a small (four regulatory documents + the fictional product catalog, low thousands of chunks) Brazilian-Portuguese legal corpus, served from a 2 vCPU ARM VPS already running several other containers (per `docs/infra-assessment.md`).

The original design (`design.md` — "Embedding model trade-offs") assumed `intfloat/multilingual-e5-small` and `granite-embedding-278m-multilingual` were available. Verifying this while implementing the change (`TextEmbedding.list_supported_models()` against the installed `fastembed==0.8.1`) found **neither is actually in `fastembed`'s supported-model catalog** — that assumption was never checked against the library itself at proposal time. The only `fastembed`-supported multilingual candidates in a comparable size class are:

| Model | Dim | Prefixing |
|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | None (symmetric model) |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | None (symmetric model) |
| `intfloat/multilingual-e5-large` | 1024 | Required: `"query: "`/`"passage: "` (asymmetric, e5-family training) |

## Decision

**`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** is the default (`RagSettings.rag_embedding_model`).

Measured locally (macOS, Intel x86_64 — this development machine, not the target ARM VPS; see Consequences) via `RAG_RUN_REAL_EMBEDDING_TESTS=1` against the real model:

- **Model load time**: ~1.8 s (first call only; the model stays resident for the process lifetime).
- **Process RSS footprint**: ~792 MB after loading (baseline Python process ~69 MB before loading → ~723 MB attributable to the model itself).
- **Query embedding latency**: ~8 ms average per query (3-query sample).
- **Passage/document batch embedding**: ~0.25 s for 50 documents (~5 ms/document) — the relevant number for `rag.ingest index`, which embeds the whole corpus's chunks (low thousands) in one run, not per-request.
- **Vector dimension**: 384, confirming `VECTOR(384)` in the `rag_chunks` migration needs no change.

This matches the design's original rationale for picking the smallest credible multilingual option: the corpus is small and domain-narrow (legal Portuguese, not open-domain), where a smaller model's main risk — missing subtle semantic distinctions — matters less than on an open-domain benchmark, and query-time latency (the number that actually affects a live conversation turn) is small in absolute terms (~8 ms, negligible next to an LLM round trip).

The real baseline eval (task 6.4, recorded in `tasks.md`) gives the actual quality signal to weigh against this footprint: recall@k 64.00% overall, but a marked lexical/colloquial split (73.68% vs. 33.33%) — colloquial questions, the more realistic signal, are meaningfully harder for this model.

## Alternatives considered

- **`intfloat/multilingual-e5-large`** (1024 dim). Best quality of the three candidates and the only `fastembed`-supported e5-family model — but rejected on footprint: a 1024-dim model is substantially heavier to load and hold resident than the 384-dim default, on a host already sharing 2 vCPUs and 11 GiB total RAM across several other personal-project containers. Kept as the documented upgrade path if the smaller models' recall@k proves insufficient after further tuning (`RAG_MIN_RELEVANCE_SCORE` calibration, `k_vector`/`k_fts`/`rrf_k` tuning) doesn't close the gap — a single-config-value swap (model name + `VECTOR(1024)` in the migration), not a retrieval-architecture change.
- **`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`** (768 dim). A middle-ground option between the default and `multilingual-e5-large`; not measured in this ADR since the default's numbers didn't force an immediate upgrade decision, but it remains the first thing to try if `multilingual-e5-large`'s footprint turns out to be unacceptable.
- **`intfloat/multilingual-e5-small`** (the original plan). Not available in `fastembed`'s catalog — see Context.

## Consequences

- No migration change needed now (`VECTOR(384)` already matches).
- The measurement above is from an Intel x86_64 Mac, not the target ARM VPS (per this change's constraints, no VPS access was used to measure this — see `proposal.md`'s Impact section: no VPS change of any kind). `onnxruntime`'s ARM-specific kernels may shift the exact latency number, but the model's parameter count (and therefore its RSS footprint, the dominant risk factor for the shared VPS) doesn't depend on CPU architecture, so this number is a reasonable proxy pending an actual on-VPS measurement in a future deploy-focused change (`add-vps-deploy`).
- The colloquial-recall gap found in the baseline eval is the number to watch after this ships; if it doesn't improve with threshold/fusion tuning, revisit this ADR's alternatives rather than treating the current default as final.

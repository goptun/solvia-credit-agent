## Why

Today `responder` answers product questions from a small static config block and has no way to answer regulatory or consumer-protection questions at all — those fall through to `out_of_scope`. A credit assistant that cannot ground its answers in the actual rules governing credit, Open Finance, and consumer protection (and that would otherwise answer from the LLM's own, unverifiable "knowledge") is not a credible demonstration of a regulated-fintech assistant. This change adds a retrieval-grounded knowledge capability — a small, curated corpus of official Brazilian regulatory texts plus the existing fictional product catalog — so the assistant answers with citations to a specific norm and article, and explicitly declines when retrieval does not support an answer.

## What Changes

- Curate and commit a small corpus manifest (`rag/corpus/manifest.yaml`) of official regulatory sources plus the fictional product catalog, with a reproducible fetch/verify CLI (`python -m rag.ingest fetch`); raw documents are gitignored, not committed.
- Structure-aware chunking of legal texts (by Capítulo/Seção/Art./§/inciso/alínea) with per-chunk provenance metadata, idempotent upsert-based indexing keyed by a deterministic chunk id and document hash.
- Hybrid retrieval in Postgres: pgvector similarity + full-text search (`portuguese` config), fused with Reciprocal Rank Fusion, behind a provider-agnostic embeddings port (fastembed adapter); an optional, disabled-by-default reranking port.
- A committed ~30-question Portuguese retrieval eval set and a CLI reporting recall@k/MRR.
- A new `knowledge_agent` graph node (smart tier) that answers only from retrieved chunks, cites norm + article per claim, validates every citation against the retrieved set, and refuses when retrieval doesn't support an answer.
- Routing: a new `regulatory_question` intent, classified alongside the existing five; both `regulatory_question` and the existing `product_question` intent route to `knowledge_agent` instead of `responder` (product-question answers stop being answered from the static config block and are instead retrieved from the product-catalog documents in the same corpus). Neither intent passes through the consent gate.
- `compliance_guard` adds an "informational content, not legal advice" disclaimer to regulatory answers, alongside its existing PII masking, approval-promise blocking, and simulation/analysis disclaimer.
- A standalone HTTP endpoint exposing the retrieval + grounded-answer chain directly (outside the conversational graph), for inspection/demo purposes independent of a full conversation turn. LangServe's fit for this is evaluated against the repo's actual `langchain-core`/FastAPI/Pydantic versions before committing to it; a plain FastAPI endpoint is an acceptable, equally-conforming outcome if LangServe doesn't fit, recorded in an ADR either way.
- Docker Compose moves to a pgvector-enabled Postgres image; a migration step creates the chunks table, HNSW vector index, and full-text index.
- `scripts/smoke_gateway.py` gains a regulatory question, a product question, and an unanswerable question, reporting citations found and whether each exists in the retrieved set.

## Capabilities

### New Capabilities
- `regulatory-knowledge-base`: corpus curation and provenance (manifest, official-sources-only, product-catalog-as-document), structure-aware ingestion/chunking, idempotent indexing, hybrid pgvector+full-text retrieval with RRF fusion, optional reranking port, and a retrieval-quality eval (recall@k, MRR).
- `regulatory-knowledge-agent`: the `knowledge_agent` node's grounded-answer behavior (citation construction and validation against the retrieved set, refusal when retrieval doesn't support an answer), and the standalone HTTP endpoint that exposes the same retrieval + answer chain outside the conversation graph.

### Modified Capabilities
- `conversation-graph`: adds the `regulatory_question` intent to intent classification and routing; routes both `product_question` and `regulatory_question` to `knowledge_agent` instead of `responder`, skipping the consent gate for both; narrows `responder`'s per-intent reply drafting to no longer cover `product_question`; extends compliance guardrails with the regulatory "informational content, not legal advice" disclaimer.

## Impact

- **New code**: `rag/` package (corpus manifest, ingest/fetch CLI, chunking, embeddings port + fastembed adapter, hybrid retrieval + RRF, optional reranker port, retrieval eval CLI), `apps/agent/nodes/knowledge_agent.py`, a new standalone FastAPI (or LangServe, pending the ADR) router.
- **Modified code**: `apps/agent/nodes/router.py` (new intent), `apps/agent/graph.py` (new node + routing matrix + `NodeName`/`GRAPH_NODE_ORDER`), `apps/agent/llm/factory.py` (`NODE_TIER_MAP["knowledge_agent"] = "smart"`), `apps/agent/nodes/responder.py` (drop `product_question` handling), `apps/agent/nodes/compliance_guard.py` (regulatory disclaimer), `docker-compose.yml` (pgvector image), `scripts/smoke_gateway.py`.
- **Dependencies**: `fastembed`, `pgvector` (Python client + Postgres extension), `pdfplumber` (BCB PDF extraction), `lxml`/`beautifulsoup4` (Planalto HTML extraction); LangServe only if the ADR concludes it fits.
- **Infrastructure**: no VPS change — this reuses the same local/CI Postgres, adding the `vector`/`unaccent` extensions and a migration step; the corpus and embedding model are never fetched or downloaded in CI (small fixtures only).
- **Out of scope** (unchanged from the foundation proposal, restated for this change): human handoff, VPS deploy, full LLM evals, prompt versioning — plus, specific to this change, no modification to `conversation-api`'s existing conversational endpoint contract (its "node-progress-only streaming" requirement already generically covers any new node, `knowledge_agent` included, without a text change) and no VPS change of any kind.

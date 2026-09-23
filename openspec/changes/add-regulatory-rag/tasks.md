## 0. Branch

- [x] 0.1 From an up-to-date `main` (`git fetch origin && git checkout main && git pull`), create branch `feat/add-regulatory-rag` — verify with `git branch --show-current`
- [x] 0.2 Commit the untracked `.claude/` OpenSpec commands/skills directory on this branch, excluding `.claude/settings.local.json` — verify `git status` shows `.claude/` tracked (minus `settings.local.json`) and `git log` shows this as a commit on the branch

## 1. Dependencies and configuration

- [x] 1.1 Add `fastembed` and a pgvector Python client (e.g. `pgvector`) to `pyproject.toml` dependencies; run `uv sync` — verify `uv run python -c "import fastembed, pgvector"` succeeds
- [x] 1.2 Add a `rag/` settings module (`pydantic-settings`, same pattern as `apps/agent/llm/settings.py`) covering: embedding model name, `chunk_max_chars`, hybrid retrieval `k_vector`/`k_fts`/`rrf_k`/`top_k`, `RAG_MIN_RELEVANCE_SCORE`, `RAG_RERANKING_ENABLED`, and the official-domains allowlist for corpus provenance — verify a unit test loads defaults and overrides from env vars
- [x] 1.3 Commit: `chore(rag): adiciona dependências e configurações do módulo de conhecimento regulatório`

## 2. Corpus manifest and fetch/verify CLI

- [x] 2.1 Confirm the current, in-force, consolidated source URL and version/consolidation date for each of these **four** regulatory documents: CDC consolidated text (Lei 8.078/1990, including the arts. 54-A–G inserted by Lei 14.181/2021), LGPD (Lei 13.709/2018), Resolução CMN nº 4.881/2020 (CET), and BCB/CMN's currently-published consolidated Resolução Conjunta nº 1/2020 (Open Finance) — verify each URL resolves (HTTP 200) from an official domain (bcb.gov.br, planalto.gov.br, gov.br, or the official Open Finance Brasil portal)
- [x] 2.2 Write `rag/corpus/manifest.yaml` with the four regulatory documents from 2.1 plus the `product_catalog` entry sourced from `apps/agent/config/product_descriptions.py`, per the schema in `design.md` — "Corpus pipeline and provenance" — verify the manifest parses against a Pydantic schema in a unit test
- [x] 2.3 Implement `rag/ingest/fetch.py` (`python -m rag.ingest fetch`): downloads each manifested URL to the gitignored `.data/rag_corpus/`, computes sha256, and fails with a clear error on a hash mismatch against a previously recorded hash — verify a unit test with a fake HTTP client covers first-fetch (records hash), unchanged re-fetch (no-op), and a mismatch (raises)
- [x] 2.4 Add `.data/rag_corpus/` to `.gitignore` — verify `git status` shows no new tracked files after running the fetch command locally
- [x] 2.5 Commit: `feat(rag): adiciona manifest do corpus e CLI de fetch/verificação`

## 3. Source format extraction and structure-aware chunking

- [x] 3.1 Add `lxml`/`beautifulsoup4` and `pdfplumber` to `pyproject.toml` dependencies — verify `uv run python -c "import bs4, pdfplumber"` succeeds
- [x] 3.2 Implement `rag/ingest/html_source.py`: decode using the source page's declared encoding (falling back to `windows-1252`), drop the text of any `<strike>`/`<del>` element or any element styled with `text-decoration: line-through` (never bare `<s>`, which the real CDC page uses cosmetically on ordinal indicators) before extracting body text, and extract amendment/revocation notes into a separate `amendment_note` field instead of leaving them in body text, per `design.md` — "Source format handling" — verify unit tests against real snippets copied from the fetched LGPD page (a genuinely revoked §1º of art. 55-A, CSS-struck) and the fetched CDC page (an in-force §1º amended by Lei 13.486/2017, plus the cosmetic `<s>` case) assert: the revoked provision's text never appears in the adapter's output, the amendment note appears only in `amendment_note` metadata, and the cosmetic `<s>º</s>` is preserved as "º" in the output, not deleted
- [x] 3.3 Implement `rag/ingest/pdf_source.py` using `pdfplumber` (see `design.md` for the pypdf-vs-pdfplumber justification): position-aware repeated header/footer removal and hyphenated line-break rejoining — verify a unit test against a real 3-page slice of the fetched Open Finance PDF asserts the repeated header/footer text is absent from every page's extracted text and a revocation note is captured as metadata; the hyphenation rejoin is verified directly against a representative case, since neither real fetched PDF contains an actual hyphenated line break
- [x] 3.4 Commit: `feat(rag): adiciona extração de HTML e PDF com remoção de texto revogado e artefatos de formatação`
- [x] 3.5 Implement `rag/ingest/chunking.py`: structure-aware splitting (Capítulo/Seção/Art./§/inciso/alínea) of the normalized text produced by the source-format adapters, keeping an in-budget article whole and splitting an oversized article by paragraph with a repeated article header, per `design.md` — "Chunking strategy" — verify unit tests using real snippets (art. 1° and the oversized art. 54-B, extracted fresh from the fetched CDC page for these tests specifically) cover: a whole in-budget article, an oversized article split with repeated headers, and a product-catalog entry (using the real `apps.agent.config.product_descriptions` content directly)
- [x] 3.6 Attach chunk metadata (`document_id`, `norm`, `article_ref`, `hierarchy_path`, `source_url`, `version_date`, `source_type`, `amendment_note`) to every chunk — verify a unit test asserts all fields are populated for both a regulatory chunk and a product-catalog chunk, and that `amendment_note` (when present) is never a substring of `content`
- [x] 3.7 Commit: `feat(rag): implementa chunking sensível à estrutura de textos legais`

## 4. Index schema and migration

- [x] 4.1 Write the `rag_chunks` table + `vector`/`unaccent` extensions + a custom `portuguese_unaccent` text search configuration + HNSW + full-text index migration SQL, per `design.md` — "Index schema" — verify `python -m rag.migrate` runs against a local pgvector-enabled Postgres, `\d rag_chunks` shows the expected columns and indexes, and a manual `SELECT to_tsvector('portuguese_unaccent', 'código') @@ to_tsquery('portuguese_unaccent', 'codigo')` returns true (note: `CREATE TEXT SEARCH CONFIGURATION` has no `IF NOT EXISTS` in PostgreSQL — verified against the real instance — so its existence is checked in Python before creating it)
- [x] 4.2 Change `docker-compose.yml`'s `postgres` service to the `pgvector/pgvector:pg16` image — verify `docker compose up` boots and `CREATE EXTENSION vector;` succeeds inside the running container
- [x] 4.3 Implement idempotent upsert-by-`chunk_id` indexing with per-document-hash reindex skipping, per `design.md` — verify an integration test (against a pgvector service container) ingests a document twice with no changes (no new rows) and once with an edited chunk size budget for one document only (only that document's chunks change, a sibling document is untouched)
- [x] 4.4 Commit: `feat(rag): adiciona schema de índice pgvector e indexação idempotente`

## 5. Embeddings port and hybrid retrieval

- [x] 5.1 Define `EmbeddingsPort` (`rag/embeddings/port.py`) with `embed_query`/`embed_documents` methods and a `FakeEmbeddings` adapter (deterministic, hash-based vectors, no network/model download) — verify a unit test asserts identical input text always yields an identical vector via either method
- [x] 5.2 Implement `FastEmbedAdapter` using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (see `design.md` — "Embedding model trade-offs" for why this replaces the originally-planned `intfloat/multilingual-e5-small`, which turned out not to be in `fastembed`'s supported-model catalog), applying the `"query: "`/`"passage: "` prefixes only when the configured model name is from the e5 family (the default symmetric model is not trained to expect them) — verify a unit test asserts the prefix-family detection for several model names with no download needed, and a separate test gated by `RAG_RUN_REAL_EMBEDDING_TESTS=1` (skipped in CI, run locally) asserts the text actually sent to the underlying model is prefixed correctly for each method and a short Portuguese sentence embeds to a 384-dim vector
- [x] 5.3 Implement `rag/retrieval/hybrid.py`: vector similarity query + full-text query (against the `portuguese_unaccent`-backed column) + Reciprocal Rank Fusion, an optional `source_type` filter applied to both queries, and returning each result's raw vector cosine similarity and `matched_fts` flag alongside the fused rank, with `k_vector`/`k_fts`/`rrf_k`/`top_k` read from settings — verify an integration test (pgvector service container, `FakeEmbeddings`) asserts: the fused ranking differs from either individual ranking on a case constructed to need both signals; a `source_type` filter excludes the other type entirely; and the returned raw similarity is independent of fused rank (note: two `psycopg` parameter-typing quirks found and fixed with explicit `::vector`/`::text` casts — see `design.md`)
- [x] 5.4 Define `RerankerPort` and a `NoopReranker` default; wire `RAG_RERANKING_ENABLED` to select it (via `get_reranker`, which always returns `NoopReranker` today — no real adapter exists, per `design.md` — "Non-Goals") — verify a unit test asserts `hybrid.py` calls the reranker unconditionally and `NoopReranker` is a pass-through
- [x] 5.5 Commit: `feat(rag): implementa embeddings, busca híbrida e fusão RRF`

## 6. Retrieval quality evaluation

- [ ] 6.1 Write a committed Portuguese evaluation set (`rag/eval/questions.yaml`): ~24 paraphrased (not copied from article text) answerable questions covering all four regulatory documents and the product catalog, each with an expected document id and article reference, plus 5–8 deliberately unanswerable questions expecting a refusal — verify the file parses against a Pydantic schema in a unit test that also asserts no answerable question's text is a substring of its cited article's stored content
- [ ] 6.2 Present a random 10-question sample from the set (including at least 2 unanswerable questions) to me for review — pause here; do not proceed to 6.4 until I confirm the sample looks correct
- [ ] 6.3 Implement `rag/eval/run.py` reporting recall@k and MRR over the answerable questions plus refusal accuracy over the unanswerable ones — verify it runs against a locally-ingested corpus and prints all three metrics
- [ ] 6.4 Run `rag/eval/run.py` against the ingested corpus and record the baseline recall@k/MRR/refusal-accuracy numbers to paste into the PR description (task 13.1)
- [ ] 6.5 Commit: `test(rag): adiciona conjunto de avaliação e CLI de recall@k/MRR/acurácia de recusa`

## 7. Knowledge agent node

- [ ] 7.1 Add `"regulatory_question"` to `Intent` in `apps/agent/state.py` — verify `mypy --strict` passes and existing `Intent`-typed code still type-checks
- [ ] 7.2 Update `apps/agent/nodes/router.py`'s `_ROUTER_INSTRUCTIONS` to classify the new intent — verify a unit test (fake LLM) asserts a regulatory-sounding message classifies as `regulatory_question`
- [ ] 7.3 Implement `apps/agent/nodes/knowledge_agent.py`: `source_type`-filtered retrieval per intent + cosine-similarity-based refusal check (using the retrieved top chunk's raw vector similarity, never the RRF fused score) + `ainvoke_structured` claim/citation extraction + citation-membership validation + template-rendered citation strings, per `design.md` — "Grounding and citation validation" — verify unit tests (fake LLM, `FakeEmbeddings`) cover: a grounded answer with valid citations, an invented citation being dropped, all-citations-invalid falling back to the deterministic refusal, a low-similarity refusal, and a case where the fused rank alone would have suggested confidence but the raw similarity does not (asserting the refusal decision follows similarity, not rank)
- [ ] 7.4 Add `"knowledge_agent": "smart"` to `NODE_TIER_MAP` in `apps/agent/llm/factory.py` — verify a unit test asserts `LLMFactory.for_node("knowledge_agent")` resolves the smart tier
- [ ] 7.5 Commit: `feat(agent): adiciona nó knowledge_agent com respostas fundamentadas e citações`

## 8. Routing and compliance integration

- [ ] 8.1 Add `"knowledge_agent"` to `apps/agent/graph.py`'s `NodeName`/`GRAPH_NODE_ORDER`, wire the node, and route both `"product_question"` and `"regulatory_question"` to it from `_route_after_router` (with a direct edge to `compliance_guard`) — verify an integration test (fake LLM) exercises both intents end-to-end and asserts the node path includes `knowledge_agent` and excludes `responder`
- [ ] 8.2 Remove `_product_question_reply` and its `PRODUCT_DESCRIPTIONS`/`GENERAL_PRODUCT_OVERVIEW` imports from `apps/agent/nodes/responder.py` — verify `uv run pytest tests/agent/nodes/test_responder.py` passes with the branch removed and no remaining reference to the old imports (`grep -rn PRODUCT_DESCRIPTIONS apps/agent/nodes/responder.py` returns nothing)
- [ ] 8.3 Add the regulatory "informational content, not legal advice" disclaimer path to `apps/agent/nodes/compliance_guard.py` for the `regulatory_question` intent — verify a unit test asserts the disclaimer is present on a regulatory reply and absent on other intents' replies
- [ ] 8.4 Commit: `feat(agent): direciona product_question e regulatory_question para o knowledge_agent`

## 9. Standalone endpoint and ADRs

- [ ] 9.1 Write `docs/adr/ADR-003-embedding-model.md` recording the trade-off table from `design.md`, the chosen model, and the measured memory/latency footprint from running `FastEmbedAdapter` locally (or in a container shaped like the target VPS) — verify the ADR includes actual measured numbers, not estimates
- [ ] 9.2 Verify LangServe's compatibility with this repo's pinned `langchain-core`/`fastapi`/`pydantic` versions (`uv pip show langserve` after a trial add, or a dependency-resolution dry run) and write `docs/adr/ADR-004-langserve-vs-fastapi.md` recording the decision (plain FastAPI, per `design.md` — "Standalone endpoint and the LangServe decision", unless the trial resolution changes the picture) — verify the ADR states the concrete version-compatibility check performed
- [ ] 9.3 Implement the standalone `POST /knowledge/answer` endpoint (`apps/api/routes_knowledge.py`) calling the same grounding function `knowledge_agent` uses — verify an `httpx.ASGITransport` test posts a question and asserts a JSON response with `answer`, `citations`, and `refused` fields
- [ ] 9.4 Commit: `feat(api): adiciona endpoint standalone de resposta fundamentada e registra ADRs`

## 10. CI and infrastructure tests

- [ ] 10.1 Add a pgvector service container to the CI workflow (alongside the existing Postgres checkpointer service, or replacing it with the pgvector image if one service can serve both) — verify the CI run shows the service healthy before the test step
- [ ] 10.2 Add small committed fixtures (a handful of chunks, not the full corpus) for unit/integration tests so CI never fetches the corpus or downloads an embedding model — verify `grep` over the CI workflow shows no invocation of `rag.ingest fetch` or a real `FastEmbedAdapter` model download
- [ ] 10.3 Run the full test suite, lint, type-check, and `gitleaks` locally — verify all pass
- [ ] 10.4 Commit: `test(ci): adiciona serviço pgvector e fixtures para os testes de recuperação`

## 11. Documentation

- [ ] 11.1 Update `README.md`'s project layout and architecture diagram to include the `rag/` package and the `knowledge_agent` node/routing — verify the Mermaid diagram renders and the new routing edges match `apps/agent/graph.py`
- [ ] 11.2 Document `python -m rag.ingest fetch`, `python -m rag.migrate`, and `python -m rag.eval.run` in the README, alongside the existing local-dev instructions — verify each documented command actually runs as written
- [ ] 11.3 Commit: `docs(readme): documenta o pipeline de conhecimento regulatório`

## 12. Smoke test extension

- [ ] 12.1 Extend `scripts/smoke_gateway.py` with a regulatory question, a product question, and a deliberately unanswerable question, reporting citations found per reply and whether each cited chunk id exists in that turn's retrieved set — verify `ruff check`/`mypy` pass on the script
- [ ] 12.2 Run `PYTHONPATH=. uv run python scripts/smoke_gateway.py` against the real gateway (via the SSH tunnel) — verify all turns report `ok`, the regulatory question's reply includes at least one citation that exists in its retrieved set, and the unanswerable question's reply is the refusal reply
- [ ] 12.3 Commit: `chore(scripts): estende o smoke test com cenários de conhecimento regulatório`

## 13. Pull Request

- [ ] 13.1 Push the branch and open a PR using the `.github/pull_request_template.md` template, including the recall@k/MRR/refusal-accuracy baseline from task 6.4 and the smoke test output from task 12.2 in the PR description — verify CI (including the pgvector service container job) passes on the PR
- [ ] 13.2 Present the PR link to me for review before merge

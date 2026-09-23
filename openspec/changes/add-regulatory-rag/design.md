## Context

See `proposal.md` — "Why" for motivation. Relevant current state:

- `apps/agent/graph.py` wires six nodes (`router`, `consent_check`, `financial_analyst`, `offer_simulator`, `responder`, `compliance_guard`) via `add_conditional_edges`, with `NodeName`/`GRAPH_NODE_ORDER` enumerating them for the API layer's generic `node_started`/`node_finished` streaming (`apps/api/routes.py` iterates the graph's own `debug`-mode stream; it never special-cases a node by name).
- `apps/agent/nodes/router.py` classifies one of five intents (`Intent` in `apps/agent/state.py`) via structured output; `product_question` currently routes straight to `responder`, which answers from a static config block (`apps/agent/config/product_descriptions.py`) — never retrieval-grounded.
- `apps/agent/llm/factory.py`'s `NODE_TIER_MAP` is a plain `{node_name: tier}` table; adding a node is one entry.
- `apps/agent/llm/port.py`'s `LLMPort` Protocol and `apps/agent/llm/structured.py`'s `ainvoke_structured` (native tool-calling first, JSON-mode + fence-stripping fallback, one repair retry) are the established patterns for any new LLM-calling node.
- Postgres is already a dependency (`langgraph-checkpoint-postgres`, `psycopg[binary]`) but only for the checkpointer, which creates its own tables via `AsyncPostgresSaver.setup()`. There is no existing migration mechanism for application-owned tables.
- `docker-compose.yml` runs plain `postgres:16-alpine`; no `vector` extension.
- No embeddings port or RAG-adjacent code exists yet — this change introduces that layer from scratch.
- The two source formats in the corpus are not uniform: Planalto's "compilado" (consolidated) pages are HTML, served historically in `windows-1252`/`ISO-8859-1` rather than UTF-8, and mark revoked provisions with `<strike>`/`<s>` (or occasionally `<del>`) elements plus trailing parenthetical amendment notes (`"(Redação dada por Lei nº ...)"`, `"(Revogado pela Lei nº ...)"`) that are still visible text on the page, not actually removed. BCB's resolutions are published as PDFs with repeated per-page headers/footers and line-break hyphenation from the original page layout.
- LangServe (`langserve` on PyPI, latest release `0.3.3`, January 2024) states in its own docs: *"We recommend using LangGraph Platform rather than LangServe for new projects... we will not be accepting new feature contributions."* Its declared dependency floors (`langchain-core<2,>=0.3`, `fastapi<1,>=0.90.1`, `pydantic<3.0,>=2.7`) are technically satisfied by this repo's pinned versions (`langchain-core==1.6.4`, `fastapi==0.141.1`, `pydantic==2.13.5`), but the project is frozen, not merely old.

## Goals / Non-Goals

**Goals:**
- Ground every regulatory/product answer in retrieved, citable text — never in the LLM's own claims.
- Keep the corpus small, reproducible without committing large binary files, and auditable (every chunk traces to an official URL and a fetch/version date).
- Reuse this repo's established patterns (ports and adapters, `ainvoke_structured`, `NODE_TIER_MAP`, structured settings) rather than introducing a second style of dependency injection or configuration.
- Keep CI hermetic: no corpus download, no embedding-model download, ever.

**Non-Goals:**
- Full legal-research-grade retrieval (e.g., citation chaining, cross-references between norms) — out of scope; a single retrieval pass per question is enough for this demo.
- General-purpose document upload or a multi-tenant knowledge base — the corpus is a fixed, curated set the maintainer controls.
- Production-grade reranking — the reranking port exists so quality can be revisited later, but this change does not have to prove it earns its cost; the ADR just has to decide whether to *enable* it, not build a novel reranker.
- Any VPS or deployment change (see proposal.md — Impact).

## Decisions

### Corpus pipeline and provenance

`rag/corpus/manifest.yaml` is the single source of truth for what belongs in the corpus:

Verified by actually fetching each URL while preparing this update (see task 2.1's checked-off verification below — the manifest is not written from assumed URLs):

```yaml
documents:
  - id: cdc-consolidada
    title: "Código de Defesa do Consumidor (Lei nº 8.078/1990), texto consolidado"
    norm: "Lei nº 8.078/1990"
    source_type: regulation
    url: "https://www.planalto.gov.br/ccivil_03/leis/l8078compilado.htm"
    retrieved_at: "2026-09-23"
    version_date: "2021-07-01"   # last consolidating amendment reflected: Lei nº 14.181/2021 (over-indebtedness), which inserted arts. 54-A–G
    sha256: "7220c4f6e957381a332edb3fc8795df6bca3ef8c46db7656cbd56a9112c65219"
  - id: lgpd
    title: "Lei Geral de Proteção de Dados Pessoais (Lei nº 13.709/2018)"
    norm: "Lei nº 13.709/2018"
    source_type: regulation
    url: "https://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm"
    retrieved_at: "2026-09-23"
    version_date: "2026-06-30"   # most recent consolidating amendment: Lei nº 15.452/2026, which redrafted art. 55-A (ANPD's institutional linkage)
    sha256: "032aea7bf183d79b8284d29bf2768d56e55f518a3061dd9c4eb7d854d612e71c"
  - id: open-finance-regulamento
    title: "Resolução Conjunta BCB/CMN nº 1/2020 (Open Finance), texto consolidado"
    norm: "Resolução Conjunta BCB/CMN nº 1, de 4 de maio de 2020"
    source_type: regulation
    url: "https://normativos.bcb.gov.br/Lists/Normativos/Attachments/51028/Res_Conj_0001_v7_L.pdf"
    retrieved_at: "2026-09-23"
    version_date: "2024-07-04"   # most recent consolidating amendment reflected: Resolução Conjunta nº 10, de 4/7/2024
    sha256: "8fcd7179b9ebe8243854d80cc3a8bca075cd926fef35d674df11306d5274dbe7"
  - id: cet-disclosure
    title: "Resolução CMN nº 4.881/2020 (Custo Efetivo Total)"
    norm: "Resolução CMN nº 4.881, de 23 de dezembro de 2020"
    source_type: regulation
    url: "https://www.bcb.gov.br/content/estabilidadefinanceira/especialnor/Resolu%C3%A7%C3%A3o4881.pdf"
    retrieved_at: "2026-09-23"
    version_date: "2020-12-23"   # unamended since publication — its only cross-references are to the older resolutions it itself revoked (3.517/2007, 3.909/2010, 4.197/2013, 4.699/2018)
    sha256: "2b8abe706b2fc482e0bf91a002a55ba6de44466e5541e40375c73fc3f3785aa6"
  - id: product-catalog
    title: "Catálogo de produtos Solvia (descrições fictícias)"
    norm: null
    source_type: product_catalog
    url: null            # not fetched externally; generated from apps/agent/config/product_descriptions.py
    retrieved_at: "2026-09-23"
    version_date: null
    sha256: "<computed from the generated text at ingestion time>"
```

Two notes on the URLs above, since a naive guess at either would have been wrong: the CET-disclosure PDF's filename contains a literal, percent-encoded `ç`/`ã` (`Resolu%C3%A7%C3%A3o4881.pdf`) — an unencoded ASCII guess 404s. The Open Finance PDF's path (`.../Attachments/51028/Res_Conj_0001_v7_L.pdf`) is BCB's own currently-published *consolidated* rendering (`_v7_L`, "L" for the amended/"com alterações" text) of Resolução Conjunta nº 1/2020, not the original 2020 publication — BCB republishes this same consolidated PDF in place as it's amended, so `python -m rag.ingest fetch`'s hash check (not the URL) is what actually detects the next amendment, not a URL change.

Three research findings from preparing this proposal, recorded here so the apply phase doesn't have to re-derive them:
- **Over-indebtedness provisions live in the CDC itself.** Lei 14.181/2021 is an *amending* law — it inserts articles 54-A through 54-G into the CDC (Lei 8.078/1990) rather than standing as its own standalone regulatory text. The corpus therefore ingests the **consolidated CDC text** (which already contains those articles) as `cdc-consolidada`, not Lei 14.181/2021 as a separate document — citations for e.g. article 54-B read `"Lei nº 8.078/1990 (CDC), art. 54-B"`, sourced from the CDC document's own metadata. (Lei 14.181/2021's promulgation date remains useful context for why those articles exist, but it is not itself a retrievable/citable document.)
- **The CET-disclosure candidate needed replacing.** The originally-listed "CMN resolution on CET disclosure" is the now-superseded Resolução CMN 3.517/2007; the current consolidated rule is **Resolução CMN nº 4.881/2020** (in force since 2021-02-01). The manifest uses 4.881/2020.
- **Open Finance's base norm has been amended repeatedly through 2024–2026** (e.g., Resolução Conjunta nº 10/2024, and 2025/2026 updates for credit-portability). The manifest ingests BCB's own **currently-published consolidated version** of Resolução Conjunta nº 1/2020 (BCB publishes a running consolidated PDF at a stable URL, republished in place as it's amended — see the manifest below), recording the consolidation date that PDF's own text prints as `version_date`; verified while preparing this update, that consolidated PDF's own cross-references currently top out at Resolução Conjunta nº 10, de 4/7/2024.

`rag/ingest/fetch.py` (invoked as `python -m rag.ingest fetch`) reads the manifest, downloads each `url` (skipping `product_catalog` entries, which are generated, not fetched), computes its sha256, and fails loudly on a mismatch against a previously-recorded hash (a real regulatory change) versus a first-time fetch (expected, records the hash). Raw downloads go to `.data/rag_corpus/` (gitignored); the product catalog's generated text is written next to them for a uniform ingestion path.

### Source format handling: HTML (Planalto) and PDF (BCB)

Both source-format adapters run *before* chunking and produce the same thing: plain, normalized, in-force-only text plus a list of per-article `amendment_note` strings (kept as metadata, never concatenated into retrievable content). Chunking (below) only ever sees this normalized output, never raw HTML or PDF bytes.

- **`rag/ingest/html_source.py`** (Planalto "compilado" pages): decodes using the encoding the response actually declares when it declares one, falling back to `windows-1252` — verified against the real fetched pages while preparing this update: `l8078compilado.htm` (CDC) declares no charset in either its `<meta>` tags or its HTTP `Content-Type` header, and its raw bytes contain `0xF3` for "ó" (`windows-1252`/`ISO-8859-1`; that byte is not valid standalone UTF-8), so decoding as UTF-8 by default would silently corrupt every accented character rather than failing loudly. It parses the decoded HTML with `lxml`/`BeautifulSoup` and **removes, entirely, any element that is a `<strike>`/`<del>` tag or carries an inline `style` with `text-decoration: ... line-through`** — this is what actually marks a revoked provision in the real fetched pages (verified against `l13709.htm` — a revoked §1º of LGPD's art. 55-A is wrapped in `<span style="...text-decoration:line-through">`, not a semantic tag), and is what keeps revoked provisions out of every downstream chunk, since chunking never sees text that was never extracted. **Bare `<s>` is deliberately excluded from this removal set** — verified against the real CDC page, Planalto's HTML uses `<s>` cosmetically (e.g. `§ 1<s>º</s>` to style the ordinal-indicator superscript), not to mark revocation; treating it as a revocation signal would silently delete the "º" from every in-force paragraph number.
- **`rag/ingest/pdf_source.py`** (BCB resolutions): extracts text via **`pdfplumber`**, chosen over `pypdf` because it returns each line's bounding box, letting header/footer normalization use *position* (top/bottom margin of the page) rather than pure text-frequency matching alone — more reliable when a repeated header/footer varies slightly page to page (e.g. embeds a page number). Both libraries are actively maintained, pure-Python, permissively licensed; `pypdf`'s plain `extract_text()` would force a frequency-only heuristic (drop a line if it repeats across ≥50% of pages), which is a weaker fallback this adapter does not need. Hyphenation: a line ending in `-` immediately followed by a continuation line is rejoined without the hyphen or an inserted space (`"exceci-"` + `"onalmente"` → `"excecionalmente"`); repeated header/footer lines (identified by page position) are dropped before the remaining lines are joined into the article's running text.
- **Amendment-note extraction is shared, not per-format**: verifying the actual fetched BCB PDF (`Resolução Conjunta nº 1/2020`, consolidated through `Resolução Conjunta nº 10/2024`) while preparing this update showed the *same* `"(Redação dada ... pela Resolução ...)"` / `"(Revogado pela Resolução ...)"` annotation style Planalto uses in HTML — BCB's own PDF consolidation already replaces a revoked item's substantive text with just the revocation note (there is no separate struck-through original text to remove in that case, unlike Planalto's HTML). So the regex over `\((Reda[çc][ãa]o dada|Inclu[íi]do|Revogado)[^)]*\)` that pulls a trailing note into `amendment_note` metadata (stripped from the body text passed to chunking) lives in one shared helper both `html_source.py` and `pdf_source.py` call on their respective extracted text, rather than being HTML-specific.

Unit tests for both adapters run against real (short) snippets saved from the actual fetched corpus — not synthetic HTML/PDF — and assert, for the HTML adapter, that struck-through sample text never appears in the adapter's output while its accompanying amendment note appears only in the returned metadata, not the body text; and, for the PDF adapter, that a revoked item's note-only text is captured as `amendment_note` rather than indexed as if it were a real, in-force provision.

### Chunking strategy

`rag/ingest/chunking.py` takes the normalized text and per-article amendment-note metadata produced by the source-format adapters (above) and splits it into a hierarchy: Capítulo → Seção → Artigo → parágrafo/inciso/alínea, using regex patterns anchored on the standard Brazilian legislative drafting conventions (`Art\. \d+`, `§ ?\d+`, `Parágrafo único`, roman-numeral incisos, lettered alíneas). An article is emitted as one chunk if its rendered text (header + body) fits the configured `chunk_max_chars` budget; otherwise it splits at paragraph/inciso boundaries, and every resulting chunk's text is prefixed with the article's header (`"Art. 54-B."`) so a chunk is independently meaningful without needing its siblings. The product catalog (already short, no legal structure, no HTML/PDF source) is chunked as one chunk per product description.

Every chunk carries: `document_id`, `norm`, `article_ref` (e.g. `"art. 54-B"`, `null` for the product catalog), `hierarchy_path` (e.g. `"Capítulo IV > Seção II > Art. 54-B"`), `source_url`, `version_date`, `source_type`, and `amendment_note` (nullable — carried through from the source-format adapter, never part of `content`). Unit tests exercise the chunker against the same real, short snippets used by the source-format adapter tests — so a regex change that breaks on a real formatting quirk is caught end to end, from raw source to final chunk.

### Index schema

One new table, `rag_chunks`, created by a small idempotent SQL migration run via a CLI (`python -m rag.migrate`), mirroring the checkpointer's own `setup()`-on-startup pattern rather than introducing a migration framework the project doesn't otherwise need:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE TABLE IF NOT EXISTS rag_chunks (
    chunk_id      TEXT PRIMARY KEY,   -- deterministic: sha256(document_id || hierarchy_path || chunk_index)
    document_id   TEXT NOT NULL,
    document_hash TEXT NOT NULL,      -- the manifest sha256 this chunk was produced from
    source_type   TEXT NOT NULL,      -- 'regulation' | 'product_catalog'
    norm          TEXT,
    article_ref   TEXT,
    hierarchy_path TEXT NOT NULL,
    source_url    TEXT,
    version_date  DATE,
    amendment_note TEXT,              -- metadata only, e.g. "(Redação dada pela Lei nº ...)" — never in `content`
    content       TEXT NOT NULL,
    embedding     VECTOR(384) NOT NULL,   -- dimension fixed by the chosen embedding model, see below
    content_tsv   TSVECTOR GENERATED ALWAYS AS (to_tsvector('portuguese_unaccent', content)) STORED
);

CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw
    ON rag_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS rag_chunks_content_tsv_gin
    ON rag_chunks USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx ON rag_chunks (document_id);
CREATE INDEX IF NOT EXISTS rag_chunks_source_type_idx ON rag_chunks (source_type);
```

`content_tsv`'s `to_tsvector('portuguese_unaccent', content)` depends on a dedicated text search configuration existing first — so an unaccented query (e.g. a customer typing "cet" or "abusao" without diacritics) still matches accented indexed content, which plain `'portuguese'` does not fold on its own. Unlike every other statement above, `CREATE TEXT SEARCH CONFIGURATION` has **no `IF NOT EXISTS` clause in PostgreSQL** (verified against a real pgvector/pg16 instance while implementing this — it's a hard syntax error, not just a warning), so the migration checks `pg_catalog.pg_ts_config` in Python first and only issues the `CREATE`/`ALTER` pair when the config doesn't exist yet, before running the `CREATE TABLE` above:

```sql
-- Only when this returns no rows:
CREATE TEXT SEARCH CONFIGURATION portuguese_unaccent (COPY = portuguese);
ALTER TEXT SEARCH CONFIGURATION portuguese_unaccent
    ALTER MAPPING FOR hword, hword_part, word WITH unaccent, portuguese_stem;
```

Idempotent indexing (spec: "Idempotent, incremental indexing") upserts by `chunk_id` (`INSERT ... ON CONFLICT (chunk_id) DO UPDATE`), and re-chunks/re-embeds a document only when its manifest `sha256` differs from the `document_hash` already stored for its existing chunks — ingestion first checks `SELECT DISTINCT document_hash FROM rag_chunks WHERE document_id = :id`.

### Hybrid retrieval and fusion

`rag/retrieval/hybrid.py` runs two queries per question against `rag_chunks` — a `pgvector` cosine-distance ORDER BY LIMIT `k_vector`, and a `ts_rank_cd`-ordered full-text query against `content_tsv` (using the `portuguese_unaccent` configuration from the migration, so an unaccented query still matches accented content) LIMIT `k_fts` — then fuses the two rank-ordered lists with Reciprocal Rank Fusion: `score(chunk) = Σ 1 / (rrf_k + rank_in_list)` over whichever lists contain it. Both queries accept an optional `source_type` filter (`WHERE source_type = :filter`, applied to both the vector and full-text query when set); `knowledge_agent` (below) always passes one — `product_catalog` for a `product_question`, `regulation` for a `regulatory_question` — so a product question can never surface regulatory chunks or vice versa. `k_vector`, `k_fts`, `rrf_k`, and the final `top_k` returned are all `Settings` fields (same `pydantic-settings` pattern as `apps/agent/llm/settings.py`), not hardcoded. Two `psycopg` parameter-typing quirks found while implementing this, both fixed with an explicit `::vector`/`::text` cast in the SQL text rather than by registering a connection-wide adapter: a bare Python list parameter against `embedding <=> %s` resolves to `double precision[]`, which has no `<=>` operator against `vector` (`UndefinedFunction`) — an INSERT works without the cast because the destination column's type drives an implicit assignment cast, but a bare comparison expression has no such destination to infer from; and a `None` parameter used in both `%(x)s IS NULL` and `= %(x)s` in the same query leaves Postgres unable to resolve the parameter's type at all (`AmbiguousParameter`) without an explicit cast on both occurrences.

Alongside the fused, RRF-ordered list, `hybrid.py` returns each result's **raw vector cosine similarity** (`1 - cosine_distance`, from the vector query alone) and whether it also appeared in the full-text query's results (`matched_fts: bool`) — RRF's fused score is a rank-based fusion weight, not a relevance measure, so it is never used for the refusal decision (see "Grounding and citation validation" below, and Risks).

An `EmbeddingsPort` Protocol (`rag/embeddings/port.py`, mirroring `LLMPort`'s shape) is the only interface retrieval code depends on, with **two** methods — `embed_query(text) -> Vector` and `embed_documents(texts) -> list[Vector]` — rather than one generic `embed`, because the chosen `multilingual-e5` model family requires different literal prefixes for queries (`"query: "`) versus indexed passages (`"passage: "`) to retrieve well; `FastEmbedAdapter` applies the correct prefix internally in each method, so no caller (ingestion, retrieval, or tests) ever handles prefixing itself, and swapping to a model that doesn't need prefixes would just make both methods no-ops on that front. `FakeEmbeddings` (the only adapter automated tests use) implements both methods identically (deterministic, hash-based vectors) since prefixing doesn't matter for a fake.

Optional reranking is a `RerankerPort` Protocol with a `NoopReranker` (default) and a real cross-encoder adapter behind it, selected by configuration (`RAG_RERANKING_ENABLED`); `hybrid.py` calls `reranker.rerank(query, fused_results)` unconditionally, so enabling reranking never changes a call site — only which adapter `NoopReranker`/real is wired in.

### Embedding model trade-offs

**Correction, made while implementing this**: the original trade-off table below named `intfloat/multilingual-e5-small` and `granite-embedding-278m-multilingual` — neither is actually in `fastembed`'s supported-model catalog (verified via `TextEmbedding.list_supported_models()` against the installed `fastembed==0.8.1`; that check was not done at proposal time, which was the actual gap). The table and choice below reflect what `fastembed` can actually load.

Candidates `fastembed` actually supports, evaluated given the target VPS is a 2 vCPU ARM host already running several other containers (per `docs/infra-assessment.md`):

| Model | Dim | Prefixing | Note |
|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | None (symmetric model) | Smallest of the credible multilingual options; well-established sentence-transformers model with broad multilingual (incl. pt) training |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | None (symmetric model) | Quality/size middle ground; candidate if the small model's recall@k is insufficient |
| `intfloat/multilingual-e5-large` | 1024 | Required: `"query: "`/`"passage: "` (asymmetric, e5-family training) | Best quality of the three but heaviest — rejected on footprint alone for a shared 2 vCPU ARM host; the only `fastembed`-supported e5 model at all (`multilingual-e5-small`/`-base` are not in its catalog) |

Default choice: **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** — smallest footprint, well-established multilingual training, and the corpus here is small and domain-narrow (legal Portuguese, not open-domain), where a small model's main risk (missing subtle semantic distinctions) matters less than on an open-domain benchmark. Because this is a *symmetric* sentence-transformers model (not asymmetric like e5), it is not trained to expect a query/passage prefix — adding one would add noise, not help — so `FastEmbedAdapter` only applies the `"query: "`/`"passage: "` prefixes when the configured model name indicates the e5 family (`"e5" in model_name.lower()`), keeping the door open to `multilingual-e5-large` as a documented, single-config-value upgrade if recall@k is insufficient, without the adapter mishandling whichever model is actually configured. The retrieval-quality eval (recall@k/MRR over the committed question set) is the actual gate: the apply-phase ADR records the measured memory footprint (`docker stats` on a comparable container) and the eval numbers, and upgrades to `paraphrase-multilingual-mpnet-base-v2` (updating `VECTOR(384)` to `VECTOR(768)` in the migration) or, if that's still short, `multilingual-e5-large` (`VECTOR(1024)`), if recall@k falls short. This is a single-parameter swap (model name + vector dimension), not a retrieval-architecture change, so deferring the final pick to measured numbers is safe.

### Retrieval quality evaluation

`rag/eval/questions.yaml` has two parts, revised after the pre-baseline sample review (see below) turned up gaps in the first draft:

- **25 answerable questions** in Portuguese, spanning all four regulatory documents and the product catalog: 9 on `cdc-consolidada` (including 4 on the over-indebtedness articles 54-A–G and 2 specifically on art. 52's credit-disclosure/late-fee-cap/early-repayment paragraphs — credit is this project's actual domain, so the set is weighted toward it rather than spread evenly), 6 on `lgpd`, 4 on `open-finance-regulamento`, 4 on `cet-disclosure`, 2 on `product-catalog`. Each carries:
  - `expected_refs`: one or more article references (a list, since one question can legitimately be answered by more than one article — e.g. "is a clause that blocks indemnification valid?" is answered by both CDC art. 51 and art. 25); `null` only for a `product_catalog` question.
  - `evidence`: a literal quote (≤200 chars) from the expected article, verified in `tests/rag/eval/test_questions.py` against the real fetched corpus text — this catches a wrong `expected_refs` before it silently corrupts the baseline, not just a wrong bounding.
  - `style`: `lexical` (default) or `colloquial`. At least 6 questions are `colloquial` — phrased the way a customer would actually ask, deliberately avoiding the article heading's own words — and 3 of those are written without diacritics (e.g. "credito", "emprestimo"), since real customer input isn't guaranteed to have them.
- **6 deliberately unanswerable questions**, each tagged `distance: far` (obviously outside the corpus's domain, e.g. tax law) or `distance: near_miss` (plausible-sounding, same subject area as the corpus, but not actually covered — e.g. asking for an overdraft interest-rate *cap* when the CET-disclosure document only describes how CET is *calculated* for overdraft, never a cap; asking about Open Finance credit-portability deadlines or a specific technical manual version, neither in the ingested consolidated resolution).

`rag/eval/run.py` reports recall@k and MRR over the answerable subset — **broken down by `style`** (lexical vs. colloquial), since a model can pass on lexical overlap alone and still fail the colloquial subset, which is the more realistic signal — and refusal accuracy over the unanswerable subset, **broken down by `distance`** (far vs. near-miss), since near-miss refusal accuracy is the harder, more meaningful number (a system that only refuses obviously-out-of-domain questions isn't actually gating on relevance). The same run doubles as the practical calibration exercise for `RAG_MIN_RELEVANCE_SCORE`: the threshold is picked as the value that maximizes near-miss refusal accuracy without materially hurting colloquial recall@k.

Before the baseline run is recorded (for the ADR and the PR description), a 10-question sample of the eval set (including at least two unanswerable questions) is presented for review — catching a bad or ambiguous question before it skews the recorded baseline is cheaper than re-baselining after the fact. That review is what produced the `evidence`/`expected_refs`/`style`/`distance` fields and the credit-weighted rebalancing above; the sample itself is not reproduced here since it was a subset of the committed file, not separate content.

### Grounding and citation validation

`apps/agent/nodes/knowledge_agent.py` (smart tier, added to `NODE_TIER_MAP`) is the only place that turns retrieved chunks into a customer-facing answer. It depends on retrieval through an injected `RetrieveFn` (`Callable[[str, Vector, SourceType], Awaitable[list[RetrievedChunk]]]`) rather than a raw database connection, so the node's own logic — intent-to-`source_type` mapping, the similarity refusal check, citation validation and rendering — is unit-testable with a fake retrieval function and no Postgres. The real implementation, `rag/retrieval/live.py`'s `hybrid_search_async`, wraps the synchronous `hybrid_search` in `asyncio.to_thread`, checking out its own connection from a `psycopg_pool.ConnectionPool` per call (a bare single `psycopg.Connection` shared across concurrent requests would not be safe to use from multiple threads at once) — wired into the node at app-startup time (task 8.1).

1. Retrieve (hybrid + optional rerank, `source_type` filtered per intent) top-k chunks for the customer's question.
2. If the best-matching chunk's **raw vector cosine similarity** (not the RRF fused score, which is a rank-based fusion weight and not a relevance measure) is below `RAG_MIN_RELEVANCE_SCORE` (configurable, a cosine-similarity threshold calibrated against the eval set — see "Retrieval quality evaluation" below), skip straight to the refusal reply — no LLM call needed for this decision, keeping it deterministic and free of an extra retryable failure mode. A full-text match on the same top chunk (`matched_fts`) is an optional secondary signal the threshold calibration can use to relax the cutoff slightly (a borderline cosine score backed by an exact keyword hit is more trustworthy than the same score alone), but cosine similarity is always the primary gate.
3. Otherwise, call the smart-tier LLM via `ainvoke_structured` with a schema `KnowledgeAnswer(claims: list[Claim])` where `Claim = {text: str, chunk_id: str}` — the LLM must attach a `chunk_id` from the retrieved set to every claim, rather than free-form citation text it could invent. This is the same native-tool-calling-first/JSON-mode-fallback path every other structured call in this repo already uses.
4. Validate: every `chunk_id` in the response must be one of the retrieved chunks' ids. Any claim citing an id outside that set is dropped; if dropping claims leaves nothing left, the node falls back to the deterministic refusal reply instead of returning a gutted answer.
5. Render the final reply by joining each surviving claim's text with a citation string built from *that chunk's own metadata* (`f"{norm}, {article_ref}"`) — the citation text is template-rendered, never the LLM's own citation string, exactly like `responder`'s numeric rendering (see `specs/conversation-graph/spec.md` — "Per-intent reply drafting").

This mirrors the CDC's own historical `responder` pattern (LLM drafts framing/content, deterministic code renders the numbers/facts it's allowed to render) — the LLM never writes citation strings itself, only says which chunk backs a given sentence.

### Routing changes

- `apps/agent/state.py`: `Intent` gains `"regulatory_question"`.
- `apps/agent/nodes/router.py`: `_ROUTER_INSTRUCTIONS` lists the sixth category with a short description ("regulatory_question: pergunta sobre regulação de crédito, Open Finance, proteção ao consumidor").
- `apps/agent/graph.py`: a new `"knowledge_agent"` node (added to `NodeName`/`GRAPH_NODE_ORDER`); `_route_after_router` sends both `"product_question"` and `"regulatory_question"` to `"knowledge_agent"` (previously `"product_question"` went to `"responder"`); `knowledge_agent` has a plain edge straight to `"compliance_guard"` (it does not pass through `responder` — its output is already the final drafted reply).
- `apps/agent/nodes/responder.py`: `_product_question_reply` and its `PRODUCT_DESCRIPTIONS`/`GENERAL_PRODUCT_OVERVIEW` imports are removed; `responder_node`'s `intent == "product_question"` branch is deleted (the `knowledge_agent` route means `responder` is never reached with that intent, but removing the dead branch keeps the node's contract matching its narrowed spec requirement).
- `apps/agent/nodes/compliance_guard.py`: a `_INFORMATIONAL_DISCLAIMER_INTENTS = {"regulatory_question"}` set (or equivalent) triggers the new disclaimer template, independent of the existing `_DISCLAIMER_INTENTS` simulation/analysis disclaimer.
- `apps/agent/config/product_descriptions.py`'s content becomes the `product_catalog` corpus document's source text (read by the ingestion CLI), rather than being imported directly by a node — `responder` no longer imports it at all after the above change.

No change to `apps/api/routes.py` or the conversation-api spec: `node_started`/`node_finished` are emitted generically from the graph's own `debug` stream (see Context), so `knowledge_agent` participates in that contract automatically, with no new branch in the route handler.

### Standalone endpoint and the LangServe decision

**Decision: plain FastAPI, not LangServe.** LangServe's own documentation (as of its latest, January-2024 release) tells adopters to prefer LangGraph Platform for new projects and states it is only accepting bug fixes, not new features. Its dependency floors are technically satisfied by this repo's current pins, but adopting a frozen library for a *new* piece of surface area — when the repo already has a working, from-scratch SSE/JSON pattern in `apps/api/` — trades a small amount of boilerplate for a dependency with no path forward. This is recorded as `docs/adr/ADR-004-langserve-vs-fastapi.md` (created during apply, per `proposal.md` — this is an "acceptable outcome, not a failure").

The standalone endpoint (`POST /knowledge/answer`, request `{"question": str}`, response `{"answer": str, "citations": [{"norm": str, "article_ref": str | null, "source_url": str | null}], "refused": bool}`) is a thin FastAPI route that calls the same retrieval + `knowledge_agent`-equivalent grounding function the graph node calls — extracted as a plain function both call, so there is exactly one grounding/citation implementation, not two. It returns a single JSON response (no SSE) since spec (`regulatory-knowledge-agent` — "Standalone retrieval-and-answer HTTP endpoint") only requires a grounded answer with citations, not a streamed one, and there is no multi-node graph traversal to report progress on for a single retrieval call.

### Module boundaries

New `rag/` top-level package (sibling to `apps/`, mirroring the project's existing ports-and-adapters style): `rag/corpus/` (manifest schema + loader), `rag/ingest/` (`html_source.py`/`pdf_source.py` format adapters, `chunking.py`, `fetch.py`, indexing CLI), `rag/embeddings/` (port + fastembed/fake adapters), `rag/retrieval/` (hybrid search, RRF, reranker port), `rag/eval/` (question set + recall@k/MRR/refusal-accuracy CLI). `apps/agent/nodes/knowledge_agent.py` and `apps/api/routes_knowledge.py` (the standalone endpoint) are the only places that import from both `apps/` and `rag/` — everything in `rag/` itself has no dependency on `apps/`, so it stays reusable/testable independent of the conversation graph, matching the existing rule that domain logic (here, retrieval) has no LangChain/FastAPI imports beyond what `EmbeddingsPort`/adapters need.

## Risks / Trade-offs

- **A small embedding model under-performs on nuanced legal Portuguese** → Mitigation: the committed retrieval eval set is the objective gate before shipping; the ADR records measured recall@k, and the design already names the upgrade path (`granite-embedding-278m-multilingual`) if the small model's numbers are unacceptable.
- **Regulatory text changes after the corpus is fetched (a norm gets amended)** → Mitigation: the manifest's `sha256` + `fetch` command surfaces drift explicitly (a hash mismatch on re-fetch) rather than silently serving stale text forever; re-running fetch + ingest picks up the new consolidated version.
- **LLM invents a citation despite instructions** → Mitigation: citation validation (Decision: "Grounding and citation validation", step 4) is deterministic Python, not another LLM call trusting the first one — an invented `chunk_id` simply fails the membership check.
- **HNSW index build cost on a small VPS as the corpus grows** → Mitigation: the corpus is deliberately small (four regulatory documents + the product catalog, low thousands of chunks at most) — HNSW build/query cost at this scale is negligible; revisit only if the corpus materially grows in a future change.
- **Reranking is speculative, disabled-by-default scope creep** → Mitigation: the port exists so the door isn't closed, but this change does not have to build or tune a real reranker to be complete — the ADR can conclude "not enabled, revisit later" and that is a satisfying answer to the requirement.

## Migration Plan

- New table (`rag_chunks`) and the `vector` extension are created by `python -m rag.migrate`, run once in CI (before tests) and documented in the README next to the existing `docker compose up` instructions; `docker-compose.yml`'s `postgres` service image changes to `pgvector/pgvector:pg16` (a strict superset of `postgres:16-alpine` with the extension preinstalled — no data migration needed for existing checkpointer tables).
- No existing table changes and no backward-incompatible API change — `product_question` answers change in *content* (retrieval-grounded instead of static-config-grounded) but not in the request/response contract, so this ships without a version bump to the conversational API.
- Rollback: dropping the `rag_chunks` table and reverting the compose image, `NODE_TIER_MAP` entry, and routing change is fully sufficient to return to pre-change behavior; no other table or data is touched.

## Open Questions

- Exact `chunk_max_chars`, `RAG_MIN_RELEVANCE_SCORE`, and RRF `k`/`top_k` defaults are tuning values best set once real chunks and the eval set exist (task-level, during apply) — they don't change the retrieval architecture, the spec, or the task breakdown above. (The four document URLs and their current consolidation dates, previously an open question here, are now confirmed and recorded in the manifest above.)

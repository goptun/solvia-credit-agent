# Solvia

Solvia is a greenfield portfolio project: an agentic credit assistant
for contact centers, powered by synthetic Open Finance Brasil data and
built with [LangGraph](https://github.com/langchain-ai/langgraph). It
demonstrates stateful agent orchestration, observability of
non-deterministic systems, and LLMOps practices in a regulated fintech
context — with no real company, customer, or credit data anywhere in
the codebase.

This repository holds the **foundation + MVP** (see
`openspec/changes/archive/2026-09-23-add-solvia-foundation-mvp/`): the
monorepo scaffold, an LLM gateway abstraction, deterministic financial
tools, a synthetic customer dataset, an MVP conversation graph, a
streaming API, and basic observability — plus **regulatory and product
Q&A** (see `openspec/changes/add-regulatory-rag/`): a small, curated
corpus of official Brazilian regulatory texts and the fictional product
catalog, retrieved with hybrid (vector + full-text) search and answered
only from cited, verified chunks — all runnable locally via
`docker compose up`.

## What the assistant does

1. Classifies intent: product question, regulatory question, loan
   simulation, profile analysis, complaint, or out of scope.
2. Gates on a synthetic Open Finance consent (asks for authorization
   when missing/expired; product/regulatory questions and
   out-of-scope/complaint messages skip this gate).
3. Analyzes a customer's synthetic financial data (income,
   debt-to-income ratio, spending categories) — entirely via
   deterministic tools, never LLM computation.
4. Simulates credit offers (Price and SAC amortization, with CET/total
   effective cost) using deterministic Python tools, sourcing rate/IOF/
   fees from a fictional product catalog.
5. Answers product and regulatory questions **only** from a curated,
   provenance-tracked corpus (Brazilian consumer/credit/data-protection
   law plus the fictional product catalog), retrieved via hybrid
   (vector + full-text) search, citing the specific norm and article
   behind every claim — refusing rather than guessing when retrieval
   doesn't support an answer. See `openspec/changes/add-regulatory-rag/`.
6. Enforces compliance guardrails: PII masking, blocking approval
   promises, and injecting mandatory disclaimers (including an
   "informational, not legal advice" note on regulatory answers) — all
   deterministic, except for an LLM-backed check for approval-promise
   language, itself backed by a deterministic keyword/regex safety net.

Human handoff, production deployment, and full LLM evaluation are out
of scope for this change — see the proposal for the planned follow-up
changes (`add-human-handoff`, `add-vps-deploy`, `add-llm-evals`,
`add-prompt-versioning`).

## Architecture

```mermaid
flowchart TD
    Client -->|POST /conversations/:id/messages| API[FastAPI]
    API -->|SSE: node_started / node_finished / final| Client
    API --> Graph[LangGraph conversation graph]
    Client2[Client] -->|POST /knowledge/answer| API
    Graph --> Router[router]
    Router -->|active_flow set| ConsentCheck
    Router -->|out_of_scope / complaint| Responder
    Router -->|product_question / regulatory_question| KnowledgeAgent[knowledge_agent]
    Router -->|loan_simulation / profile_analysis| ConsentCheck[consent_check]
    ConsentCheck -->|valid| FinancialAnalyst[financial_analyst]
    ConsentCheck -->|valid, loan_simulation| OfferSimulator[offer_simulator]
    ConsentCheck -->|missing/expired/ambiguous| Responder
    FinancialAnalyst --> Responder[responder]
    OfferSimulator --> Responder
    Responder --> ComplianceGuard[compliance_guard]
    KnowledgeAgent --> ComplianceGuard
    ComplianceGuard --> END

    Graph -.checkpoints.-> Postgres[(Postgres)]
    FinancialAnalyst -.tools.-> Tools[apps/agent/tools: income, DTI, spending, CET/IRR, Price/SAC]
    OfferSimulator -.tools.-> Tools
    KnowledgeAgent -.hybrid search.-> RagChunks[(rag_chunks: pgvector + full-text)]
    Router -.LLM fast.-> Gateway[9router gateway]
    ComplianceGuard -.LLM fast.-> Gateway
    FinancialAnalyst -.LLM smart, unused in MVP.-> Gateway
    OfferSimulator -.LLM smart.-> Gateway
    Responder -.LLM smart.-> Gateway
    KnowledgeAgent -.LLM smart.-> Gateway
    Graph -.traces.-> LangFuse[LangFuse Cloud]
```

## Key decisions

- **LangGraph over CrewAI/autonomous agents** — deterministic routing,
  a fixed compliance step, and a Postgres-backed checkpointer for
  resumable conversations. See `docs/adr/ADR-001-langgraph-orchestration.md`.
- **Model aliases, not model names** — the app only ever references
  `solvia-fast`/`solvia-smart`; the `9router` gateway owns the actual
  model fallback chain behind each alias. See
  `docs/adr/ADR-002-llm-gateway-aliases.md`.
- **The LLM never computes numbers** — income, DTI, spending
  categories, and Price/SAC/CET simulations are 100% deterministic
  Python (`apps/agent/tools/`); the LLM only classifies intent,
  extracts slots, and drafts non-numeric reply framing.
- **CET as annualized IRR** — the total effective cost is the
  internal rate of return of the net cash flow (amount released minus
  IOF/fees vs. installments), computed with `Decimal` and documented
  rounding (see `specs/credit-simulation/spec.md`).
- **Deterministic compliance where possible** — PII masking and
  disclaimer injection are regex/template-based; only approval-promise
  detection uses an LLM, backed by a deterministic keyword check.
- **Grounded answers, verified citations, never invented** — the
  `knowledge_agent` node only ever answers from retrieved corpus
  chunks; the LLM attaches a `chunk_id` to every claim, which is then
  validated against the retrieved set (an invented citation is
  dropped, not trusted) and rendered from the chunk's own metadata,
  never from LLM-generated citation text. See
  `openspec/changes/add-regulatory-rag/design.md` — "Grounding and
  citation validation".

## Local development

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python 3.12 is pinned via `.python-version`)
- Docker (for `docker compose up`, or to run just Postgres locally)
- SSH access to the VPS hosting the `9router` gateway (for real LLM
  calls — optional if you only want to run with `LLM_PROVIDER=fake`)

### Local development against the gateway

The `9router` gateway is **never exposed publicly** — it's reachable
only through an SSH tunnel:

```bash
ssh -i /path/to/your/key -L 8080:localhost:<gateway-port> <user>@<vps-host>
```

Keep that tunnel running, then set in your local `.env`:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=<your gateway API key>
```

`GET /health/ready` reports whether the gateway is reachable through
the tunnel.

### Regulatory knowledge base

The regulatory/product knowledge base lives in Postgres (`rag_chunks`,
`vector` + `unaccent` extensions) and is built in three steps:

```bash
uv run python -m rag.migrate               # idempotent: creates rag_chunks + indexes
PYTHONPATH=. uv run python -m rag.ingest fetch    # re-fetches the corpus, verifies hashes
PYTHONPATH=. uv run python -m rag.ingest index    # chunks, embeds, and indexes every document
```

`rag.ingest fetch` needs no gateway/LLM access (it downloads directly
from official sources — `planalto.gov.br`, `bcb.gov.br`); its output
goes to the gitignored `.data/rag_corpus/`, never committed. Re-run
`rag.ingest index` any time the manifest or a fetched document changes
— it only reindexes documents whose hash actually changed (see
`specs/regulatory-knowledge-base/spec.md` — "Idempotent, incremental
indexing").

Retrieval quality and grounding are measured by the evaluation harness
below.

### Evaluation harness

`evals/` is the single measuring instrument for the assistant's components
(`python -m evals --help`). It has versioned, Pydantic-validated datasets
(`evals/datasets/`, with labelling rules in its README), metrics with 95%
confidence intervals, committed baselines with explicit tolerances
(`evals/baselines/`), and two modes:

| | Offline | Live |
|---|---|---|
| Suites | `retrieval`, `compliance` | `router`, `slots`, `compliance-llm`, `grounding` |
| Needs | Postgres + pgvector, embedding model, a committed chunk fixture | + the gateway through the SSH tunnel |
| Runs | on every PR (`Evals (offline)` job) and locally | locally only, budgeted and paced |

```bash
# offline: deterministic compliance suite and retrieval over the committed fixture
DATABASE_URL=... PYTHONPATH=. uv run python -m evals run --suite compliance,retrieval --mode offline --compare-baseline

# live (needs the tunnel and a real gateway; estimates and refuses to start over budget)
PYTHONPATH=. uv run python -m evals run --suite router,slots,compliance-llm,grounding --mode live --sample 0.3 --seed 42

python -m evals report --run run.json --baseline evals/baselines/retrieval.json
python -m evals baseline update --suite compliance --from-run run.json
```

Live runs report their gateway-call count, record the resolved model of every
call and are checked for contamination (upstream quota errors, unexpected
models, fallbacks): a sampled, incomplete or contaminated run is reported but
can never become a baseline. The offline retrieval baseline is recorded only
from the CI job's artifact, never from a laptop. The design, its trade-offs and
the follow-ups it found are in
[`docs/adr/ADR-006-evaluation-harness.md`](docs/adr/ADR-006-evaluation-harness.md);
running the live suite from CI is a recorded, unimplemented decision
([`ADR-007`](docs/adr/ADR-007-live-evals-in-ci-over-tailscale.md)).

Current baselines (regenerated by `python -m evals report --update-readme`):

<!-- evals:metrics:start -->
**compliance-llm** (live, commit `dfe89f5`, compliance v2)

| metric | value | 95% CI | n |
|---|---|---|---|
| llm.precision | 1.0000 | [0.772, 1.000] | 13 |
| llm.recall | 0.9286 | [0.685, 0.987] | 14 |
| llm.false_positive_rate | 0.0000 | [0.000, 0.125] | 27 |

**router** (live, commit `5519fce`, router v2)

| metric | value | 95% CI | n |
|---|---|---|---|
| router.accuracy | 1.0000 | [0.947, 1.000] | 69 |

**compliance** (offline, commit `8060f36`, compliance v2)

| metric | value | 95% CI | n |
|---|---|---|---|
| keyword.precision | 0.5714 | [0.250, 0.842] | 7 |
| keyword.recall | 0.2857 | [0.117, 0.546] | 14 |
| keyword.false_positive_rate | 0.1111 | [0.039, 0.281] | 27 |
| keyword.false_positive_rate_on_approval_mentions | 0.2500 | [0.089, 0.532] | 12 |
| strict.precision | 0.6923 | [0.424, 0.873] | 13 |
| strict.recall | 0.6429 | [0.388, 0.837] | 14 |
| strict.false_positive_rate | 0.1481 | [0.059, 0.325] | 27 |
| strict.false_positive_rate_on_approval_mentions | 0.3333 | [0.138, 0.609] | 12 |
| masking.exact_match | 0.7895 | [0.567, 0.915] | 19 |

**retrieval** (offline, commit `dab1db7`, retrieval v2)

| metric | value | 95% CI | n |
|---|---|---|---|
| recall_at_k | 0.7397 | [0.629, 0.827] | 73 |
| mrr | 0.5219 | [0.429, 0.615] | 73 |
| false_refusal_rate_at_threshold | 0.2192 | [0.140, 0.327] | 73 |
| unanswerable_refusal_rate_at_threshold | 0.7200 | [0.524, 0.857] | 25 |
<!-- evals:metrics:end -->

### Manual smoke test against the real gateway

`scripts/smoke_gateway.py` sends one real message per classifiable
intent (product question, regulatory question is exercised via the
same `knowledge_agent` path, loan simulation, profile analysis,
complaint, out of scope) through the full graph, against the real
`9router` gateway. It is **not** part of the automated test suite and
is never run by CI — it needs the SSH tunnel above, a real
`LLM_API_KEY`, and `DATABASE_URL` pointing at a Postgres with the
corpus already migrated and indexed (see "Regulatory knowledge base"
above):

```bash
PYTHONPATH=. uv run python scripts/smoke_gateway.py
```

It prints, per turn, the intent the router actually classified, the
sequence of nodes executed, `reply_status` (`ok`/`unavailable`/`empty`),
the resolved underlying model when available, and the first 120
characters of the reply — replies are drawn entirely from the
synthetic dataset, the fictional product catalog, and public
regulatory text, so a short local preview is fine; this script's
*output* is never committed, only the script itself.

### Run everything with Docker Compose

```bash
cp .env.example .env   # fill in LLM_API_KEY, DATABASE_URL passwords, etc.
docker compose up
```

This starts `api` (FastAPI on `:8000`) and `postgres`. LangFuse is
Cloud-hosted by default (set `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`
in `.env`); omit them to use an in-memory fake tracer with no external
calls.

### Run without Docker

```bash
uv sync
docker run -d --name solvia-postgres -e POSTGRES_USER=solvia -e POSTGRES_PASSWORD=solvia \
  -e POSTGRES_DB=solvia -p 5432:5432 pgvector/pgvector:pg16
cp .env.example .env
uv run uvicorn apps.api.app:app --reload
```

### Try it

```bash
curl -N -X POST http://localhost:8000/conversations/demo-1/messages \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "cust-0000", "message": "Quero simular um empréstimo de 5000 reais em 12 meses, Price"}'
```

`customer_id` must be a synthetic customer id — see
`data/fixtures/customers.json` for a small committed set (`cust-0000`
through `cust-0008`), or generate the full ~200-customer dataset:

```bash
uv run python -m apps.agent.synthetic_data
```

(writes to `data/generated/`, gitignored).

### Tests, lint, type-check

```bash
uv run pytest                 # unit + integration tests (fake LLM only, never the real gateway)
DATABASE_URL=postgresql://solvia:solvia@localhost:5432/solvia_test uv run pytest  # + the Postgres checkpointer integration test
uv run ruff check . && uv run ruff format --check .
uv run mypy .
pre-commit run --all-files    # ruff, mypy, gitleaks
```

## Project layout

```
apps/
  api/            FastAPI service (routes, SSE streaming, health checks)
    routes_knowledge.py  Standalone POST /knowledge/answer endpoint
  agent/
    graph.py       LangGraph wiring and routing matrix
    state.py       Typed conversation state
    nodes/         router, consent_check, financial_analyst, offer_simulator,
                    responder, compliance_guard, knowledge_agent
    tools/         Deterministic financial calculations (income, DTI, spending, CET/IRR, Price/SAC)
    llm/           Provider-agnostic LLM factory (OpenAI-compatible, Google, fake)
    synthetic_data/ Reproducible Open Finance Brasil–shaped data generator
    observability/  Structured logging + LangFuse tracing
    config/         Product catalog, product descriptions, fixed reference date
    repositories/   Customer data access
  console/        Streamlit agent console (scaffold only in this change)
services/
  crm_mock/       Mock CRM microservice (scaffold only in this change)
rag/
  corpus/         manifest.yaml + provenance model for the curated regulatory/product corpus
  ingest/         Fetch, HTML/PDF extraction, amendment-note handling, chunking, indexing
  embeddings/     EmbeddingsPort + fastembed adapter (multilingual, e5-prefix aware)
  retrieval/      Hybrid (pgvector + full-text) search, RRF fusion, reranker port
  settings.py     RagSettings (embedding model, top-k, fusion weights, refusal threshold)
  migrate.py      Idempotent rag_chunks schema + index migration
evals/            Evaluation harness: datasets, metrics, offline/live suites, baselines, reports
data/
  fixtures/       Small, committed synthetic dataset used by tests
  generated/      Full generated dataset (gitignored)
docs/adr/         Architecture decision records
openspec/         Specs, design, and tasks for this and future changes
```

## Documentation

- `docs/infra-assessment.md` — VPS pre-flight assessment (redacted).
- `docs/adr/` — architecture decision records.
- `openspec/changes/archive/2026-09-23-add-solvia-foundation-mvp/` — proposal, design, specs, and tasks for the foundation + MVP.
- `openspec/changes/add-regulatory-rag/` — proposal, design, specs, and tasks for the regulatory/product knowledge capability.
- `openspec/changes/add-llm-evals/` — proposal, design, specs, and tasks for the evaluation harness.
- `CONTRIBUTING.md` / `CLAUDE.md` — engineering conventions.

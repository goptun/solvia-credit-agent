# Solvia

Solvia is a greenfield portfolio project: an agentic credit assistant
for contact centers, powered by synthetic Open Finance Brasil data and
built with [LangGraph](https://github.com/langchain-ai/langgraph). It
demonstrates stateful agent orchestration, observability of
non-deterministic systems, and LLMOps practices in a regulated fintech
context — with no real company, customer, or credit data anywhere in
the codebase.

This repository holds the **foundation + MVP** (see
`openspec/changes/add-solvia-foundation-mvp/`): the monorepo scaffold,
an LLM gateway abstraction, deterministic financial tools, a synthetic
customer dataset, an MVP conversation graph, a streaming API, and basic
observability — all runnable locally via `docker compose up`.

## What the assistant does

1. Classifies intent: product question, loan simulation, profile
   analysis, complaint, or out of scope.
2. Gates on a synthetic Open Finance consent (asks for authorization
   when missing/expired; product questions and out-of-scope/complaint
   messages skip this gate).
3. Analyzes a customer's synthetic financial data (income,
   debt-to-income ratio, spending categories) — entirely via
   deterministic tools, never LLM computation.
4. Simulates credit offers (Price and SAC amortization, with CET/total
   effective cost) using deterministic Python tools, sourcing rate/IOF/
   fees from a fictional product catalog.
5. Enforces compliance guardrails: PII masking, blocking approval
   promises, and injecting mandatory disclaimers — all deterministic,
   except for an LLM-backed check for approval-promise language, itself
   backed by a deterministic keyword/regex safety net.

Human handoff, regulatory RAG, production deployment, and evaluation
are out of scope for this change — see the proposal for the planned
follow-up changes (`add-regulatory-rag`, `add-human-handoff`,
`add-vps-deploy`, `add-llm-evals`, `add-prompt-versioning`).

## Architecture

```mermaid
flowchart TD
    Client -->|POST /conversations/:id/messages| API[FastAPI]
    API -->|SSE: node_started / node_finished / final| Client
    API --> Graph[LangGraph conversation graph]
    Graph --> Router[router]
    Router -->|active_flow set| ConsentCheck
    Router -->|out_of_scope / complaint / product_question| Responder
    Router -->|loan_simulation / profile_analysis| ConsentCheck[consent_check]
    ConsentCheck -->|valid| FinancialAnalyst[financial_analyst]
    ConsentCheck -->|valid, loan_simulation| OfferSimulator[offer_simulator]
    ConsentCheck -->|missing/expired/ambiguous| Responder
    FinancialAnalyst --> Responder[responder]
    OfferSimulator --> Responder
    Responder --> ComplianceGuard[compliance_guard]
    ComplianceGuard --> END

    Graph -.checkpoints.-> Postgres[(Postgres)]
    FinancialAnalyst -.tools.-> Tools[apps/agent/tools: income, DTI, spending, CET/IRR, Price/SAC]
    OfferSimulator -.tools.-> Tools
    Router -.LLM fast.-> Gateway[9router gateway]
    ComplianceGuard -.LLM fast.-> Gateway
    FinancialAnalyst -.LLM smart, unused in MVP.-> Gateway
    OfferSimulator -.LLM smart.-> Gateway
    Responder -.LLM smart.-> Gateway
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

### Manual smoke test against the real gateway

`scripts/smoke_gateway.py` sends one real message per classifiable
intent (product question, loan simulation, profile analysis, complaint,
out of scope) through the full graph, against the real `9router`
gateway. It is **not** part of the automated test suite and is never
run by CI — it needs the SSH tunnel above and a real `LLM_API_KEY`:

```bash
PYTHONPATH=. uv run python scripts/smoke_gateway.py
```

It prints only the intent the router actually classified, the sequence
of nodes executed, and whether a final reply was produced for each
intent — never the reply content or the message text, to avoid leaking
model output into terminal history.

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
  -e POSTGRES_DB=solvia -p 5432:5432 postgres:16-alpine
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
  agent/
    graph.py       LangGraph wiring and routing matrix
    state.py       Typed conversation state
    nodes/         router, consent_check, financial_analyst, offer_simulator, responder, compliance_guard
    tools/         Deterministic financial calculations (income, DTI, spending, CET/IRR, Price/SAC)
    llm/           Provider-agnostic LLM factory (OpenAI-compatible, Google, fake)
    synthetic_data/ Reproducible Open Finance Brasil–shaped data generator
    observability/  Structured logging + LangFuse tracing
    config/         Product catalog, product descriptions, fixed reference date
    repositories/   Customer data access
  console/        Streamlit agent console (scaffold only in this change)
services/
  crm_mock/       Mock CRM microservice (scaffold only in this change)
data/
  fixtures/       Small, committed synthetic dataset used by tests
  generated/      Full generated dataset (gitignored)
docs/adr/         Architecture decision records
openspec/         Specs, design, and tasks for this and future changes
```

## Documentation

- `docs/infra-assessment.md` — VPS pre-flight assessment (redacted).
- `docs/adr/` — architecture decision records.
- `openspec/changes/add-solvia-foundation-mvp/` — proposal, design, specs, and tasks for this change.
- `CONTRIBUTING.md` / `CLAUDE.md` — engineering conventions.

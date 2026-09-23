## 0. Repository Hygiene and Branch

- [x] 0.0 Create branch `feat/add-solvia-foundation-mvp` off `main` — verify with `git branch --show-current`
- [x] 0.1 Add `.gitignore` (ignoring `.env` and `.env.*` except `.env.example`, `.DS_Store`, `.venv`, `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, coverage files, `data/generated/`, `.claude/settings.local.json`) and commit the existing OpenSpec planning artifacts under `openspec/changes/add-solvia-foundation-mvp` on this branch — verify with `git log` showing this as the branch's first commit and `git status` showing no untracked files matching the ignored patterns

## 1. VPS Pre-Flight Assessment (BLOCKING GATE — requires explicit go-ahead before Task Group 2)

- [x] 1.1 Read-only: collect OS/kernel, CPU cores, total/available RAM, swap, disk usage/free space, Docker and Docker Compose versions, running containers with `docker stats --no-stream`, ports in use (especially 80/443), existing reverse proxy (Caddy/Traefik/Nginx) and its TLS/subdomain handling, firewall rules, and uptime/load average — verify by producing the raw findings (no VPS mutation of any kind)
- [x] 1.2 Verify the `9router` gateway is reachable from where the API container will run, that `/v1/models` lists `solvia-fast` and `solvia-smart`, and that a minimal chat completion (including a simple tool-call / JSON-output check) succeeds on each alias — verify by recording the request/response outcome for each alias
- [x] 1.3 Verify local development access to the gateway through an SSH tunnel (`ssh -L`): establish the tunnel, set `LLM_BASE_URL=http://localhost:<port>/v1` locally, and confirm `/v1/models` responds through it, without the gateway being exposed publicly — verify by recording the tunnel command used and the successful response through it
- [x] 1.4 Estimate the resource footprint of the planned stack (api, postgres+pgvector, streamlit console, crm_mock, fastembed in memory, optional self-hosted LangFuse) and compare against available headroom with a safety margin for existing services — verify the estimate is documented with numbers, not just a verdict
- [x] 1.5 Produce `docs/infra-assessment.md` (IPs/hostnames/usernames redacted) with findings, a go/no-go verdict, risks, and recommended adjustments (e.g., LangFuse Cloud instead of self-host, per-container memory limits, reuse existing reverse proxy) — verify the file exists and gitleaks reports no findings on it
- [x] 1.6 STOP: present the verdict and `docs/infra-assessment.md` to me and wait for explicit go-ahead. If the verdict is no-go or has blocking risks, first adjust `design.md` to match the actual infrastructure (e.g., reuse existing reverse proxy/Docker network) and re-present before any further task begins
- [x] 1.7 Commit: `docs(infra): adiciona avaliação da infraestrutura da VPS` — verify with `git log` showing `docs/infra-assessment.md` committed on the branch

## 2. Repository Scaffold

- [x] 2.1 Create the monorepo layout (`apps/api`, `apps/agent/{graph.py,state.py,nodes,tools,prompts,llm,config}`, `apps/console`, `services/crm_mock`, `data/`, `rag/`, `evals/`, `infra/`, `docs/adr/`) with placeholder `__init__.py`/README stubs — verify the directory tree matches `design.md`
- [x] 2.2 Initialize the `uv` project (Python 3.12) and add `ruff`, `mypy`, `pytest` as dev dependencies — verify `uv run pytest --collect-only` runs without error
- [x] 2.3 Commit: `chore(repo): adiciona estrutura monorepo e ferramentas base`

## 3. Engineering Foundation

- [x] 3.1 Add pre-commit config (`ruff format`, `ruff check`, `mypy`, `gitleaks`) — verify `pre-commit run --all-files` passes on the current scaffold
- [x] 3.2 Add a GitHub Actions CI workflow running lint, type-check, tests (with a Postgres service container for integration tests that need a real database, e.g., the checkpointer resume test), and `gitleaks` as a required check on every PR — verify the workflow file is valid YAML and a first CI run once pushed succeeds
- [x] 3.3 Write `CONTRIBUTING.md` documenting branch naming (`<type>/<change-id>`), Conventional Commits (English type/scope, Portuguese description), PR process, and squash-merge policy
- [x] 3.4 Write `CLAUDE.md` recording all engineering conventions from this change (version control, clean code, testing, tooling) for future sessions
- [x] 3.5 Add `.github/pull_request_template.md` in Portuguese (resumo, change OpenSpec vinculada, como testar, prints/traces quando relevante) — verify the template renders when opening a PR
- [x] 3.6 List as manual steps for me: (a) enable branch protection on `main` requiring the CI checks (lint, type-check, tests, gitleaks) to pass before merge, and (b) enable GitHub secret scanning and push protection on the repository — neither can be automated from here
- [x] 3.7 Commit: `chore(repo): configura pre-commit, CI e convenções de contribuição`

## 4. Settings and LLM Gateway (`llm-gateway`)

- [x] 4.1 Implement the pydantic-settings module reading `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_FAST`, `LLM_MODEL_SMART`, `LLM_PROVIDER`, and Google/Vertex settings — verify a unit test loads settings from environment variables with no hardcoded values
- [x] 4.2 Define the `LLMPort` protocol and the fake LLM adapter — verify a unit test drives a node through the fake adapter only
- [x] 4.3 Implement the OpenAI-compatible adapter (`ChatOpenAI` wrapper using `base_url`/`api_key` from settings) — verify a unit test asserts client construction uses only settings-derived values
- [x] 4.4 Implement the Google adapter (`langchain-google-genai`) supporting both Gemini Developer API and Vertex AI via `vertexai`/`project`/`location` settings — verify unit tests cover both construction modes
- [x] 4.5 Implement `LLMFactory.for_node(node_name)` resolving tier→alias→adapter from configuration — verify unit tests cover the fast/smart tier mapping table for every MVP node (`router`: fast, `financial_analyst`: smart, `offer_simulator`: smart, `responder`: smart, `compliance_guard`: fast)
- [x] 4.6 Implement timeout + exponential-backoff retry wrapper around LLM calls — verify a unit test simulates a transient failure followed by success
- [x] 4.7 Implement smart→fast degradation and the fixed "demo temporarily unavailable" fallback — verify unit tests for: smart succeeds; smart fails then fast succeeds; both fail and the fallback message is returned
- [x] 4.8 Attach the resolved underlying model to the call's trace span when present in response metadata — verify unit tests for both the present and absent cases
- [x] 4.9 Write `docs/adr/ADR-002-llm-gateway-aliases.md` documenting the gateway/alias/degradation strategy
- [x] 4.10 Commit: `feat(llm): adiciona settings e factory de LLM com tiers e degradação`

## 5. Synthetic Data Generator (`synthetic-data`)

- [x] 5.1 Define Open Finance Brasil–shaped Pydantic models for account, transaction, credit card, and consent (including a consent `status` field: valid, missing, or expired) — verify unit tests validate each model's required fields
- [x] 5.2 Implement the fixed-seed generator producing ~200 customers across salaried/self-employed/over-indebted/thin-file profiles, with every generated date derived from a fixed reference date in config (never `datetime.now()`/wall clock) — verify a unit test asserts two runs with the same seed and reference date produce identical output (including dates) and all four profiles are present
- [x] 5.3 Ensure the generated customer population includes at least one customer with each consent status (`valid`, `missing`, `expired`) — verify a unit test asserts all three statuses are present
- [x] 5.4 Write generator output to `data/generated/` (gitignored) and commit a small, explicit set of fixture records under `data/fixtures/` for use by the automated test suite — verify tests only read `data/fixtures/` and `data/generated/` is empty/untracked after a fresh generation run
- [x] 5.5 Verify every generated record validates against its Pydantic schema — add a unit test asserting no validation errors across the generated dataset
- [x] 5.6 Commit: `feat(data): adiciona gerador de dados sintéticos Open Finance`

## 6. Deterministic Financial Tools (`credit-simulation`)

- [x] 6.1 Implement income estimation from transaction history — verify unit tests including a no-recurring-credits edge case
- [x] 6.2 Implement debt-to-income ratio calculation — verify unit tests including a zero-income edge case
- [x] 6.3 Implement the deterministic spending categorization tool from transaction codes/descriptions in the synthetic data, with a defined default/uncategorized category for unmatched transactions — verify unit tests cover known categories and the fallback case
- [x] 6.4 Define the fictional product catalog config (interest rate, IOF fixed rate + capped daily rate, fees — all configurable) as static configuration, never sourced from the LLM — verify a unit test loads and validates the catalog
- [x] 6.5 Implement the CET calculation as the annualized IRR of the net cash flow (amount released minus IOF and fees, versus the installment schedule), using `Decimal` with documented rounding rules — verify unit tests against documented, hand-computed reference cases
- [x] 6.6 Implement Price amortization simulation using catalog-sourced rate/IOF/fees and the CET/IRR calculation — verify unit tests against the documented reference cases
- [x] 6.7 Implement SAC amortization simulation using catalog-sourced rate/IOF/fees and the CET/IRR calculation — verify unit tests against the documented reference cases, including that installments decrease over time
- [x] 6.8 Add input validation (non-positive amount/term, negative rate, unrecognized amortization type) raising a domain exception instead of returning a result — verify unit tests assert the exception is raised for each case
- [x] 6.9 Verify domain/tools code coverage is ≥ 80% via `pytest --cov`
- [x] 6.10 Commit: `feat(tools): adiciona simulação de crédito determinística (CET/IRR), catálogo de produto e categorização de gastos`

## 7. MVP Conversation Graph (`conversation-graph`)

- [x] 7.1 Define the typed conversation state (Pydantic/TypedDict) per `design.md`, including `customer_id`, `intent`, `pending_intent`, `active_flow` (`none`/`consent_confirmation`/`slot_filling`), `consent_status`, `simulation_slots`, `draft_reply`, and `compliance_flags` — verify a unit test constructs and validates the state
- [x] 7.2 Implement customer id binding: require `customer_id` on the first message, look it up against the synthetic customer records, reject unknown ids, persist it in state, and reject any later message that supplies a different `customer_id` for the same conversation — verify unit tests for unknown id, valid id, and immutability across turns
- [x] 7.3 Implement the `router` node (fast tier: product question / loan simulation / profile analysis / complaint / out of scope), including a "new request" signal in its structured output. When `active_flow` is `none`, classify normally with conditional routing (out-of-scope, complaint, and product question short-circuit directly to `responder`; profile analysis never reaches `offer_simulator`). When `active_flow` is not `none`, skip classification and route directly to the flow's owning node (`consent_check` for `consent_confirmation`, `offer_simulator` for `slot_filling`) unless the "new request" signal fires, in which case clear `active_flow`/`pending_intent` and classify normally — verify unit tests with the fake LLM cover each intent branch, the active-flow skip-and-route case for both flow types, and the new-request-clears-the-flow case
- [x] 7.4 Implement `consent_check` with no LLM call: read the bound customer's synthetic consent status; when missing or expired, set `pending_intent` to the triggering intent, set `active_flow = consent_confirmation`, and route to `responder` to request authorization; on the customer's next routed-back message, apply a deterministic, normalized keyword/regex match against known Brazilian Portuguese authorization phrases — a match grants a synthetic consent (`consent_status = just_granted`), clears `active_flow`, and resumes `pending_intent`; a clear refusal clears the flow; anything ambiguous re-asks without changing `consent_status` — verify unit tests for missing, expired, valid, just-granted, explicit-confirmation, clear-refusal, and ambiguous-reply cases
- [x] 7.5 Implement `financial_analyst` (smart tier) delegating to the `credit-simulation` tools for income/DTI/spending categories — verify a unit test with the fake LLM and real tools
- [x] 7.6 Implement `offer_simulator`: extract amount/term/amortization-type slots via structured LLM output (smart tier); when a slot is missing or invalid, set `active_flow = slot_filling` and route back to `responder` for a targeted follow-up question; consume the customer's next routed-back message (per the `router`'s active-flow handling) as the answer to the pending slot; once all slots are valid, clear `active_flow` and delegate to the Price/SAC tools using catalog-sourced terms — verify unit tests for the follow-up-question path, the slot-answer-consumed path, and the complete-slots path, and that the LLM never produces the numeric simulation result directly
- [x] 7.7 Implement `responder` (smart tier): draft the reply from state and tool outputs per intent — data-backed reply (loan simulation/profile analysis) rendered from tool output via templates, product-catalog-only reply (product question), complaint acknowledgment noting human handoff is a future capability, and polite refusal (out of scope) — verify unit tests per intent asserting the drafted content's source
- [x] 7.8 Implement `compliance_guard`: deterministic regex-based PII masking (CPF, account/card numbers, phone, email), deterministic template-based disclaimer injection, and approval-promise detection via the fast-tier LLM backed by a deterministic keyword/regex check (either mechanism blocking is sufficient) — verify unit tests for masking, disclaimer injection, and promise detection (including a case the LLM check misses but the regex check catches)
- [x] 7.9 Wire graph edges with the Postgres checkpointer (`router → consent_check → financial_analyst → offer_simulator → responder → compliance_guard → END`, plus conditional shortcuts to `responder` for out-of-scope/complaint/product-question, and the active-flow routes from `router` straight to `consent_check`/`offer_simulator` when a flow is pending) — verify an integration test resumes a conversation (including its bound `customer_id` and any pending `active_flow`) from persisted state after a simulated process restart
- [x] 7.10 Write `docs/adr/ADR-001-langgraph-orchestration.md` documenting the choice of LangGraph over CrewAI/autonomous agents
- [x] 7.11 Commit: `feat(agent): adiciona grafo MVP de conversa com identidade de cliente, consentimento sintético e checkpointer Postgres`

## 8. Conversation API (`conversation-api`)

- [x] 8.1 Implement `POST /conversations/{id}/messages` requiring `customer_id` on the first message of a conversation, rejecting unknown customer ids and mismatched customer ids on existing conversations — verify API tests for the unknown-id, mismatched-id, new-conversation, and continued-conversation cases
- [x] 8.2 Implement SSE streaming restricted to `node_started`/`node_finished` progress events (no message content) plus a single terminal `final` event carrying the reply produced after `compliance_guard` — verify an API test asserts no event before the last one carries reply content and exactly one `final` event closes the stream
- [x] 8.3 Implement `GET /health/live` (process-only, no external calls) and `GET /health/ready` (includes the gateway `/v1/models` check) — verify API tests cover live-always-ok and ready reachable/unreachable gateway cases
- [x] 8.4 Commit: `feat(api): adiciona endpoint de conversa vinculado a cliente com streaming seguro e health checks`

## 9. Observability (`observability`)

- [x] 9.1 Configure structlog JSON logging with a trace-id correlation processor — verify a unit test asserts the trace id is present in emitted log records
- [x] 9.2 Implement the secret/PII redaction processor — verify a unit test asserts known secret and PII fields are redacted before emission
- [x] 9.3 Integrate LangFuse tracing (one trace per turn, one span per node, one span per LLM call) — verify a test using a fake LangFuse client asserts span counts match the nodes and LLM calls executed for a turn
- [x] 9.4 Verify no secrets or infra details appear in trace payloads — add a unit test on span serialization
- [x] 9.5 Commit: `feat(observability): adiciona logging estruturado e tracing LangFuse`

## 10. Local Environment and Documentation

- [ ] 10.1 Write `docker-compose.yml` (dev profile: `api`, `postgres`; LangFuse via env) — verify `docker compose config` validates and `docker compose up` boots `api` + `postgres` locally
- [ ] 10.2 Write `.env.example` with every required variable and safe placeholders, including `LLM_BASE_URL` documented as pointing at the local SSH tunnel (e.g., `http://localhost:<port>/v1`) — verify `gitleaks` passes and no real secret, host, or IP is present
- [ ] 10.3 Write `README.md` with an architecture diagram (Mermaid), key decisions, run instructions, and an explicit section on establishing the SSH tunnel to the `9router` gateway for local development (the gateway is never exposed publicly) — verify all listed commands actually run against the scaffold
- [ ] 10.4 Run the full test suite, lint, type-check, and `gitleaks` locally — verify all pass
- [ ] 10.5 Commit: `docs(repo): adiciona docker-compose, env de exemplo e README`

## 11. Pull Request

- [ ] 11.1 Push the branch and open a PR using the `.github/pull_request_template.md` template — verify CI (including the Postgres service container job) passes on the PR
- [ ] 11.2 Present the PR link to me for review before merge

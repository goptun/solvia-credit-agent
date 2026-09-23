## Context

See proposal.md - Why. This is a greenfield repository (no existing code, no existing specs). The project runs against an already-running OpenAI-compatible LLM gateway (`9router`) on the author's VPS, exposing two model aliases (`solvia-fast`, `solvia-smart`), each backed by an ordered fallback chain of Gemini Developer API free-tier models on the gateway side. The application must never hardcode or depend on the underlying model names — only the aliases. All code, docs, and identifiers avoid referencing any real company. Secrets and infra details (IPs, hostnames, keys) must never enter the repo, logs, or traces (gitleaks enforced in pre-commit and CI).

A read-only VPS infrastructure assessment (task group 1 in tasks.md) gates all implementation: it must produce a go/no-go verdict before any implementation task starts, and this design assumes a go verdict, reusing the existing reverse proxy and Docker networks discovered during that assessment where applicable. `.gitignore` and the feature branch are created first (task group 0), independent of that gate.

## Goals / Non-Goals

**Goals:**
- Define the module boundaries and dependency-injection seams so financial/domain logic stays LLM- and framework-free.
- Define the conversation graph's node contracts, typed state shape (including the bound synthetic customer identity), and checkpointing strategy.
- Define the LLM provider abstraction, tier mapping, and degradation behavior precisely enough to implement and test with fakes.
- Define the API contract (request/response/SSE event shapes) for the single MVP endpoint, including how a conversation is bound to a customer and how streaming is kept safe.
- Define the observability approach (what is traced, what is logged, what is redacted).
- Define concrete MVP behavior for every classified intent, the synthetic consent flow, slot extraction for simulation requests, the CET calculation, and which parts of compliance enforcement are deterministic versus LLM-assisted.
- Define the explicit routing matrix (intent × consent status → path) and how an active flow (consent confirmation or slot filling) takes priority over intent reclassification for the customer's next message.

**Non-Goals:**
- Regulatory RAG, human handoff (interrupt/resume, mock CRM, Streamlit console wiring), VPS deployment, evals, and prompt versioning are separate future changes (see proposal.md) and are not designed here beyond the extension points they need.
- Real aggregator/sandbox integration for Open Finance data — only a synthetic generator and an extension point.
- Multi-tenant, multi-user auth/authorization — `customer_id` identifies which synthetic customer record a conversation is bound to, not an authenticated principal; the MVP API is unauthenticated, for local/demo use only (auth hardening is part of `add-vps-deploy`).
- Real consent redirection/collection UX (e.g., an actual Open Finance consent journey) — the consent flow here only records a synthetic grant in state.

## Decisions

### Module boundaries and dependency injection
- `apps/agent/tools/` (financial calculations, spending categorization) and `apps/agent/state.py` contain pure Python with no LangChain/FastAPI imports — testable without any framework.
- `apps/agent/config/` holds the fictional product catalog (interest rates, IOF fixed/daily rates, fees, product descriptions) and the fixed reference date used by the synthetic data generator, as static configuration — never sourced from the LLM and never read by nodes except through this module.
- `apps/agent/nodes/` are thin: each node reads/writes typed state and delegates to `tools/`, `config/`, or `llm/`, no business logic inline.
- `apps/agent/llm/` defines a `LLMPort` protocol plus concrete adapters (OpenAI-compatible, Google, fake) and a factory that resolves the adapter + tier mapping from settings. Everything crossing a boundary (LLM client, embeddings client, CRM client stub, repositories) is injected, never constructed inline in nodes, so all of it substitutes with fakes in tests.
- Alternative considered: let LangGraph nodes call `ChatOpenAI` directly. Rejected — it would leak provider details into node code and block the Google adapter / fake-LLM substitution needed for CI (which never calls the real gateway).

### LLM gateway strategy and provider abstraction
- `LLMFactory.for_node(node_name) -> LLMPort` looks up a per-node tier (`fast` | `smart`) from config (a plain mapping, not code): `router` → fast, `consent_check` → no LLM call (deterministic), `financial_analyst` → smart, `offer_simulator` → smart (slot extraction), `responder` → smart, `compliance_guard` → fast (approval-promise detection only). It then resolves the tier to a model alias (`solvia-fast` / `solvia-smart`) via `LLM_MODEL_FAST` / `LLM_MODEL_SMART` env vars, then builds the adapter selected by `LLM_PROVIDER` (default: `openai_compatible`).
- OpenAI-compatible adapter wraps `langchain_openai.ChatOpenAI` with `base_url`/`api_key` from settings; Google adapter wraps `langchain-google-genai`, supporting both the Gemini Developer API and Vertex AI (toggled by `vertexai=true` + `project`/`location` settings) — same `LLMPort` interface either way.
- Resilience is layered on top of the gateway, not a replacement for it: the app applies its own timeout + exponential-backoff retry per call; if every retry against `solvia-smart` is exhausted, the app degrades once to `solvia-fast` for that call; if `solvia-fast` also fails, the node returns a fixed "demo temporarily unavailable" response instead of raising.
- The resolved underlying model, when present in response metadata, is attached to the LangFuse span for that call (cost/quality analysis), without the app ever branching logic on it.
- Alternative considered: implement fallback chains inside the app instead of relying on the gateway. Rejected per ADR-002 — the gateway already owns the fallback chain across free-tier models; the app only needs the coarser fast/smart degradation and must stay ignorant of underlying model identities.

### Local development access to the gateway
- The `9router` gateway is never exposed publicly. Local development reaches it through an SSH tunnel (`ssh -L <local-port>:<gateway-host>:<gateway-port> <vps-user>@<vps-host>`, run out-of-band by the developer, not by the application), with `LLM_BASE_URL=http://localhost:<local-port>/v1` in the local `.env`. Task group 1 (VPS pre-flight assessment) verifies the tunnel works before any other implementation task starts.
- Alternative considered: expose the gateway on the VPS's public interface for convenience. Rejected — it would put an unauthenticated LLM proxy on the public internet; the tunnel keeps it reachable only to whoever holds the SSH key.

### Customer identity binding
- Every conversation is bound to exactly one synthetic `customer_id` (an id from the synthetic dataset). The API requires `customer_id` on the first message of a conversation; the graph looks it up against the synthetic customer records and rejects the request if the id is unknown. Once bound, `customer_id` is persisted in state and is immutable for the lifetime of that conversation id — a later message that supplies a different `customer_id` for the same conversation id is rejected.
- Alternative considered: make `customer_id` optional and default to a demo customer. Rejected — the assistant's core value (financial analysis, simulation, consent) only makes sense against a specific customer's synthetic data, and an implicit default would hide bugs where the wrong customer's data is used.

### Conversation graph and state
- Graph: `router → consent_check → financial_analyst → offer_simulator → responder → compliance_guard → END`, with `router` able to route non-linearly — out-of-scope and complaint intents short-circuit directly to `responder` (skipping consent/analysis/simulation), and product question short-circuits directly to `responder` too (skipping the consent gate entirely, since it only needs the product catalog) — rather than a strict pipeline; edges are conditional on `state.intent`, `state.consent_status`, and `state.active_flow`. See "Routing matrix" below for the full intent × consent-status → path table.
- State is a single typed object (Pydantic model, used as LangGraph state) with: `customer_id`, `messages` (running transcript), `intent`, `pending_intent` (the intent being fulfilled once an active flow completes, e.g., the loan simulation or profile analysis that triggered a consent request), `active_flow` (`none` | `consent_confirmation` | `slot_filling` — which node currently owns the next customer message), `consent_status` (`valid` | `missing` | `expired` | `just_granted`), `simulation_slots` (amount, term_months, amortization_type, and which of these are still missing/invalid), `customer_profile` (financial summary once computed: income, DTI, spending categories), `simulation_result`, `draft_reply` (set by `responder`, consumed and possibly rewritten by `compliance_guard`), `compliance_flags`, and `trace_id`. No untyped dicts cross node boundaries.
- **Active-flow routing**: when `active_flow` is not `none`, `router` skips intent classification and routes the message directly to the node that owns the flow (`consent_check` for `consent_confirmation`, `offer_simulator` for `slot_filling`). The only classification `router` still performs while a flow is active is a lightweight "new request" check in its structured output: if the message is clearly a new, unrelated request rather than a reply to the pending flow, `router` clears `active_flow` (and `pending_intent`) and classifies the message normally instead of forwarding it to the flow owner. When `consent_check` grants a synthetic consent, it clears `active_flow` and resumes `pending_intent` (routing on to `financial_analyst` or `offer_simulator` as that intent requires) rather than falling through to `responder`.
- Checkpointing uses LangGraph's Postgres checkpointer so conversations survive process restarts; checkpoint id = conversation id from the API path. `customer_id` is part of the checkpointed state, so its immutability check applies across restarts too.
- Human-in-the-loop `interrupt`/`Command` is an explicit extension point (a `compliance_guard` → future `handoff` edge, triggered today only by acknowledging complaints in `responder`) but is not implemented in this change — `add-human-handoff` implements it. The graph ends at `END` for every path in this MVP.
- Alternative considered: encode routing as a fixed linear sequence. Rejected — intent classification (loan simulation vs. product question vs. complaint vs. out-of-scope) requires conditional routing from the start, and retrofitting it later would change the state schema.
- Alternative considered: always re-run full intent classification on every message, even during an active flow. Rejected — a slot answer like "24 meses" or a confirmation like "sim, autorizo" is not itself classifiable as one of the five top-level intents, and re-classifying it would either misroute it or require the classifier to know about flow-internal states, coupling it to `offer_simulator`/`consent_check` internals.

### Routing matrix (intent × consent status → path)

| Intent | Consent status | Path |
|---|---|---|
| `product_question` | any | `router` → `responder` (no consent gate, no analysis, no simulation) |
| `loan_simulation` | `valid` / `just_granted` | `router` → `consent_check` → `offer_simulator` (slot filling as needed) → `responder` |
| `loan_simulation` | `missing` / `expired` | `router` → `consent_check` (requests consent, sets `active_flow = consent_confirmation`, `pending_intent = loan_simulation`) → *(customer confirms)* → `offer_simulator` → `responder` |
| `profile_analysis` | `valid` / `just_granted` | `router` → `consent_check` → `financial_analyst` → `responder` (never reaches `offer_simulator`) |
| `profile_analysis` | `missing` / `expired` | `router` → `consent_check` (requests consent, sets `active_flow = consent_confirmation`, `pending_intent = profile_analysis`) → *(customer confirms)* → `financial_analyst` → `responder` |
| `complaint` | any | `router` → `responder` |
| `out_of_scope` | any | `router` → `responder` |

### Concrete per-intent MVP behavior
- **Loan simulation**: `offer_simulator` extracts `amount`, `term_months`, and `amortization_type` (Price|SAC) from the conversation via structured LLM output (smart tier); any missing or invalid slot sets `active_flow = slot_filling` and causes `responder` to ask a targeted follow-up question instead of simulating. The customer's next message (e.g., "24 meses") is then routed by `router` straight back to `offer_simulator` (per "Active-flow routing" above) rather than being reclassified, and is applied to the pending slot. Once all slots are valid, `active_flow` is cleared, the simulation tool (credit-simulation capability) computes the schedule and CET using rate/IOF/fees from the product catalog, and `responder` renders the numeric result via a template — it never generates numbers itself.
- **Profile analysis**: `financial_analyst` computes income, DTI, and spending categories via tools; `responder` drafts a summary from those tool outputs.
- **Product question**: `responder` answers only from the fictional product catalog's product descriptions (config, not the LLM's own knowledge); full regulatory RAG is out of scope for this change (`add-regulatory-rag`).
- **Complaint**: `responder` acknowledges the complaint and states that a human handoff capability is planned but not yet available in this demo; no ticket is opened (the mock CRM and handoff are `add-human-handoff`).
- **Out of scope**: `responder` issues a polite refusal without invoking financial analysis, simulation, or the product catalog.

### Synthetic consent flow
- `consent_check` reads the bound customer's synthetic consent status from the synthetic dataset (`valid`, `missing`, or `expired`). `consent_check` never calls an LLM — every decision it makes, including confirmation detection, is deterministic.
- If `valid` (or `just_granted`), the conversation proceeds to `financial_analyst`/`offer_simulator` as needed for the classified intent.
- If `missing` or `expired`, `consent_check` records the triggering intent in `pending_intent`, sets `active_flow = consent_confirmation`, and routes to `responder`, which asks the customer to authorize access to their financial data.
- The customer's next message is routed by `router` straight back to `consent_check` (per "Active-flow routing" above), which classifies it deterministically: a normalized keyword/regex match against known Brazilian Portuguese authorization phrases (e.g., "sim, autorizo", "eu autorizo", "autorizo o acesso") records a synthetic consent grant (`consent_status = just_granted`), clears `active_flow`, and resumes `pending_intent` by routing on to `financial_analyst` or `offer_simulator`. A reply that clearly refuses (e.g., "não autorizo") clears the flow and lets `responder` acknowledge the refusal. Any other, ambiguous reply does not change `consent_status` or clear the flow — `consent_check` routes back to `responder` to ask again for an explicit yes/no.
- A message that `router`'s "new request" check identifies as clearly unrelated to the pending confirmation clears `active_flow`/`pending_intent` and is classified normally instead (per "Active-flow routing" above), rather than being forced through `consent_check`.
- Redirecting to a real Open Finance consent journey is explicitly out of scope — this flow only ever manipulates the synthetic record in state.

### API contract
- `POST /conversations/{id}/messages`: body `{ "customer_id": string (required on the first message of a conversation, optional and must match on later messages), "message": string }`. `{id}` is client-supplied (used as the LangGraph thread/checkpoint id); an unknown id starts a new conversation and requires `customer_id`; a known id with an unknown `customer_id` that doesn't match the conversation's bound customer is rejected with a 4xx error. An unknown `customer_id` (not present in the synthetic dataset) is always rejected.
- Response is `text/event-stream` (SSE). To guarantee no unguarded model output ever reaches the client, the stream carries only two event types: `node_started` / `node_finished` (`{"node": str, "type": "node_started"|"node_finished"}`, no message content) for each graph node as it executes, and exactly one terminal `final` event (`{"type": "final", "data": {"reply": str}}`) carrying the reply produced after `compliance_guard` has run. Raw LLM tokens are never streamed.
- `GET /health/live`: process-only liveness check (no external calls) — always returns healthy if the process is running.
- `GET /health/ready`: readiness check that additionally verifies the `9router` gateway is reachable and `/v1/models` lists both configured aliases; used by the VPS assessment and later by deploy healthchecks.
- Alternative considered: stream raw LLM tokens for a more "live" typing effect. Rejected — the compliance guardrails (PII masking, approval-promise blocking, disclaimers) must run on the complete draft reply before anything reaches the customer; streaming tokens would leak unguarded content ahead of that check.
- Alternative considered: WebSockets instead of SSE. Rejected for the MVP — SSE is simpler to proxy through a reverse proxy and sufficient for the restricted progress/final event model above; no client→server streaming is needed.

### CET as annualized IRR
- CET (total effective cost) is computed as the annualized internal rate of return (IRR) of the net cash flow: the amount actually released to the customer (loan amount minus IOF and fees) versus the installment payments (Price or SAC schedule). All monetary and rate arithmetic uses `Decimal`, with rounding applied only at the boundary of each reported figure (installment values, IOF, CET percentage), per rules documented alongside the calculation.
- IOF is computed as a fixed rate plus a daily rate, both capped per the current regulation, with both cap values configurable (not hardcoded), so they can be updated without code changes if the regulation changes.
- Alternative considered: report a simple nominal rate instead of an IRR-based CET. Rejected — CET is the customer-facing legal disclosure figure in Brazilian consumer credit and must reflect the true cost including IOF/fees, not just the nominal interest rate.

### Deterministic compliance enforcement
- PII masking (CPF, account/card numbers, phone, email) is implemented with regex pattern matching against the draft reply — fully deterministic, no LLM involved.
- Mandatory disclaimer injection is implemented via fixed templates selected by conversation context (e.g., a simulation-specific disclaimer) — fully deterministic.
- Approval-promise detection (blocking language that guarantees loan approval) is the one guardrail that uses the LLM (fast tier), because it requires judging free-form phrasing; it is backed by a deterministic keyword/regex check that catches an approval-promise pattern even if the LLM check misses it, so a promise is blocked if either mechanism flags it.
- Alternative considered: use the LLM for PII masking and disclaimer selection too, for flexibility. Rejected — these are exactly the guardrails that must never depend on non-deterministic behavior; regex and templates give 100% reproducible enforcement and are trivially unit-testable.

### Observability
- structlog JSON logging everywhere, with a processor that redacts known secret/PII field names (API keys, tokens, Authorization headers, raw customer PII) before a record is emitted, and injects `trace_id`.
- LangFuse trace per conversation turn, span per graph node and per LLM call; the same `trace_id` is used as the structlog correlation key so a log line and a trace can be cross-referenced.
- Alternative considered: rely on LangFuse alone for debugging. Rejected — structured logs are needed for infra-level issues (gateway unreachable, checkpointer errors) that occur outside any LangFuse span.

## Risks / Trade-offs

- [Gateway's free-tier Gemini models return inconsistent structured output] → Tool/router/slot-extraction outputs use native tool-calling first with a JSON-mode-parse-and-validate-and-retry fallback (Pydantic), so the app tolerates a mid-conversation model swap by the gateway.
- [VPS may lack headroom for the full stack (api + postgres + LangFuse self-host + console)] → Task group 1's assessment produces a go/no-go verdict and may recommend LangFuse Cloud instead of self-host or per-container memory limits before any other task proceeds.
- [Free-tier LLM quota exhaustion during the demo or during development] → Degradation logic (smart→fast→friendly-unavailable) contains the failure mode to a graceful message rather than an error; per-IP/session hardening is deferred to `add-vps-deploy` since this change is not internet-facing yet.
- [Synthetic data generator drifting from real Open Finance Brasil schemas] → Fixed seed plus schema-shaped fixtures reviewed against the public Open Finance Brasil spec; real aggregator integration is explicitly out of scope, so schema fidelity only needs to support the MVP's own tools.
- [LangServe is in maintenance mode] → Not used in this change at all (only planned for a simple RAG chain in `add-regulatory-rag`); ADR captures the rationale for that later decision now while it's fresh.
- [SSH tunnel drops or isn't running locally] → `GET /health/ready` and the VPS assessment both surface gateway unreachability explicitly, and the app's own degradation/fallback logic keeps the demo usable (with the friendly unavailable message) rather than hanging.
- [LLM-based approval-promise detection misses a novel phrasing] → The deterministic keyword/regex check runs in parallel and blocks on its own if it matches, so detection doesn't depend solely on the LLM call succeeding or being accurate.
- [Router's "new request" check wrongly keeps an active flow alive for an unrelated message, or wrongly clears a flow for a genuine reply] → Consent confirmation itself is decided deterministically regardless of the "new request" signal (an ambiguous reply always re-asks rather than silently misinterpreting), bounding the blast radius of a routing misclassification to one extra follow-up turn.

## Migration Plan

Not applicable — greenfield repository, first change, nothing to migrate. Deployment (VPS, reverse proxy, CI/CD to production) is entirely deferred to `add-vps-deploy`; this change only needs to run via `docker compose up` locally (against the gateway via an SSH tunnel) after the VPS pre-flight assessment gate passes.

## Open Questions

None — the VPS-dependent unknowns (exact headroom, existing reverse proxy, gateway reachability, tunnel viability) are resolved by task group 1's assessment before implementation proceeds, not deferred past this plan.

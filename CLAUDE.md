# CLAUDE.md

Engineering conventions for AI-assisted work in this repository. See also
`CONTRIBUTING.md` (same content, human-facing) and
`openspec/changes/*/design.md` for feature-level decisions.

## Project

Solvia: an agentic credit assistant for contact centers, powered by Open
Finance Brasil data and LangGraph. A greenfield portfolio project — a
public live demo, no real company or real customer data anywhere.

## Version control

- Trunk-based: `main` is always green and protected; never commit
  directly to it.
- One short-lived branch per OpenSpec change: `<type>/<change-id>`.
- Conventional Commits: English type/scope, **Portuguese description**,
  imperative mood, subject ≤ 72 chars, body explains the *why* when not
  obvious.
- Small, atomic commits — one logical change each, all passing lint,
  type-check, and tests. Never mix a refactor with a behavior change.
  Commit after each completed task (or coherent group of tasks) in a
  change's `tasks.md`, referencing the task in the commit body.
- One PR per OpenSpec change, using `.github/pull_request_template.md`
  (Portuguese). Squash-merge into `main`.
- Never commit secrets, `.env` files, or generated data — see the
  secrets policy below.

## Secrets and sensitive data (public repository — non-negotiable)

No API keys, tokens, passwords, SSH keys, IP addresses, internal
hostnames, usernames, or internal gateway URLs in code, docs, tests,
fixtures, notebooks, logs, commit messages, PR descriptions, or CI
output — ever.

- Secrets and environment-specific values come only from environment
  variables. `.env` is gitignored; `.env.example` holds only obvious
  placeholders.
- Docs/examples use placeholders or reserved documentation
  ranges/domains (`203.0.113.0/24`, `example.com`), never real values.
- CI/CD deploy credentials live only in GitHub Actions encrypted
  secrets (`${{ secrets.* }}`); workflows never echo them.
- Logs and LangFuse traces never contain secrets or infra details —
  structlog processors redact keys, tokens, and Authorization headers.
- gitleaks runs in pre-commit and as a required CI check on every PR.
- If a secret or IP is ever committed: stop immediately, rotate the
  credential first (removing it from history alone is not enough), then
  rewrite history before any push.

## Clean code

- Code, identifiers, docstrings, and technical docs in **English**;
  user-facing assistant messages in **Brazilian Portuguese**.
- Domain logic (financial calculations, rules) is pure Python with no
  LangChain/FastAPI imports; graph nodes are thin and delegate to
  `tools/`/`config/`/`llm/`; I/O lives at the edges (ports and adapters).
- Dependency injection for LLM, embeddings, vector store, CRM client,
  and repositories — everything testable with fakes.
- Full type hints, `mypy --strict`, Pydantic models at every boundary
  (API, tool I/O, LLM structured output); no untyped dicts crossing
  module boundaries.
- Small functions, single responsibility, intention-revealing names, no
  magic numbers (named constants/config via pydantic-settings), no dead
  or commented-out code.
- Explicit error handling with domain exceptions; no bare `except`;
  retries/timeouts on external calls.
- Docstrings on public modules/classes/functions (Google style);
  comments only for the *why*.

## Testing

- Unit tests for domain and tools, including edge cases.
- Integration tests for graph paths using the fake LLM adapter.
- API tests with `httpx`.
- CI never calls the real LLM gateway — only the fake adapter
  (`LLM_PROVIDER=fake`).
- Coverage target ≥ 80% on domain code (`pytest --cov`).
- Integration tests that need a real database (e.g., the checkpointer
  resume test) run against the Postgres service container in CI.

## LLM gateway conventions

- The app only ever knows model **aliases** (`solvia-fast`,
  `solvia-smart`), never underlying model names — resolved through the
  `LLMFactory`, never a concrete provider SDK imported in node code.
- `solvia-smart` resolves to a reasoning model that spends
  `reasoning_tokens` from the same `max_tokens` budget before emitting
  visible content — always request a generous `max_tokens` (≥256) on
  `smart`-tier calls, and treat an empty completion with
  `finish_reason == "length"` as retryable, not a hard failure (see
  `docs/infra-assessment.md`).
- JSON-mode responses may be wrapped in a ` ```json ` Markdown fence —
  strip it before `json.loads`.
- Local development reaches the gateway only through an SSH tunnel
  (`ssh -L`); the gateway is never exposed publicly.

## Tooling

Enforced via pre-commit (`ruff format`, `ruff check`, `mypy`,
`gitleaks`) and CI on every PR (lint, type-check, tests, gitleaks).

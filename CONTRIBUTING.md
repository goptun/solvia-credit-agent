# Contributing to Solvia

This is a solo portfolio project, but it follows the same engineering
discipline as a production repository. These conventions are also recorded
in `CLAUDE.md` for AI-assisted sessions.

## Branching

- `main` is always green and protected. Never commit directly to it.
- One short-lived branch per OpenSpec change, named `<type>/<change-id>`
  (e.g. `feat/add-regulatory-rag`, `fix/…`, `chore/…`, `docs/…`).
- Rebase on `main` before opening a PR if it has moved.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/), with the
**type and scope in English** and the **description in Brazilian
Portuguese**, imperative mood, subject ≤ 72 chars. Explain the *why* in
the body when it isn't obvious.

```
feat(agent): adiciona nó consent_check ao grafo
test(tools): cobre casos de borda da simulação SAC
fix(api): corrige encerramento do stream SSE em caso de erro
```

Keep commits small and atomic: one logical change per commit, each one
passing lint, type-check, and tests. Never mix a refactor with a behavior
change. Commit after completing each task in a change's `tasks.md` (or a
coherent group of them), and reference the task in the commit body.

## Pull requests

- One PR per OpenSpec change.
- Use the PR template (`.github/pull_request_template.md`).
- All CI checks (lint, type-check, tests, gitleaks) must pass before merge.
- Squash-merge into `main`.

## Secrets and sensitive data

This repository is public. Never commit API keys, tokens, passwords, SSH
keys, IP addresses, internal hostnames, usernames, or internal gateway
URLs — in code, docs, tests, fixtures, notebooks, logs, commit messages,
PR descriptions, or CI output. All secrets and environment-specific
values come only from environment variables (`.env`, gitignored;
`.env.example` holds only placeholders). Run `git diff --staged` and
gitleaks before every commit if you're ever unsure.

## Code quality

- Python 3.12, `uv`, `ruff` (lint + format), `mypy --strict`, `pytest`.
- Full type hints; Pydantic models at every boundary; no untyped dicts
  crossing module boundaries.
- Domain logic (financial calculations, rules) is pure Python with no
  framework imports; graph nodes are thin and delegate to services/tools.
- Tests: unit tests for domain and tools (including edge cases),
  integration tests for graph paths with a fake LLM, API tests with
  `httpx`. CI never calls the real LLM gateway.
- `pre-commit run --all-files` and `uv run pytest` must pass locally
  before pushing.

## Releases

Tag releases with SemVer and keep `CHANGELOG.md` generated from
Conventional Commits.

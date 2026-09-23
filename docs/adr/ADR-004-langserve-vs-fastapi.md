# ADR-004: Plain FastAPI, not LangServe, for the standalone knowledge endpoint

## Status

Accepted

## Context

`add-regulatory-rag` needs a standalone HTTP endpoint exposing the retrieval-and-answer chain directly (`POST /knowledge/answer`), independent of the conversational graph's SSE contract — see `specs/regulatory-knowledge-agent/spec.md` — "Standalone retrieval-and-answer HTTP endpoint". The original proposal named LangServe as the planned mechanism, conditional on verifying its fit against this repo's actual pinned dependency versions first.

**Verification performed** (this repo, `langchain-core==1.6.4`, `fastapi==0.141.1`, `pydantic==2.13.5`):

- `uv add --no-sync langserve` resolves cleanly: `langserve==0.3.3` (its latest release, dated 2025-10-17) declares `langchain-core<2,>=0.3` for its non-optional dependencies, `httpx`/`orjson`/`pydantic` — no conflict with this repo's pins.
- The actual FastAPI integration (`add_routes()`) lives behind the `all` extra, not the base package. `uv add --no-sync "langserve[all]"` resolves too, pulling in `fastapi` (already a dependency, no new constraint) and a **new** dependency, `sse-starlette==1.8.2` — the version LangServe's own pin (`sse-starlette<2.0.0,>=1.3.0`) caps it to, one major version behind the current `sse-starlette` release line.
- LangServe's own documentation states: *"We recommend using LangGraph Platform rather than LangServe for new projects. We will continue to accept bug fixes for LangServe from the community; however, we will not be accepting new feature contributions."* This repo already uses LangGraph directly (`apps/agent/graph.py`), not LangGraph Platform, so this isn't a straightforward "migrate to the recommended alternative" situation either — but it does confirm LangServe is a legacy-maintenance-only project, not a currently developed one.
- This repo already has a working, hand-rolled SSE/JSON pattern in `apps/api/routes.py` and `apps/api/events.py` for the conversational endpoint, with no `sse-starlette` dependency at all.

## Decision

**Plain FastAPI**, not LangServe, for `POST /knowledge/answer`.

The dependency floors are technically satisfiable, but adopting a feature-frozen library for a *new* piece of surface area — when it would add a new, one-major-version-behind dependency (`sse-starlette`) for capability this endpoint doesn't even need (the spec requires a single grounded JSON answer, not a streamed one — see `design.md` — "Standalone endpoint and the LangServe decision") — trades a small amount of boilerplate LangServe would have saved for a dependency with no path forward and a benefit this endpoint can't use anyway.

`apps/api/routes_knowledge.py` (task 9.3) is a thin FastAPI route calling the same grounding function `knowledge_agent` uses, returning one JSON response — no SSE, no LangServe, no new dependency.

## Alternatives considered

- **LangServe (bare, no extras).** Rejected — doesn't even include the FastAPI route-registration integration (`add_routes()`) the whole point of adopting it would be for; using it would still require the `all` extra.
- **LangServe with the `all` extra.** Rejected per Decision above — resolves and would work, but adds a stale, capped dependency (`sse-starlette<2.0.0`) for a feature-frozen library, for a streaming capability this endpoint's spec doesn't require.

## Consequences

- No new runtime dependency for this endpoint.
- If a future change needs to expose several LangChain `Runnable`s as routes with LangServe's playground/schema-introspection conveniences, this ADR's verification (the dependency floors are satisfiable) remains useful — but LangServe's frozen-feature status should be re-weighed against whatever LangGraph Platform or a newer alternative looks like at that time, not assumed unchanged from this ADR's date.

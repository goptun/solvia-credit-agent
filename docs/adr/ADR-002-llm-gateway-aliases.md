# ADR-002: LLM access through an OpenAI-compatible gateway with model aliases

## Status

Accepted

## Context

Solvia needs LLM access for intent classification, financial analysis,
slot extraction, reply drafting, and compliance checks. The author runs
a personal OpenAI-compatible gateway (`9router`) on their VPS, in front
of the Gemini Developer API free tier, and wants:

- The application to stay ignorant of which underlying model actually
  answered a call, so the gateway's model choices can change without a
  code change.
- Two cost/latency tiers (a fast one for classification-style tasks, a
  smart one for reasoning-heavy tasks), not a single fixed model.
- The app to keep working, in a degraded but graceful way, if a tier —
  or the gateway itself — is temporarily unavailable (free-tier quota
  exhaustion is a real, expected failure mode).
- To swap providers (e.g. call Gemini directly, or a different
  OpenAI-compatible endpoint) without touching agent node code.
- CI to never depend on network access to any real LLM.

## Decision

- The application only ever references two **model aliases**:
  `solvia-fast` and `solvia-smart` (env `LLM_MODEL_FAST` /
  `LLM_MODEL_SMART`). It never hardcodes or branches on an underlying
  model name. The gateway owns an ordered fallback chain of concrete
  models behind each alias; the app is deliberately unaware of that
  chain.
- A `LLMPort` protocol (`apps/agent/llm/port.py`) is the only interface
  agent nodes depend on. An `LLMFactory` (`apps/agent/llm/factory.py`)
  resolves `for_node(node_name)` to a tier (`fast`/`smart`) via a plain
  data table (`NODE_TIER_MAP`), then to a concrete adapter chosen by
  `LLM_PROVIDER`: an OpenAI-compatible adapter (default, talks to
  `9router`), a Google adapter (Gemini Developer API or Vertex AI, via
  `langchain-google-genai`), or a fake adapter for tests.
- **Application-level resilience is layered on top of, not instead of,
  the gateway's own fallback chain**: every call gets a timeout and
  exponential-backoff retries (`apps/agent/llm/resilience.py`); if a
  `smart`-tier call is exhausted, it degrades once to the `fast` tier;
  if that also fails, the node returns a fixed "demo temporarily
  unavailable" message instead of raising. `fast`-tier nodes have no
  further fallback — if `fast` itself fails, the same unavailable
  message is returned directly.
- The resolved underlying model, when the gateway reports it in
  response metadata, is captured (`LLMCallResult.resolved_model`) for
  observability (task group 9) to attach to a trace span — the
  application never branches logic on this value.
- CI and the automated test suite use only the fake adapter
  (`LLM_PROVIDER=fake`); no test ever calls the real gateway.

## Alternatives considered

- **Implement the model fallback chain inside the application.**
  Rejected — the gateway already owns this, and duplicating it in the
  app would mean two systems disagreeing about which model is "next" on
  failure. The app only needs the coarser fast/smart degradation.
- **Call `ChatOpenAI`/`ChatGoogleGenerativeAI` directly from graph
  nodes.** Rejected — it would leak provider details into node code and
  block swapping providers or substituting the fake adapter in tests.
- **A single model tier instead of fast/smart.** Rejected — router-style
  classification and compliance checks are latency-sensitive and don't
  need a reasoning model; financial analysis, slot extraction, and reply
  drafting benefit from a stronger model. A fixed single tier would
  either overpay in latency everywhere or underpower the reasoning
  tasks.

## Consequences

- Moving a node between tiers, or swapping the active provider, is a
  configuration change, not a node code change.
- `docs/infra-assessment.md` documents a real finding from this
  decision: `solvia-smart` resolves to a reasoning model that spends
  `reasoning_tokens` from the same `max_tokens` budget before emitting
  visible content, and its JSON-mode output can be wrapped in a
  Markdown fence — both must be handled by the smart-tier call sites and
  the JSON-mode parsing fallback, not by this ADR's resilience layer.
- The gateway is reached locally only through an SSH tunnel, never a
  public endpoint (see `design.md` — "Local development access to the
  gateway").

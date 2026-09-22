## Purpose

Gives every agent node a uniform, provider-agnostic way to call an LLM through the `9router` gateway's model aliases, with tier-based routing, resilience, and graceful degradation when the gateway is unavailable or a tier is exhausted.

## ADDED Requirements

### Requirement: Provider-agnostic LLM access
The system SHALL expose LLM access to agent nodes only through a provider-agnostic interface, resolved by a factory from configuration, with no node importing a concrete LLM client library directly.

#### Scenario: Node requests an LLM without knowing the provider
- **WHEN** an agent node needs to call an LLM
- **THEN** it obtains an LLM instance from the factory and calls it through the provider-agnostic interface only, without referencing any concrete provider SDK

#### Scenario: Switching provider requires no node changes
- **WHEN** the configured LLM provider is changed from the OpenAI-compatible adapter to the Google adapter (or vice versa) via configuration only
- **THEN** all agent nodes continue to function without any code changes to node logic

### Requirement: Per-node model tier mapping
The system SHALL resolve, for each agent node, a configured tier (`fast` or `smart`) that maps to the `solvia-fast` or `solvia-smart` gateway alias, with the mapping defined in configuration rather than hardcoded per node.

#### Scenario: Router and compliance guard use the fast tier
- **WHEN** the `router` or `compliance_guard` node requests an LLM
- **THEN** the factory resolves it to the model alias configured for the `fast` tier

#### Scenario: Financial analyst and responder use the smart tier
- **WHEN** the `financial_analyst` or `responder` node requests an LLM
- **THEN** the factory resolves it to the model alias configured for the `smart` tier

#### Scenario: Tier mapping changes without code changes
- **WHEN** the tier assigned to a node is changed in configuration (e.g., moving a node from `fast` to `smart`)
- **THEN** the node uses the newly configured tier on its next call, with no changes to the node's source code

### Requirement: Application-level resilience on top of the gateway
The system SHALL apply a timeout and exponential-backoff retries to every LLM call, in addition to whatever fallback the gateway performs internally.

#### Scenario: Transient failure is retried
- **WHEN** an LLM call fails with a transient error (timeout or 5xx-equivalent)
- **THEN** the system retries the call with exponential backoff up to the configured retry limit before treating the call as failed

### Requirement: Graceful degradation from smart to fast
The system SHALL degrade a failed `smart`-tier call to the `fast`-tier alias exactly once before treating the call as unavailable.

#### Scenario: Smart tier exhausted, fast tier succeeds
- **WHEN** all retries against the `smart` alias fail for a given call
- **THEN** the system retries the same request once against the `fast` alias and returns its result if it succeeds

### Requirement: Friendly unavailable response when both tiers fail
The system SHALL return a fixed, user-facing "demo temporarily unavailable" response instead of raising an error to the end user when both the `smart` and `fast` aliases fail for a call.

#### Scenario: Both tiers exhausted
- **WHEN** the `smart` alias fails and the subsequent degraded `fast` alias call also fails
- **THEN** the node returns the fixed unavailable message as the conversation turn's output instead of propagating an exception to the API layer

### Requirement: Resolved model recorded for observability
The system SHALL record, on the tracing span for each LLM call, the underlying model name reported by the gateway response metadata whenever the gateway provides it, without the application branching any logic on that value.

#### Scenario: Gateway reports the underlying model
- **WHEN** an LLM call succeeds and the response metadata includes the resolved underlying model
- **THEN** that model name is attached to the call's trace span

#### Scenario: Gateway does not report the underlying model
- **WHEN** an LLM call succeeds and the response metadata does not include a resolved model
- **THEN** the call proceeds normally and no resolved-model field is attached to the trace span

### Requirement: Fake LLM for automated tests
The system SHALL provide a fake LLM adapter, selectable via configuration, that returns deterministic canned responses and SHALL be the only adapter used by the automated test suite and CI.

#### Scenario: Tests never call the real gateway
- **WHEN** the automated test suite exercises any agent node that calls an LLM
- **THEN** the fake LLM adapter is used and no network call is made to the real gateway

## Purpose

Makes the non-deterministic agent's behavior inspectable: every conversation turn produces correlated structured logs and an LLM trace, without ever leaking secrets or PII into either.

## ADDED Requirements

### Requirement: Structured logging with correlation id
The system SHALL emit structured (JSON) logs for each request and graph node execution, each carrying a trace id that correlates it with the corresponding LangFuse trace.

#### Scenario: Log line carries the trace id
- **WHEN** a graph node executes as part of handling a conversation turn
- **THEN** the log line for that execution includes the same trace id used for that turn's LangFuse trace

### Requirement: Secret and PII redaction in logs
The system SHALL redact known secret fields (API keys, tokens, Authorization headers) and customer PII from all log output before it is emitted.

#### Scenario: API key is redacted
- **WHEN** a log record would otherwise include an LLM gateway API key or bearer token
- **THEN** that value is redacted before the log record is emitted

#### Scenario: Customer PII is redacted
- **WHEN** a log record would otherwise include a customer's PII (e.g., document number, full account number)
- **THEN** that value is redacted before the log record is emitted

### Requirement: LangFuse tracing per turn and LLM call
The system SHALL create a LangFuse trace for each conversation turn, with a span for each graph node and each LLM call made during that turn.

#### Scenario: Turn produces one trace with per-node spans
- **WHEN** a conversation turn is processed through multiple graph nodes
- **THEN** a single LangFuse trace is created for the turn, containing one span per node executed and one span per LLM call made

### Requirement: Traces never contain secrets or infra details
The system SHALL NOT include secrets, API keys, or infrastructure details (internal hostnames, IPs) in any LangFuse trace payload.

#### Scenario: Trace payload excludes gateway credentials
- **WHEN** an LLM call span is recorded on a trace
- **THEN** the span excludes the gateway API key and any internal hostname or IP used to reach it

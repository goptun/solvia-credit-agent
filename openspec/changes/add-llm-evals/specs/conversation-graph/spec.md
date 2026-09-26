## MODIFIED Requirements

### Requirement: Compliance guardrails
The system SHALL enforce compliance guardrails on every outgoing reply: masking PII, blocking language that promises loan approval, injecting mandatory disclaimers for simulation/analysis replies, and injecting an "informational content, not legal advice" disclaimer for regulatory-question replies. The approval-promise check SHALL fail closed: when its LLM-based verdict cannot be obtained, the system SHALL decide with a stricter deterministic screen, SHALL log a warning that carries no reply content, and SHALL mark the turn as degraded, never skipping the check silently.

#### Scenario: PII is masked in the reply
- **WHEN** an outgoing reply would otherwise include a customer's PII (CPF, account or card number, phone, or email)
- **THEN** the compliance guard deterministically masks that PII, via pattern matching, before the reply is sent

#### Scenario: Approval promises are blocked
- **WHEN** a draft reply contains language that promises or guarantees loan approval
- **THEN** the compliance guard blocks or rewrites that language before the reply is sent, whether the language was flagged by the LLM-based check, the deterministic keyword/regex check, or both

#### Scenario: Mandatory disclaimer is present
- **WHEN** any reply involving a simulated offer or financial analysis is sent
- **THEN** it includes the mandatory regulatory disclaimer, injected deterministically via template

#### Scenario: Informational disclaimer is present on regulatory answers
- **WHEN** a reply answering a regulatory question is sent
- **THEN** it includes an "informational content, not legal advice" disclaimer, injected deterministically via template

#### Scenario: Unavailable LLM verdict blocks an unhedged approval mention
- **WHEN** the LLM-based approval check cannot produce a valid result (a structured-output failure, a timeout, or an exhausted turn deadline) and the draft mentions approval without an explicit hedge phrase, such as "seu crédito está aprovado, veja a simulação" or "após análise, seu empréstimo foi aprovado"
- **THEN** the compliance guard blocks or rewrites that language, logs a warning containing no reply content, and marks the turn as degraded

#### Scenario: Unavailable LLM verdict lets an explicitly hedged reply through
- **WHEN** the LLM-based approval check cannot produce a valid result and the draft mentions approval only with an explicit hedge phrase, such as "sujeito à análise", "depende de análise" or "não posso garantir"
- **THEN** the reply is sent unchanged, with a warning logged and the turn marked as degraded

#### Scenario: Unavailable LLM verdict never skips the check silently
- **WHEN** the LLM-based approval check cannot produce a valid result for any reason
- **THEN** a warning is logged and the turn is marked as degraded, whatever the reply says

## MODIFIED Requirements

### Requirement: Intent classification and routing
The system SHALL classify each incoming customer message into one of: product question, regulatory question, loan simulation, profile analysis, complaint, or out of scope, and SHALL route the conversation accordingly.

#### Scenario: Loan simulation intent routes to simulation path
- **WHEN** a customer message is classified as a loan simulation request
- **THEN** the conversation proceeds toward the offer simulation step (subject to the consent gate)

#### Scenario: Out-of-scope intent short-circuits
- **WHEN** a customer message is classified as out of scope
- **THEN** the conversation routes directly to the responder step without invoking financial analysis or simulation

#### Scenario: Complaint intent short-circuits
- **WHEN** a customer message is classified as a complaint
- **THEN** the conversation routes directly to the responder step without invoking financial analysis or simulation

#### Scenario: Product question intent skips the consent gate
- **WHEN** a customer message is classified as a product question
- **THEN** the conversation routes directly to the knowledge agent step without passing through the consent gate, financial analysis, or simulation

#### Scenario: Regulatory question intent skips the consent gate
- **WHEN** a customer message is classified as a regulatory question
- **THEN** the conversation routes directly to the knowledge agent step without passing through the consent gate, financial analysis, or simulation

#### Scenario: Profile analysis intent does not invoke offer simulation
- **WHEN** a customer message is classified as profile analysis
- **THEN** the conversation proceeds through the consent gate and financial analyst step but does not invoke the offer simulation step

### Requirement: Per-intent reply drafting
The system SHALL draft a reply appropriate to the classified intent before compliance checking: a data-backed reply for loan simulation and profile analysis, an acknowledgment noting that human handoff is a future capability for complaints, and a polite refusal for out-of-scope requests. Product questions and regulatory questions are answered by the knowledge agent instead of this drafting step.

#### Scenario: Loan simulation or profile analysis reply is data-backed
- **WHEN** the responder drafts a reply for a loan simulation or profile analysis intent
- **THEN** the numeric values in the reply come from the simulation or analysis tool output, rendered via a template, never generated as numbers by the LLM

#### Scenario: Complaint reply acknowledges and defers to future handoff
- **WHEN** the responder drafts a reply for a complaint intent
- **THEN** the reply acknowledges the complaint and states that human handoff is a planned but not-yet-available capability

#### Scenario: Out-of-scope reply is a polite refusal
- **WHEN** the responder drafts a reply for an out-of-scope intent
- **THEN** the reply politely declines to help with that request without invoking financial analysis, simulation, or the product catalog

#### Scenario: Product question reply uses only the product catalog
- **WHEN** a customer message is classified as a product question
- **THEN** the reply is produced by the knowledge agent from the product-catalog documents in the regulatory knowledge base, not by this per-intent drafting step, and not from the LLM's general knowledge or an external source

### Requirement: Compliance guardrails
The system SHALL enforce compliance guardrails on every outgoing reply: masking PII, blocking language that promises loan approval, injecting mandatory disclaimers for simulation/analysis replies, and injecting an "informational content, not legal advice" disclaimer for regulatory-question replies.

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

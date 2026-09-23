## Purpose

Defines the MVP conversational agent orchestration: how an incoming customer message, bound to a synthetic customer identity, is classified, gated on consent, analyzed, simulated, drafted into a reply, and compliance-checked, with conversation state persisted across turns.

## ADDED Requirements

### Requirement: Customer identity binding
The system SHALL bind every conversation to exactly one synthetic customer id, SHALL reject the conversation if that id does not match a known synthetic customer, and SHALL treat the bound customer id as immutable for the lifetime of that conversation.

#### Scenario: Unknown customer id is rejected
- **WHEN** a conversation is started (or continued) with a `customer_id` that does not match any synthetic customer record
- **THEN** the system rejects the request and does not advance the graph

#### Scenario: Known customer id binds the conversation
- **WHEN** a conversation is started with a `customer_id` that matches a synthetic customer record
- **THEN** the graph records that `customer_id` in state and uses it for every subsequent step of the conversation

#### Scenario: Customer id is immutable within a conversation
- **WHEN** a later message for an existing conversation supplies a `customer_id` different from the one already bound to that conversation
- **THEN** the system rejects that message and does not change the conversation's bound customer id

### Requirement: Intent classification and routing
The system SHALL classify each incoming customer message into one of: product question, loan simulation, profile analysis, complaint, or out of scope, and SHALL route the conversation accordingly.

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
- **THEN** the conversation routes directly to the responder step without passing through the consent gate, financial analysis, or simulation

#### Scenario: Profile analysis intent does not invoke offer simulation
- **WHEN** a customer message is classified as profile analysis
- **THEN** the conversation proceeds through the consent gate and financial analyst step but does not invoke the offer simulation step

### Requirement: Active-flow routing takes priority over reclassification
The system SHALL, whenever a conversation has an active flow awaiting a specific customer response (a pending consent confirmation or a pending simulation slot), route the customer's next message directly to the node that owns that flow instead of reclassifying its intent — unless the message is clearly a new, unrelated request, in which case the active flow is cleared and normal intent classification resumes for that message.

#### Scenario: Consent confirmation resumes the original request
- **WHEN** the customer replies "sim, autorizo" while a consent authorization request is pending for a loan simulation
- **THEN** the system grants synthetic consent and resumes the loan simulation request that triggered the consent flow, without reclassifying "sim, autorizo" as a new intent

#### Scenario: Slot answer is consumed by the pending slot-filling flow
- **WHEN** the customer replies "24 meses" while the offer simulator is waiting on a missing term slot
- **THEN** the system applies "24 meses" as the term slot's value instead of reclassifying it as a new intent

#### Scenario: An unrelated new request clears the active flow
- **WHEN** the customer sends a message that is clearly a new, unrelated request while a consent-confirmation or slot-filling flow is active
- **THEN** the system clears the active flow and classifies the new message's intent normally, rather than forcing it through the flow's owning node

### Requirement: Synthetic Open Finance consent flow
The system SHALL read the bound customer's synthetic consent status before analyzing their financial data, and SHALL drive a request-and-confirm flow that records a synthetic consent grant in state when consent is missing or expired.

#### Scenario: Missing consent blocks financial analysis
- **WHEN** a customer requests profile analysis or loan simulation and their recorded synthetic consent status is missing or expired
- **THEN** the conversation routes to asking the customer to authorize access instead of proceeding to the financial analyst step

#### Scenario: Explicit confirmation grants synthetic consent and resumes the original request
- **WHEN** the customer has been asked to authorize access and their next message is an explicit confirmation
- **THEN** the system records a synthetic consent grant in state and the conversation proceeds to the financial analyst or offer simulator step for the intent that originally triggered the consent request

#### Scenario: Valid consent allows analysis to proceed
- **WHEN** a customer has a valid recorded consent (including one just granted in this conversation)
- **THEN** the conversation proceeds to the financial analyst step for that request

### Requirement: Deterministic consent confirmation detection
The system SHALL detect an explicit consent confirmation using a deterministic, normalized keyword/regex match against known Brazilian Portuguese authorization phrases, SHALL make no LLM call to do so, and SHALL re-ask for confirmation when a reply is ambiguous rather than guessing.

#### Scenario: Explicit confirmation is recognized deterministically
- **WHEN** the customer's reply, while consent is pending, matches a known authorization phrase (e.g., "sim, autorizo", "eu autorizo")
- **THEN** the system grants synthetic consent using only the deterministic keyword/regex match, with no LLM call involved in that decision

#### Scenario: Ambiguous reply re-asks instead of guessing
- **WHEN** the customer's reply, while consent is pending, matches neither a known authorization phrase nor a clear refusal
- **THEN** the system asks again for explicit confirmation instead of granting or denying consent

### Requirement: Financial profile analysis
The system SHALL analyze a consented customer's financial data to estimate income, debt-to-income ratio, and spending categories, using the credit-simulation tools rather than LLM computation.

#### Scenario: Profile analysis produces income, DTI, and spending categories
- **WHEN** the financial analyst step runs for a consented customer
- **THEN** it produces an estimated income, a debt-to-income ratio, and spending categories using the deterministic financial tools

### Requirement: Simulation slot extraction
The system SHALL extract loan amount, term in months, and amortization type (Price or SAC) from the conversation via structured LLM output, and SHALL ask a follow-up question instead of simulating when any slot is missing or invalid.

#### Scenario: All slots present and valid
- **WHEN** the customer's messages provide a valid amount, term, and amortization type
- **THEN** the conversation proceeds to the credit offer simulation step with those slot values

#### Scenario: A slot is missing
- **WHEN** the customer's messages do not provide one of amount, term, or amortization type
- **THEN** the system asks a follow-up question targeting the missing slot instead of invoking the simulation

#### Scenario: A slot is invalid
- **WHEN** an extracted slot value fails validation (e.g., a non-positive amount or an unrecognized amortization type)
- **THEN** the system asks a follow-up question targeting that slot instead of invoking the simulation

### Requirement: Credit offer simulation
The system SHALL simulate credit offers (Price and SAC, with CET) via the deterministic financial tools, using product terms from the fictional product catalog, once all simulation slots are valid, and the LLM SHALL never compute simulation values itself.

#### Scenario: Simulation delegates to deterministic tools
- **WHEN** the offer simulator step runs with a complete, valid set of simulation slots
- **THEN** it calls the Price/SAC simulation tools, using the interest rate, IOF, and fees from the product catalog, to obtain the numeric result rather than having the LLM produce the numbers directly

### Requirement: Per-intent reply drafting
The system SHALL draft a reply appropriate to the classified intent before compliance checking: a data-backed reply for loan simulation and profile analysis, a product-catalog-only reply for product questions, an acknowledgment noting that human handoff is a future capability for complaints, and a polite refusal for out-of-scope requests.

#### Scenario: Loan simulation or profile analysis reply is data-backed
- **WHEN** the responder drafts a reply for a loan simulation or profile analysis intent
- **THEN** the numeric values in the reply come from the simulation or analysis tool output, rendered via a template, never generated as numbers by the LLM

#### Scenario: Product question reply uses only the product catalog
- **WHEN** the responder drafts a reply for a product question intent
- **THEN** the reply content is drawn only from the fictional product catalog, not from the LLM's general knowledge or an external source

#### Scenario: Complaint reply acknowledges and defers to future handoff
- **WHEN** the responder drafts a reply for a complaint intent
- **THEN** the reply acknowledges the complaint and states that human handoff is a planned but not-yet-available capability

#### Scenario: Out-of-scope reply is a polite refusal
- **WHEN** the responder drafts a reply for an out-of-scope intent
- **THEN** the reply politely declines to help with that request without invoking financial analysis, simulation, or the product catalog

### Requirement: Compliance guardrails
The system SHALL enforce compliance guardrails on every outgoing reply: masking PII, blocking language that promises loan approval, and injecting mandatory disclaimers.

#### Scenario: PII is masked in the reply
- **WHEN** an outgoing reply would otherwise include a customer's PII (CPF, account or card number, phone, or email)
- **THEN** the compliance guard deterministically masks that PII, via pattern matching, before the reply is sent

#### Scenario: Approval promises are blocked
- **WHEN** a draft reply contains language that promises or guarantees loan approval
- **THEN** the compliance guard blocks or rewrites that language before the reply is sent, whether the language was flagged by the LLM-based check, the deterministic keyword/regex check, or both

#### Scenario: Mandatory disclaimer is present
- **WHEN** any reply involving a simulated offer or financial analysis is sent
- **THEN** it includes the mandatory regulatory disclaimer, injected deterministically via template

### Requirement: Typed, checkpointed conversation state
The system SHALL represent conversation state as a typed schema and SHALL persist it via a checkpointer keyed by conversation id, so a conversation can resume after a process restart.

#### Scenario: Conversation resumes after restart
- **WHEN** a conversation with prior turns is continued after the process has restarted
- **THEN** the graph resumes from the persisted state for that conversation id, including its bound `customer_id`, rather than starting over

#### Scenario: State crossing node boundaries is typed
- **WHEN** any node reads or writes conversation state
- **THEN** it does so through the typed state schema, not an untyped dictionary

## Purpose

Exposes the conversation graph over HTTP so a client (the demo console, or later the public site) can send a customer-bound message and receive a safely-streamed reply, and can check whether the service and its LLM gateway dependency are healthy.

## ADDED Requirements

### Requirement: Streamed conversation endpoint bound to a customer
The system SHALL expose `POST /conversations/{id}/messages` requiring a `customer_id` on the first message of a conversation, rejecting unknown or mismatched customer ids, and SHALL stream the resulting graph progress to the caller via Server-Sent Events, ending with a single final event containing the complete assistant reply.

#### Scenario: New conversation id starts a new conversation
- **WHEN** a message is posted to `/conversations/{id}/messages` with an id not seen before and a valid `customer_id`
- **THEN** a new conversation is started, bound to that `customer_id`, and the graph's progress events are streamed back

#### Scenario: Existing conversation id continues the conversation
- **WHEN** a message is posted to `/conversations/{id}/messages` with an id from a prior conversation
- **THEN** the graph resumes from that conversation's persisted state and streams the new turn's progress events

#### Scenario: Unknown customer id is rejected
- **WHEN** a message is posted with a `customer_id` that does not match any synthetic customer record
- **THEN** the request is rejected with an error and no conversation is started or advanced

#### Scenario: Mismatched customer id on an existing conversation is rejected
- **WHEN** a message is posted to an existing conversation id with a `customer_id` different from the one already bound to that conversation
- **THEN** the request is rejected with an error and the conversation's bound customer id is unchanged

### Requirement: Streaming carries no unguarded model output
The system SHALL restrict the SSE stream to node-progress events (`node_started`, `node_finished`) that carry no message content, plus exactly one terminal `final` event carrying the reply produced after the compliance guard has run, and SHALL NOT stream raw or intermediate model tokens.

#### Scenario: Stream contains only progress and final events
- **WHEN** a conversation turn is processed
- **THEN** every event on the stream before the last one is a `node_started` or `node_finished` event with no reply content, and the last event is a single `final` event

#### Scenario: Final event follows compliance guard
- **WHEN** the `final` event is emitted
- **THEN** its reply content is the one produced after the compliance guard step, not an earlier draft

### Requirement: Liveness health endpoint
The system SHALL expose `GET /health/live` reporting only whether the service process is running, making no external calls.

#### Scenario: Live process reports healthy
- **WHEN** `/health/live` is called while the service process is running
- **THEN** the response indicates the service is healthy, regardless of the LLM gateway's reachability

### Requirement: Readiness health endpoint with gateway reachability
The system SHALL expose `GET /health/ready` reporting whether the configured LLM gateway is reachable, in addition to the service's own status.

#### Scenario: Ready and reachable gateway
- **WHEN** `/health/ready` is called while the service is running and the gateway responds
- **THEN** the response indicates the service is ready and the gateway is reachable

#### Scenario: Gateway unreachable is reflected in readiness
- **WHEN** `/health/ready` is called while the configured LLM gateway does not respond
- **THEN** the response indicates the service is not ready due to the unreachable gateway, without the endpoint itself erroring

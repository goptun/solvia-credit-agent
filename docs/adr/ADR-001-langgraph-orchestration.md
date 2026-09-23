# ADR-001: LangGraph over CrewAI / autonomous agents for orchestration

## Status

Accepted

## Context

Solvia's assistant must: classify intent, gate on Open Finance consent,
run deterministic financial tools, extract structured slots across
multiple turns, draft a reply, and enforce compliance guardrails —
always in that general shape, with explicit conditional routing (e.g.
skipping the consent gate for product questions, short-circuiting
out-of-scope/complaint messages straight to a reply). The conversation
must also survive a process restart mid-flow (e.g., while waiting on a
consent confirmation), and every numeric value must come from a
deterministic tool, never from the LLM.

This calls for an orchestration framework that gives explicit control
over state and transitions, not one built around an LLM autonomously
deciding what to do next.

## Decision

Use **LangGraph**: a directed graph of nodes (`router`, `consent_check`,
`financial_analyst`, `offer_simulator`, `responder`, `compliance_guard`)
over an explicit typed state (`apps/agent/state.py`), with conditional
edges encoding the routing matrix in `design.md`, and a Postgres
checkpointer so a conversation resumes from its last state after a
restart — verified directly against a real Postgres instance (task
7.9's integration test), not just an in-memory fake.

## Alternatives considered

- **CrewAI (or a similar "autonomous agent" framework).** Rejected —
  these frameworks are built around agents deciding their own next step
  and tool calls, which fits open-ended research/task-completion work
  well but fights against Solvia's requirements: a fixed compliance
  step that must always run last, a consent gate that must always run
  before financial data access, and calculations that must never be
  delegated to an LLM's own judgment. Getting deterministic ordering and
  guaranteed steps out of an autonomous-agent framework means fighting
  its core abstraction rather than using it.
- **A hand-rolled state machine (no framework).** Rejected — LangGraph
  already provides the two things a hand-rolled version would have to
  reimplement carefully: a well-tested Postgres-backed checkpointer for
  exactly this kind of resumable, multi-turn state, and a typed-state
  graph structure that integrates with LangChain's chat model and
  structured-output interfaces (used throughout `apps/agent/nodes/`).
- **A single mega-prompt with no graph at all.** Rejected — this is
  precisely the "the LLM never computes values, it only calls tools"
  requirement's opposite: it would blur classification, tool use, and
  compliance enforcement into one non-deterministic call with no
  guaranteed steps.

## Consequences

- Every node is a plain async function over typed state
  (`ConversationState`, a `TypedDict`), independently unit-testable with
  the fake LLM adapter (see `specs/llm-gateway/spec.md`), with no
  framework-specific mocking needed.
- The checkpointer requirement is explicit and satisfied by
  `apps/agent/checkpointer.py`, with an allowlist for the custom
  Pydantic types in state (`SimulationSlots`, `CustomerProfileSummary`,
  `SimulationSummary`) rather than relying on LangGraph's permissive
  (and, per its own deprecation notice, soon-to-be-blocked) default
  deserialization behavior.
- Human-in-the-loop (interrupt/resume for handoff) is a natural
  extension of this same graph in `add-human-handoff` — LangGraph's
  `interrupt`/`Command` primitives are designed for exactly this
  continuation pattern, so no re-architecture is expected there.

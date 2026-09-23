# regulatory-knowledge-agent Specification

## Purpose
Answers regulatory, Open Finance, and product questions strictly from retrieved corpus chunks, citing the specific norm and article behind every claim, and refuses to answer when retrieval does not support one — never from the LLM's own unverified knowledge.

## Requirements

### Requirement: Answers are grounded exclusively in retrieved chunks
The system SHALL construct every regulatory or product-question answer only from the content of chunks retrieved for that question, and SHALL NOT answer such a question from the LLM's general knowledge.

#### Scenario: Answer content traces to retrieved chunks
- **WHEN** the knowledge agent answers a regulatory or product question
- **THEN** every factual claim in the answer is traceable to the content of a chunk retrieved for that question

### Requirement: Every claim is cited by norm and article
The system SHALL render, for every claim in a regulatory answer, a citation identifying the source norm and article it comes from, built from the retrieved chunk's own metadata rather than generated freely by the LLM.

#### Scenario: Citation reflects chunk metadata
- **WHEN** the knowledge agent's answer includes a citation
- **THEN** the citation's norm and article identifiers are taken from the metadata of a chunk that was actually retrieved for that answer

### Requirement: Unsupported or invented citations are rejected
The system SHALL validate that every citation appearing in a drafted answer refers to a chunk that was part of the retrieved set for that question, and SHALL drop or regenerate the answer when a citation does not.

#### Scenario: Citation not in the retrieved set is caught
- **WHEN** a drafted answer contains a citation to a norm/article that is not among the chunks retrieved for that question
- **THEN** the system does not return that answer as-is; it drops the unsupported citation or regenerates the answer

### Requirement: Refuses when retrieval does not support an answer
The system SHALL reply that the information is not available in the knowledge base and suggest rephrasing, instead of answering, whenever the best-matching retrieved chunk's similarity to the question falls below a configured threshold or the retrieved chunks do not support an answer to the question asked. This similarity is a direct measure of how closely the chunk matches the question (e.g. vector similarity, optionally corroborated by a full-text match), never a rank-based fusion score, since a fusion rank reflects relative ordering among retrieved chunks and not whether any of them are actually relevant.

#### Scenario: Low-similarity retrieval yields a refusal, not a guess
- **WHEN** the best-matching retrieved chunk's similarity to the question falls below the configured threshold for a regulatory or product question
- **THEN** the knowledge agent replies that the information is not available in the knowledge base and suggests rephrasing, rather than attempting an answer

#### Scenario: Retrieved chunks unrelated to the question yield a refusal
- **WHEN** chunks are retrieved but do not address the substance of the customer's question
- **THEN** the knowledge agent declines to answer rather than answering from unrelated content or its own knowledge

### Requirement: Standalone retrieval-and-answer HTTP endpoint
The system SHALL expose the retrieval-and-answer chain via a standalone HTTP endpoint, independent of the conversational graph and its Server-Sent Events contract, so the grounded-answer behavior can be exercised directly.

#### Scenario: Standalone endpoint answers independently of a conversation
- **WHEN** a request is made to the standalone retrieval-and-answer endpoint with a question, without an associated conversation id
- **THEN** the endpoint returns a grounded answer with citations, applying the same grounding, citation, and refusal behavior as the conversational path

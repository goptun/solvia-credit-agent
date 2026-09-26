## MODIFIED Requirements

### Requirement: Retrieval quality evaluation
The system SHALL provide a committed, versioned set of about 100 Portuguese evaluation questions — most with an expected source document and one or more acceptable article references, and a smaller subset that are deliberately unanswerable from the corpus (including near-miss in-domain questions) and expect a refusal — with every question paraphrased rather than copied verbatim from source article text, and the evaluation command (`evaluation` capability) SHALL report recall@k and Mean Reciprocal Rank (MRR) over the answerable questions plus refusal accuracy over the unanswerable ones, against the current index.

#### Scenario: Evaluation reports recall@k and MRR
- **WHEN** the retrieval evaluation command runs against the committed question set's answerable questions
- **THEN** it reports recall@k and MRR computed from whether and where each question's expected document/article appears in the retrieved results

#### Scenario: Evaluation reports refusal accuracy on unanswerable questions
- **WHEN** the retrieval evaluation command runs against the committed question set's deliberately unanswerable questions
- **THEN** it reports refusal accuracy: the fraction of those questions for which the system correctly declines to answer

#### Scenario: Questions are paraphrased, not copied from source text
- **WHEN** an evaluation question is added to the committed set
- **THEN** its wording is a paraphrase of how a customer would actually ask, not a verbatim copy of the cited article's own sentence

### Requirement: Automated tests never require the corpus or embedding model to be fetched
The system SHALL run its default automated unit and integration tests using small committed fixtures and a fake embeddings adapter, and SHALL NOT download the corpus or an embedding model as part of that suite. A separate, path-scoped offline evaluation job MAY download and cache the embedding model from the public model hub, but SHALL NOT download the corpus, call the gateway, or require any secret; it indexes a committed, processed fixture of the corpus instead.

#### Scenario: CI runs without network access to the corpus or model
- **WHEN** the automated test suite runs in CI
- **THEN** it completes using committed fixtures and the fake embeddings adapter, without fetching the corpus or downloading an embedding model

#### Scenario: The offline evaluation job uses the fixture, not the corpus
- **WHEN** the offline evaluation job indexes documents for retrieval evaluation
- **THEN** it indexes the committed chunk fixture and does not fetch the corpus from its official sources

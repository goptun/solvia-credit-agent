## Purpose

Provides the measuring instrument every quality change is judged against: versioned evaluation datasets, reproducible per-component metrics with confidence intervals, a deterministic offline regression gate for pull requests, and budgeted, contamination-checked live runs against the gateway whose results can become committed baselines.

## ADDED Requirements

### Requirement: Versioned, schema-validated evaluation datasets
The system SHALL keep its evaluation datasets in the repository as versioned files, each validated against a schema by automated tests, and SHALL contain no real personal data.

#### Scenario: A malformed dataset item fails validation
- **WHEN** a dataset item is missing a required field, has an unknown label, or references a document that does not exist
- **THEN** the automated tests fail and name the dataset and item

#### Scenario: Datasets carry a version
- **WHEN** a dataset's content changes
- **THEN** its version changes with it, and every report and baseline records the dataset version it was produced from

#### Scenario: No real personal data
- **WHEN** a dataset contains a personal identifier (CPF, phone, email, account or card number)
- **THEN** it is a synthetic or reserved-format value, never a real person's data

### Requirement: Retrieval and grounding dataset composition
The system SHALL provide a retrieval/grounding dataset of about 100 Portuguese questions: about 75 answerable questions spread across every corpus document and the product catalog, and about 25 unanswerable questions of which at least 15 are near-miss in-domain questions the corpus does not cover.

#### Scenario: Answerable questions are stratified
- **WHEN** the dataset is validated
- **THEN** at least 40% of the answerable questions are tagged colloquial, some are written without accents, some list more than one acceptable article reference, and every corpus document and the product catalog each have questions

#### Scenario: Every item is tagged
- **WHEN** a question is added to the dataset
- **THEN** it carries a style (lexical or colloquial), its target document (or none for unanswerable), and a difficulty derived by the documented labelling rule

#### Scenario: Evidence quotes are verifiable
- **WHEN** an answerable question is validated
- **THEN** its short literal evidence quote is found in the text of the expected document and article, and validation fails if it is not

### Requirement: Router, slot-extraction and compliance datasets
The system SHALL provide a router dataset of about 60 messages labelled with an intent (covering all six intents, ambiguous messages, and active-flow continuations), a slot-extraction dataset of about 30 loan requests with expected amount, term and amortization type (including missing and invalid slots), and a compliance dataset of about 40 draft replies labelled promise, hedge or neutral (including adversarial phrasings and the fail-closed examples) plus personal-data masking cases with the expected masked output.

#### Scenario: Router set covers every intent and continuations
- **WHEN** the router dataset is validated
- **THEN** each of the six intents has items, and it includes short active-flow continuation messages that must not be reclassified as new requests

#### Scenario: Slot set records missing and invalid values
- **WHEN** a slot item omits a value or gives an invalid one
- **THEN** its expected value for that field is recorded as absent, so extraction is scored on not inventing values

#### Scenario: Compliance set contains the fail-closed examples
- **WHEN** the compliance dataset is validated
- **THEN** it includes approval promises that reuse words such as analysis and simulation, and explicitly hedged sentences that must not be flagged

### Requirement: Dataset labelling rules and review gate
The system SHALL document the labelling rules of every dataset in a README, and SHALL NOT record a baseline for a dataset version until a stratified sample of 15 items per dataset has been reviewed and approved by the maintainer.

#### Scenario: Reviewers get a stratified sample
- **WHEN** the review sample command runs for a dataset with a seed
- **THEN** it prints 15 items covering the dataset's tags and labels, and the same seed always prints the same items

#### Scenario: Retrieval items show supporting evidence offline
- **WHEN** the review sample command runs for the retrieval dataset
- **THEN** each printed item is accompanied by the top three chunks the retrieval index returns for it (document, article reference, similarity and the first 200 characters of the chunk), computed from the committed corpus fixture without any gateway call, so the maintainer can check that near-miss questions are not answered by the corpus

#### Scenario: Baseline refused for an unapproved dataset version
- **WHEN** a baseline update is requested for a dataset whose current content does not match the approved version recorded in the repository
- **THEN** the update is refused and says the dataset must be reviewed again

### Requirement: Retrieval metrics
The system SHALL report retrieval recall@k and MRR overall and broken down by style and by document, the distribution of best-match similarity for answerable versus unanswerable questions, and a table of false-refusal rate versus unanswerable-refusal rate across a range of similarity thresholds, without choosing a threshold.

#### Scenario: Recall and MRR by stratum
- **WHEN** the retrieval suite runs
- **THEN** recall@k and MRR are reported overall, per style, and per document, counting a question as a hit when any acceptable article reference of the expected document is retrieved

#### Scenario: Threshold trade-off table
- **WHEN** the retrieval suite runs
- **THEN** it reports, for each threshold in a documented range, the share of answerable questions that would be refused and the share of unanswerable questions that would be refused, and it does not recommend or apply a threshold

### Requirement: Grounding end-to-end metrics
The system SHALL report, for the grounding suite, the false-refusal rate on answerable questions, refusal accuracy on unanswerable questions split into far and near-miss, citation validity, the expected-reference hit rate, and a decomposition of each false refusal by cause.

#### Scenario: Refusal decomposition by cause
- **WHEN** an answerable question is refused end to end
- **THEN** it is attributed to exactly one cause: refused by the similarity threshold, the expected source was not retrieved, or the model refused although the expected source was in the retrieved context

#### Scenario: Citation validity
- **WHEN** an answer cites a reference
- **THEN** the run records whether that reference exists in the retrieved context of that same turn, and reports the share of answers whose every citation does

### Requirement: Classification, extraction and compliance metrics
The system SHALL report router accuracy with a per-intent confusion matrix, slot exact-match per field, precision and recall of promise detection separately for the keyword screen, the strict fail-closed screen and the LLM check, and exact-match of personal-data masking.

#### Scenario: Router confusion matrix
- **WHEN** the router suite runs
- **THEN** it reports overall accuracy and a matrix of expected versus predicted intent

#### Scenario: Screens are scored independently
- **WHEN** the compliance suite runs
- **THEN** precision and recall of promise detection are reported separately for each screen, and the strict screen's false-positive rate on hedge and neutral replies that mention approval is reported

#### Scenario: Masking exact match
- **WHEN** a masking case is evaluated
- **THEN** it passes only if the masked output equals the expected masked output exactly

### Requirement: Operational metrics with the resolved model
The system SHALL report, per node, the single-attempt latency (p50 and p95), the number of attempts, the JSON-fallback rate and the no-tool-call rate, and SHALL record the resolved underlying model for every LLM call, including calls made through structured output.

#### Scenario: Resolved model captured for structured-output calls
- **WHEN** a node makes a structured-output call during a live run
- **THEN** the run records the underlying model the gateway reported for that call, not only for plain completions

#### Scenario: No-tool-call and fallback rates
- **WHEN** a structured-output call answers without a tool call or falls back to JSON mode
- **THEN** it is counted in that node's no-tool-call and JSON-fallback rates

### Requirement: Confidence intervals on proportions
The system SHALL report a confidence interval alongside every proportion metric, computed with a method suited to small samples.

#### Scenario: Small-sample proportion
- **WHEN** a proportion is reported from a small sample
- **THEN** it is shown with its confidence interval so a difference of one or two items is not read as a change

### Requirement: Single evaluation command with reproducible output
The system SHALL provide one command that runs selected suites in offline or live mode with deterministic seeds and machine-readable output.

#### Scenario: Suites are selectable
- **WHEN** the command is run with one or more suite names and a mode
- **THEN** only those suites run, and the output is a machine-readable report that records the mode, git commit, dataset versions and seed

#### Scenario: Live runs can be stratified samples
- **WHEN** a live run is started with a sample fraction and a seed
- **THEN** it runs a stratified, seeded subset of each selected dataset covering every stratum it can, records the fraction and seed in its report, and the same fraction and seed always select the same items

#### Scenario: Offline runs are reproducible
- **WHEN** an offline suite is run twice on the same commit with the same inputs
- **THEN** the reported metrics are identical

### Requirement: Offline mode needs no gateway and no new secrets
The system SHALL run every offline suite without calling the gateway and without any secret beyond what continuous integration already has.

#### Scenario: Offline suites make no gateway call
- **WHEN** an offline suite runs in continuous integration
- **THEN** no request is made to the gateway and no credential is read

### Requirement: Offline regression gate on pull requests
The system SHALL compare offline metrics to a committed baseline with explicit per-metric tolerances and SHALL fail the pull request check on a regression beyond tolerance, printing the metric-by-metric difference.

#### Scenario: Regression beyond tolerance fails the check
- **WHEN** an offline metric is worse than its baseline by more than its tolerance
- **THEN** the check fails and prints the baseline value, the new value, the tolerance and the difference

#### Scenario: Improvement is reported, not enforced
- **WHEN** an offline metric is better than its baseline by more than its tolerance
- **THEN** the check passes and notes that the baseline can be updated

#### Scenario: Local retrieval runs are informational
- **WHEN** the retrieval suite runs outside continuous integration
- **THEN** its metrics and its difference from the baseline are printed for information, the run never fails a check because of that difference, and it cannot be used to record a baseline

#### Scenario: No baseline yet
- **WHEN** a suite has no committed baseline
- **THEN** the check passes with a prominent notice that no baseline exists, and it starts comparing as soon as one is committed

#### Scenario: Deterministic checks have zero tolerance
- **WHEN** a deterministic metric (keyword screen results, masking exact-match) changes at all
- **THEN** the check fails

### Requirement: Offline gate scope
The system SHALL run the deterministic compliance and masking checks on every pull request, and SHALL run the retrieval evaluation only when retrieval code, evaluation code or the datasets change, using a committed processed fixture of the corpus so the corpus itself is never downloaded.

#### Scenario: Compliance checks always run
- **WHEN** any pull request is opened
- **THEN** the keyword compliance and masking checks run against their baseline

#### Scenario: Retrieval evaluation runs only when relevant
- **WHEN** a pull request changes none of the retrieval code, the evaluation code or the datasets
- **THEN** the retrieval evaluation is skipped and the check still reports success

#### Scenario: Corpus fixture stays in sync with the manifest
- **WHEN** the recorded source hash of a document in the fixture differs from the corpus manifest
- **THEN** the automated tests fail, telling the maintainer to rebuild the fixture

### Requirement: Live mode is local and never part of pull request checks
The system SHALL run live suites (router, slots, compliance LLM check, grounding end to end) only on demand against the gateway, and SHALL NOT run them in pull request continuous integration.

#### Scenario: Live suites are not run by continuous integration
- **WHEN** a pull request check runs
- **THEN** no live suite is started and no gateway call is made

### Requirement: Live run call budget and pacing
The system SHALL enforce a configurable hard budget on gateway calls per live run and SHALL pace calls, so a run cannot exhaust the provider's quota.

#### Scenario: Budget exhausted stops the run
- **WHEN** a live run reaches its call budget before completing its suites
- **THEN** it stops making calls, marks the report incomplete, and that run cannot become a baseline

#### Scenario: Calls are paced
- **WHEN** a live run makes consecutive gateway calls
- **THEN** a configured minimum interval separates them

#### Scenario: Estimate before spending
- **WHEN** a live run is started
- **THEN** it states the expected number of gateway calls for the selected suites and refuses to start if that exceeds the budget

### Requirement: Per-call logging and contamination detection in live runs
The system SHALL log, for every gateway call in a live run, its status and resolved underlying model, and SHALL mark the run contaminated when the share of rate-limit or unavailability responses exceeds a threshold, when calls resolve to models outside the expected set, or when the share resolved to fallback models exceeds a threshold.

#### Scenario: Rate limiting contaminates a run
- **WHEN** the share of calls that received a quota or unavailability response exceeds the configured threshold
- **THEN** the run is marked contaminated with the reason

#### Scenario: Unexpected model mix contaminates a run
- **WHEN** calls resolve to underlying models outside the expected set for that alias
- **THEN** the run is marked contaminated and the unexpected models are listed

#### Scenario: A contaminated run is still reported
- **WHEN** a run is contaminated
- **THEN** its report is written and clearly marked contaminated, and it is never accepted as a baseline

### Requirement: LangFuse dataset mirroring and live run results
The system SHALL mirror each dataset item to a LangFuse dataset idempotently, and SHALL publish each live run to LangFuse as a dataset run with per-item scores, without placing secrets or infrastructure details in any payload.

#### Scenario: Mirroring is idempotent
- **WHEN** the mirror command runs twice on an unchanged dataset
- **THEN** the second run creates no duplicate items

#### Scenario: Live results are published as dataset runs
- **WHEN** a live run completes and LangFuse credentials are configured
- **THEN** the run appears as a dataset run with scores per item and its aggregate metrics

#### Scenario: Missing LangFuse credentials do not fail the run
- **WHEN** a live run completes without LangFuse credentials configured
- **THEN** the committed report is still written and the missing publication is logged

### Requirement: Baselines are explicit, uncontaminated and traceable
The system SHALL store one committed baseline per suite, updated only through an explicit command, only from a complete, unsampled and uncontaminated run of an approved dataset version, and SHALL record the git commit, dataset version, gateway alias and resolved-model mix in it. The deterministic compliance baseline MAY be recorded locally, but the offline retrieval baseline SHALL be recorded only from the run produced by continuous integration, because embedding results can differ between processor architectures.

#### Scenario: Update from a clean run
- **WHEN** the baseline update command is given an uncontaminated, complete run of an approved dataset version
- **THEN** the suite's baseline is rewritten with the metrics, tolerances, git commit, dataset version, alias and resolved-model mix

#### Scenario: Update refused for a contaminated or incomplete run
- **WHEN** the baseline update command is given a contaminated or incomplete run
- **THEN** it is refused and states why

#### Scenario: Offline retrieval baseline only from a continuous-integration run
- **WHEN** a baseline update for the offline retrieval suite is requested from a local run, or from a run artifact that was not produced by continuous integration, or whose commit is not an ancestor of the current commit with only baseline files changed since
- **THEN** it is refused and says the baseline must come from the continuous-integration run of the code it will gate

#### Scenario: Update refused for a sampled run
- **WHEN** the baseline update command is given a run that used only a sample of a dataset
- **THEN** it is refused and states that baselines need the full suites

#### Scenario: Update refused from a dirty working tree
- **WHEN** the baseline update command runs with uncommitted changes
- **THEN** it is refused, so a baseline always maps to a real commit

### Requirement: Reports for pull requests and the README
The system SHALL generate a concise Markdown report of a run's metrics with their confidence intervals, suitable for a pull request description, and SHALL be able to refresh the metrics section of the README from the committed baselines.

#### Scenario: Report tables
- **WHEN** the report command runs on a run or baseline
- **THEN** it produces Markdown tables of the metrics per suite with confidence intervals and, when a baseline is given, the difference from it

#### Scenario: README metrics section
- **WHEN** the README refresh command runs
- **THEN** only the delimited metrics section is rewritten from the committed baselines

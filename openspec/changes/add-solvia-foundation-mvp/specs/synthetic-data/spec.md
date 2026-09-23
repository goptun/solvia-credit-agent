## Purpose

Provides a reproducible, synthetic Open Finance Brasil–shaped dataset (customers, accounts, transactions, credit cards, consents) so the agent and its tools can be developed, tested, and demoed without any real customer data.

## ADDED Requirements

### Requirement: Reproducible generation
The system SHALL generate the synthetic dataset deterministically from a fixed seed, such that repeated generation runs with the same seed produce identical output.

#### Scenario: Same seed produces identical dataset
- **WHEN** the generator is run twice with the same fixed seed
- **THEN** both runs produce byte-for-byte identical customer, account, transaction, credit-card, and consent records

### Requirement: Dates derived from a fixed reference date
The system SHALL derive every generated date (account opening dates, transaction dates, consent grant/expiry dates, etc.) from a fixed reference date held in configuration, and SHALL NOT derive any generated date from the current system clock.

#### Scenario: Generation is independent of wall-clock time
- **WHEN** the generator is run at two different real-world moments with the same seed and the same configured reference date
- **THEN** all generated dates are identical between the two runs

### Requirement: Customer population and profile coverage
The system SHALL generate approximately 200 synthetic customers distributed across at least the following profiles: salaried, self-employed, over-indebted, and thin credit file.

#### Scenario: Dataset includes all required profiles
- **WHEN** the dataset is generated
- **THEN** it contains customers belonging to each of the salaried, self-employed, over-indebted, and thin-credit-file profiles

### Requirement: Open Finance Brasil–shaped records
The system SHALL generate accounts, transactions, credit cards, and consents whose fields match the shapes defined by Open Finance Brasil schemas, so downstream tools can consume them without translation.

#### Scenario: Generated account matches expected shape
- **WHEN** an account record is generated for any customer
- **THEN** it includes the fields required by the Open Finance Brasil account schema (e.g., account type, balances, currency)

#### Scenario: Generated consent matches expected shape
- **WHEN** a consent record is generated for any customer
- **THEN** it includes the fields required to represent a valid Open Finance consent, including its status and scope

### Requirement: Consent status coverage
The system SHALL generate customers whose synthetic consent status covers at least `valid`, `missing`, and `expired`, so the consent flow can be exercised for each case.

#### Scenario: Dataset includes each consent status
- **WHEN** the dataset is generated
- **THEN** it contains at least one customer with a `valid` consent, one with a `missing` consent, and one with an `expired` consent

### Requirement: Generated output separated from committed fixtures
The system SHALL write generated data to a gitignored output location, distinct from a small, explicitly committed set of fixture records used by automated tests.

#### Scenario: Generation output is not committed
- **WHEN** the generator is run locally
- **THEN** its output lands under the gitignored generated-data location and does not appear in `git status` as a new tracked file

#### Scenario: Tests use committed fixtures, not generated output
- **WHEN** the automated test suite exercises code that depends on synthetic data
- **THEN** it reads the small committed fixture records rather than requiring the generator to have been run

### Requirement: No real customer data
The system SHALL NOT include any real customer, account, or transaction data; all generated PII SHALL be synthetic.

#### Scenario: Generated PII is synthetic
- **WHEN** any customer record is generated
- **THEN** all personally identifiable fields (name, document numbers, contact info) are synthetic values, not derived from any real individual

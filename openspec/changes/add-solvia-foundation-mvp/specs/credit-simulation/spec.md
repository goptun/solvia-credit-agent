## Purpose

Provides deterministic, LLM-free financial calculations (income estimation, debt-to-income, spending categorization, and Price/SAC loan simulation with CET) that the agent calls as tools rather than computing itself, sourcing commercial terms from a fictional, versioned product catalog.

## ADDED Requirements

### Requirement: Deterministic, LLM-free calculations
The system SHALL compute all financial calculations (income estimation, debt-to-income ratio, spending categorization, and amortization simulations) using deterministic Python logic, with no LLM involved in producing the numeric or categorical result.

#### Scenario: Same inputs always produce the same output
- **WHEN** any financial calculation tool is called twice with identical inputs
- **THEN** it returns identical results both times

### Requirement: Income estimation from transaction history
The system SHALL estimate a customer's income from their account and transaction data.

#### Scenario: Income estimated from recurring credits
- **WHEN** a customer's transaction history contains recurring credit deposits
- **THEN** the tool returns an estimated income figure derived from that history

### Requirement: Debt-to-income ratio
The system SHALL compute a customer's debt-to-income ratio from their estimated income and their outstanding debt obligations (including credit card balances).

#### Scenario: Ratio computed from income and debts
- **WHEN** a customer's estimated income and total debt obligations are provided
- **THEN** the tool returns the debt-to-income ratio as a numeric value

### Requirement: Deterministic spending categorization
The system SHALL categorize a customer's transactions into spending categories based on their transaction codes and descriptions, using deterministic rules.

#### Scenario: Transaction categorized by code or description
- **WHEN** a transaction's code or description matches a known spending category pattern
- **THEN** the tool assigns that transaction to the corresponding category

#### Scenario: Unrecognized transaction falls back to a default category
- **WHEN** a transaction's code and description match no known spending category pattern
- **THEN** the tool assigns it to a defined default/uncategorized category rather than failing

### Requirement: Fictional product catalog as the source of commercial terms
The system SHALL source interest rate, IOF (fixed rate plus a daily rate, both capped per current regulation), and fees for any simulation exclusively from a fictional product catalog held in configuration, and the LLM SHALL NOT supply or alter these values.

#### Scenario: Simulation reads terms from the catalog
- **WHEN** a simulation is requested for a given amortization type
- **THEN** the interest rate, IOF parameters, and fees used are read from the product catalog configuration, not from the LLM or the request

### Requirement: CET as annualized IRR of the net cash flow
The system SHALL compute CET (total effective cost) as the annualized internal rate of return of the net cash flow — the amount released to the customer (loan amount minus IOF and fees) against the installment schedule — using `Decimal` arithmetic with documented rounding rules applied only at reported figures.

#### Scenario: CET matches a documented reference case
- **WHEN** a simulation is run with the inputs of a documented reference case
- **THEN** the computed CET matches the reference case's expected CET within the documented rounding tolerance

### Requirement: Price amortization simulation with CET
The system SHALL simulate a loan offer using the Price (constant installment) amortization method and SHALL report the total effective cost (CET) alongside the installment schedule.

#### Scenario: Price simulation returns installment schedule and CET
- **WHEN** a loan amount, term, and the Price amortization type are provided for a simulation
- **THEN** the tool returns the per-installment schedule and the CET for the loan, using catalog-sourced rate/IOF/fees

### Requirement: SAC amortization simulation with CET
The system SHALL simulate a loan offer using the SAC (constant amortization) method and SHALL report the total effective cost (CET) alongside the installment schedule.

#### Scenario: SAC simulation returns installment schedule and CET
- **WHEN** a loan amount, term, and the SAC amortization type are provided for a simulation
- **THEN** the tool returns the per-installment schedule (with decreasing installment values) and the CET for the loan, using catalog-sourced rate/IOF/fees

### Requirement: Invalid simulation inputs are rejected
The system SHALL reject simulation requests with invalid inputs (e.g., non-positive loan amount, non-positive term, negative rate, or an unrecognized amortization type) with a clear error rather than returning a numeric result.

#### Scenario: Non-positive loan amount is rejected
- **WHEN** a simulation is requested with a loan amount of zero or less
- **THEN** the tool raises a validation error and returns no simulation result

#### Scenario: Unrecognized amortization type is rejected
- **WHEN** a simulation is requested with an amortization type other than Price or SAC
- **THEN** the tool raises a validation error and returns no simulation result

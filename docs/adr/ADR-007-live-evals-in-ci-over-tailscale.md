# ADR-007: Running the live evaluation from GitHub Actions over Tailscale

## Status

**Proposed — not implemented.** Nothing in this ADR is built or configured; implementing any option
other than A requires the maintainer's explicit approval and would be a separate OpenSpec change.
Recommended outcome: **A, local only.**

## Context

The live suites (`router`, `slots`, `compliance-llm`, `grounding`) call the `9router` gateway, which is
never exposed publicly and is reachable only through an SSH tunnel from the maintainer's machine. The
repository is public. Running the live suite from GitHub Actions would need a runner that can reach the
gateway, which means joining a private network from CI. See [ADR-006](ADR-006-evaluation-harness.md)
for the harness and its budget and contamination controls.

## Options

- **A. Local only (current).** The maintainer opens the tunnel (`ssh -N solvia-tunnel`), runs the suite
  at most a few times per change, and commits the report and baselines.
- **B. `workflow_dispatch` over an ephemeral Tailscale node.** A manually triggered workflow joins the
  tailnet with an ephemeral, tag-scoped auth key kept in an environment secret, restricted by ACL to
  the gateway's single port, and runs the suite.

## Trade-offs of B on a public repository

- It needs at least two secrets (the tailnet key and the gateway API key), which breaks this change's
  rule that CI needs no new secrets.
- Anyone with write access can edit a workflow on a branch and try to exfiltrate secrets, unless the
  workflow is pinned to `main` and the secrets live in an *environment* with required reviewers.
- A runner on the tailnet is a lateral-movement path if the ACL is ever loosened; an ephemeral key
  expires, but a leaked one is usable until then.
- Unattended runs would consume the shared free-tier quota, reproducing the contamination problem
  ([ADR-005](ADR-005-llm-latency-and-timeouts.md)).
- Runner egress adds latency variance that pollutes the latency metrics.
- Benefit: repeatable runs from a clean machine, independent of the maintainer's laptop.

## Decision

Stay local-only (A). Revisit only when live evaluation is run often enough to justify the risk. If it is
pursued, do it as a separate change with: environment protection with required reviewers, a one-hour
ephemeral tagged key, an ACL to one port, a `main`-only workflow, and a hard call budget.

## Consequences

CI never touches the gateway; live baselines are refreshed by the maintainer on demand, and the
`Evals (offline)` job remains the only evaluation gate on pull requests.

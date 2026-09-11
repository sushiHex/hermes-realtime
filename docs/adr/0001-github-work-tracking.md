# ADR 0001: GitHub owns actionable work

Status: Accepted by the project owner. [Adoption record](https://github.com/sushiHex/hermes-realtime/issues/45).

## Context

The public collaborator roadmap combined design constraints, implementation
history, and an actionable backlog. Maintaining work order and completion status
in both Markdown and GitHub would give contributors competing records.

## Decision

| Artifact | Responsibility |
| --- | --- |
| GitHub Issues | Actionable work, discussion, ownership, dependencies, and acceptance criteria |
| GitHub Milestones | Groups of issues that deliver an observable outcome |
| Reviewed repository plans and accepted ADRs | Design intent and accepted decisions |
| Pull requests and implementation records | Changes, review, and completion evidence |
| Optional GitHub Project | A view of the same issues |

One work item has one issue; one decision has one authoritative document.
Use native issue dependencies and sub-issues. Keep implementation work,
investigations, and owner decisions distinct. Tracking a proposal never approves
its design, and closing an investigation does not imply that a feature qualified.

Existing contracts remain authoritative in their implementation guides. New
cross-cutting plans can live under `docs/plans/`; accepted architectural decisions
live under `docs/adr/`. Link existing authority rather than copying it into a new
document. Explicit exclusions and rejected designs remain in those records until
a concrete, authorized proposal warrants a new decision.

Repository Markdown must not duplicate mutable issue state or become a second
backlog. Historical candidate and qualification records remain useful evidence;
they do not determine current ownership or priority. A Project board and custom
synchronization service are unnecessary for this migration.

## Consequences

Contributors and agents query GitHub to select, claim, hand off, and close work.
The [tracking guide](../work-tracking.md) supplies the operational workflow and
links. Issue forms prompt for scope, authority, and completion evidence; they are
not approval or enforcement mechanisms. Existing review, privacy, cancellation,
and qualification requirements remain in force.

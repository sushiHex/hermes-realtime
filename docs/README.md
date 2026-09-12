# Collaborator guide

Start with the [project overview](../README.md) for installation and supported-use
boundaries. Use these pages to choose work and understand the implementation:

| Question | Start here |
| --- | --- |
| What should we work on next? | [GitHub Issues](https://github.com/sushiHex/hermes-realtime/issues) and [Milestones](https://github.com/sushiHex/hermes-realtime/milestones); follow the [tracking guide](work-tracking.md) |
| What exists, and what has been verified? | [Implementation status](implementation-status.md): source, tests, defaults, and outstanding qualification |
| How do I contribute a change? | [Contributing](../CONTRIBUTING.md): setup, public-repository boundaries, and PR expectations |
| Which engineering rules apply? | [Development guide](../AGENTS.md): protocol, ownership, cancellation, and change discipline |

## Implementation guides

| Area | Guide |
| --- | --- |
| Host, browser, and media composition | [Local LiveKit and full-host setup](local-livekit.md) |
| Hermes dispatch, approvals, and cancellation | [Hermes bridge and API integration](hermes-bridge.md) |
| Foreground lookup, consent, and latency | [Source-backed routing](source-backed-latency.md) |
| Candidate-bound source conversation comparison | [Deterministic equivalence](deterministic-equivalence.md) |
| Wheel binding, revocation, admission closure, and purge | [Packaged revocation race](revoke-race.md) |
| Session budgets, durable rollover, and successor conversation | [Packaged capacity rollover](capacity-rollover.md) |
| Capture queue overflow, durable exclusion, and rejected-source absence | [Packaged admission overflow](over-budget-turn.md) |
| Spool crash boundaries, durable seals, and recovery purge | [Packaged spool crash matrix](spool-crash-matrix.md) |
| Synthetic capacity, filtering, rollback, SQLite faults, and blocked drain | [Synthetic fault matrix](synthetic-fault-matrix.md) |
| Exact artifact deletion, preserved decoys, and idempotent purge | [Full-purge cleanup](full-purge-cleanup.md) |
| Close retries, caller cancellation, and paired conversation facts | [Owned-close faults](owned-close-faults.md) |
| Evidence consent, admission, storage, and purge | [Evidence capture boundary](evidence-capture.md) |
| Frozen inputs, scenario authority, operator observations, and cleanup | [Slice 0 execution protocol](qualification-execution.md) |
| Admitted tools, retained input files, independent builds, and installed runtimes | [Qualification input bindings](qualification-input-files.md) |
| Packaging, CI, publication, and installed-path qualification | [Release gates](release-gates.md) |
| Checkpoint exit diagnostics and their evidence limits | [Windows child-exit investigation](windows-checkpoint-child-exit.md) |

Detailed contracts belong in these guides. Reviewed plans and accepted
[architectural decisions](adr/0001-github-work-tracking.md) hold design authority;
issues link to them rather than repeating their contracts.

## Working together

Read the owning issue, or PR for an incidental correction, along with its
prerequisites, ownership, and linked design before proposing a change.
Use one bounded issue per substantive outcome and link its PR.
Check [open PRs](https://github.com/sushiHex/hermes-realtime/pulls) for overlapping
work. The [tracking guide](work-tracking.md) explains triage and handoffs; an issue
or milestone does not approve a design or reserve work.

Update implementation evidence with the change that affects it. Link the relevant source and tests,
record the candidate and run supporting any new qualification claim, and state
what remains unverified. Follow the [documentation maintenance rules](../CONTRIBUTING.md#documentation-maintenance)
to keep plans, implementation, and evidence aligned.

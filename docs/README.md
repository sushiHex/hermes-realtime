# Collaborator guide

Start with the [project overview](../README.md) for installation and supported-use
boundaries. Use these pages to choose work and understand the implementation:

| Question | Start here |
| --- | --- |
| What should we work on next? | [Roadmap](roadmap.md): ordered milestones, dependencies, and completion criteria |
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
| Evidence consent, admission, storage, and purge | [Evidence capture boundary](evidence-capture.md) |
| Packaging, CI, publication, and installed-path qualification | [Release gates](release-gates.md) |
| Checkpoint exit diagnostics and their evidence limits | [Windows child-exit investigation](windows-checkpoint-child-exit.md) |

Detailed contracts belong in these guides. The roadmap links to them rather than
repeating timeout values, schemas, or operational procedures.

## Working together

Choose the earliest unblocked roadmap item and check its implementation status
before proposing new code. Use a [bounded issue](https://github.com/sushiHex/hermes-realtime/issues)
to discuss substantial changes and link the implementation PR to that issue.
Check [open PRs](https://github.com/sushiHex/hermes-realtime/pulls) for overlapping
work; a roadmap entry does not reserve work or authorize a merge.

Update status with the change that affects it. Link the relevant source and tests,
record the candidate and run supporting any new qualification claim, and state
what remains unverified. Follow the [documentation maintenance rules](../CONTRIBUTING.md#documentation-maintenance)
to keep plans, implementation, and evidence aligned.

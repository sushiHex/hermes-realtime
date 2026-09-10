# Packaged capture admission overflow

[Collaborator guide](README.md) Â· [Qualification status](implementation-status.md#qualification-producer-status)

The `over_budget_turn` producer exercises the ordinary capture queue's real
64-record admission limit during a valid conversation turn. It compares the
same synthetic conversation with capture disabled and with participant consent,
using the source-bound Pure candidate wheel and owned Windows process harness.
Capture remains disabled by default. [Issue #35](https://github.com/sushiHex/hermes-realtime/issues/35)
tracks the implementation and exact qualification evidence.

This is a bounded queue-admission proof. It does not qualify the separate
cumulative 16 MiB per-turn quota. Session-capacity rollover has its own
[producer](capacity-rollover.md); neither scenario substitutes for the other.

## Conversation and admission

The first response generates 80 distinct short segments and delivers them through
the real LiveKit transport. A second ordinary input and response must also
complete. The producer holds the real writer's first ordinary completion until
both conversations finish, then releases the writer and observes its durable
prefix. It delegates all production operations; no queue, quota, counter,
admission decision, or timeout is replaced.

The independent parent compares both complete production traces: committed
context, 81 generated segments, 81 transport confirmations, and normal host
return. Each accepted input must appear as the latest user message in its own
production inference snapshot; later segments may evict older messages from the
unchanged bounded context window. All browser completions and successful close
stages are required.

The production capacity trace must show 58 ordinary admissions, one refusal at
64 reserved records and 59 physical records, 59 ordinary completions, and a
completed owner drain with zero remaining record or byte credits. Queue space
becoming available cannot repair a source publication that was already lost.

When a one-shot generated or transport-confirmed publication is refused for
capacity, admission permanently marks that capture session incomplete. Once
the turn completes, is cancelled, or fails, the exact live evidence lease can
retire without creating an accepted evidence terminal. It releases unused terminal credits and
advances any already-durable revocation waiting for that lease. Other live
operations retain their own leases; healthy delivery-count validation and
retryable command/terminal queue behavior remain unchanged. The
[admission implementation](../src/hermes_realtime/evidence/admission.py) and its
[regression tests](../tests/evidence/test_capture_admission_overflow.py) define
that runtime behavior.

## Durable evidence and rejected source

The observer reads the real SQLite store independently before and after owned
close. Both reads must agree on one active epoch and one open session with 61
events: the two opening records, the first turn's opening and accepted typed
input, and its first 57 generated segments. No transport, snapshot, settlement,
seal, conflict, or erasure record may appear. Retained text without the required
completed turn and sealed-session authority is persisted but excluded from
eligible evidence.

The shared reader validates canonical payload bytes, full reconstructed
snapshots, HRE1 chains, aggregates, consent and binding lineage, retention,
installation/epoch authority, and the durable clock high-water. Accepted HTTP
consent and typed-input provenance provide independent source anchors. The
create command also binds to the agreeing browser-session and LiveKit
participant identities and generations read from their actual owners before
and after consent. The child exports both roles separately at both reads; the
parent requires complete matching role and create-generation commitments
unchanged across consent.
The actual host consent callback participant, generation, request, and consent
are read by the existing writer factory and independently matched to those
live observations and the accepted HTTP request.

The real queue is observed before the dispatcher strips each envelope. Its
create item uses ordinal 2, ordinary records use 3 through 61, and the final
owner drain uses 62 with watermark 61. Complete payloads must match transport
dispatch and SQLite spool receipt; full ordinary snapshots must match the
independently reconstructed durable records. The exact SQLite daemon, factory
and spool calls, dispatcher, and event loop retain distinct owner threads.
The runtime owner must call the exact production dispatcher method, bound to
the actual queue, admission controller, and observed transport. These bindings
are checked after activation and after close. Both retained writer threads must
be stopped without a dispatcher failure, SQLite sticky fault, or suppressed
spool-close exception. Only keyed
thread commitments and bounded call stages leave the child.

After close, the producer scans every present file allowed by the evidence-file
manifest for 25 rejected source values: the 24 generated values after the
accepted prefix, including the follow-up response, and the second user input.
Unknown files, indirect files, missing database, an exceeded scan bound, or any
match block acceptance. Previously accepted generated text is not classified as
absent merely because its later transport-evidence record was refused; absence
of transport records is checked separately.

Raw content, SQL, identifiers, timestamps, paths, and the invocation's ephemeral
HMAC key remain inside the child. The parent receives bounded keyed observations
and derives exactly `persisted_but_excluded` and `rejected_source_absent`.

The [worker](../scripts/over_budget_turn_worker.py),
[independent validator](../scripts/over_budget_turn.py), and
[shared observations](../scripts/evidence_observation.py) implement these checks;
[producer tests](../tests/test_over_budget_turn.py) exercise their refusal boundary.

## Candidate and ownership boundary

The producer uses the same [package/source binding](revoke-race.md#package-and-source-binding)
and retained-process cleanup boundary as revocation and rollover. Every runtime
source blob in the supplied Pure wheel must match the exact committed archive.
Normal worker exit, all five teardown milestones, waits on retained processes,
complete Windows Job cleanup, and owned workspace removal remain required.

On Windows with the locked development environment:

```powershell
uv run --frozen --group dev python -m scripts.qualify_over_budget_turn `
  --candidate . --baseline origin/main `
  --candidate-wheel $candidateWheel --wheel-sha256 $candidateWheelSha256 `
  --livekit-executable $livekitExecutable --livekit-sha256 $livekitSha256
```

Only independent acceptance produces `over-budget-turn-summary-v1`, containing
the exact source commit, tree, archive/wheel/observation digests, process count,
and derived assertions. Native CI runs this producer after source equivalence,
revocation, and rollover, consuming the same Pure candidate wheel.

Sixteen governed producers and the remaining installed and human-assisted
prerequisites remain unavailable or unqualified. These results cannot establish
an accepted full `qualification-report-v1`, offline installed dependency closure,
physical audibility, or general browser readiness.

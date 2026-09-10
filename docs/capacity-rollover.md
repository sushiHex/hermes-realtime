# Packaged capacity rollover

[Collaborator guide](README.md) · [Qualification status](implementation-status.md#qualification-producer-status)

The `capacity_rollover` producer runs three ordinary typed turns through the real
consented host, LiveKit transport, admission controller, and SQLite writer from
the source-bound Pure candidate wheel. The existing per-turn session reservations
trigger rollover after the second turn. No quota, counter, timeout, or production
lifecycle is replaced. [Issue #33](https://github.com/sushiHex/hermes-realtime/issues/33)
tracks the bounded implementation and exact qualification evidence.

## Durable transition and source equality

A delegating observer runs on the real SQLite owner thread. It reads the store
through a separate read-only connection before rollover and at SQLite's trace
callback immediately before the real `COMMIT`. Both readers must see the same
open predecessor with 14 events. The callback retains only transaction-control
names and write verbs; SQL statements and their values never leave the child.
Every mutating statement must run while the real connection is inside the one
observed transaction. Writes before its begin or after its commit, unknown
statement kinds, and writable PRAGMAs block acceptance. The two exact read-only
quota PRAGMAs remain allowed. A scoped Python audit observes connection opens
throughout the real rollover call, including aliases and other threads. Only
the observer's one read-only callback connection is allowed; any other open
blocks acceptance. The audit does not change SQLite execution or retain paths
or handles. Callback failure
blocks acceptance even though SQLite itself suppresses callback exceptions.

After that single transaction returns, another read must see the sealed
predecessor with 16 events and its open successor with two opening events. The
third ordinary turn must finish in the successor. The observer waits for the
real writer's third settlement result before reading its eight durable events;
foreground completion alone does not imply writer completion. The sealed
predecessor and the successor's opening chain must remain unchanged.

Every store snapshot checks SQLite integrity, foreign keys, exact canonical
payload bytes, contiguous sequences, recomputed HRE1 hashes, session aggregates,
and terminal authority. Each snapshot independently checks the installation's
clock high-water against the latest durable event time, including immediately
after rollover, before a later turn can repair a missed update. Installation and
epoch opening times must match the original session; purge authority is forbidden
in this fresh scenario. Event times cannot move backward, and all four rollover
control records share one timestamp. Only keyed commitments to clock and store
authority cross the child boundary. The static event validator checks each session's turn
history; separate rollover checks bind its close/seal records and the successor
to the complete consent envelope and production binding: consent version,
disclosure digest, retention, source acceptance, and source availability. A
separate commitment preserves that envelope across all four observations.
Another commitment is captured from the exact consent request bytes dispatched
through the browser API, only after its accepted acknowledgment. Every stored
session must match that independent source commitment. Consistently rewriting
all store consent fields and recomputing their hashes cannot replace the request
that was actually accepted. The create command also binds to the generation
read from the existing browser session and LiveKit owners. They must agree
in both participant identity and generation before and after consent; the
browser snapshot is read under its real authority lock. The keyed pair must
also remain unchanged across consent. Both anchors are independently compared in the parent. This establishes the observed live
activation; it does not qualify reconnect or every stale-binding rejection.
The reader also requires one active epoch and no
conflicts or pending/completed erasure authority in this fresh scenario.

Before transport delegation and again on the SQLite owner thread, observers
commit the complete create and rollover command DTOs. The accepted DTO retained
by SQLite must also match. No command field is omitted: request sequence and
fingerprint, admission ordinal, identities, expiry, and all nested snapshots are
bound. The create request sequence/fingerprint must match the exact accepted
browser request. Ordered dispatch must match every actual record ordinal,
including the two ordinals consumed by atomic epoch creation.
The observer delegates the real blocking dequeue and checks its complete queue
envelope before the dispatcher strips it: protocol, lane, ordinal, and the
complete payload commitment. The create item must use ordinal 2, rollover 15,
and drain 22 with watermark 21. Ordinary and rollover payload ordinals must
match their envelopes; payload commitments must also match transport dispatch.
The queue and dispatcher retain their production implementations.
The SQLite owner-thread observer also commits every complete ordinary record
before calling the real spool. Those DTOs must match dequeue and transport;
their full snapshots must match independently reconstructed durable snapshots,
including event identity and every payload field. Every append must commit.
Validly rehashing a substituted event cannot satisfy this comparison.
The transport must be the exact production SQLite daemon. Factory and spool
calls are checked against its retained thread object, separately from the
observed dispatcher and event-loop threads. The dispatcher is bound to the real
runtime owner after activation. Its callable must be the exact production
dispatcher method, with the observed queue, runtime admission controller, and
delegating transport. Those bindings must still match after close; arbitrary
callbacks cannot substitute for typed dispatch and admission completion.
After host close, both retained worker threads
must actually be stopped without a recorded dispatcher exception or sticky
SQLite fault. A suppressed spool-close exception also blocks acceptance; the
observer records the failure before the daemon can suppress it. Only keyed
thread commitments, bounded call stages,
and observed stop states cross the child boundary; thread IDs and names do not.

The observer also commits complete dispatched control snapshots, including event
identity, binding identity and generation, source availability, and consent lineage. The reader reconstructs
those snapshots from SQLite; both observer and parent compare their commitments
with the dispatched commands. The successor's stored expiry must match the
rollover command's exact deadline. Canonical opening and expiry timestamps,
opening-event timestamp equality, the initial retention interval, and unchanged
timestamps across subsequent observations are also required. The successor's
retention interval must match accepted consent, allowing only zero to five
seconds between the runtime's deadline sample and the writer's opening sample.
The deadline cannot extend consent; a shorter interval beyond the existing
five-second observation authority also fails. Command/store agreement alone
cannot establish retention policy. A consistent
rewrite within the store cannot substitute different command authority.

Session IDs, timestamps, and raw records stay in the child. An invocation-local HMAC key commits session lineage, chain entries,
and content without exporting that key or identifiers.

The independent parent validator compares the persisted user, generated, and
transport-confirmed text commitments with the accepted inputs and actual
production conversation observations. Each user commitment binds source type
together with text to the accepted typed HTTP input, so a valid, consented
microphone record containing the same text cannot substitute for that input.
It requires all five ordered rollover
stages, three successful settlements and durable writer results, no queue
rejections, complete traces, all twelve successful close stages, and released
capacity. It derives exactly `persisted_source_equal` and `rollover_atomic`.
Missing, duplicate, reordered, partial, foreign, or contradictory observations
cannot produce an accepted receipt.

## Candidate and process ownership

The producer shares the [packaged binding and ownership boundary](revoke-race.md#package-and-source-binding)
with revocation. The runner, source archive, and pure wheel must match the same
clean committed candidate. The parent verifies its executing acceptance modules
against that archive, checks imported source locations, and revalidates the
materialized source and wheel after execution. Normal worker exit, waits on all
retained processes, complete Windows Job cleanup, and workspace removal remain
required. Raw content exists only in the owned child workspace and memory.

On Windows with the locked development environment:

```powershell
uv run --frozen --group dev python -m scripts.qualify_capacity_rollover `
  --candidate . --baseline origin/main `
  --candidate-wheel $candidateWheel --wheel-sha256 $candidateWheelSha256 `
  --livekit-executable $livekitExecutable --livekit-sha256 $livekitSha256
```

The command prints `capacity-rollover-summary-v1` only after independent
acceptance. It includes the exact source commit, tree, archive digest, wheel
digest, observation digest, process count, and derived assertions. The Native CI
job consumes the existing Pure candidate wheel and runs this producer after
source equivalence and revocation.

This qualifies one successful session-capacity transition. Queue saturation,
cumulative turn-budget overflow, crash recovery, and physical audibility remain
separate claims. The inherited development dependencies do not establish an
offline installed dependency closure or native-library provenance. Seventeen
governed producers and the remaining installed and human-assisted prerequisites
are still unavailable or unqualified; these partial results cannot establish an
accepted full `qualification-report-v1`.

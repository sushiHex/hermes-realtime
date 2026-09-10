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
names; SQL statements and their values never leave the child. Callback failure
blocks acceptance even though SQLite itself suppresses callback exceptions.

After that single transaction returns, another read must see the sealed
predecessor with 16 events and its open successor with two opening events. The
third ordinary turn must finish in the successor. The observer waits for the
real writer's third settlement result before reading its eight durable events;
foreground completion alone does not imply writer completion. The sealed
predecessor and the successor's opening chain must remain unchanged.

Every store snapshot checks SQLite integrity, foreign keys, exact canonical
payload bytes, contiguous sequences, recomputed HRE1 hashes, session aggregates,
and terminal authority. The static event validator checks each session's turn
history; separate rollover checks bind its close/seal records and the successor
to the complete consent envelope and production binding: consent version,
disclosure digest, retention, source acceptance, and source availability. A
separate commitment preserves that envelope across all four observations.
Another commitment is captured from the exact consent request bytes dispatched
through the browser API, only after its accepted acknowledgment. Every stored
session must match that independent source commitment. Consistently rewriting
all store consent fields and recomputing their hashes cannot replace the request
that was actually accepted. The reader also requires one active epoch and no
conflicts or pending/completed erasure authority in this fresh scenario.

Before delegating epoch creation and rollover, the observer commits the complete
dispatched control snapshots, including event identity, binding identity and
generation, source availability, and consent lineage. The reader reconstructs
those snapshots from SQLite; both observer and parent compare their commitments
with the dispatched commands. The successor's stored expiry must match the
rollover command's exact deadline. Canonical opening and expiry timestamps,
opening-event timestamp equality, the initial retention interval, and unchanged
timestamps across subsequent observations are also required. A consistent
rewrite within the store cannot substitute different command authority.

Session IDs, timestamps, and raw records stay in the child. An invocation-local HMAC key commits session lineage, chain entries,
and content without exporting that key or identifiers.

The independent parent validator compares the persisted user, generated, and
transport-confirmed text commitments with the accepted inputs and actual
production conversation observations. It requires all five ordered rollover
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

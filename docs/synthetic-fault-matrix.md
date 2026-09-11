# Synthetic fault and capacity matrix

[Collaborator guide](README.md) · [Execution protocol](qualification-execution.md)

The `synthetic_fault_matrix` producer occupies canonical ordinal 15 with proof
class `synthetic_injected`. It drives five closed injections against the real
candidate scheduler, SQLite spool, and writer daemon. [Issue #59](https://github.com/sushiHex/hermes-realtime/issues/59)
records the implementation and exact candidate evidence. Capture remains disabled
by default; this fixture creates synthetic consent and source records privately.

## Five independently checked cases

| Case | Real operation and required observations |
| --- | --- |
| `queue_capacity_coupled` | The production scheduler reaches 64 record credits and 64 physical items, refuses the next source as `record_capacity`, persists all 64 accepted records, and releases every credit. |
| `deny_filter` | A generated credential-shaped synthetic input is refused by the real filter. The session is durably tainted, the rejected source is absent, and full purge completes. |
| `clock_rollback` | An injected clock regression latches durable purge authority. Recovery with a later fixture clock completes physical purge before any new capture. The OS clock is unchanged. |
| `sqlite_injected_fault` | One event insertion raises an injected SQLite error. The real transaction rolls back, the writer latches its fault, the rejected source is absent, and full purge completes. |
| `writer_drain_blocked` | A bounded gate holds the real daemon's drain while its owned caller remains pending. Release delegates to the production drain; both threads finish before purge and process completion. |

The capacity fixture first commits the two opening records, a turn, and one
synthetic typed input. It primes the real scheduler's ordinal allocator with that
committed prefix, without assigning counters or changing limits. It then queues
64 generated records, observes refusal of the next, and drains the accepted
prefix through the real spool. Independent reads require exactly 68 durable
events with the complete expected source and valid record chain.

The parent derives capacity algebra from the observed production configuration:
64 records, 64 physical items, 32,768 maximum canonical bytes per record, and
2,097,152 aggregate canonical bytes. The aggregate ceiling equals the lawful
record-count product; neither aggregate bytes nor physical items can overflow
before the record limit for this ordered record path. The fixture's actual queue
charge is independently reconstructed from its canonical snapshots. It never
lowers limits to manufacture separate overflows.

This is a synthetic scheduler/spool proof. It does not establish full-host
conversation equivalence, cumulative 16 MiB turn-quota exhaustion, physical audio,
browser behavior, or real hardware clock/disk failures. Those retain their own
governed scenarios and evidence boundaries.

## Source, storage, and cleanup authority

The [worker](../scripts/synthetic_fault_worker.py) delegates production operations.
The [independent reader](../scripts/synthetic_fault_oracle.py) imports no candidate
models or persistence implementation: it verifies the pinned schema, canonical
payloads, HRE1 record chains, exact synthetic source, session aggregates, consent
lineage, retention and clock ordering through separate read-only SQLite access.
Sentinel bytes are inspected after their owner releases the live lock.

The [parent validator](../scripts/synthetic_fault_matrix.py) requires all five
cases in exact order and derives the seven governed assertions. It rejects
missing observations, altered source, wrong capacity/refusal, incorrect taint or
purge authority, incomplete drain, and changed cleanup. Nested booleans cannot
stand in for integers. Labels, supplied success flags and serialized observations
cannot mint a producer receipt.

Each case runs in a fresh retained Windows process using the existing
[storage owner](../scripts/storage_process.py) and source-bound Pure wheel. The
archive authority binds every executing script; the wheel authority binds runtime
blobs. Five deliberately synthetic sidecars accompany the real database during
purge, providing all six deletion inputs; they make no journal-recovery claim.
Before deletion, the parent retains their NTFS identities and a live marker as a
positive control. After deletion, each identity must be retired, the marker and
adjacent decoys must remain unchanged, and the live child must have released
storage. Normal exit, retained-handle waits, zero active Job processes, and owned
workspace removal are required before a receipt is issued.

Failure preserves the existing private-workspace retention limit in
[the execution protocol](qualification-execution.md#finalize-validate-and-retain).
This producer does not implement the complete runner's durable recovery journal.
Do not call an unrecorded failed workspace automatically recoverable or publish
its raw files. Public summaries contain only permitted source/artifact/observation
digests, counts, proof class, and derived assertion codes.

## Run the exact candidate

On Windows with the locked development environment and the exact Pure wheel:

```powershell
uv run --frozen --group dev python -m scripts.qualify_synthetic_fault_matrix `
  --candidate . --baseline origin/main `
  --candidate-wheel $candidateWheel --wheel-sha256 $candidateWheelSha256
```

Only independent acceptance emits `synthetic-fault-matrix-summary-v1`. The Native
CI job exercises this producer against the same candidate wheel as the other
packaged scenarios. [Tests](../tests/test_synthetic_fault_matrix.py) include real
Windows cases and adversarial observation/receipt refusals; test-only observations
are not candidate qualification evidence.

An accepted matrix remains one bounded scenario. Full installed dependency
closure, all other scenarios, operator observations, and complete report
acceptance must refer to one frozen candidate before desktop qualification can be
claimed. This matrix alone cannot produce an accepted full qualification report.

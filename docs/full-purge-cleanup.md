# Full-purge cleanup

[Collaborator guide](README.md) · [Qualification status](implementation-status.md#qualification-producer-status)

The `full_purge_cleanup` producer exercises the real V1 full-purge operation on
Windows. Its two separately owned workers import the candidate's source-bound
Pure wheel. It occupies governed ordinal 18 with proof class
`real_windows_filesystem`; scenario IDs and report fields are unchanged.

## Fixture and execution

The first worker creates a real consent epoch with its two opening events, then
closes the fixture owner. It seeds five synthetic database sidecars, three decoys
in the evidence root, and two decoys in an adjacent directory. Two root decoys
deliberately resemble database and journal prefixes.

An independent observer checks the closed fixture before the worker calls
`SQLiteEvidenceSpool.purge_full_store` with the exact V1 command. It requires
`purge_completed`, closes the owner, and observes the result. The second worker
opens the same fixture in a fresh process, calls the command again, and requires
`already_absent` with an unchanged durable image.

The [process owner](../scripts/storage_process.py) uses explicit normal-exit
operations for these workers. Their expected exit code is zero. Before allowing
each worker to exit, the parent independently opens the root and remaining files
with exclusive sharing while the retained process is still alive. A missing file
or retained owner refuses acceptance; termination cannot supply release evidence.
Both workers enter their private Windows Jobs before executing, and both must be
waited, leave empty Jobs, and close all retained process handles.

## Independent acceptance

The [observer](../scripts/full_purge_observation.py) uses the retained-handle
[Windows storage oracle](../scripts/windows_storage_oracle.py), without candidate
probe or manifest imports. Its bounded file inventory, path, stream, link, volume,
and permission checks retain the [spool observer's documented profile](spool-crash-matrix.md).
SQLite inspection opens a disposable database-only copy outside the evidence
root. The deliberately synthetic sidecars are deletion inputs, not journal
recovery evidence; the crash observer's journal behavior is unchanged.

The [validator](../scripts/full_purge_cleanup.py) requires:

- The exact two-event source history, schema and installation authority, using
  the independent storage reader and fixed V1 source commitments.
- All six exact database artifact names before purge and their absence afterward,
  with no added files or deletion of prefix lookalikes.
- The unchanged root marker, exact clear sentinel image including both slots,
  and unchanged bytes for every root and adjacent-directory decoy.
- The exact two phase results, fresh process identities, live release observations,
  normal exits, and complete process cleanup.
- Equality between the first final image and both observations of the repeated
  purge, making idempotence a durable-state check.

The producer pins the complete reviewed V1 report contract and binds its executing
scripts, candidate source archive, and Pure wheel. Supplied observations or success
flags cannot construct an accepted producer receipt. The registry follows canonical
scenario-ID order, so adding this producer does not shift another ordinal.

## Running and evidence limits

Use a clean committed checkout and its Pure candidate wheel:

```powershell
python -m scripts.qualify_full_purge_cleanup `
  --candidate <checkout> `
  --candidate-wheel <candidate-wheel> --wheel-sha256 <sha256>
```

The Native CI job uses the wheel artifact from the same workflow. Its summary
contains source, archive, wheel and observation commitments, two owned processes,
and the derived assertions `adjacent_decoys_preserved`, `purge_verified`, and
`purged`. It contains no transcript, host path, or process identifier. Successful
temporary workspaces are removed; failed workspaces remain private for diagnosis.

This fixture does not qualify initialization-temporary deletion, an active host's
consent/revocation orchestration, arbitrary sidecar contents, volume-full behavior,
installed-host recovery, or physical devices. Those contracts remain separate.
Capture remains disabled by default. Fourteen producers remain unavailable and
full Slice 0 acceptance remains pending.

[Issue #41](https://github.com/sushiHex/hermes-realtime/issues/41) tracks exact
candidate and run evidence. Issues #13, #24, and #27 retain their historical evidence
limits; the separate close defect in #40 belongs to the owned-close work.

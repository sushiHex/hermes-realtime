# Packaged spool crash recovery

[Collaborator guide](README.md) · [Qualification status](implementation-status.md#qualification-producer-status)

The `spool_crash_matrix` producer exercises the 41 governed storage checkpoints
in both prescribed exit modes: child `os._exit(197)` and parent
`TerminateProcess(..., 198)`. Each crash is followed by recovery in a fresh
process. The four rollback checkpoints additionally use both caught-up and
regressed restart clocks: 82 governed cases, 90 recovery attempts, and 180
separately retained processes.

## Execution and acceptance

The [process owner](../scripts/storage_process.py) verifies the executing scripts
against the candidate archive and uses the existing source-bound pure-wheel
capability. Children import the wheel's runtime and the archive's test driver.
The Windows GUI interpreter avoids an incidental console helper; protocol I/O
uses two explicitly inherited pipes. Each child enters its private Job before
its first instruction. The parent binds its checkpoint to the retained process
identity, checks the prescribed exit code, waits the process, observes an empty
Job, and closes owned handles. Recovery requires normal exit zero.

The [driver](../scripts/storage_worker.py) uses the production Windows storage
probe. It delegates the named durable operations to the existing real SQLite
crash driver. The [observer](../scripts/storage_observation.py) separately reads
the filesystem and database before and after production recovery. SQLite itself
may recover a hot rollback journal during the observer's cold database open;
the observer issues no application SQL writes.

The [validator](../scripts/spool_crash_matrix.py) requires the complete ordered
matrix. It checks the exact precommit event counts and unsealed sessions,
revalidates committed event hashes using an [independent HRE1 oracle](../scripts/evidence_protocol_oracle.py)
pinned to fixed protocol vectors, checks complete seal lineage and recovery
erasure receipts, and verifies sentinel states and each full-purge deletion
prefix. Each stored event must also match one of seven [pinned synthetic source
events](../scripts/spool_crash_oracle.py), independently of the candidate's
payload parser. Purge must leave every database artifact absent while preserving the
root marker and adjacent decoys. Malformed initialization images must be refused
without mutation; completed initialization temporaries must be removed. The
empty, partial, complete, and flushed images are pinned by independent V1 byte
encoding. Erasure receipt commitments bind every column, including the erased
epoch, request identity, control fingerprint, admission authority, and timestamp;
pending requests and logically deleted requests likewise bind their complete
authority. Receipt identities remain inside the reader; exported receipts carry
only state/reason, deletion counts, and commitments. Final root-marker bytes
and both sentinel slots are independently pinned at every boundary, including
their generations, predecessor slots, and fixture authority IDs. The
original logical database must remain unchanged through every pre-latch rollback
checkpoint. The sentinel commitment must remain unchanged before its write.

The rollback fixture deliberately includes an older closed epoch alongside an
active epoch. The existing driver creates that older epoch through the real
spool in a separate database, then seeds its rows into the recovery fixture.
Independent chain validation checks the resulting history. This establishes
recovery behavior for the fixture; it does not establish that ordinary consent
can create multiple epochs in one store.

## Running the producer

Use a clean committed candidate and its source-bound pure candidate wheel:

```powershell
python -m scripts.qualify_spool_crash_matrix `
  --candidate <checkout> `
  --candidate-wheel <candidate-wheel> --wheel-sha256 <sha256>
```

The Native release job runs this command using the Pure candidate wheel artifact
from the same workflow. The summary contains source commit, tree, archive and
wheel digests, an observation digest, process count, and derived machine
assertions. It contains no transcript, output, host paths, or process identifiers.
Successful workspaces are removed; failed runs retain their private temporary
workspace for investigation. No workflow rerun or timeout change supplies
acceptance.

## Evidence boundary

This producer covers the governed packaged spool crash class. It does not supply
the separate installed-host crash, full-purge cleanup, owned-close fault,
Windows filesystem/volume-full, or physical qualification producers. Capture
remains disabled by default and full Slice 0 acceptance remains pending.

Implementation and exact candidate/run evidence are tracked in
[issue #38](https://github.com/sushiHex/hermes-realtime/issues/38). The historical
causes in issues #13, #24, and #27 remain unproven.

# Packaged spool crash recovery

[Collaborator guide](README.md) · [Qualification status](implementation-status.md#qualification-producer-status)

The `spool_crash_matrix` producer exercises the 41 governed storage checkpoints
in both prescribed exit modes: child `os._exit(197)` and parent
`TerminateProcess(..., 198)`. Each crash is followed by recovery in a fresh
process. The four rollback checkpoints additionally use both caught-up and
regressed restart clocks: 82 governed cases, 90 recovery attempts, and 180
separately retained processes.

The V1 report contract pins the actual sentinel state at each recovery-entry
checkpoint: clear, absent final sentinel, first-create pending, clock-rollback
pending, or full-purge pending. This corrects the earlier placeholder that
required `clear` for every case. Case identities and order are unchanged. The
producer pins the complete reviewed report-schema commitment, so changing any
case field or transitive constraint requires an explicit producer review.

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
the filesystem and database before and after production recovery. SQLite opens
only a disposable copy of the database and rollback journal, outside the evidence
root. The original crash image remains untouched for production recovery.

Filesystem acceptance uses an [independent Windows observer](../scripts/windows_storage_oracle.py),
with no candidate probe or artifact-manifest imports. It retains the root and
each leaf without delete sharing while checking the exact paths, fixed local
volume, reparse attributes, single file links, bounded sizes, streams, and DACLs.
The bounded fixture profile permits ordinary allow ACEs for the current user,
SYSTEM, Administrators, and Windows owner identities; other ACE forms or trustees
refuse qualification. Owner identities must resolve to the permitted owner set.
This is a fixture security check, not a general Windows permission evaluator.
The API authorities are Microsoft's [handle security descriptor](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo),
[stream enumeration](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-findfirststreamw),
and [owner identity](https://learn.microsoft.com/en-us/windows-server/identity/ad-ds/manage/understand-special-identities-groups) contracts.

The [validator](../scripts/spool_crash_matrix.py) requires the complete ordered
matrix. It checks the exact precommit event counts and unsealed sessions,
revalidates committed event hashes using an [independent HRE1 oracle](../scripts/evidence_protocol_oracle.py)
pinned to fixed protocol vectors, checks complete seal lineage and recovery
erasure receipts, and verifies sentinel states and each full-purge deletion
prefix. Each stored event must also match one of seven [pinned synthetic source
events](../scripts/spool_crash_oracle.py), independently of the candidate's
payload parser. Each checkpoint also pins the complete ordered source history,
so another valid fixture event or session cannot substitute for its input.
Those commitments include absolute timestamps from the fixed driver clock;
installation commitments also pin creation time and the clock high-water state.
Persisted session and epoch identities, consent version,
retention expiry, and lifecycle timestamps must agree with those source events;
the fixture permits no conflict rows or orphan epochs. Purge must leave every database artifact absent while preserving the
root marker and adjacent decoys. Malformed initialization images must be refused
without mutation; completed initialization temporaries must be removed. The
ordinary recovered store must retain exactly its database, final authority
files, and original decoys, with no journal, vacuum, or initialization residue. The
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
The full-purge latch checkpoint retains the complete two-event source history
and installation authority, and its database bytes must match a commitment
recorded before latching. Its deliberately synthetic sidecars are deletion
fixtures; this checkpoint inspects a database-only copy. Every committed schema,
including empty creation checkpoints, must match an independent V1 commitment
covering all 25 `sqlite_master` entries, their exact SQL, and persisted page size,
encoding, auto-vacuum, journal mode, application ID, and user version.

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

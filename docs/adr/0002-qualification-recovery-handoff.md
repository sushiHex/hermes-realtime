# ADR 0002: Recovery refuses rather than resumes

Status: Proposed for review. [Work item](https://github.com/sushiHex/hermes-realtime/issues/81).
Reviewed source: [`431e465f`](https://github.com/sushiHex/hermes-realtime/commit/431e465f4b0f14d603bd437849b2f078e5ac6558),
the merged commit of [#76](https://github.com/sushiHex/hermes-realtime/pull/76) and the newest
commit touching the process owner, the run journal and the execution protocol.
Nothing below is implemented; this record maps a handoff for a later slice of
[#62](https://github.com/sushiHex/hermes-realtime/issues/62).

## Context

The Windows process owner and the private run journal merged as independent
primitives with no code path between them. The owner's `resume_root()` docstring
states that it "supplies no durable journal, consent, or dispatch authority", and
neither module imports the other. Connecting them is where a recovery design can
go wrong in one specific way: by letting a durable record stand in for a live
capability, so that a restarted controller believes it may adopt a child, assert
an exit, or accept a scenario it did not observe.

The accepted [execution protocol](../qualification-execution.md#finalize-validate-and-retain)
already fixes the launch shape. A launch intent is durable before creation; the
child is created suspended with its noninherited kill-on-close Job associated at
creation; the observed identity is durable before resume. The protocol draws the
consequence: a controller death before the identity update "leaves an unresumed
child owned by the closing Job, never an unassigned running worker."

That consequence is stronger than it first appears, and it is what makes this
design small.

## Decision

The kernel decides process liveness. The journal decides only whether an
attempt's effects were possible. Recovery is therefore a refusal function over
recorded facts, not a controller that adopts, resumes, or reconciles a process.

Because the Job carries `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` and is associated at
creation, a controller death at any boundary terminates the child. No recovery
path ever needs to find, adopt, or kill a process. What survives a restart is not
a process question but an effects question: could this child have executed
instructions before it died? The journal answers exactly that and nothing more.

The liveness guarantee is scoped to the single-Job topology the owner implements.
It sets no breakaway limit and never queries whether the controller is itself
already inside a Job, so nested-Job composition is neither checked nor reasoned
about here. The claim is also specifically about termination on last-handle
close, not about Job membership queries proving that a process exited; the owner
handles that separately for its own bookkeeping, and recovery never inspects a
process at all.

### Ordered boundaries

Each row is one boundary, the live capability at it, the durable transition that
brackets it, and what is true if the controller dies there.

| # | Boundary | Live capability | Durable transition | On controller death |
| --- | --- | --- | --- | --- |
| 0 | Bind journal to run and recovery location | none | `opened` | Attempt never started |
| 1 | Before creating an owned file | — | `filesystem_intent` → `intended` | File may exist; the intent names it |
| 2 | After creation | — | `filesystem_bound` → `bound` | File exists at a recorded identity |
| 3 | Creation did not occur | — | `filesystem_absent` → `absent` | Nothing to clean |
| 4 | Before `create_process_suspended` | `launch_root_suspended()` unused | `process_intent` → `intended` | Child may exist, suspended in the Job; dies with it |
| 5 | After creation, before resume | Suspended root retained; resume possible | `process_bound` → `bound` | Child terminated without executing |
| 6 | Creation failed | Attempt consumed | `process_absent` → `absent` | Nothing exists |
| 7 | Immediately before `resume_root()` | Resume still possible | `resume_intent` → `resume_intended`, `execution_uncertain` set | Child may have executed; effects possible |
| 8 | After `resume_root()` returns | Root resumed | `resumed`, uncertainty cleared | Child executed and is terminated |
| 9 | Exit observed | — | `process_exited` from `bound`, `resume_intended` or `resumed`, identity must equal the bound identity | Resolved as an obligation, but see below |
| 10 | Owned file removed | — | `filesystem_removed` | Resolved |
| 11 | Every obligation resolved | `finalize()` | `sequence_complete` | Complete |

Obligations are addressed by the sequence number of the frame that declared them,
so no separate identifier space is introduced. Row 7 is the only boundary that
admits doubt about execution. A restart must reason about that row and,
separately, about whether the journal is readable at all; the dispositions below
keep those two questions apart.

Rows 7 to 9 are not a chain. An exit may be recorded from `bound`,
`resume_intended` or `resumed`, and recording one never clears uncertainty —
only row 8 does. This is the right refusal, because a suspended child that is
terminated also exits, so an exit observation alone cannot say whether the child
ran. It does impose a hard obligation on the implementation: record `resumed`
immediately after `resume_root()` returns. A sequence that reaches an exit from
`resume_intended` satisfies every obligation check and still fails completion at
"sequence has unresolved execution", with no transition able to clear it
afterwards. Resolving a process obligation and resolving doubt are separate
events, and only the second can close a run.

### Dispositions

| State | Recorded evidence | Disposition |
| --- | --- | --- |
| Same-process retry | Writer live and unpoisoned | Permitted for cleanup only. `launch_root_suspended()` refuses a second root once `_root_launch_attempted` is set, which happens before the kernel call, so a failed creation consumes the attempt |
| Controller interruption, same process | Writer live | Cleanup proceeds through `finalize()`, and the owner already finalizes itself when its own launch or resume raises. Once finalization starts, resume is permanently revoked, including when cleanup needs a retry. The attempt fails |
| Controller death before `resume_intent` | `execution_uncertain` false; process `intended` or `bound` | The child never executed. Clean recorded filesystem obligations; refuse the attempt. No process action is available or needed |
| Controller death after `resume_intent`, before `resumed` | `execution_uncertain` true | Effects were possible. Refuse the attempt and refuse cleanup acceptance: unknown exit still prevents it |
| Stale or ambiguous identity | Re-queried identity differs from the bound identity | Refuse. `resume_root()` already refuses a changed identity on the live path; recovery must never adopt on a PID alone |
| Failed cleanup | `pending_filesystems` non-empty | Retain the workspace and refuse acceptance, as existing storage producers already do |
| Unconfirmed tail, chain intact | `integrity_complete` true with `unconfirmed_frames` above zero | Reopening for writing refuses. Inspection is trustworthy up to the committed head: obligations recorded there are authoritative and must be cleaned. Frames past the head are never applied and must not be read as facts |
| Unreadable or broken journal | `integrity_complete` false | Reopening refuses and the obligation lists stop being an inventory. A failure found before frame decoding resets the parsed state, so the lists return empty; a failure found mid-sequence keeps only what was applied before the break, so they return partial. Neither is complete. Retain the recovery root as holding unknown residue and refuse; never read a short or empty list as nothing to clean |
| Journal capacity exhausted | Append refused at 2,048 frames or 8 MiB | The writer is not poisoned and stays usable, but no further obligation can be recorded. Since recording precedes creation, refuse all further dispatch, then finalize and clean what is already recorded |
| Successful completion | `recorded_complete` true | The attempt may be accepted on its own producer evidence. The journal contributes none |

### Why no recorded fact is authority

Three claims, each resting on a mechanism rather than on policy.

A durable fact cannot resume a child. `resume_root()` requires that the passed
root be the identical retained object, that the owner still hold its Job handle,
and that both the process and thread handles still be owned. Those are an
in-process object reference and live kernel handles. A restarted controller holds
bytes, so resume is unreachable by construction.

A durable fact cannot establish exit or removal. `recorded_complete` is a
property of the recorded sequence, and the facts type says so of itself:
"Structural private facts only; never cleanup, recovery, or acceptance
authority." A `process_exited` frame records an observation some earlier live
controller made; after a restart no such observation is in hand.

A durable fact cannot mint acceptance. Receipts are minted by the registered
producer that observed the scenario — the deterministic equivalence producer
refuses any receipt it did not mint itself — and no producer imports the journal;
the coupling between the two modules is currently zero in both directions. The
journal is a private side record that cannot enter Git, public reports, logs,
comments, or attachments, so it has no path into an accepted result even if a
later caller wished to give it one.

### Seam

Reuse both primitives unchanged. Add no public surface, no recovery custody
object, no orchestration framework, no caller-supplied cleanup claim, and no
second schema.

The journal exposes public types but no public functions: creation, inspection
and reopening are all private today. That asymmetry is the seam. Bind the restart
path to inspection alone; reopening for writing belongs only to a controller that
is still live and still holds its own handles.

The owner needs no new method. The gap is that `launch_root()` composes creation
and resume with no durable step between them, which is correct for the existing
producers in `scripts/equivalence_process.py`,
`scripts/qualification_tool_process.py` and `scripts/storage_process.py`. They
keep it. A later controller calls `launch_root_suspended()` and `resume_root()`
directly and writes rows 4, 5, 7 and 8 between them.

That ordering is the load-bearing obligation of this whole design, and nothing
enforces it. The owner has no journal awareness and no ordering hook, and the
journal has no knowledge of the kernel calls it brackets. Every row above assumes
the journal write has returned durably before the owner call it precedes. An
implementation that issues the call first, or that treats an unflushed write as
recorded, keeps every individual guarantee here and still loses the property they
were combined to produce. State it as a requirement on the controller and test it
directly.

One naming hazard is worth stating because it inverts the model if missed:
reopening the journal resumes a writer, not a child. A controller that reads a
reopenable journal as a resumable process has granted a record the authority this
record denies it.

### RED cases for the later slice

Each names the production path it must exercise.

1. Recording fails before dispatch; dispatch refuses.
2. `process_intent` durable, controller death before `process_bound`; the restart
   finds no resume intent, refuses, and cleans only recorded filesystem obligations.
3. `resume_intent` durable, controller death before `resumed`; the restart reports
   uncertain execution and refuses both the attempt and cleanup acceptance.
4. A chain-valid tail past the committed head; reopening refuses, and the
   obligations recorded up to the head still read as pending.
5. A journal broken before frame decoding and one broken mid-sequence; both refuse
   reopening, and neither the empty list nor the partial one is accepted as an
   inventory.
6. A bound identity that no longer matches; refusal without PID-only adoption.
7. Completion attempted with a pending obligation; refused.
8. Completion attempted while execution is uncertain; refused.
9. An exit recorded from `resume_intended` without `resumed`; completion refuses
   and no later transition rescues it. This is the case the implementation must be
   written never to reach.
10. The frame or byte bound reached; the append refuses without poisoning the
    writer, and dispatch stops rather than continuing unrecorded.
11. The owner call issued before its journal write is durable; refused, because
    every guarantee above rests on that ordering and nothing else enforces it.

No test should assert that an orphaned running worker is cleaned up after
controller death. The at-creation kill-on-close Job makes that state
unrepresentable, so such a test would assert on the kernel rather than on this
project's code. For the same reason no recovery-side process termination path is
proposed.

## Consequences

Recovery becomes a pure function from recorded facts to a refusal and a list of
filesystem obligations, which is testable without launching a child. The list is
authoritative only while the journal reads as intact; otherwise it is a floor,
not an inventory. An entire
category of machinery — custody transfer, adoption, reconciliation, recovery-side
termination — is not built, because the kernel already guarantees what it would
have provided.

The cost is that row 7 can never be narrowed by better recording. A controller
death between resume intent and observed resume is permanently ambiguous about
effects, and the accepted conservative reading is refusal. Attempts lost that way
are failed attempts, retained with their bounded conclusions, never reclassified
later by a green run.

The existing producers are unaffected: they continue to use the composed launch
and gain no durable recovery authority from these primitives.

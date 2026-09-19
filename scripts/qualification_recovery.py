"""Connect the Windows process owner to the run journal without granting either authority.

[ADR 0002](../docs/adr/0002-qualification-recovery-handoff.md) fixes the shape this module
implements. The kernel decides process liveness: the Job carries
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` and is associated at creation, so a controller death
at any boundary terminates the child. What survives a restart is therefore not a process
question but an effects question, and the journal answers only that.

Two things follow, and they are the whole module. Acquisition has one call site, so the
durable write that brackets each kernel call cannot be reordered or skipped by a caller.
Recovery is a pure function from recorded facts to a refusal, because no recorded fact can
resume a child, establish an exit, or mint acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from scripts.qualification_run_journal import (
    ProcessIdentityFactsV1,
    ProcessIntentV1,
    RunJournalFactsV1,
)


class _RootOwnerV1(Protocol):
    """The owner surface acquisition needs, named so this module cannot reach further."""

    def launch_root_suspended(self) -> object: ...

    def resume_root(self, root: object, /) -> None: ...


class _ProcessRecorderV1(Protocol):
    """The journal surface acquisition needs, in the order the protocol requires it."""

    def intend_process(self, intent: ProcessIntentV1, /) -> int: ...

    def bind_suspended_process(self, ordinal: int, identity: ProcessIdentityFactsV1, /) -> None: ...

    def intend_resume(self, ordinal: int, /) -> None: ...

    def record_resumed(self, ordinal: int, /) -> None: ...


@dataclass(frozen=True, slots=True)
class RecordedRootV1:
    """One resumed root and the journal ordinal that owns its obligations."""

    ordinal: int
    root: object

    def __post_init__(self) -> None:
        if type(self) is not RecordedRootV1 or type(self.ordinal) is not int or self.ordinal <= 0:
            raise TypeError("recorded root is not exact")


class AttemptDispositionV1(StrEnum):
    """What a restarted controller may do with an attempt, which is never to resume it."""

    COMPLETE = "complete"
    REFUSE_RETAINING = "refuse_retaining"
    REFUSE_CLEANING = "refuse_cleaning"


def _identity_of(root: object, scenario_id: str) -> ProcessIdentityFactsV1:
    """Restate the owner's proven root identity in the journal's vocabulary.

    Every field is copied from what the kernel reported through a handle the owner still
    holds. Nothing here observes the process independently, and nothing invents a value.
    """
    identity = root.identity  # type: ignore[attr-defined]
    return ProcessIdentityFactsV1(
        scenario_id,
        root.role,  # type: ignore[attr-defined]
        identity.pid,
        identity.parent_pid,
        identity.parent_creation_filetime,
        identity.creation_filetime,
        identity.image_basename,
        identity.image_sha256,
    )


def acquire_recorded_root(
    owner: _RootOwnerV1, recorder: _ProcessRecorderV1, intent: ProcessIntentV1
) -> RecordedRootV1:
    """Acquire a resumed root whose every boundary is durable before it is crossed.

    The ordering below is the load-bearing obligation ADR 0002 identifies, and nothing in
    either primitive enforces it: the owner has no journal awareness and the journal knows
    nothing of the kernel calls around it. Composing them here is what makes the wrong
    order unreachable, because there is no caller left to interleave them.

    Failure needs no handling, and that is deliberate. A raise anywhere below leaves the
    last durable record short of the boundary it was about to cross, and the owner
    finalizes itself, which permanently revokes resume. A restart then reads a pending
    intent and refuses. Recording an optimistic ``process_absent`` here would replace that
    conservative refusal with a claim this module cannot support: after a failed launch it
    does not know whether a child was created before the owner terminated the Job.
    """
    if type(intent) is not ProcessIntentV1:
        raise TypeError("process intent must be exact")
    ordinal = recorder.intend_process(intent)
    root = owner.launch_root_suspended()
    recorder.bind_suspended_process(ordinal, _identity_of(root, intent.scenario_id))
    recorder.intend_resume(ordinal)
    owner.resume_root(root)
    recorder.record_resumed(ordinal)
    return RecordedRootV1(ordinal, root)


def disposition_of(facts: RunJournalFactsV1) -> AttemptDispositionV1:
    """Decide what a restart may do, from recorded facts alone.

    The order of these tests is their meaning. A journal that does not read as intact
    reports obligation lists that are a floor rather than an inventory, whether they come
    back empty or partial, so its root is retained as unknown residue. Uncertain execution
    survives an intact journal: the child may have run before it died, and an unknown
    effect still prevents cleanup acceptance. Only then can a complete sequence be
    distinguished from one whose recorded obligations are authoritative and clearable.

    No branch resumes anything. Resume requires the identical retained root object and live
    kernel handles, which a restarted controller does not have, so refusal is not a policy
    this function applies but the only outcome available to it.
    """
    if type(facts) is not RunJournalFactsV1:
        raise TypeError("run journal facts must be exact")
    if not facts.integrity_complete:
        return AttemptDispositionV1.REFUSE_RETAINING
    if facts.execution_uncertain:
        return AttemptDispositionV1.REFUSE_RETAINING
    if facts.recorded_complete:
        return AttemptDispositionV1.COMPLETE
    return AttemptDispositionV1.REFUSE_CLEANING

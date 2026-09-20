"""Connect the Windows process owner to the run journal without granting either authority.

The [accepted execution protocol](../docs/qualification-execution.md#finalize-validate-and-retain)
is the authority here. It requires that process acquisition be made recoverable by durably
recording a launch intent before creation, creating the child suspended with its
noninherited kill-on-close Job already associated, persisting the observed creation and
image identity while it remains suspended, and resuming only after that update is durable.
[ADR 0002](../docs/adr/0002-qualification-recovery-handoff.md) maps that requirement onto
the merged primitives and is proposed rather than accepted; nothing below depends on it.

The protocol draws the consequence that makes this module small: a controller death before
the identity update leaves an unresumed child owned by the closing Job, never an unassigned
running worker. What survives a restart is therefore not a process question but an effects
question, and the journal answers only that.

Two things follow. Acquisition has one call site, so the durable write that brackets each
kernel call cannot be reordered or skipped by a caller. Recovery is a pure function from
recorded facts to a refusal, because no recorded fact can resume a child, establish an
exit, or mint acceptance.
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

    def finalize(self) -> object: ...


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


def _identity_of(root: object) -> ProcessIdentityFactsV1:
    """Restate the owner's proven root identity in the journal's vocabulary.

    Every field comes from the owner, including the scenario and role. Taking the scenario
    from the caller's intent instead would launder that value into both sides of the
    journal's own ``process_bound`` comparison, which checks the recorded identity against
    the recorded intent: the check would compare the caller's value with itself and could
    never refuse a root belonging to another scenario. Copying only what the owner proved
    is what leaves that guard able to fire.
    """
    identity = root.identity  # type: ignore[attr-defined]
    return ProcessIdentityFactsV1(
        identity.scenario_id,
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

    The owner finalizes itself only when one of its own operations raises. A recorder that
    raises leaves it untouched, so every such path finalizes here instead. Without that a
    failed ``record_resumed`` would be the worst case in the module: the resume already
    returned, so the child is running, and the caller receives an exception in place of the
    root it needed to shut down. Finalizing on any failure is what keeps "no path leaves a
    live child behind" true rather than merely usual.

    No path records ``process_absent``. A raise leaves the last durable record short of the
    boundary it was about to cross, and a restart reads a pending intent and refuses.
    Claiming absence would replace that conservative refusal with a statement this module
    cannot support: after a failed launch it does not know whether a child was created
    before the Job was terminated.
    """
    if type(intent) is not ProcessIntentV1:
        raise TypeError("process intent must be exact")
    try:
        ordinal = recorder.intend_process(intent)
        root = owner.launch_root_suspended()
        recorder.bind_suspended_process(ordinal, _identity_of(root))
        recorder.intend_resume(ordinal)
        owner.resume_root(root)
        recorder.record_resumed(ordinal)
    except BaseException as primary:
        try:
            owner.finalize()
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "recorded root acquisition and finalization both failed", [primary, cleanup]
            ) from None
        raise
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

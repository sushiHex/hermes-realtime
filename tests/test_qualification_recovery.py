"""The recovery seam records every boundary before crossing it, and never resumes.

These cover the two production paths ADR 0002 names: acquisition, whose only job is an
ordering nothing else enforces, and the restart disposition, whose only job is to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from runpy import run_path

import pytest

from scripts import qualification_recovery as recovery
from scripts import qualification_run_journal as journal

_HELPERS = run_path(str(Path(__file__).with_name("test_qualification_run_journal.py")))


def _intent() -> journal.ProcessIntentV1:
    return journal.ProcessIntentV1(
        scenario_id="deterministic_equivalence",
        role="root",
        image_basename="python.exe",
        image_sha256="a" * 64,
    )


@dataclass(frozen=True, slots=True)
class _Identity:
    scenario_id: str = "deterministic_equivalence"
    pid: int = 4242
    parent_pid: int = 17
    parent_creation_filetime: int = 130_000_000_000_000_000
    creation_filetime: int = 130_000_000_000_000_001
    image_basename: str = "python.exe"
    image_sha256: str = "a" * 64


@dataclass(frozen=True, slots=True)
class _Root:
    identity: _Identity = _Identity()
    role: str = "root"


class _TracingOwner:
    """A root owner that records when it was called, and can fail at one named boundary."""

    def __init__(self, trace: list[str], *, fail_at: str | None = None) -> None:
        self._trace = trace
        self._fail_at = fail_at

    def _step(self, name: str) -> None:
        self._trace.append(name)
        if name == self._fail_at:
            raise RuntimeError(f"owner refused at {name}")

    def launch_root_suspended(self) -> object:
        self._step("launch_root_suspended")
        return _Root()

    def resume_root(self, root: object, /) -> None:
        del root
        self._step("resume_root")

    def finalize(self) -> object:
        self._trace.append("finalize")
        return None


class _TracingRecorder:
    """A journal writer that records its transitions in order, and can fail at one."""

    def __init__(self, trace: list[str], *, fail_at: str | None = None) -> None:
        self._trace = trace
        self._fail_at = fail_at

    def _step(self, name: str) -> None:
        self._trace.append(name)
        if name == self._fail_at:
            raise RuntimeError(f"recorder refused at {name}")

    def intend_process(self, intent: journal.ProcessIntentV1, /) -> int:
        del intent
        self._step("intend_process")
        return 1

    def bind_suspended_process(
        self, ordinal: int, identity: journal.ProcessIdentityFactsV1, /
    ) -> None:
        del ordinal, identity
        self._step("bind_suspended_process")

    def intend_resume(self, ordinal: int, /) -> None:
        del ordinal
        self._step("intend_resume")

    def record_resumed(self, ordinal: int, /) -> None:
        del ordinal
        self._step("record_resumed")


def test_every_boundary_is_durable_before_the_kernel_call_that_crosses_it() -> None:
    """The ordering is the whole point of the seam, so it is asserted exactly.

    Neither primitive can enforce this: the owner has no journal awareness and the journal
    knows nothing of the kernel calls around it. A caller free to interleave them could
    satisfy every individual guarantee and still lose the property they combine to produce.
    """
    trace: list[str] = []
    recorded = recovery.acquire_recorded_root(
        _TracingOwner(trace), _TracingRecorder(trace), _intent()
    )

    assert trace == [
        "intend_process",
        "launch_root_suspended",
        "bind_suspended_process",
        "intend_resume",
        "resume_root",
        "record_resumed",
    ]
    assert recorded.ordinal == 1


def test_a_refused_launch_leaves_the_intent_pending_rather_than_claiming_absence() -> None:
    """After a failed launch this module does not know whether a child was created.

    The owner finalizes itself on failure, which terminates the Job, but recording
    ``process_absent`` would state that creation did not occur. A pending intent instead
    refuses the attempt on restart, which is the conservative reading the protocol requires.
    """
    trace: list[str] = []
    with pytest.raises(RuntimeError):
        recovery.acquire_recorded_root(
            _TracingOwner(trace, fail_at="launch_root_suspended"),
            _TracingRecorder(trace),
            _intent(),
        )

    assert trace == ["intend_process", "launch_root_suspended", "finalize"]


def test_a_refused_resume_leaves_execution_uncertain() -> None:
    """Resume intent is durable before the call, so a death there cannot read as clean.

    The child may have executed before the owner terminated the Job. That doubt is the one
    thing the journal exists to carry, and clearing it requires the ``resumed`` record that
    this path never reaches.
    """
    trace: list[str] = []
    with pytest.raises(RuntimeError):
        recovery.acquire_recorded_root(
            _TracingOwner(trace, fail_at="resume_root"), _TracingRecorder(trace), _intent()
        )

    assert trace == [
        "intend_process",
        "launch_root_suspended",
        "bind_suspended_process",
        "intend_resume",
        "resume_root",
        "finalize",
    ]
    assert "record_resumed" not in trace


def test_a_recorder_failure_after_the_resume_still_finalizes_the_owner() -> None:
    """The owner self-finalizes only for its own failures, so this path must do it.

    ``resume_root`` has already returned, which means the child is running and the owner
    saw nothing go wrong. Raising without finalizing would hand the caller an exception in
    place of the root it needed to shut down, leaving a live worker behind.
    """
    trace: list[str] = []
    with pytest.raises(RuntimeError):
        recovery.acquire_recorded_root(
            _TracingOwner(trace), _TracingRecorder(trace, fail_at="record_resumed"), _intent()
        )

    assert trace[-2:] == ["record_resumed", "finalize"]


def test_finalization_failure_preserves_the_original_refusal() -> None:
    """A cleanup failure must not hide what actually went wrong."""

    class _UnfinalizableOwner(_TracingOwner):
        def finalize(self) -> object:
            raise OSError("finalization refused")

    trace: list[str] = []
    with pytest.raises(BaseExceptionGroup) as raised:
        recovery.acquire_recorded_root(
            _UnfinalizableOwner(trace), _TracingRecorder(trace, fail_at="intend_resume"), _intent()
        )

    assert [type(error) for error in raised.value.exceptions] == [RuntimeError, OSError]


def test_a_root_from_another_scenario_is_refused_by_the_journals_own_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam records what the owner proved, which is what leaves that guard able to fire.

    Copying the scenario from the caller's intent would put the same value on both sides of
    the journal's ``process_bound`` comparison, so a root belonging to another scenario
    would bind under a false identity and then be resumed. A real writer is required here:
    a tracing double cannot refuse anything.
    """

    class _ForeignOwner(_TracingOwner):
        def launch_root_suspended(self) -> object:
            self._step("launch_root_suspended")
            return _Root(_Identity(scenario_id="revoke_race"))

    io = _HELPERS["_FakeIo"]()
    monkeypatch.setattr(journal, "_journal_io", lambda: io)
    writer = journal._create_run_journal(41, _HELPERS["_binding"](), _HELPERS["_location"]())

    with pytest.raises(ValueError, match="process identity differs"):
        recovery.acquire_recorded_root(_ForeignOwner([]), writer, _intent())


def _facts(**overrides: object) -> journal.RunJournalFactsV1:
    base: dict[str, object] = {
        "frame_count": 4,
        "unconfirmed_frames": 0,
        "byte_count": 512,
        "integrity_complete": True,
        "partial_tail": False,
        "recorded_complete": False,
        "execution_uncertain": False,
        "pending_filesystems": (),
        "pending_processes": (),
        "reason": None,
    }
    return journal.RunJournalFactsV1(**(base | overrides))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param(
            {"integrity_complete": False, "reason": "head"},
            recovery.AttemptDispositionV1.REFUSE_RETAINING,
            id="unreadable-journal-reports-no-inventory",
        ),
        pytest.param(
            {"integrity_complete": False, "partial_tail": True, "pending_processes": (1,)},
            recovery.AttemptDispositionV1.REFUSE_RETAINING,
            id="partial-journal-lists-a-floor-not-an-inventory",
        ),
        pytest.param(
            {"execution_uncertain": True, "pending_processes": (1,)},
            recovery.AttemptDispositionV1.REFUSE_RETAINING,
            id="uncertain-execution-prevents-cleanup-acceptance",
        ),
        pytest.param(
            {"pending_processes": (1,), "pending_filesystems": (2,)},
            recovery.AttemptDispositionV1.REFUSE_CLEANING,
            id="intact-and-certain-obligations-are-authoritative",
        ),
        pytest.param(
            {"recorded_complete": True},
            recovery.AttemptDispositionV1.COMPLETE,
            id="a-complete-sequence-needs-no-refusal",
        ),
    ],
)
def test_the_restart_disposition_refuses_on_recorded_facts_alone(
    overrides: dict[str, object], expected: recovery.AttemptDispositionV1
) -> None:
    """Every outcome is a refusal or a completion; none of them resumes anything."""
    assert recovery.disposition_of(_facts(**overrides)) is expected


def test_an_unresumed_acquisition_reads_back_as_uncertain_from_a_real_journal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two halves compose: what acquisition stops recording is what recovery refuses on.

    Driving a real writer rather than a double proves the transition names line up, which a
    tracing recorder cannot establish.
    """
    io = _HELPERS["_FakeIo"]()
    monkeypatch.setattr(journal, "_journal_io", lambda: io)
    writer = journal._create_run_journal(41, _HELPERS["_binding"](), _HELPERS["_location"]())

    with pytest.raises(RuntimeError):
        recovery.acquire_recorded_root(
            _TracingOwner([], fail_at="resume_root"), writer, _intent()
        )

    facts = journal._inspect_run_journal(41, _HELPERS["_binding"](), _HELPERS["_location"]())
    assert facts.execution_uncertain is True
    assert facts.recorded_complete is False
    assert recovery.disposition_of(facts) is recovery.AttemptDispositionV1.REFUSE_RETAINING

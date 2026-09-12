"""At-creation Job membership must precede child execution or owner observation."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import replace
from pathlib import Path
from runpy import run_path
from typing import Any

import pytest


def _helpers() -> dict[str, Any]:
    return run_path(
        str(Path(__file__).resolve().parents[1] / "tests/test_qualify_evidence_slice_zero.py")
    )


def test_suspended_creation_binds_the_job_in_the_native_attribute_list() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    api = fixture["_ConcreteApiShim"]()
    kernel = fixture["_concrete_kernel"](runner, api)

    result = kernel.create_process_suspended(
        ("C:/candidate/python.exe", "-I"),
        (("LANG", "C"),),
        "C:/candidate",
        (11, 12),
        4,
        job=101,
    )

    handle_update = (
        "update",
        0x00020002,
        2 * ctypes.sizeof(ctypes.c_void_p),
        (11, 12),
    )
    job_update = ("update", 0x0002000D, ctypes.sizeof(ctypes.c_void_p), (101,))
    assert [event for event in api.events if event[0] == "initialize"] == [
        ("initialize", False, 2),
        ("initialize", True, 2),
    ]
    assert [event for event in api.events if event[0] == "update"] == [
        handle_update,
        job_update,
    ]
    assert api.events.index(job_update) < next(
        index for index, event in enumerate(api.events) if event[0] == "create_process"
    )
    assert result.pid == 42


@pytest.mark.parametrize("job", [None, False, 0, -1, 11, ctypes.c_void_p(-1).value])
def test_invalid_or_inherited_job_is_rejected_before_process_creation(job: object) -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    api = fixture["_ConcreteApiShim"]()

    with pytest.raises(runner._WindowsScenarioJobError):
        fixture["_concrete_kernel"](runner, api).create_process_suspended(
            ("C:/candidate/python.exe",),
            (("LANG", "C"),),
            "C:/candidate",
            (11, 12),
            4,
            job=job,
        )
    assert api.events == []


@pytest.mark.parametrize("previous", [0, 2, 0xFFFFFFFF])
def test_concrete_resume_requires_exactly_one_previous_suspension(
    monkeypatch: pytest.MonkeyPatch, previous: int
) -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    api = fixture["_ConcreteApiShim"]()
    calls: list[int] = []

    def resume(handle: Any) -> int:
        calls.append(int(handle.value))
        return previous

    api.ResumeThread = fixture["_ApiFunction"](resume)
    kernel = fixture["_concrete_kernel"](runner, api)
    if previous == 0xFFFFFFFF:
        failure = OSError("private ResumeThread failure")
        monkeypatch.setattr(runner.ctypes, "get_last_error", lambda: 5, raising=False)
        monkeypatch.setattr(runner.ctypes, "WinError", lambda _code: failure, raising=False)
        expected: type[BaseException] = OSError
    else:
        failure = None
        expected = runner._WindowsScenarioJobError

    with pytest.raises(expected) as raised:
        kernel.resume_thread(502)

    if failure is not None:
        assert raised.value is failure
    assert calls == [502]


def test_concrete_resume_accepts_one_previous_suspension() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    api = fixture["_ConcreteApiShim"]()
    calls: list[int] = []
    api.ResumeThread = fixture["_ApiFunction"](
        lambda handle: (calls.append(int(handle.value)), 1)[1]
    )

    assert fixture["_concrete_kernel"](runner, api).resume_thread(502) is None
    assert calls == [502]


def test_bad_concrete_resume_count_finalizes_the_owned_suspended_root() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    api = fixture["_ConcreteApiShim"]()
    api.ResumeThread = fixture["_ApiFunction"](lambda _handle: 2)
    concrete = fixture["_concrete_kernel"](runner, api)
    kernel = fixture["_FakeWindowsKernel"](runner)
    kernel.resume_thread = concrete.resume_thread
    job = fixture["_job"](runner, kernel)

    with pytest.raises(runner._WindowsScenarioJobError):
        job.launch_root()

    assert job.last_finalization is not None and job.last_finalization.closed
    assert ("terminate_job", 101) in kernel.events
    assert all(event[0] != "terminate_process_pid" for event in kernel.events)


@pytest.mark.skipif(os.name != "nt", reason="real Windows at-creation Job membership")
def test_real_unresumed_child_is_already_owned_and_dies_when_job_closes(tmp_path: Path) -> None:
    import msvcrt

    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = runner._CtypesWindowsKernelV1()
    job = kernel.create_job(runner._WindowsJobLimitsV1(4, 512 * 1024**2, 512 * 1024**2))
    marker = tmp_path / "child-executed"
    read_fd, write_fd = os.pipe()
    read_handle = msvcrt.get_osfhandle(read_fd)
    os.set_handle_inheritable(read_handle, True)
    child = None
    job_closed = False
    try:
        flags = (
            runner._CREATE_SUSPENDED
            | runner._EXTENDED_STARTUPINFO_PRESENT
            | runner._CREATE_UNICODE_ENVIRONMENT
            | runner._CREATE_NO_WINDOW
        )
        child = kernel.create_process_suspended(
            (
                sys._base_executable,
                "-I",
                "-S",
                "-c",
                "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')",
                str(marker),
            ),
            tuple(
                sorted(
                    (key, value)
                    for key, value in os.environ.items()
                    if key in {"SystemRoot", "WINDIR"}
                )
            ),
            str(tmp_path),
            (read_handle,),
            flags,
            job=job,
        )
        assert kernel.query_job_processes(job) == (child.pid,)
        kernel.close_handle(job)
        job_closed = True
        kernel.wait(child.process_handle, 5_000)
        assert not marker.exists()
    finally:
        if not job_closed:
            kernel.close_handle(job)
        if child is not None:
            kernel.wait(child.process_handle, 5_000)
            kernel.close_handle(child.thread_handle)
            kernel.close_handle(child.process_handle)
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("primary", ["identity:201", "resume"])
def test_post_creation_failure_retains_wait_failure_for_cleanup_retry(primary: str) -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    kernel.fail.update({primary, "wait:201"})
    job = fixture["_job"](runner, kernel)

    with pytest.raises(BaseExceptionGroup):
        job.launch_root()
    assert ("terminate_job", 101) in kernel.events
    assert kernel.events.count(("terminate_job", 101)) == 1
    assert ("close", 201) not in kernel.events

    kernel.fail.difference_update({primary, "wait:201"})
    assert job.finalize().closed
    assert ("close", 201) in kernel.events


@pytest.mark.skipif(os.name != "nt", reason="Windows attribute failure error reporting")
def test_job_attribute_failure_never_creates_a_process_and_releases_the_attribute_list() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    api = fixture["_ConcreteApiShim"]()
    original = api.UpdateProcThreadAttribute

    def update(attributes: object, flags: int, attribute: int, *args: object) -> int:
        if attribute == 0x0002000D:
            return 0
        return original(attributes, flags, attribute, *args)

    api.UpdateProcThreadAttribute = update
    with pytest.raises(OSError):
        fixture["_concrete_kernel"](runner, api).create_process_suspended(
            ("C:/candidate/python.exe",),
            (("LANG", "C"),),
            "C:/candidate",
            (11, 12),
            4,
            job=101,
        )
    assert all(event[0] != "create_process" for event in api.events)
    assert ("delete",) in api.events


def test_controller_can_record_the_suspended_identity_before_resume() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    job = fixture["_job"](runner, kernel)

    root = job.launch_root_suspended()
    assert root.identity.pid == 42
    assert not any(event[0] == "resume" for event in kernel.events)
    kernel.events.append(("identity_recorded", root.identity.creation_filetime))
    job.resume_root(root)

    assert kernel.events[-2:] == [("identity", 201), ("resume", 202)]
    assert kernel.events.index(("identity_recorded", 420)) < kernel.events.index(("resume", 202))
    job.finalize()


@pytest.mark.parametrize(
    "seam", ["suspended_store", "compatibility_handoff", "resume_consume"]
)
def test_control_interruption_during_root_handoff_finalizes_the_owned_child(seam: str) -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    template = fixture["_job"](runner, kernel)
    interruption = KeyboardInterrupt(f"private {seam} detail")

    class InterruptingJob(runner._WindowsScenarioJobV1):
        armed = False

        def __setattr__(self, name: str, value: object) -> None:
            super().__setattr__(name, value)
            if (
                self.armed
                and name == "_suspended_root"
                and ((seam == "suspended_store" and value is not None)
                     or (seam == "resume_consume" and value is None))
            ):
                self.armed = False
                raise interruption

        def launch_root_suspended(self) -> Any:
            root = super().launch_root_suspended()
            if seam == "compatibility_handoff":
                raise interruption
            return root

    job = InterruptingJob(kernel, template._spec, template._runner_identity, template._rules)
    job.armed = seam == "suspended_store"

    with pytest.raises(KeyboardInterrupt) as raised:
        if seam == "resume_consume":
            root = job.launch_root_suspended()
            job.armed = True
            job.resume_root(root)
        elif seam == "compatibility_handoff":
            job.launch_root()
        else:
            job.launch_root_suspended()

    assert raised.value is interruption
    assert job.last_finalization is not None and job.last_finalization.closed
    assert ("terminate_job", 101) in kernel.events
    assert not any(event[0] == "resume" for event in kernel.events)


def test_compatibility_second_launch_refusal_does_not_finalize_prior_root() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    job = fixture["_job"](runner, kernel)
    root = job.launch_root_suspended()
    before = list(kernel.events)

    with pytest.raises(runner._WindowsScenarioJobError):
        job.launch_root()

    assert kernel.events == before
    assert job.last_finalization is None
    job.resume_root(root)
    job.finalize()


def test_interrupted_handoff_preserves_primary_and_retryable_cleanup_failure() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    template = fixture["_job"](runner, kernel)
    interruption = KeyboardInterrupt("private handoff detail")

    class InterruptingJob(runner._WindowsScenarioJobV1):
        def launch_root_suspended(self) -> Any:
            super().launch_root_suspended()
            raise interruption

    job = InterruptingJob(kernel, template._spec, template._runner_identity, template._rules)
    kernel.fail.add("wait:201")

    with pytest.raises(BaseExceptionGroup) as raised:
        job.launch_root()

    assert raised.value.exceptions[0] is interruption
    cleanup = raised.value.exceptions[1]
    assert isinstance(cleanup, runner._WindowsFinalizationError)
    assert kernel.events.count(("terminate_job", 101)) == 1
    kernel.fail.remove("wait:201")
    assert job.finalize().closed


def test_finalization_attempt_permanently_revokes_suspended_resume_authority() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    job = fixture["_job"](runner, kernel)
    root = job.launch_root_suspended()
    kernel.fail.add("active")
    kernel.fail_close.update({root.process_handle, root.thread_handle})

    with pytest.raises(runner._WindowsFinalizationError):
        job.finalize()
    before = list(kernel.events)
    with pytest.raises(runner._WindowsScenarioJobError):
        job.resume_root(root)
    assert kernel.events == before

    kernel.fail.remove("active")
    kernel.fail_close.clear()
    assert job.finalize().closed


def test_changed_suspended_identity_refuses_before_resume_and_remains_retryable() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    job = fixture["_job"](runner, kernel)
    root = job.launch_root_suspended()
    original = kernel.identities[root.process_handle]
    kernel.identities[root.process_handle] = kernel._identity(
        43, 7, 430, "root.exe", "a" * 64
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        job.resume_root(root)
    assert any(
        isinstance(error, runner._WindowsScenarioJobError) for error in raised.value.exceptions
    )
    assert any(
        isinstance(error, runner._WindowsFinalizationError) for error in raised.value.exceptions
    )
    assert not any(event[0] == "resume" for event in kernel.events)

    kernel.identities[root.process_handle] = original
    assert job.finalize().closed


@pytest.mark.parametrize("mutation", ["foreign", "replayed", "closed", "second_root"])
def test_suspended_root_authority_cannot_be_replaced_reused_or_reopened(mutation: str) -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    job = fixture["_job"](runner, kernel)
    root = job.launch_root_suspended()
    supplied = replace(root) if mutation == "foreign" else root
    if mutation == "replayed":
        job.resume_root(root)
    if mutation == "closed":
        job.finalize()
    before = list(kernel.events)

    with pytest.raises(runner._WindowsScenarioJobError):
        if mutation == "second_root":
            job.launch_root_suspended()
        else:
            job.resume_root(supplied)
    assert kernel.events == before
    if mutation != "closed":
        job.finalize()


def test_finalized_empty_job_cannot_start_a_new_root() -> None:
    fixture = _helpers()
    runner = fixture["_load_runner"]()
    kernel = fixture["_FakeWindowsKernel"](runner)
    job = fixture["_job"](runner, kernel)

    assert job.finalize().closed
    before = list(kernel.events)
    with pytest.raises(runner._WindowsScenarioJobError):
        job.launch_root_suspended()
    assert kernel.events == before

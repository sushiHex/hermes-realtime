"""Malformed private frames and partial native snapshots cannot supply authority."""

from __future__ import annotations

import ctypes
import os
from typing import Any

import pytest


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"sequence": False},
        {"sequence": 1},
        {"nonce": "b" * 64},
        {"extra": True},
        {"version": 2},
    ],
)
def test_acknowledgment_requires_exact_types_identity_and_order(change: dict[str, Any]) -> None:
    from scripts.equivalence_process import _require_ack

    nonce = "a" * 64
    _require_ack({"version": 1, "nonce": nonce, "sequence": 0}, nonce, 0)
    with pytest.raises(ValueError):
        _require_ack({"version": 1, "nonce": nonce, "sequence": 0} | change, nonce, 0)


def test_successful_but_partial_job_membership_is_rejected() -> None:
    from scripts import qualify_evidence_slice_zero as core

    class Api:
        def QueryInformationJobObject(
            self, job: Any, kind: int, buffer: Any, size: int, returned: Any
        ) -> int:
            assert kind == 3
            header = ctypes.cast(
                buffer, ctypes.POINTER(core._JOBOBJECT_BASIC_PROCESS_ID_LIST_HEADER_V1)
            ).contents
            header.NumberOfAssignedProcesses = 2
            header.NumberOfProcessIdsInList = 1
            ids = ctypes.cast(
                ctypes.byref(buffer._obj, ctypes.sizeof(header)), ctypes.POINTER(ctypes.c_size_t)
            )
            ids[0] = 101
            return 1

    kernel = core._CtypesWindowsKernelV1(platform="nt")
    kernel._kernel32 = Api()
    kernel._job_max_active_processes[1] = 4
    with pytest.raises(core._WindowsScenarioJobError):
        kernel.query_job_processes(1)


@pytest.mark.skipif(os.name != "nt", reason="Windows exit-race error handling")
@pytest.mark.parametrize("exit_observed", [True, False])
def test_image_query_exit_race_requires_the_retained_handle_to_signal(exit_observed: bool) -> None:
    from scripts import qualify_evidence_slice_zero as core

    class Api:
        def __init__(self) -> None:
            self.waits: list[int] = []

        def GetProcessTimes(self, handle: Any, creation: Any, *unused: Any) -> int:
            creation._obj.value = 420
            return 1

        def GetProcessId(self, handle: Any) -> int:
            return 42

        def WaitForSingleObject(self, handle: Any, timeout: int) -> int:
            self.waits.append(timeout)
            return 0 if exit_observed and len(self.waits) == 2 else 258

        def QueryFullProcessImageNameW(self, *unused: Any) -> int:
            ctypes.set_last_error(5)
            return 0

    api = Api()
    kernel = core._CtypesWindowsKernelV1(platform="nt")
    kernel._kernel32 = api
    retained = core._WindowsKernelProcessV1(42, 7, 70, 420, "python.exe", "a" * 64)
    kernel._retained_process_identities[201] = retained
    if exit_observed:
        assert kernel.query_process_identity(201) is retained
    else:
        with pytest.raises(OSError):
            kernel.query_process_identity(201)
    assert api.waits == [0, 1000]


@pytest.mark.skipif(os.name != "nt", reason="Windows diagnostic pipe")
@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_exit_observations_survive_normal_and_abnormal_process_exit(exit_code: int) -> None:
    import msvcrt
    import subprocess
    import sys

    from scripts.equivalence_process import _read_worker_exit_events

    read_fd, write_fd = os.pipe()
    handle = msvcrt.get_osfhandle(write_fd)
    os.set_handle_inheritable(handle, True)
    startup = subprocess.STARTUPINFO()
    startup.lpAttributeList = {"handle_list": [handle]}
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import os,sys,msvcrt; "
             "f=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_WRONLY|os.O_BINARY); "
             "os.write(f,b'DTSPAX'); sys.exit(int(sys.argv[2]))", str(handle), str(exit_code)],
            close_fds=True, startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.close(write_fd)
        write_fd = -1
        assert child.wait(timeout=5) == exit_code
        assert _read_worker_exit_events(read_fd) == [
            "done_acknowledged", "server_stop_entered", "server_stop_returned",
            "protocol_closed", "atexit_entered", "atexit_callbacks_returned",
        ]
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        for fd in (read_fd, write_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.skipif(os.name != "nt", reason="Windows diagnostic pipe")
@pytest.mark.parametrize(
    ("stop", "exit_code", "expected"),
    [
        ("callback", 9, ["atexit_entered"]),
        ("finalization", 0, ["atexit_entered", "atexit_callbacks_returned"]),
    ],
)
def test_worker_exit_milestones_separate_exit_callbacks_from_finalization(
    stop: str, exit_code: int, expected: list[str],
) -> None:
    import msvcrt
    import subprocess
    import sys
    from pathlib import Path

    from scripts.equivalence_process import _read_worker_exit_events

    read_fd, write_fd = os.pipe()
    handle = msvcrt.get_osfhandle(write_fd)
    os.set_handle_inheritable(handle, True)
    startup = subprocess.STARTUPINFO()
    startup.lpAttributeList = {"handle_list": [handle]}
    child = None
    try:
        # Same registration order as the worker: X before the scenario, a library
        # exit callback during it, and A in the worker's final cleanup. A callback
        # that never returns stands in for a native crash inside library teardown.
        child = subprocess.Popen(
            [sys.executable, "-I", "-c",
             "import atexit,os,sys,msvcrt; sys.path.insert(0,sys.argv[3]); "
             "from scripts.equivalence_worker import _mark_exit; "
             "f=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_WRONLY|os.O_BINARY); "
             "atexit.register(_mark_exit,f,b'X'); "
             "sys.argv[2]=='callback' and atexit.register(os._exit,9); "
             "atexit.register(_mark_exit,f,b'A')",
             str(handle), stop, str(Path(__file__).resolve().parent.parent)],
            close_fds=True, startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.close(write_fd)
        write_fd = -1
        assert child.wait(timeout=10) == exit_code
        assert _read_worker_exit_events(read_fd) == expected
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        for fd in (read_fd, write_fd):
            if fd >= 0:
                os.close(fd)


def test_worker_registers_callbacks_returned_milestone_before_any_scenario_runs() -> None:
    import inspect

    from scripts import equivalence_worker

    source = inspect.getsource(equivalence_worker.main)
    registration = source.index('atexit.register(_mark_exit, progress_fd, b"X")')
    assert registration < source.index("_redirect_diagnostics()")
    assert registration < source.index("_read_frame(request_fd")
    assert source.count('b"X"') == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows diagnostic pipe")
@pytest.mark.parametrize(
    "raw", [b"", b"DT", b"AX", b"private-output", b"DD", b"AD", b"XA", b"DTSPAD", b"DTSPAXX"],
)
def test_worker_exit_observations_are_bounded_even_with_a_live_writer(raw: bytes) -> None:
    from scripts.equivalence_process import _read_worker_exit_events

    read_fd, write_fd = os.pipe()
    try:
        if raw:
            os.write(write_fd, raw)
        expected = {
            b"": [],
            b"DT": ["done_acknowledged", "server_stop_entered"],
            b"AX": ["atexit_entered", "atexit_callbacks_returned"],
        }.get(raw)
        assert _read_worker_exit_events(read_fd) == expected
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.skipif(os.name != "nt", reason="Windows worker teardown")
@pytest.mark.parametrize("failure", ["terminate", "wait"])
def test_worker_server_stop_failure_still_closes_protocol_pipes(failure: str) -> None:
    from scripts.equivalence_process import _read_worker_exit_events
    from scripts.equivalence_worker import _stop_server_and_close_protocol

    class Server:
        def terminate(self) -> None:
            if failure == "terminate":
                raise OSError("controlled stop failure")

        def wait(self, *, timeout: float) -> None:
            assert timeout == 5
            raise OSError("controlled stop failure")

    request_read, request_write = os.pipe()
    response_read, response_write = os.pipe()
    progress_read, progress_write = os.pipe()
    try:
        with pytest.raises(OSError, match="controlled stop failure"):
            _stop_server_and_close_protocol(Server(), request_read, response_write, progress_write)
        for fd in (request_read, response_write):
            with pytest.raises(OSError):
                os.fstat(fd)
        assert _read_worker_exit_events(progress_read) == [
            "server_stop_entered", "protocol_closed",
        ]
    finally:
        from contextlib import suppress

        for fd in (request_read, request_write, response_read, response_write,
                   progress_read, progress_write):
            with suppress(OSError):
                os.close(fd)


@pytest.mark.skipif(os.name != "nt", reason="Windows archived runner setup")
@pytest.mark.parametrize("failure", ["allocation", "parent", "materialize"])
def test_archived_runner_owns_resources_before_workspace_validation(
    tmp_path, monkeypatch, failure,
) -> None:
    import hashlib
    from pathlib import Path
    from types import SimpleNamespace

    from scripts import equivalence_process as owner

    closed = []
    kernel = SimpleNamespace(open_process=lambda *_: 101, close_handle=closed.append)
    monkeypatch.setattr(owner.core, "_CtypesWindowsKernelV1", lambda: kernel)
    root = Path(owner.__file__).resolve().parents[1]
    metadata = SimpleNamespace(manifest=tuple(
        SimpleNamespace(path=path.relative_to(root).as_posix(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in (root / "scripts").glob("*.py")
    ))
    monkeypatch.setattr(
        owner.archives, "verified_candidate_source_archive_metadata", lambda _: metadata,
    )
    monkeypatch.setattr(owner.archives, "_archive_bytes_for_consumer", lambda *_: b"synthetic")
    monkeypatch.setattr(owner, "_executable", lambda path, _: path)
    monkeypatch.setattr(owner.tempfile, "gettempdir", lambda: str(tmp_path))
    workspace = tmp_path / "workspace" if failure != "parent" else tmp_path / "other" / "workspace"

    def allocate(**_):
        if failure == "allocation":
            raise OSError("synthetic allocation failure")
        workspace.mkdir(parents=True)
        return str(workspace)

    def materialize(*_):
        raise RuntimeError("synthetic materialization failure")

    monkeypatch.setattr(owner.tempfile, "mkdtemp", allocate)
    monkeypatch.setattr(owner, "_materialize_archive", materialize)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        owner._run_archived_scenario(
            None, None, scenario="deterministic_equivalence", wheel=None,
            livekit_executable=Path(owner.sys.executable), livekit_sha256="a" * 64,
        )
    assert closed == [101]
    # An unproven parent must remain untouched; a verified owned workspace is removed.
    assert workspace.exists() is (failure == "parent")

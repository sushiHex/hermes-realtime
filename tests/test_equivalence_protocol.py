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
             "os.write(f,b'DTSPA'); sys.exit(int(sys.argv[2]))", str(handle), str(exit_code)],
            close_fds=True, startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.close(write_fd)
        write_fd = -1
        assert child.wait(timeout=5) == exit_code
        assert _read_worker_exit_events(read_fd) == [
            "done_acknowledged", "server_stop_entered", "server_stop_returned",
            "protocol_closed", "atexit_entered",
        ]
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=5)
        for fd in (read_fd, write_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.skipif(os.name != "nt", reason="Windows diagnostic pipe")
@pytest.mark.parametrize("raw", [b"", b"DT", b"private-output", b"DD", b"AD", b"DTSPAD"])
def test_worker_exit_observations_are_bounded_even_with_a_live_writer(raw: bytes) -> None:
    from scripts.equivalence_process import _read_worker_exit_events

    read_fd, write_fd = os.pipe()
    try:
        if raw:
            os.write(write_fd, raw)
        expected = (
            [] if not raw else
            ["done_acknowledged", "server_stop_entered"] if raw == b"DT" else None
        )
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

"""Real Win32 regression coverage for retained scenario process ownership."""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="real retained Windows process lifecycle")
@pytest.mark.parametrize("hidden", [False, True])
@pytest.mark.parametrize("redirect", [False, True])
def test_real_windows_job_retains_and_closes_an_exited_root(
    tmp_path: Path, hidden: bool, redirect: bool
) -> None:
    import msvcrt

    from scripts import qualify_evidence_slice_zero as core
    from scripts.equivalence_process import _read_frame

    kernel = core._CtypesWindowsKernelV1()
    runner_handle = kernel.open_process(
        os.getpid(), core._SYNCHRONIZE | core._PROCESS_QUERY_LIMITED_INFORMATION
    )
    raw = kernel.query_process_identity(runner_handle)
    python = Path(sys._base_executable).resolve()
    console = Path(os.environ["SYSTEMROOT"]) / "System32" / "conhost.exe"
    read_fd, write_fd = os.pipe()
    ack_read, ack_write = os.pipe()
    os.set_inheritable(write_fd, True)
    os.set_inheritable(ack_read, True)
    handles = (msvcrt.get_osfhandle(write_fd), msvcrt.get_osfhandle(ack_read))
    code = (
        "import os,sys,msvcrt;"
        "w=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_WRONLY|os.O_BINARY);"
        "r=msvcrt.open_osfhandle(int(sys.argv[2]),os.O_RDONLY|os.O_BINARY);"
        "sys.path.insert(0,sys.argv[3]);"
        "from scripts.equivalence_worker import _redirect_diagnostics;"
        "_redirect_diagnostics() if sys.argv[4]=='redirect' else None;"
        "sys.stdout.write('synthetic buffered diagnostic');"
        "os.write(w,b'{}\\n');os.read(r,1)"
    )
    try:
        spec = core._WindowsScenarioSpecV1(
            "deterministic_equivalence",
            (
                str(python),
                "-I",
                "-c",
                code,
                *map(str, handles),
                str(Path(__file__).resolve().parents[1]),
                "redirect" if redirect else "ordinary",
            ),
            (("SystemRoot", os.environ["SYSTEMROOT"]),),
            str(tmp_path),
            2000,
            handles,
            core._WindowsJobLimitsV1(4, 1024**3, 2 * 1024**3),
            no_window=hidden,
        )
        job = core._WindowsScenarioJobV1(
            kernel,
            spec,
            core._WindowsRunnerIdentityV1(raw.pid, raw.creation_filetime),
            (
                core._WindowsRoleRuleV1(
                    "host_root",
                    python.name,
                    hashlib.sha256(python.read_bytes()).hexdigest(),
                    frozenset(),
                    True,
                ),
                core._WindowsRoleRuleV1(
                    "console_owned_descendant",
                    console.name,
                    hashlib.sha256(console.read_bytes()).hexdigest(),
                    frozenset({"host_root"}),
                    False,
                ),
            ),
        )

        def exercise() -> core._WindowsBoundProcessV1:
            root = job.launch_root()
            assert _read_frame(read_fd, time.monotonic() + 5) == {}
            assert root in job.checkpoint("running").members
            os.write(ack_write, b"a")
            kernel.wait(root.process_handle, 2000)
            code = ctypes.c_uint32()
            api = kernel._api()
            api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            api.GetExitCodeProcess.restype = ctypes.c_int
            assert api.GetExitCodeProcess(root.process_handle, ctypes.byref(code))
            assert code.value == 0
            return root

        root = core._run_with_windows_scenario_job_finalization_v1(job, exercise)
        assert job.last_finalization is not None
        assert job.last_finalization.closed and job.last_finalization.zero_active_observed
        assert root.process_handle in job.last_finalization.waited_handles
    finally:
        for fd in (read_fd, write_fd, ack_read, ack_write):
            os.close(fd)
        kernel.close_handle(runner_handle)

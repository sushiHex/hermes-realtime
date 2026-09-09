"""Real Win32 regression coverage for retained scenario process ownership."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "nt", reason="real retained Windows process lifecycle")
def test_real_windows_job_retains_and_closes_an_exited_root(tmp_path: Path) -> None:
    import msvcrt

    from scripts import qualify_evidence_slice_zero as core

    kernel = core._CtypesWindowsKernelV1()
    runner_handle = kernel.open_process(
        os.getpid(), core._SYNCHRONIZE | core._PROCESS_QUERY_LIMITED_INFORMATION
    )
    raw = kernel.query_process_identity(runner_handle)
    python = Path(sys._base_executable).resolve()
    image_hash = hashlib.sha256(python.read_bytes()).hexdigest()
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    try:
        spec = core._WindowsScenarioSpecV1(
            "deterministic_equivalence",
            (str(python), "-I", "-c", "pass"),
            (("SystemRoot", os.environ["SYSTEMROOT"]),),
            str(tmp_path),
            2000,
            (msvcrt.get_osfhandle(write_fd),),
            core._WindowsJobLimitsV1(4, 1024**3, 2 * 1024**3),
        )
        job = core._WindowsScenarioJobV1(
            kernel,
            spec,
            core._WindowsRunnerIdentityV1(raw.pid, raw.creation_filetime),
            (core._WindowsRoleRuleV1("host_root", python.name, image_hash, frozenset(), True),),
        )
        root = job.launch_root()
        try:
            result = core._run_with_windows_scenario_job_finalization_v1(
                job, lambda: kernel.wait(root.process_handle, 2000)
            )
        except core._WindowsFinalizationError as error:
            raise AssertionError(
                [(failure.operation, failure.message) for failure in error.result.failures]
            ) from error
        assert result is None
        assert job.last_finalization is not None
        assert job.last_finalization.closed and job.last_finalization.zero_active_observed
        assert root.process_handle in job.last_finalization.waited_handles
    finally:
        os.close(read_fd)
        os.close(write_fd)
        kernel.close_handle(runner_handle)



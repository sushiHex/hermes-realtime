"""Malformed private frames and partial native snapshots cannot supply authority."""

from __future__ import annotations

import ctypes
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

"""Archive-bound storage workers, owned before their first instruction on Windows."""

from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts import candidate_source_archive_oracle as archives
from scripts import qualify_evidence_slice_zero as core
from scripts.candidate_e2e_fast_track import _materialize_archive
from scripts.candidate_wheel import VerifiedCandidateWheelV1, _wheel_for_consumer
from scripts.equivalence_process import (
    _read_frame,
    _require,
    _verify_tree,
    _verify_wheel_tree,
    _write_frame,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1
from scripts.windows_storage_oracle import audit_drain_release, audit_storage_release


@dataclass(frozen=True, slots=True)
class _StorageInvocation:
    observation: bytes
    process: core._WindowsBoundProcessV1
    cleanup: core._WindowsFinalizationResultV1
    exit_code: int
    expected_exit: int
    storage_released: bool = False


@dataclass(frozen=True, slots=True)
class _StorageArchive:
    workspace: Path
    source: Path
    package: Path
    metadata: archives.CandidateSourceArchiveMetadataV1
    wheel_sha256: str


@contextmanager
def _storage_archive(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
) -> Iterator[_StorageArchive]:
    _require(
        os.name == "nt" and ctypes.sizeof(ctypes.c_void_p) == 8 and not sys.flags.optimize,
        "storage qualification requires nonoptimized 64-bit Windows Python",
    )
    metadata = archives.verified_candidate_source_archive_metadata(archive)
    payload = archives._archive_bytes_for_consumer(archive, identity)
    package = _wheel_for_consumer(wheel, archive, identity)
    runner = Path(__file__).resolve().parent.parent
    required = {
        "scripts/storage_process.py",
        "scripts/storage_worker.py",
        "scripts/storage_observation.py",
        "scripts/windows_storage_oracle.py",
        "scripts/spool_crash_oracle.py",
        "scripts/spool_crash_matrix.py",
        "scripts/full_purge_cleanup.py",
        "scripts/full_purge_observation.py",
        "scripts/full_purge_worker.py",
        "scripts/qualify_evidence_slice_zero.py",
        "scripts/equivalence_process.py",
        "tests/evidence/spool_crash_worker.py",
    }
    _require(
        required <= {member.path for member in metadata.manifest},
        "storage runner closure is incomplete",
    )
    # Bind every executing script, including transitive ownership and archive helpers.
    for member in metadata.manifest:
        if member.path.startswith("scripts/") and member.path.endswith(".py"):
            _require(
                hashlib.sha256((runner / member.path).read_bytes()).hexdigest() == member.sha256,
                "executing storage runner differs from its candidate archive",
            )
    workspace = Path(tempfile.mkdtemp(prefix="hermes-storage-")).resolve(strict=True)
    parent = Path(tempfile.gettempdir()).resolve(strict=True)
    _require(workspace.parent == parent, "storage workspace parent differs")
    completed = False
    try:
        source = _materialize_archive(workspace, payload, metadata)
        packaged = workspace / "wheel-package"
        packaged.mkdir()
        for name, raw in package.members.items():
            path = packaged / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        _verify_tree(source, metadata)
        _verify_wheel_tree(packaged, package.members)
        yield _StorageArchive(workspace, source, packaged, metadata, package.wheel_sha256)
        _verify_tree(source, metadata)
        _verify_wheel_tree(packaged, package.members)
        completed = True
    finally:
        _require(
            workspace.resolve(strict=True).parent == parent and not workspace.is_symlink(),
            "storage cleanup target differs",
        )
        # A failed run retains its private workspace for diagnosis. In particular,
        # never delete storage beneath a process whose cleanup could not be proved.
        if completed:
            shutil.rmtree(workspace)


def _exit_code(kernel: Any, handle: int) -> int:
    value = ctypes.c_uint32()
    api = kernel._api()
    api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    api.GetExitCodeProcess.restype = ctypes.c_int
    if not api.GetExitCodeProcess(handle, ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(value.value)


def _validate_invocation(record: _StorageInvocation) -> Any:
    cleanup = record.cleanup
    _require(
        type(record.exit_code) is int
        and type(record.expected_exit) is int
        and record.expected_exit in {0, 197, 198}
        and record.exit_code == record.expected_exit,
        "storage worker exit differs from its prescribed mode",
    )
    _require(
        cleanup.closed
        and cleanup.zero_active_observed
        and not cleanup.failures
        and not cleanup.failed_handles
        and record.process.process_handle in cleanup.waited_handles,
        "storage worker retained ownership cleanup is incomplete",
    )
    observation: Any = core.load_strict_canonical_json(
        record.observation, source="storage observation"
    )
    _require(type(observation) is dict, "storage observation is not an object")
    _require(
        type(record.storage_released) is bool
        and record.storage_released
        is (
            (
                record.expected_exit in {197, 198}
                and observation.get("checkpoint") == "after_drain_ack_before_exit"
            )
            or (record.expected_exit == 0 and observation.get("phase") in {"purge", "repeat_purge"})
        ),
        "storage release observation differs",
    )
    return observation


def _run_storage_worker(
    archive: _StorageArchive, *, point: str, mode: int, action: str, clock: str = "caught-up"
) -> _StorageInvocation:
    """Crash codes are accepted only for a bound crash checkpoint, never recovery."""
    purge = action in {"purge", "repeat_purge"}
    _require(action in {"crash", "recover", "purge", "repeat_purge"}, "storage action differs")
    _require(
        type(mode) is int and (mode == 0 if purge else mode in {197, 198}),
        "storage exit mode differs",
    )
    _require(clock in {"caught-up", "regressed"}, "storage recovery clock differs")
    _require(
        not purge or (point == "full_purge_cleanup" and clock == "caught-up"),
        "full-purge worker configuration differs",
    )
    expected_exit = mode if action == "crash" else 0
    # The GUI interpreter uses the same runtime without creating a console host.
    # All protocol I/O travels over the two explicitly inherited pipe handles.
    python = Path(sys._base_executable).with_name("pythonw.exe").resolve(strict=True)  # type: ignore[attr-defined]
    python_hash = hashlib.sha256(python.read_bytes()).hexdigest()
    kernel = core._CtypesWindowsKernelV1()
    runner_handle = kernel.open_process(
        os.getpid(), core._SYNCHRONIZE | core._PROCESS_QUERY_LIMITED_INFORMATION
    )
    job = None
    try:
        raw = kernel.query_process_identity(runner_handle)
        runner = core._WindowsRunnerIdentityV1(raw.pid, raw.creation_filetime)
        with ExitStack() as pipes:
            request_read, request_write = os.pipe()
            response_read, response_write = os.pipe()
            endpoints = {
                fd: pipes.enter_context(
                    os.fdopen(
                        fd, "rb" if fd in {request_read, response_read} else "wb", buffering=0
                    )
                )
                for fd in (request_read, request_write, response_read, response_write)
            }
            import msvcrt

            for fd in (request_read, response_write):
                os.set_inheritable(fd, True)
            handles = tuple(msvcrt.get_osfhandle(fd) for fd in (request_read, response_write))
            module = "scripts.full_purge_worker" if purge else "scripts.storage_worker"
            bootstrap = (
                "import sys,runpy;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];"
                f"runpy.run_module('{module}',run_name='__main__')"
            )
            command = (
                str(python),
                "-I",
                "-B",
                "-c",
                bootstrap,
                str(archive.package),
                str(archive.source),
                *(str(handle) for handle in handles),
            )
            environment = {
                name: os.environ[name]
                for name in ("SystemRoot", "SystemDrive", "WINDIR")
                if name in os.environ
            } | {
                "TEMP": str(archive.workspace),
                "TMP": str(archive.workspace),
                "PATH": str(python.parent),
            }
            spec = core._WindowsScenarioSpecV1(
                "full_purge_cleanup" if purge else "spool_crash_matrix",
                command,
                tuple(sorted(environment.items(), key=lambda item: (item[0].casefold(), item[0]))),
                str(archive.workspace),
                10_000,
                handles,
                core._WindowsJobLimitsV1(8, 512 * 1024**2, 512 * 1024**2),
                no_window=True,
            )
            rules = (
                core._WindowsRoleRuleV1(
                    "storage_root", python.name, python_hash, frozenset(), True
                ),
            )
            job = core._WindowsScenarioJobV1(kernel, spec, runner, rules)
            root = job.launch_root()
            endpoints[request_read].close()
            endpoints[response_write].close()
            nonce = secrets.token_hex(32)

            def execute() -> tuple[Any, int, bool]:
                _write_frame(
                    request_write,
                    {
                        "version": 1,
                        "nonce": nonce,
                        "point": point,
                        "mode": mode,
                        "action": action,
                        "clock": clock,
                        "workspace": str(archive.workspace),
                    },
                )
                frame = _read_frame(response_read, time.monotonic() + 10)
                _require(
                    type(frame) is dict
                    and set(frame)
                    == {"version", "nonce", "point", "mode", "action", "pid", "observation"}
                    and type(frame["version"]) is int
                    and frame["version"] == 1
                    and frame["nonce"] == nonce
                    and frame["point"] == point
                    and type(frame["mode"]) is int
                    and frame["mode"] == mode
                    and frame["action"] == action
                    and type(frame["pid"]) is int
                    and frame["pid"] == root.identity.pid,
                    "storage frame is disconnected from its retained process or checkpoint",
                )
                members = job.checkpoint("checkpoint").members
                _require(members == (root,), "storage worker has unexpected descendants")
                observation = frame["observation"]
                if type(observation) is dict and set(observation) == {"failure", "source_line"}:
                    _require(
                        observation["failure"] in {"timeout", "value", "os", "other"}
                        and type(observation["source_line"]) is int
                        and 0 <= observation["source_line"] <= 100_000,
                        "storage failure diagnostics differ",
                    )
                    raise ValueError(
                        f"storage worker failed: {observation['failure']} "
                        f"at source line {observation['source_line']}"
                    )
                drain_released = action == "crash" and point == "after_drain_ack_before_exit"
                if drain_released:
                    audit_drain_release(
                        archive.workspace / f"{point}-exit{mode}-{clock}" / "evidence",
                        root.process_handle,
                    )
                if purge:
                    audit_storage_release(
                        archive.workspace / "full_purge_cleanup-exit0-caught-up/evidence",
                        root.process_handle,
                        (
                            ".hermes-realtime-evidence-root-v1",
                            "capture-v1.owner",
                            "purge-decoy.bin",
                            "capture-v1.sqlite3.backup",
                            "capture-v1.sqlite3-wal.backup",
                        ),
                    )
                if expected_exit == 198:
                    api = kernel._api()
                    api.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
                    api.TerminateProcess.restype = ctypes.c_int
                    if not api.TerminateProcess(root.process_handle, 198):
                        raise ctypes.WinError(ctypes.get_last_error())
                else:
                    _write_frame(request_write, {"version": 1, "nonce": nonce, "sequence": 0})
                kernel.wait(root.process_handle, 10_000)
                code = _exit_code(kernel, root.process_handle)
                _require(code == expected_exit, "storage worker did not reach its prescribed exit")
                return frame["observation"], code, drain_released or purge

            observation, code, storage_released = (
                core._run_with_windows_scenario_job_finalization_v1(job, execute)
            )
            cleanup = job.last_finalization
            assert cleanup is not None
            record = _StorageInvocation(
                core.canonical_json_bytes(observation),
                root,
                cleanup,
                code,
                expected_exit,
                storage_released,
            )
            _validate_invocation(record)
            _require(
                hashlib.sha256(python.read_bytes()).hexdigest() == python_hash,
                "storage interpreter changed",
            )
            return record
    finally:
        if job is not None and job.last_finalization is None:
            job.finalize()
        kernel.close_handle(runner_handle)

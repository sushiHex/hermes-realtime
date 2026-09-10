"""Private Windows process and pipe owner for the archived equivalence harness."""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import secrets
import shutil
import socket
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from scripts import candidate_source_archive_oracle as archives
from scripts import qualify_evidence_slice_zero as core
from scripts.candidate_e2e_fast_track import _materialize_archive
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_MAX_FRAME = 256 * 1024
if TYPE_CHECKING:
    from scripts.candidate_wheel import VerifiedCandidateWheelV1

_ArchivedResult = tuple[
    archives.CandidateSourceArchiveMetadataV1, bytes,
    tuple[core._WindowsBoundProcessV1, ...], core._WindowsFinalizationResultV1, int,
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_ack(frame: Any, nonce: str, sequence: int) -> None:
    _require(
        type(frame) is dict
        and set(frame) == {"version", "nonce", "sequence"}
        and type(frame["version"]) is int
        and frame["version"] == 1
        and type(frame["nonce"]) is str
        and frame["nonce"] == nonce
        and type(frame["sequence"]) is int
        and frame["sequence"] == sequence,
        "child acknowledgment identity differs",
    )


def _require_owned_listener(port: int, pid: int) -> None:
    """Match the selected signaling socket to an already retained Job member."""
    _require(type(port) is int and 1 <= port <= 65535, "invalid signaling port")
    api = ctypes.WinDLL("iphlpapi", use_last_error=True)
    api.GetExtendedTcpTable.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    ]
    api.GetExtendedTcpTable.restype = ctypes.c_uint32
    size = ctypes.c_uint32()
    _require(
        api.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 5, 0) == 122,
        "TCP owner table size is unavailable",
    )
    _require(4 <= size.value <= 8 * 1024 * 1024, "TCP owner table exceeds its bound")
    buffer = (ctypes.c_byte * size.value)()
    _require(
        api.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 5, 0) == 0,
        "TCP owner table changed or failed",
    )
    values = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint32))
    count = int(values[0])
    _require(4 + count * 24 <= size.value, "TCP owner table is truncated")
    listeners = []
    for index in range(count):
        row = tuple(int(values[1 + index * 6 + offset]) for offset in range(6))
        if row[0] == 2 and socket.ntohs(row[2] & 0xFFFF) == port:
            listeners.append((socket.inet_ntoa(row[1].to_bytes(4, "little")), row[5]))
    _require(
        listeners == [("127.0.0.1", pid)],
        "signaling listener is not the retained loopback LiveKit process",
    )


def _read_frame(fd: int, deadline: float) -> Any:
    """One reader, bounded frame and deadline; never block waiting for pipe EOF."""
    import msvcrt

    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.PeekNamedPipe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
    ]
    api.PeekNamedPipe.restype = ctypes.c_int
    frame = bytearray()
    while time.monotonic() < deadline:
        available = ctypes.c_uint32()
        if not api.PeekNamedPipe(
            msvcrt.get_osfhandle(fd), None, 0, None, ctypes.byref(available), None
        ):
            raise ValueError("equivalence pipe closed before its complete frame")
        if available.value:
            # Read only one byte through the LF boundary so the next frame can
            # never be consumed accidentally. Frames have a fixed small cap.
            value = os.read(fd, 1)
            _require(bool(value), "equivalence pipe EOF")
            frame.extend(value)
            _require(len(frame) <= _MAX_FRAME, "equivalence frame exceeds its bound")
            if value == b"\n":
                return core.load_strict_canonical_json(bytes(frame), source="equivalence frame")
        else:
            time.sleep(0.01)
    raise TimeoutError("equivalence frame deadline expired")


def _read_worker_exit_events(fd: int) -> list[str] | None:
    """Sample five content-free milestones without waiting for pipe EOF."""
    import _winapi
    import msvcrt

    names = {
        ord("D"): "done_acknowledged",
        ord("T"): "server_stop_entered",
        ord("S"): "server_stop_returned",
        ord("P"): "protocol_closed",
        ord("A"): "atexit_entered",
    }
    try:
        peek = cast(tuple[bytes, int, int], _winapi.PeekNamedPipe(msvcrt.get_osfhandle(fd), 1))
        available = peek[1]
        raw = os.read(fd, min(available, 6)) if available else b""
    except BrokenPipeError:
        raw = b""
    except OSError:
        return None
    if any(value not in names for value in raw):
        return None
    order = [list(names).index(value) for value in raw]
    if order != sorted(set(order)):
        return None
    return [names[value] for value in raw]


def _write_frame(fd: int, document: Any) -> None:
    raw = core.canonical_json_bytes(document)
    _require(len(raw) <= _MAX_FRAME, "equivalence frame exceeds its bound")
    while raw:
        count = os.write(fd, raw)
        _require(count > 0, "equivalence pipe write failed")
        raw = raw[count:]


def _verify_tree(root: Path, metadata: archives.CandidateSourceArchiveMetadataV1) -> None:
    expected = {item.path: item for item in metadata.manifest if item.kind == "file"}
    actual = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}
    _require(set(actual) == set(expected), "materialized candidate file set changed")
    for name, path in actual.items():
        info = path.lstat()
        _require(
            not path.is_symlink() and not (getattr(info, "st_file_attributes", 0) & 0x400),
            "materialized candidate contains a reparse point",
        )
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected[name].sha256,
            "materialized candidate bytes changed",
        )


def _executable(path: Path, digest: str) -> Path:
    _require(type(path) is type(Path()) and type(digest) is str, "executable pin types are invalid")
    _require(
        path.is_absolute() and path.suffix.lower() == ".exe",
        "executable pin must select an absolute .exe",
    )
    _require(
        len(digest) == 64 and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
        "executable pin differs from actual bytes",
    )
    return path.resolve(strict=True)


def run_archived_equivalence(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    *, livekit_executable: Path, livekit_sha256: str,
) -> _ArchivedResult:
    return _run_archived_scenario(
        archive, identity, scenario="deterministic_equivalence", wheel=None,
        livekit_executable=livekit_executable, livekit_sha256=livekit_sha256,
    )


def run_archived_revoke_race(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    wheel: VerifiedCandidateWheelV1,
    *, livekit_executable: Path, livekit_sha256: str,
) -> _ArchivedResult:
    return _run_archived_scenario(
        archive, identity, scenario="revoke_race", wheel=wheel,
        livekit_executable=livekit_executable, livekit_sha256=livekit_sha256,
    )


def _verify_wheel_tree(root: Path, members: dict[str, bytes]) -> None:
    files = {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}
    _require(set(files) == set(members), "materialized wheel member set changed")
    for name, path in files.items():
        _require(not path.is_symlink() and path.resolve().is_relative_to(root)
                 and not (getattr(path.lstat(), "st_file_attributes", 0) & 0x400),
                 "materialized wheel contains a reparse point")
        _require(path.read_bytes() == members[name], "materialized wheel bytes changed")


def _run_archived_scenario(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    *, scenario: str, wheel: VerifiedCandidateWheelV1 | None,
    livekit_executable: Path, livekit_sha256: str,
) -> _ArchivedResult:
    """Launch only verified archive bytes; retain all observed processes to close."""
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8 or sys.flags.optimize:
        raise ValueError("equivalence requires nonoptimized 64-bit Windows Python")
    _require(
        scenario in {"deterministic_equivalence", "revoke_race", "capacity_rollover"},
        "unknown archived scenario",
    )
    _require(
        (scenario != "deterministic_equivalence") == (wheel is not None),
        "scenario package binding differs",
    )
    package = None
    if wheel is not None:
        from scripts.candidate_wheel import _wheel_for_consumer

        package = _wheel_for_consumer(wheel, archive, identity)
    metadata = archives.verified_candidate_source_archive_metadata(archive)
    payload = archives._archive_bytes_for_consumer(archive, identity)
    # The trusted parent must be the same runner implementation as the candidate.
    module_root = Path(__file__).resolve().parent.parent
    for name in (
        "scripts/equivalence_process.py",
        "scripts/deterministic_equivalence.py",
        "scripts/qualify_evidence_slice_zero.py",
        "scripts/candidate_wheel.py",
        "scripts/revoke_race.py",
        "scripts/packaged_scenario.py",
        "scripts/capacity_rollover.py",
    ):
        matching = [member for member in metadata.manifest if member.path == name]
        _require(
            len(matching) == 1
            and matching[0].sha256 == hashlib.sha256((module_root / name).read_bytes()).hexdigest(),
            "executing runner differs from candidate archive",
        )
    livekit = _executable(livekit_executable, livekit_sha256)
    python = Path(sys.executable).resolve(strict=True)
    base_python = Path(sys._base_executable).resolve(strict=True)  # type: ignore[attr-defined]
    python_hash = hashlib.sha256(python.read_bytes()).hexdigest()
    base_hash = hashlib.sha256(base_python.read_bytes()).hexdigest()
    console_host = Path(os.environ["SYSTEMROOT"]) / "System32" / "conhost.exe"
    console_hash = hashlib.sha256(console_host.read_bytes()).hexdigest()
    kernel = core._CtypesWindowsKernelV1()
    runner_handle = kernel.open_process(
        os.getpid(), core._SYNCHRONIZE | core._PROCESS_QUERY_LIMITED_INFORMATION
    )
    workspace = Path(tempfile.mkdtemp(prefix="hermes-equivalence-")).resolve(strict=True)
    expected_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    _require(
        workspace.parent == expected_parent,
        "scenario workspace is outside its owned temporary parent",
    )
    job: core._WindowsScenarioJobV1 | None = None
    finalized = False
    try:
        root = _materialize_archive(workspace, payload, metadata)
        _verify_tree(root, metadata)
        package_root = root / "src"
        if package is not None:
            package_root = workspace / "wheel-package"
            package_root.mkdir()
            for name, raw in package.members.items():
                path = package_root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            _verify_wheel_tree(package_root, package.members)
        raw_runner = kernel.query_process_identity(runner_handle)
        runner = core._WindowsRunnerIdentityV1(raw_runner.pid, raw_runner.creation_filetime)
        with ExitStack() as pipes:
            request_read, request_write = os.pipe()
            response_read, response_write = os.pipe()
            progress_read, progress_write = os.pipe()
            endpoints = {
                fd: pipes.enter_context(
                    os.fdopen(
                        fd, "rb" if fd in {request_read, response_read, progress_read} else "wb",
                        buffering=0
                    )
                )
                for fd in (
                    request_read, request_write, response_read, response_write,
                    progress_read, progress_write,
                )
            }
            os.set_inheritable(request_read, True)
            os.set_inheritable(response_write, True)
            os.set_inheritable(progress_write, True)
            import msvcrt

            handles = tuple(
                msvcrt.get_osfhandle(fd) for fd in (request_read, response_write, progress_write)
            )
            bootstrap = (
                "import sys,runpy;sys.path[:0]=[sys.argv.pop(1),sys.argv.pop(1)];"
                "runpy.run_module('scripts.equivalence_worker',run_name='__main__')"
            )
            command = (
                str(python),
                "-I",
                "-B",
                "-c",
                bootstrap,
                str(package_root),
                str(root),
                *(str(handle) for handle in handles),
            )
            environment = {
                name: os.environ[name]
                for name in ("SystemRoot", "SystemDrive", "WINDIR")
                if name in os.environ
            } | {"TEMP": str(workspace), "TMP": str(workspace), "PATH": str(base_python.parent)}
            spec = core._WindowsScenarioSpecV1(
                scenario,
                command,
                tuple(sorted(environment.items(), key=lambda item: (item[0].casefold(), item[0]))),
                str(workspace),
                10_000,
                handles,
                core._WindowsJobLimitsV1(8, 2 * 1024**3, 4 * 1024**3),
                no_window=True,
            )
            rules = (
                core._WindowsRoleRuleV1("host_root", python.name, python_hash, frozenset(), True),
                core._WindowsRoleRuleV1(
                    "host_runtime", base_python.name, base_hash, frozenset({"host_root"}), False
                ),
                core._WindowsRoleRuleV1(
                    "livekit_owned_descendant",
                    livekit.name,
                    livekit_sha256,
                    frozenset({"host_root", "host_runtime"}),
                    False,
                ),
                core._WindowsRoleRuleV1(
                    "console_owned_descendant",
                    console_host.name,
                    console_hash,
                    frozenset({"host_root", "host_runtime", "livekit_owned_descendant"}),
                    False,
                ),
            )
            job = core._WindowsScenarioJobV1(kernel, spec, runner, rules)
            nonce = secrets.token_hex(32)
            rows: list[Any] = []
            members: dict[int, core._WindowsBoundProcessV1] = {}
            root_process = job.launch_root()
            endpoints[request_read].close()
            endpoints[response_write].close()
            endpoints[progress_write].close()
            observed_exit_code: int | None = None
            exchange_completed = False
            deadline = time.monotonic() + 600

            def execute() -> int:
                nonlocal observed_exit_code, exchange_completed
                _write_frame(
                    request_write,
                    {
                        "version": 1,
                        "scenario": scenario,
                        "nonce": nonce,
                        "livekit": str(livekit),
                        "livekitSha256": livekit_sha256,
                        "workspace": str(workspace),
                        "sourceCommit": metadata.candidate_head_oid,
                        "sourceTree": metadata.candidate_tree_oid,
                    },
                )
                from scripts.deterministic_equivalence import ARMS_V1

                signaling_port = 0
                arms = ARMS_V1 if scenario == "deterministic_equivalence" else (scenario,)
                for sequence, stage in enumerate(("ready", *arms, "done")):
                    frame = _read_frame(response_read, deadline)
                    if type(frame) is dict and frame.get("stage") == "failed":
                        failure = frame.get("observation")
                        if (
                            type(failure) is dict
                            and set(failure) == {"failure", "sourceLine"}
                            and failure["failure"]
                            in {"assertion", "timeout", "runtime", "value", "os", "other"}
                            and type(failure["sourceLine"]) is int
                            and 0 <= failure["sourceLine"] <= 100_000
                        ):
                            raise ValueError(
                                f"archived worker failed during {stage}: {failure['failure']} "
                                f"at source line {failure['sourceLine']}"
                            )
                        raise ValueError("archived worker failed with malformed diagnostics")
                    _require(
                        type(frame) is dict
                        and set(frame) == {"version", "nonce", "sequence", "stage", "observation"},
                        "equivalence envelope fields are not closed",
                    )
                    _require(
                        type(frame["version"]) is int
                        and frame["version"] == 1
                        and frame["nonce"] == nonce
                        and type(frame["sequence"]) is int
                        and frame["sequence"] == sequence
                        and frame["stage"] == stage,
                        "equivalence frame identity or order mismatch",
                    )
                    snapshot = job.checkpoint(stage.replace("_", "-"))
                    for member in snapshot.members:
                        members[member.identity.pid] = member
                    _require(
                        any(
                            member.role == "livekit_owned_descendant" for member in snapshot.members
                        ),
                        "owned LiveKit process is absent at checkpoint",
                    )
                    livekit_members = [
                        member
                        for member in snapshot.members
                        if member.role == "livekit_owned_descendant"
                    ]
                    _require(len(livekit_members) == 1, "LiveKit process ownership is ambiguous")
                    if stage == "ready":
                        _require(
                            type(frame["observation"]) is dict
                            and set(frame["observation"]) == {"port"},
                            "readiness has no exact signaling port",
                        )
                        signaling_port = frame["observation"]["port"]
                    elif stage != "done":
                        _require(
                            type(frame["observation"]) is dict
                            and frame["observation"].get("arm") == stage,
                            "frame is disconnected from its arm",
                        )
                        rows.append(frame["observation"])
                    else:
                        _require(frame["observation"] == {}, "unexpected lifecycle payload")
                    _require_owned_listener(signaling_port, livekit_members[0].identity.pid)
                    _write_frame(
                        request_write, {"version": 1, "nonce": nonce, "sequence": sequence}
                    )
                exchange_completed = True
                kernel.wait(root_process.process_handle, 10_000)
                code = ctypes.c_uint32()
                api = kernel._api()
                api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
                api.GetExitCodeProcess.restype = ctypes.c_int
                if not api.GetExitCodeProcess(root_process.process_handle, ctypes.byref(code)):
                    raise ctypes.WinError(ctypes.get_last_error())
                observed_exit_code = int(code.value)
                _require(code.value == 0, f"archived worker exit code is {code.value}")
                return int(code.value)

            try:
                exit_code = core._run_with_windows_scenario_job_finalization_v1(job, execute)
            finally:
                # Diagnostic milestones never supply qualification authority. In
                # particular, a complete exchange cannot accept an abnormal exit.
                print("[archived-worker] " + json.dumps({
                    "version": 1,
                    "scenario": scenario,
                    "exchange_completed": exchange_completed,
                    "exit_code": observed_exit_code,
                    "child_events": _read_worker_exit_events(progress_read),
                }))
            finalization = job.last_finalization
            assert finalization is not None
            finalized = finalization.closed and finalization.zero_active_observed
        _verify_tree(root, metadata)
        if package is not None:
            _verify_wheel_tree(package_root, package.members)
        _executable(livekit, livekit_sha256)
        _executable(python, python_hash)
        _executable(base_python, base_hash)
        _executable(console_host, console_hash)
        return (
            metadata,
            core.canonical_json_bytes(rows),
            tuple(members.values()),
            finalization,
            exit_code,
        )
    except BaseException as primary:
        # Retry cleanup once, never the scenario. Preserve the original failure
        # even if the retry releases everything. A further failure retains its
        # Job owner on the exception for explicit subsequent cleanup.
        failed_cleanup = job.last_finalization if job is not None else None
        if job is not None and failed_cleanup is not None and not failed_cleanup.closed:
            try:
                job.finalize()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "equivalence failed and retained cleanup still needs attention",
                    [primary, cleanup],
                ) from None
        raise
    finally:
        kernel.close_handle(runner_handle)
        # Never remove a running scenario's files. The known, fresh temporary
        # directory is the only recursive cleanup target.
        completed_cleanup = job.last_finalization if job is not None else None
        if (
            job is None
            or finalized
            or (
                completed_cleanup is not None
                and completed_cleanup.closed
                and completed_cleanup.zero_active_observed
                and not completed_cleanup.failures
            )
        ):
            _require(
                workspace.resolve(strict=True).parent == expected_parent
                and not workspace.is_symlink(),
                "scenario cleanup target changed",
            )
            shutil.rmtree(workspace)

"""Opted-in Windows controller containment, not provider qualification authority.

The base-Python guardian escapes ancestor Jobs; the explicitly bound controller
keeps its dependency environment. The generic Windows kernel owns the controller
atomically before resume. This proves containment/quiescence, not an authenticated
descendant inventory or recovery of an interrupted qualification.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"[0-9a-f]{64}")
_MAX_JSON = 65536
_MAX_OUTPUT = 8 * 1024**2
EXECUTION_ROOT_ENV = "HERMES_QUALIFICATION_EXECUTION_ROOT"
# Kept alive until abrupt guardian exit when Job settlement is unproven. Never
# finalize these generators: that would thaw/delete namespaces with live users.
_UNSETTLED_RESOURCES: list[ExitStack] = []
# Reviewed import-time local dependency closure of the two reused resource
# primitives (including the PluginManager helper's dynamic authority import).
# This follows the existing trusted, fixed-checkout bootstrap premise; checking
# source files is not protection against concurrent hostile Python replacement.
_RESOURCE_SOURCE_NAMES = (
    "__init__",
    "candidate_e2e_fast_track",
    "candidate_source_archive_oracle",
    "equivalence_process",
    "qualification_execution_files",
    "qualification_file_seals",
    "qualify_evidence_slice_zero",
    "qualify_hermes_v020_pluginmanager",
    "release_gate",
    "source_archive_authority",
    "task13_artifact_orchestrator",
    "windows_storage_oracle",
)
# Shared by the guardian and outer qualification callers/proofs. DETACHED_PROCESS
# must not be added: Windows then ignores CREATE_NO_WINDOW for that process.
WINDOWLESS_BREAKAWAY_FLAGS = 0x1000000 | 0x200 | 0x8000000


def _read(path: Path, maximum: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("launch_input_bound")
    return raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise ValueError("launch_duplicate_field")
            value[key] = item
        return value

    return json.loads(raw, object_pairs_hook=pairs)


def _validate_sealed_resources(value: Any) -> dict[str, Any]:
    """A bounded declarative selection; no model or launch policy is inferred."""
    from scripts.qualification_file_seals import _relative_windows_member

    if (
        type(value) is not dict
        or set(value) != {"schema_version", "execution", "inputs"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["inputs"]) is not list
        or len(value["inputs"]) > 16
    ):
        raise ValueError("launch_resources_schema")
    total_files = total_bytes = 0
    for index, tree in enumerate([value["execution"], *value["inputs"]]):
        if (
            type(tree) is not dict
            or set(tree) != {"root", "files"}
            or type(tree["root"]) is not str
            or "\0" in tree["root"]
            or not Path(tree["root"]).is_absolute()
            or ".." in Path(tree["root"]).parts
            or tree["root"].startswith("\\\\")
            or type(tree["files"]) is not list
            or not 0 < len(tree["files"]) <= 16384
        ):
            raise ValueError("launch_resources_tree")
        names = []
        size = 0
        for item in tree["files"]:
            if (
                type(item) is not dict
                or set(item) != {"path", "sha256", "size_bytes"}
                or type(item["sha256"]) is not str
                or _SHA.fullmatch(item["sha256"]) is None
                or type(item["size_bytes"]) is not int
                or not 0 <= item["size_bytes"] <= (512 * 1024**2 if index == 0 else 16 * 1024**3)
            ):
                raise ValueError("launch_resources_file")
            names.append(_relative_windows_member(item["path"]))
            size += item["size_bytes"]
        directories = {
            str(parent)
            for name in names
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        folded = {name.casefold() for name in names}
        if (
            names != sorted(names)
            or len(folded) != len(names)
            or folded.intersection(name.casefold() for name in directories)
            or len(directories) > 16384
            or len({name.casefold() for name in directories}) != len(directories)
            or (index == 0 and size > 4 * 1024**3)
        ):
            raise ValueError("launch_resources_inventory")
        total_files += len(names)
        total_bytes += size
    if total_files > 16384 or total_bytes > 64 * 1024**3:
        raise ValueError("launch_resources_bound")
    # Detach the nested declaration from mutable caller-owned containers.
    return cast(dict[str, Any], _json(json.dumps(value, allow_nan=False).encode()))


def _acquire_sealed_resources(value: Any, owned: ExitStack) -> str:
    from scripts.qualification_execution_files import (
        _execution_files_for_consumer,
        owned_execution_files,
    )
    from scripts.qualification_file_seals import (
        retain_file_seals,
        sealed_file_bytes,
        sealed_file_metadata,
    )

    declaration = _validate_sealed_resources(value)
    contents = {}
    for index, tree in enumerate([declaration["execution"], *declaration["inputs"]]):
        members = tuple(item["path"] for item in tree["files"])
        seals = owned.enter_context(retain_file_seals(Path(tree["root"]), members))
        expected = tuple(
            (item["path"], item["sha256"], item["size_bytes"]) for item in tree["files"]
        )
        if sealed_file_metadata(seals) != expected:
            raise ValueError("launch_resources_held_binding")
        if index == 0:
            for name, digest, size in expected:
                raw = sealed_file_bytes(seals, name, size)
                if len(raw) != size or _sha(raw) != digest:
                    raise ValueError("launch_resources_read_binding")
                contents[name] = raw
    snapshot = owned.enter_context(owned_execution_files(contents))
    return str(_execution_files_for_consumer(snapshot))


def _resource_sources() -> dict[str, str]:
    return {
        name: _sha(_read(_ROOT / "scripts" / (name + ".py"), 4 * 1024**2))
        for name in _RESOURCE_SOURCE_NAMES
    }


def _verify_resource_sources(value: Any, *, imported: bool = False) -> None:
    if type(value) is not dict or set(value) != set(_RESOURCE_SOURCE_NAMES):
        raise ValueError("launch_resources_source_schema")
    for name in _RESOURCE_SOURCE_NAMES:
        path = _ROOT / "scripts" / (name + ".py")
        if (
            path.resolve(strict=True) != path
            or type(value[name]) is not str
            or _SHA.fullmatch(value[name]) is None
            or _sha(_read(path, 4 * 1024**2)) != value[name]
        ):
            raise ValueError("launch_resources_source_digest")
        if imported:
            module_names = ("scripts",) if name == "__init__" else ("scripts." + name, name)
            modules = [sys.modules[key] for key in module_names if key in sys.modules]
            if not modules or any(
                (origin := getattr(module, "__file__", None)) is None
                or Path(origin).resolve() != path
                for module in modules
            ):
                raise ValueError("launch_resources_source_origin")


def _resource_summary(value: Any) -> dict[str, Any]:
    declaration = _validate_sealed_resources(value)
    return {
        "sealed_resources_sha256": _sha(
            json.dumps(declaration, sort_keys=True, separators=(",", ":")).encode()
        ),
        "execution_file_count": len(declaration["execution"]["files"]),
        "input_file_count": sum(len(tree["files"]) for tree in declaration["inputs"]),
        "input_tree_count": len(declaration["inputs"]),
    }


@dataclass(frozen=True)
class InterpreterBinding:
    guardian: str
    controller: str
    guardian_sha256: str
    controller_sha256: str


@dataclass(frozen=True)
class LaunchResult:
    exit_code: int
    stdout: bytes


def load_binding(path: Path, digest: str, *, controller: bool = True) -> InterpreterBinding:
    raw = _read(path, _MAX_JSON)
    if type(digest) is not str or _SHA.fullmatch(digest) is None or _sha(raw) != digest:
        raise ValueError("launch_binding_digest")
    value = _json(raw)
    if type(value) is not dict or set(value) != {"schema_version", "guardian", "controller"}:
        raise ValueError("launch_binding_schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("launch_binding_schema")
    for role in ("guardian", "controller"):
        item = value[role]
        if (
            type(item) is not dict
            or set(item) != {"path", "sha256"}
            or type(item["path"]) is not str
            or not Path(item["path"]).is_absolute()
            or type(item["sha256"]) is not str
            or _SHA.fullmatch(item["sha256"]) is None
        ):
            raise ValueError("launch_interpreter_schema")
        try:
            actual = _sha(_read(Path(item["path"]), 64 * 1024**2))
        except OSError as error:
            raise ValueError("launch_interpreter_unavailable") from error
        if actual != item["sha256"]:
            raise ValueError("launch_interpreter_digest")
    if Path(value["guardian"]["path"]).resolve() != Path(cast(Any, sys)._base_executable).resolve():
        raise ValueError("launch_base_interpreter")
    if controller and Path(value["controller"]["path"]).resolve() != Path(sys.executable).resolve():
        raise ValueError("launch_controller_interpreter")
    return InterpreterBinding(
        value["guardian"]["path"],
        value["controller"]["path"],
        value["guardian"]["sha256"],
        value["controller"]["sha256"],
    )


def guardian_command(binding: InterpreterBinding) -> tuple[str, ...]:
    return binding.guardian, "-B", str(Path(__file__).resolve())


def _outside_jobs() -> bool:
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    api.IsProcessInJob.restype = ctypes.c_int
    flag = ctypes.c_int()
    if not api.IsProcessInJob(api.GetCurrentProcess(), None, ctypes.byref(flag)):
        raise OSError("launch_job_membership_unavailable")
    return not bool(flag.value)


def _record(path: Path, value: dict[str, Any], *, reserve: bool = False) -> None:
    raw = (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()
    if len(raw) > _MAX_JSON:
        raise ValueError("launch_receipt_bound")
    with path.open("xb" if reserve else "ab", buffering=0) as stream:
        if stream.write(raw) != len(raw):
            raise OSError("launch_receipt_short_write")
        os.fsync(stream.fileno())


def completion_status(value: Any, *, cleanup_verified: bool) -> str:
    return (
        "complete"
        if (
            type(value) is dict
            and set(value) == {"normal_exit", "exit_code"}
            and value["normal_exit"] is True
            and type(value["exit_code"]) is int
            and 0 <= value["exit_code"] <= 255
            and cleanup_verified is True
        )
        else "incomplete"
    )


def _resource_release_winerror(error: BaseException) -> int | None:
    """Sample at most sixteen exception objects; never retain messages or paths."""
    pending = [error]
    for _ in range(16):
        if not pending:
            break
        current = pending.pop(0)
        code = getattr(current, "winerror", None)
        if type(code) is int and 0 <= code <= 65535:
            return code
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions[: 16 - len(pending)])
    return None


def _retain_cleanup_members(kernel: Any, job: int, owned: list[int]) -> tuple[int, ...]:
    """Pin a pre-termination Job checkpoint for synchronization, not authentication.

    This does not claim to capture a process created after the checkpoint. Job
    zero and successful physical resource removal remain mandatory acceptance.
    """
    api = kernel._api()
    api.IsProcessInJob.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    api.IsProcessInJob.restype = ctypes.c_int
    for pid in kernel.query_job_processes(job):
        handle = kernel.open_process(pid, 0x101000)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
        owned.append(handle)
        member = ctypes.c_int()
        if (
            api.GetProcessId(ctypes.c_void_p(handle)) != pid
            or not api.IsProcessInJob(
                ctypes.c_void_p(handle), ctypes.c_void_p(job), ctypes.byref(member)
            )
            or not member.value
        ):
            raise ValueError("launch_cleanup_member_binding")
    return tuple(owned)


# No tooling package is imported into the external controller namespace.
_CONTROLLER = """
import ctypes,hashlib,json,msvcrt,os,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_bytes())
sys.stdin=open(os.devnull,'r')
sys.stdout=open(p['stdout'],'a',encoding='utf-8')
sys.stderr=open(p['stderr'],'a',encoding='utf-8')
for fd,stream in enumerate((sys.stdin,sys.stdout,sys.stderr)):
    os.dup2(stream.fileno(),fd)
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.SetStdHandle.argtypes=[ctypes.c_uint32,ctypes.c_void_p]
    api.SetStdHandle.restype=ctypes.c_int
    if not api.SetStdHandle((-10,-11,-12)[fd],msvcrt.get_osfhandle(fd)):
        raise OSError('controller_stdio')
target=Path(p['script'])
with target.open('rb') as f: raw=f.read(4194305)
if len(raw)>4194304 or hashlib.sha256(raw).hexdigest()!=p['script_sha256']:
    raise ValueError('controller_source_binding')
sys.path.insert(0,str(target.parent)); sys.argv=[str(target),*p['arguments']]
normal=False; code=2
try:
    exec(compile(raw,str(target),'exec'),{'__name__':'__main__','__file__':str(target),'__package__':None})
    normal=True; code=0
except SystemExit as e:
    normal=e.code is None or type(e.code) is int and 0<=e.code<=255
    code=0 if e.code is None else e.code if normal else 2
finally:
    sys.stdout.flush(); sys.stderr.flush()
    with Path(p['completion']).open('xb',buffering=0) as f:
        f.write((json.dumps({'normal_exit':normal,'exit_code':code})+'\\n').encode())
        os.fsync(f.fileno())
"""


def _owned_controller(
    plan: dict[str, Any], binding: InterpreterBinding, evidence: dict[str, Any]
) -> tuple[bytes, Any, bool]:
    import msvcrt

    from scripts import qualify_evidence_slice_zero as core

    if (
        Path(core.__file__).resolve() != Path(plan["kernel_path"]).resolve()
        or _sha(_read(Path(core.__file__), 4 * 1024**2)) != plan["kernel_sha256"]
    ):
        raise ValueError("launch_kernel_binding")
    kernel = core._CtypesWindowsKernelV1()
    job = None
    root = None
    cleanup = False
    settled = False
    resources = ExitStack() if "sealed_resources" in plan else None
    primary: BaseException | None = None
    deadline = time.monotonic() + plan["timeout_seconds"]
    folder = Path(plan["work"])
    output, errors = folder / "stdout", folder / "stderr"
    try:
        environment_values = dict(os.environ)
        if resources is not None:
            if any(key.casefold() == EXECUTION_ROOT_ENV.casefold() for key in environment_values):
                raise ValueError("launch_resources_environment")
            environment_values[EXECUTION_ROOT_ENV] = _acquire_sealed_resources(
                plan["sealed_resources"], resources
            )
            _verify_resource_sources(plan["sealed_resources_sources"], imported=True)
            if time.monotonic() >= deadline:
                raise TimeoutError("launch_controller_deadline")
        with open(os.devnull, "rb") as null, output.open("xb") as out, errors.open("xb") as err:
            handles = tuple(msvcrt.get_osfhandle(stream.fileno()) for stream in (null, out, err))
            for stream in (null, out, err):
                os.set_inheritable(stream.fileno(), True)
            job = kernel.create_job(core._WindowsJobLimitsV1(64, 64 * 1024**3, 128 * 1024**3))
            command = (binding.controller, "-B", "-c", _CONTROLLER, plan["plan_path"])
            environment = tuple(
                sorted(environment_values.items(), key=lambda item: (item[0].casefold(), item[0]))
            )
            flags = 0x4 | 0x200 | 0x400 | 0x80000 | 0x8000000
            root = kernel.create_process_suspended(
                command, environment, plan["cwd"], handles, flags, job=job
            )
            kernel.resume_thread(root.thread_handle)
            while True:
                if output.stat().st_size > _MAX_OUTPUT or errors.stat().st_size > _MAX_JSON:
                    raise ValueError("launch_output_bound")
                api = kernel._api()
                status = api.WaitForSingleObject(ctypes.c_void_p(root.process_handle), 50)
                if time.monotonic() >= deadline:
                    raise TimeoutError("launch_controller_deadline")
                if status == 0:
                    api.GetExitCodeProcess.argtypes = [
                        ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_uint32),
                    ]
                    api.GetExitCodeProcess.restype = ctypes.c_int
                    code = ctypes.c_uint32()
                    if (
                        not api.GetExitCodeProcess(
                            ctypes.c_void_p(root.process_handle), ctypes.byref(code)
                        )
                        or code.value != 0
                    ):
                        raise RuntimeError("launch_controller_exit")
                    break
                if status != 258:
                    raise OSError("launch_controller_wait")
    except BaseException as error:
        primary = error
    finally:
        failures: list[BaseException] = []
        if job is not None:
            until = time.monotonic() + 30
            member_handles: list[int] = []
            members: tuple[int, ...] = ()
            member_waits_complete = True
            if resources is not None:
                try:
                    members = _retain_cleanup_members(kernel, job, member_handles)
                except BaseException as error:
                    member_waits_complete = False
                    failures.append(error)
            try:
                kernel.terminate_job(job)
            except BaseException as error:
                failures.append(error)
            if root is not None:
                try:
                    kernel.wait(root.process_handle, max(1, int((until - time.monotonic()) * 1000)))
                except BaseException as error:
                    if resources is not None:
                        member_waits_complete = False
                    failures.append(error)
            # TerminateJobObject initiates TerminateProcess for each member.
            # Job accounting zero can precede process signaling/image teardown;
            # retain before termination, then wait within the same cleanup bound.
            for handle in members:
                try:
                    remaining = int((until - time.monotonic()) * 1000)
                    if remaining <= 0:
                        raise TimeoutError("launch_cleanup_deadline")
                    kernel.wait(handle, remaining)
                except BaseException as error:
                    member_waits_complete = False
                    failures.append(error)
            try:
                while True:
                    active = kernel.query_job_active_process_count(job)
                    if type(active) is not int:
                        raise ValueError("launch_cleanup_count")
                    if time.monotonic() >= until:
                        raise TimeoutError("launch_cleanup_deadline")
                    if active == 0:
                        settled = member_waits_complete
                        cleanup = settled and not failures
                        break
                    time.sleep(0.01)
            except BaseException as error:
                failures.append(error)
            if resources is not None and not settled:
                # Keep contexts strongly referenced and never restore/delete an
                # executable namespace when any consumer may still be alive.
                _UNSETTLED_RESOURCES.append(resources)
            if cleanup or resources is not None:
                handles_to_close = (
                    () if root is None else (root.thread_handle, root.process_handle)
                ) + tuple(member_handles) + (job,)
                if resources is not None and not settled:
                    # Request kernel kill-on-close before the guardian's hard
                    # exit. Complete settlement is unproven: no acceptance follows.
                    handles_to_close = (job,) + handles_to_close[:-1]
                for handle in handles_to_close:
                    try:
                        kernel.close_handle(handle)
                    except BaseException as error:
                        failures.append(error)
        if resources is not None and (job is None or settled):
            try:
                resources.close()
            except BaseException as error:
                evidence.update(
                    failure_phase="resource_release",
                    failure_category="cleanup_failed",
                    resource_release_winerror=_resource_release_winerror(error),
                )
                failures.append(error)
        evidence["cleanup_verified"] = cleanup and not failures
        if primary is not None:
            failures.insert(0, primary)
        if failures:
            raise BaseExceptionGroup("controller execution or cleanup failed", failures)
    # Descendants may write until Job quiescence; the final acceptance bound
    # must follow cleanup, not merely the root's signaled wait.
    _read(errors, _MAX_JSON)
    completion = (
        _json(_read(Path(plan["completion"]), _MAX_JSON))
        if Path(plan["completion"]).exists()
        else None
    )
    return _read(output, _MAX_OUTPUT), completion, cleanup


def _guardian(path: Path, digest: str) -> int:
    raw = _read(path, _MAX_JSON)
    if _sha(raw) != digest:
        raise ValueError("launch_plan_digest")
    plan = _json(raw)
    receipt = Path(plan["receipt"])
    final: dict[str, Any] = {
        "stage": "incomplete",
        "cleanup_verified": False,
        "controller_completed": False,
    }
    try:
        if not _outside_jobs():
            raise RuntimeError("guardian_inherited_job")
        binding = load_binding(Path(plan["binding"]), plan["binding_sha256"], controller=False)
        if _sha(_read(Path(__file__), 4 * 1024**2)) != plan["launcher_sha256"]:
            raise ValueError("launch_source_digest")
        if "sealed_resources" in plan:
            _verify_resource_sources(plan["sealed_resources_sources"])
            final.update(_resource_summary(plan["sealed_resources"]))
        _record(receipt, {"stage": "started", "guardian_sha256": binding.guardian_sha256})
        stdout, completion, cleanup = _owned_controller(plan, binding, final)
        final.update(cleanup_verified=cleanup, controller_completed=completion is not None)
        final["stage"] = completion_status(completion, cleanup_verified=cleanup)
        final["guardian_sha256"] = binding.guardian_sha256
        if final["stage"] == "complete":
            Path(plan["result"]).write_bytes(stdout)
            final.update(
                exit_code=completion["exit_code"],
                stdout_bytes=len(stdout),
                stdout_sha256=_sha(stdout),
            )
    except BaseException as error:
        final["stage"] = "incomplete"
        final["category"] = (
            "guardian_inherited_job"
            if isinstance(error, RuntimeError) and str(error) == "guardian_inherited_job"
            else "timed_out"
            if isinstance(error, TimeoutError)
            or isinstance(error, BaseExceptionGroup)
            and error.subgroup(TimeoutError) is not None
            else "invalid_input"
            if isinstance(error, ValueError)
            else "execution_failed"
        )
    finally:
        try:
            _record(receipt, final)
        finally:
            if _UNSETTLED_RESOURCES:
                # Normal interpreter finalization can collect generator-based
                # contexts and thaw namespaces. Let kernel process teardown
                # release retained handles instead. Immutability after this
                # unsettled exit is not established and cannot be accepted.
                os._exit(2)
    return 0 if final["stage"] == "complete" else 2


def run_owned_controller(
    script: Path,
    arguments: tuple[str, ...],
    *,
    binding_path: Path,
    binding_sha256: str,
    receipt: Path,
    timeout_seconds: int = 14400,
    sealed_resources: dict[str, Any] | None = None,
) -> LaunchResult:
    """Run a bound controller; optional v1 resources belong to its guardian.

    ``sealed_resources`` selects one execution tree and up to 16 input trees.
    Each tree declares ``root`` and sorted ``files`` containing exactly ``path``,
    ``sha256``, and ``size_bytes``. The controller receives the fresh execution
    tree only through :data:`EXECUTION_ROOT_ENV`; input paths stay as declared.
    No resource context belongs to this observing caller. This owns no model
    policy and does not authenticate the caller's declared selection.
    """
    category = "refused"
    try:
        if os.name != "nt":
            raise ValueError("launch_requires_windows")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 14400:
            raise ValueError("launch_deadline_bound")
        if type(arguments) is not tuple or any(
            type(arg) is not str or "\0" in arg for arg in arguments
        ):
            raise ValueError("launch_arguments")
        binding = load_binding(binding_path, binding_sha256)
        resource_plan = (
            _validate_sealed_resources(sealed_resources) if sealed_resources is not None else None
        )
        script = script.resolve(strict=True)
        script_sha = _sha(_read(script, 4 * 1024**2))
        _record(
            receipt,
            {"stage": "reserved", "script_sha256": script_sha, "binding_sha256": binding_sha256},
            reserve=True,
        )
        # The detached guardian may outlive this caller, including graceful
        # interruption. Its plan/evidence must never be removed by caller unwind.
        folder = receipt.with_name(receipt.name + ".work").resolve()
        folder.mkdir()
        work = str(folder)
        plan = {
            "binding": str(binding_path.resolve()),
            "binding_sha256": binding_sha256,
            "receipt": str(receipt.resolve()),
            "script": str(script),
            "script_sha256": script_sha,
            "arguments": arguments,
            "timeout_seconds": timeout_seconds,
            "work": work,
            "cwd": str(Path.cwd()),
            "launcher_sha256": _sha(_read(Path(__file__), 4 * 1024**2)),
            "completion": str(folder / "completion.json"),
            "result": str(folder / "result"),
        }
        if resource_plan is not None:
            plan["sealed_resources"] = resource_plan
            plan["sealed_resources_sources"] = _resource_sources()
        kernel_path = Path(__file__).with_name("qualify_evidence_slice_zero.py")
        plan.update(
            kernel_path=str(kernel_path), kernel_sha256=_sha(_read(kernel_path, 4 * 1024**2))
        )
        plan.update(stdout=str(folder / "stdout"), stderr=str(folder / "stderr"))
        path = folder / "plan.json"
        plan["plan_path"] = str(path)
        raw = json.dumps(plan).encode()
        if len(raw) > _MAX_JSON:
            raise ValueError("launch_plan_bound")
        path.write_bytes(raw)
        try:
            process = subprocess.Popen(
                (*guardian_command(binding), "--guardian", str(path), _sha(raw)),
                cwd=_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=WINDOWLESS_BREAKAWAY_FLAGS,
                close_fds=True,
            )
        except OSError as error:
            category = (
                "breakaway_not_permitted"
                if getattr(error, "winerror", None) == 5
                else "guardian_spawn_failed"
            )
            _record(
                receipt,
                {
                    "stage": "refused",
                    "category": category,
                    "controller_completed": False,
                    "cleanup_verified": None,
                },
            )
            raise RuntimeError(category) from error
        try:
            process.wait(timeout=timeout_seconds + 60)
        except subprocess.TimeoutExpired as primary:
            category = "guardian_timeout"
            failures: list[BaseException] = [primary]
            try:
                for operation in (process.kill, lambda: process.wait(timeout=30)):
                    try:
                        operation()
                    except BaseException as error:
                        failures.append(error)
            finally:
                try:
                    _record(
                        receipt,
                        {
                            "stage": "observer_timeout",
                            "category": category,
                            "controller_completed": False,
                            "cleanup_verified": None,
                        },
                    )
                except BaseException as error:
                    failures.append(error)
            if len(failures) > 1:
                raise BaseExceptionGroup("guardian timeout and cleanup failed", failures) from None
            raise TimeoutError("launch_guardian_deadline") from primary
        except KeyboardInterrupt:
            category = "interrupted"
            _record(receipt, {"stage": "observer_interrupted", "category": category})
            raise
        rows = _read(receipt, _MAX_JSON).splitlines()
        final = _json(rows[-1])
        if process.returncode != 0 or final.get("stage") != "complete":
            raise RuntimeError("launch_incomplete")
        stdout = _read(folder / "result", _MAX_OUTPUT)
        if _sha(stdout) != final.get("stdout_sha256") or len(stdout) != final.get("stdout_bytes"):
            raise ValueError("launch_result_binding")
        category = "complete"
        return LaunchResult(final["exit_code"], stdout)
    finally:
        print(
            "[qualification-launch] " + json.dumps({"version": 1, "category": category}),
            file=sys.stderr,
        )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--guardian":
        raise SystemExit(2)
    sys.path.insert(0, str(_ROOT))
    raise SystemExit(_guardian(Path(sys.argv[2]), sys.argv[3]))

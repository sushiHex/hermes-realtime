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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ROOT = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"[0-9a-f]{64}")
_MAX_JSON = 65536
_MAX_OUTPUT = 8 * 1024**2


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
    primary: BaseException | None = None
    deadline = time.monotonic() + plan["timeout_seconds"]
    folder = Path(plan["work"])
    output, errors = folder / "stdout", folder / "stderr"
    try:
        with open(os.devnull, "rb") as null, output.open("xb") as out, errors.open("xb") as err:
            handles = tuple(msvcrt.get_osfhandle(stream.fileno()) for stream in (null, out, err))
            for stream in (null, out, err):
                os.set_inheritable(stream.fileno(), True)
            job = kernel.create_job(core._WindowsJobLimitsV1(64, 64 * 1024**3, 128 * 1024**3))
            command = (binding.controller, "-B", "-c", _CONTROLLER, plan["plan_path"])
            environment = tuple(
                sorted(os.environ.items(), key=lambda item: (item[0].casefold(), item[0]))
            )
            flags = 0x4 | 0x200 | 0x400 | 0x80000 | 0x8000000
            root = kernel.create_process_suspended(
                command, environment, plan["cwd"], handles, flags, job=job
            )
            kernel.resume_thread(root.thread_handle)
            while True:
                if output.stat().st_size > _MAX_OUTPUT or errors.stat().st_size > _MAX_JSON:
                    raise ValueError("launch_output_bound")
                if time.monotonic() >= deadline:
                    raise TimeoutError("launch_controller_deadline")
                api = kernel._api()
                status = api.WaitForSingleObject(ctypes.c_void_p(root.process_handle), 50)
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
            try:
                kernel.terminate_job(job)
            except BaseException as error:
                failures.append(error)
            if root is not None:
                try:
                    kernel.wait(root.process_handle, max(1, int((until - time.monotonic()) * 1000)))
                except BaseException as error:
                    failures.append(error)
            try:
                while True:
                    active = kernel.query_job_active_process_count(job)
                    if type(active) is not int:
                        raise ValueError("launch_cleanup_count")
                    if active == 0:
                        cleanup = not failures
                        break
                    if time.monotonic() >= until:
                        raise TimeoutError("launch_cleanup_deadline")
                    time.sleep(0.01)
            except BaseException as error:
                failures.append(error)
            if cleanup:
                for handle in (
                    () if root is None else (root.thread_handle, root.process_handle)
                ) + (job,):
                    try:
                        kernel.close_handle(handle)
                    except BaseException as error:
                        failures.append(error)
        evidence["cleanup_verified"] = cleanup and not failures
        if primary is not None:
            failures.insert(0, primary)
        if failures:
            raise BaseExceptionGroup("controller execution or cleanup failed", failures)
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
            else "invalid_input"
            if isinstance(error, ValueError)
            else "execution_failed"
        )
    finally:
        _record(receipt, final)
    return 0 if final["stage"] == "complete" else 2


def run_owned_controller(
    script: Path,
    arguments: tuple[str, ...],
    *,
    binding_path: Path,
    binding_sha256: str,
    receipt: Path,
    timeout_seconds: int = 14400,
) -> LaunchResult:
    """Caller binds its controller/dependencies; this function owns no model policy."""
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
                creationflags=0x1000000 | 0x200 | 0x8000000,
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

"""Start the maintained preview owner independently of its Windows caller.

The caller binds the interpreter/active plan and supplies an existing private
output directory. A spawn is only a launch request; serve owns readiness, its
Containment and settlement. No fallback to an attached process is permitted.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

from scripts.qualification_launch import WINDOWLESS_BREAKAWAY_FLAGS, _outside_jobs

Category = Literal[
    "windows_required", "output_exists", "owner_output_unavailable",
    "breakaway_not_permitted", "owner_spawn_failed", "owner_inherited_job",
    "membership_unavailable",
    "interpreter_probe_failed", "owner_runtime_mismatch",
]

_BINDING_ENV = "HERMES_PREVIEW_OWNER_RUNTIME"
_IDENTITY_ENV = ("__PYVENV_LAUNCHER__", "PYTHONEXECUTABLE", _BINDING_ENV)
_FIELDS = {"executable", "base_executable", "prefix", "base_prefix",
           "executable_sha256", "base_sha256"}
# Fixed, trusted-interpreter code emits only six bounded path/digest fields.
_PROBE = (
    "import sys,json,hashlib,pathlib; "
    "h=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest(); "
    "print(json.dumps(dict(executable=sys.executable,base_executable=sys._base_executable,"
    "prefix=sys.prefix,base_prefix=sys.base_prefix,executable_sha256=h(sys.executable),"
    "base_sha256=h(sys._base_executable))))"
)


def _environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _IDENTITY_ENV}


def _digest(path: str) -> str:
    with Path(path).open("rb") as stream:
        raw = stream.read(64 * 1024**2 + 1)
    if len(raw) > 64 * 1024**2:
        raise ValueError("image_bound")
    return hashlib.sha256(raw).hexdigest()


def _validate_binding(value: object) -> dict[str, str]:
    if (type(value) is not dict or set(value) != _FIELDS
            or any(type(item) is not str or not 1 <= len(item) <= 4096
                   for item in value.values())):
        raise ValueError("runtime_binding")
    for key in ("executable", "base_executable", "prefix", "base_prefix"):
        if not Path(value[key]).is_absolute():
            raise ValueError("runtime_path")
    for path, digest in (("executable", "executable_sha256"), ("base_executable", "base_sha256")):
        if _digest(value[path]) != value[digest]:
            raise ValueError("runtime_image")
    return value


def _probe_interpreter(interpreter: Path) -> dict[str, str]:
    try:
        before = _digest(str(interpreter))
        result = subprocess.run(
            [str(interpreter), "-I", "-B", "-c", _PROBE], stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
            creationflags=WINDOWLESS_BREAKAWAY_FLAGS, close_fds=True, env=_environment(),
        )
        if result.returncode or len(result.stdout) > 8192:
            raise ValueError("probe_result")
        binding = _validate_binding(json.loads(result.stdout))
        if (Path(binding["executable"]).resolve() != interpreter.resolve()
                or binding["executable_sha256"] != before):
            raise ValueError("probe_identity")
        return binding
    except (OSError, ValueError, subprocess.TimeoutExpired):
        raise PreviewOwnerLaunchRefusal("interpreter_probe_failed") from None


def _actual_image() -> Path:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_ulong]
    kernel.GetModuleFileNameW.restype = ctypes.c_ulong
    buffer = ctypes.create_unicode_buffer(32768)
    length = kernel.GetModuleFileNameW(None, buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise OSError("image_unavailable")
    return Path(buffer.value)


def _verify_owner_runtime(interpreter: Path) -> None:
    try:
        raw = os.environ.pop(_BINDING_ENV, "")
        if not 0 < len(raw.encode("utf-8")) <= 8192:
            raise ValueError("binding_bound")
        binding = _validate_binding(json.loads(raw))
        if Path(binding["executable"]).resolve() != interpreter.resolve():
            raise ValueError("plan_interpreter")
        actual = {"executable": sys.executable, "base_executable": str(_actual_image()),
                  "prefix": sys.prefix, "base_prefix": sys.base_prefix}
        if any(Path(value).resolve() != Path(binding[key]).resolve()
               for key, value in actual.items()):
            raise ValueError("owner_identity")
        # CPython consumes this variable during initialization. Never propagate it.
        if "__PYVENV_LAUNCHER__" in os.environ:
            raise ValueError("launcher_not_consumed")
    except (OSError, ValueError):
        raise PreviewOwnerLaunchRefusal("owner_runtime_mismatch") from None


class PreviewOwnerLaunchRefusal(RuntimeError):
    def __init__(self, category: Category) -> None:
        super().__init__(category)
        self.category = category


def assert_independent_owner(*, interpreter: Path) -> None:
    """Call in the actual serve process before creating its own Containment."""
    try:
        outside = _outside_jobs()
    except OSError:
        raise PreviewOwnerLaunchRefusal("membership_unavailable") from None
    if not outside:
        raise PreviewOwnerLaunchRefusal("owner_inherited_job")
    _verify_owner_runtime(interpreter)


def launch_preview_owner(
    *, interpreter: Path, active: Path, output: Path,
) -> subprocess.Popen[bytes]:
    if sys.platform != "win32":
        raise PreviewOwnerLaunchRefusal("windows_required")
    binding = _probe_interpreter(interpreter)
    environment = _environment()
    # Reproduce CPython's own venv redirection without its extra child Job.
    environment["__PYVENV_LAUNCHER__"] = binding["executable"]
    environment[_BINDING_ENV] = json.dumps(binding, separators=(",", ":"))
    try:
        # Inherit the prepared output directory's private ACL. Exclusive files
        # preserve earlier evidence and never depend on a caller-owned pipe.
        with (output / "owner.private.log").open("xb") as stdout, (
            output / "owner.stderr.private.log"
        ).open("xb") as stderr:
            try:
                return subprocess.Popen(
                    [binding["base_executable"], "-I", "-B",
                     str(Path(__file__).with_name("preview_cutover.py")),
                     "serve", "--active", str(active)],
                    cwd=Path(__file__).resolve().parents[1],
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=WINDOWLESS_BREAKAWAY_FLAGS, close_fds=True,
                    env=environment,
                )
            except OSError as error:
                category: Category = (
                    "breakaway_not_permitted" if getattr(error, "winerror", None) == 5
                    else "owner_spawn_failed"
                )
                raise PreviewOwnerLaunchRefusal(category) from None
    except FileExistsError:
        raise PreviewOwnerLaunchRefusal("output_exists") from None
    except OSError:
        raise PreviewOwnerLaunchRefusal("owner_output_unavailable") from None

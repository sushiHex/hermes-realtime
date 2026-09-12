"""Run only the three literal PluginManager stages selected from a sealed harness."""

from __future__ import annotations

import ast
import contextlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

_NAMES = ("_DISABLED_CHILD", "_ENABLE_CHILD", "_ENABLED_CHILD")
_DIST_INFO = re.compile(
    r"hermes_realtime-0\.0\.3\.dist-info/"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}"
    r"(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,255})*"
)
_MISSING = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z0-9_.]{1,128})'")


class _StageFailure(ValueError):
    def __init__(self, stage: str, module: str | None = None) -> None:
        self.stage = stage
        self.module = module
        super().__init__("PluginManager sealed stage failed")


class _Readable(Protocol):
    def read(self, size: int) -> bytes: ...


def _fail() -> int:
    return 1


def _stages(harness: Path) -> dict[str, str]:
    tree = ast.parse(harness.read_text(encoding="utf-8"), filename=str(harness))
    values: dict[str, str] = {}
    writes = {
        name: [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ]
        for name in _NAMES
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in _NAMES:
            continue
        if (
            target.id in values
            or not isinstance(node.value, ast.Constant)
            or type(node.value.value) is not str
        ):
            raise ValueError("sealed harness stages differ")
        values[target.id] = node.value.value
    if set(values) != set(_NAMES) or any(len(writes[name]) != 1 for name in _NAMES) or any(
        not values[name] or len(values[name]) > 32_768 for name in _NAMES
    ):
        raise ValueError("sealed harness stages are incomplete")
    return values


def _drain(
    stream: _Readable,
    retained: bytearray,
    state: list[bool | None],
    index: int,
) -> None:
    try:
        while True:
            chunk = stream.read(4096)
            if not chunk:
                state[index] = True
                return
            remaining = 4097 - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])
    except OSError:
        state[index] = False
        return


def _stage(
    name: str, code: str, arguments: tuple[str, ...], environment: dict[str, str]
) -> dict[str, object]:
    child = subprocess.Popen(
        (sys.executable, "-I", "-S", "-B", "-c", code, *arguments),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
        cwd=environment["TEMP"],
    )
    assert child.stdout is not None and child.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    drain_state: list[bool | None] = [None, None]
    readers = (
        threading.Thread(
            target=_drain, args=(child.stdout, stdout, drain_state, 0), daemon=True
        ),
        threading.Thread(
            target=_drain, args=(child.stderr, stderr, drain_state, 1), daemon=True
        ),
    )
    for reader in readers:
        reader.start()
    finalization_deadline = time.monotonic() + 190
    timed_out = False
    try:
        returncode = child.wait(timeout=180)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = None
        with contextlib.suppress(OSError):
            child.kill()
        remaining = finalization_deadline - time.monotonic()
        if remaining > 0:
            with contextlib.suppress(subprocess.TimeoutExpired):
                returncode = child.wait(timeout=remaining)
    for reader in readers:
        reader.join(timeout=max(0.0, finalization_deadline - time.monotonic()))
    if (
        timed_out
        or returncode is None
        or any(reader.is_alive() for reader in readers)
        or drain_state != [True, True]
    ):
        raise _StageFailure(name)
    raw_stdout, raw_stderr = bytes(stdout), bytes(stderr)
    if returncode != 0 or len(raw_stdout) > 4096 or len(raw_stderr) > 4096:
        match = _MISSING.search(raw_stderr.decode("utf-8", errors="replace"))
        raise _StageFailure(name, match.group(1) if match is not None else None)
    value = json.loads(raw_stdout.decode("utf-8"))
    if json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n" != raw_stdout:
        raise ValueError("PluginManager sealed stage output differs")
    if type(value) is not dict:
        raise ValueError("PluginManager sealed stage output differs")
    return value


def _write(output: Path, value: dict[str, object]) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with output.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def main(arguments: list[str]) -> int:
    if len(arguments) != 7:
        return _fail()
    try:
        harness, source, packages, profile, default = (
            Path(item).resolve(strict=True) for item in arguments[:5]
        )
        output = Path(arguments[5]).resolve(strict=False)
        expected_dist_info = arguments[6]
        if (
            not harness.is_file()
            or not 0 < harness.stat().st_size <= 2 * 1024**2
            or not source.is_dir()
            or not packages.is_dir()
            or not profile.is_dir()
            or not default.is_dir()
            or output.exists()
            or output.parent.resolve(strict=True) != profile.parent
            or default.parent != profile.parent
            or default == profile
        ):
            return _fail()
        if len(expected_dist_info) > 32_768:
            raise ValueError("installed distribution inventory is unbounded")
        inventory = json.loads(expected_dist_info)
        if (
            type(inventory) is not list
            or not 1 <= len(inventory) <= 1024
            or inventory != sorted(inventory)
            or len(inventory) != len(set(inventory))
            or any(
                type(item) is not str or _DIST_INFO.fullmatch(item) is None
                for item in inventory
            )
            or json.dumps(inventory, separators=(",", ":")) != expected_dist_info
        ):
            raise ValueError("installed distribution inventory differs")
        stages = _stages(harness)
        environment = {
            key: os.environ[key]
            for key in ("SystemRoot", "WINDIR", "SystemDrive", "PATH", "TEMP", "TMP", "TMPDIR")
            if key in os.environ
        } | {
            "HERMES_HOME": str(profile),
            "HOME": str(default),
            "LOCALAPPDATA": str(default),
            "USERPROFILE": str(default),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        values = {
            "disabled": _stage(
                "disabled",
                stages["_DISABLED_CHILD"], (str(source), str(profile), str(packages)), environment
            ),
            "enable": _stage(
                "enable",
                stages["_ENABLE_CHILD"], (str(source), str(profile), str(packages)), environment
            ),
            "enabled": _stage(
                "enabled",
                stages["_ENABLED_CHILD"],
                (
                    str(source),
                    str(profile),
                    str(packages),
                    json.dumps(inventory, separators=(",", ":")),
                ),
                environment,
            ),
        }
        value = {"pid": os.getpid(), "stages": values, "version": 1}
        _write(output, value)
    except _StageFailure as error:
        if "output" in locals() and not output.exists():
            _write(
                output,
                {
                    "error": "ModuleNotFoundError" if error.module is not None else "StageFailure",
                    "module": error.module,
                    "pid": os.getpid(),
                    "stage": error.stage,
                    "version": 1,
                },
            )
        return _fail()
    except (
        OSError,
        SyntaxError,
        ValueError,
        subprocess.SubprocessError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return _fail()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

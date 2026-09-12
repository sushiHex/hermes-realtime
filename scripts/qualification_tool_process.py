"""Retain completion of a tool invocation under the existing Windows Job owner.

This private primitive records the actual command and environment; a governed
recipe must separately bind those to its source, inputs and outputs. It grants no
qualification or workspace-cleanup authority. Callers must keep tool seals live
through every owned consumer's verified exit, including retryable cleanup errors.
The complete runner's durable recovery and atomic acquisition remain separate.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from weakref import WeakKeyDictionary

from scripts import qualify_evidence_slice_zero as core
from scripts.qualification_tool_environment import (
    ImmutableToolEnvironmentV1,
    _tool_image_for_consumer,
    tool_environment_metadata,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _exit_code(kernel: Any, handle: int) -> int:
    api = kernel._api()
    api.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    api.GetExitCodeProcess.restype = ctypes.c_int
    code = ctypes.c_uint32()
    _require(
        bool(api.GetExitCodeProcess(handle, ctypes.byref(code))), "tool exit code is unavailable"
    )
    return int(code.value)


def _wait_for_tool_exit(kernel: Any, process: int, job: int, timeout_ms: int) -> None:
    """Require root exit and Job quiescence within one shared wait budget."""
    deadline = monotonic() + timeout_ms / 1000
    kernel.wait(process, timeout_ms)
    code = _exit_code(kernel, process)
    while True:
        active = kernel.query_job_active_process_count(job)
        _require(type(active) is int and active >= 0, "tool Job accounting is invalid")
        _require(monotonic() <= deadline, "tool retains active consumers beyond its wait budget")
        if active == 0:
            break
        remaining = deadline - monotonic()
        _require(remaining > 0, "tool retains active consumers beyond its wait budget")
        sleep(min(0.01, remaining))
    _require(code == 0, "tool did not exit normally")


@dataclass(frozen=True, slots=True)
class ToolInvocationMetadataV1:
    role: str
    image_sha256: str
    command_sha256: str
    environment_sha256: str
    distribution_sha256s: tuple[tuple[str, str], ...]
    exit_code: int


class CompletedToolInvocationV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("tool invocation receipts are owner-minted only")


@dataclass(frozen=True, slots=True)
class _Invocation:
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    working_directory: str
    process: core._WindowsProcessIdentityV1
    cleanup: core._WindowsFinalizationResultV1
    metadata: ToolInvocationMetadataV1


_COMPLETED: WeakKeyDictionary[CompletedToolInvocationV1, _Invocation] = WeakKeyDictionary()


def _run_tool(
    tools: ImmutableToolEnvironmentV1,
    role: str,
    arguments: tuple[str, ...],
    workspace: Path,
    *,
    timeout_milliseconds: int = 60_000,
) -> CompletedToolInvocationV1:
    _require(os.name == "nt", "tool invocation requires Windows")
    _require(role in {"build_python", "uv"}, "tool invocation role is unavailable")
    _require(
        type(arguments) is tuple
        and 0 < len(arguments) <= 128
        and all(type(value) is str and value and "\x00" not in value for value in arguments),
        "tool arguments differ from the bounded command profile",
    )
    _require(
        type(timeout_milliseconds) is int and 1 <= timeout_milliseconds <= 600_000,
        "tool invocation timeout is outside its bound",
    )
    image, digest, _ = _tool_image_for_consumer(tools, role)
    distributions = tool_environment_metadata(tools)
    _require(
        isinstance(workspace, Path) and workspace.is_absolute(), "tool workspace is not absolute"
    )
    directory = workspace.resolve(strict=True)
    _require(directory == workspace and directory.is_dir(), "tool workspace is indirect")
    isolation = ("-I", "-S", "-B") if role == "build_python" else ()
    command = (str(image), *isolation, *arguments)
    environment = {
        name: os.environ[name]
        for name in ("SystemRoot", "SystemDrive", "WINDIR")
        if name in os.environ
    } | {
        "PATH": str(image.parent),
        "TEMP": str(directory),
        "TMP": str(directory),
        "TMPDIR": str(directory),
        "UV_NO_CONFIG": "1",
        "UV_NO_CACHE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    pairs = tuple(sorted(environment.items(), key=lambda item: (item[0].casefold(), item[0])))
    kernel = core._CtypesWindowsKernelV1()
    runner_handle = kernel.open_process(
        os.getpid(), core._SYNCHRONIZE | core._PROCESS_QUERY_LIMITED_INFORMATION
    )
    job = None
    try:
        raw = kernel.query_process_identity(runner_handle)
        runner = core._WindowsRunnerIdentityV1(raw.pid, raw.creation_filetime)
        # The fixed recipes use output artifacts; no ambient stdin/stdout is an
        # observation channel. Retain one exact inherited null handle for Win32.
        with open(os.devnull, "r+b", buffering=0) as null:
            import msvcrt

            os.set_inheritable(null.fileno(), True)
            handle = msvcrt.get_osfhandle(null.fileno())
            spec = core._WindowsScenarioSpecV1(
                "qualification_tool",
                command,
                pairs,
                str(directory),
                timeout_milliseconds,
                (handle,),
                core._WindowsJobLimitsV1(8, 2 * 1024**3, 4 * 1024**3),
                no_window=True,
            )
            rules = tuple(
                [core._WindowsRoleRuleV1("tool_root", image.name, digest, frozenset(), True)]
                + [
                    core._WindowsRoleRuleV1(
                        child_role,
                        _tool_image_for_consumer(tools, child_role)[0].name,
                        _tool_image_for_consumer(tools, child_role)[1],
                        frozenset({"tool_root", "build_python", "uv"}),
                        False,
                    )
                    for child_role in ("build_python", "uv")
                ]
            )
            job = core._WindowsScenarioJobV1(kernel, spec, runner, rules)
            root = job.launch_root()

            def observe() -> None:
                assert job is not None and job._job_handle is not None
                _wait_for_tool_exit(
                    kernel,
                    root.process_handle,
                    job._job_handle,
                    timeout_milliseconds,
                )
                _require(
                    tool_environment_metadata(tools) == distributions, "tool environment changed"
                )

            core._run_with_windows_scenario_job_finalization_v1(job, observe)
            cleanup = job.last_finalization
            _require(
                cleanup is not None
                and cleanup.closed
                and cleanup.zero_active_observed
                and not cleanup.failures
                and not cleanup.failed_handles
                and root.process_handle in cleanup.waited_handles,
                "tool retained ownership cleanup is incomplete",
            )
            assert cleanup is not None
            _require(tool_environment_metadata(tools) == distributions, "tool environment changed")
    finally:
        try:
            if job is not None and job.last_finalization is None:
                job.finalize()
        finally:
            kernel.close_handle(runner_handle)
    metadata = ToolInvocationMetadataV1(
        role,
        digest,
        _digest(command),
        _digest(pairs),
        tuple((item.role, item.distribution_sha256) for item in distributions),
        0,
    )
    receipt = object.__new__(CompletedToolInvocationV1)
    _COMPLETED[receipt] = _Invocation(
        command, pairs, str(directory), root.identity, cleanup, metadata
    )
    return receipt


def _tool_invocation_for_consumer(receipt: CompletedToolInvocationV1) -> _Invocation:
    if type(receipt) is not CompletedToolInvocationV1:
        raise TypeError("tool invocation capability type differs")
    _require(receipt in _COMPLETED, "tool invocation capability is unregistered")
    return _COMPLETED[receipt]


def tool_invocation_metadata(receipt: CompletedToolInvocationV1) -> ToolInvocationMetadataV1:
    return _tool_invocation_for_consumer(receipt).metadata

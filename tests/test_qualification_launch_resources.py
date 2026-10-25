"""Synthetic guardian resource ownership; never execute a provider or native child."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import qualification_launch as launch


def _resources(tmp_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution": {
            "root": str(tmp_path / "source"),
            "files": [
                {"path": "tool.exe", "sha256": hashlib.sha256(b"tool").hexdigest(), "size_bytes": 4}
            ],
        },
        "inputs": [
            {
                "root": str(tmp_path / "inputs"),
                "files": [
                    {
                        "path": "weights.bin",
                        "sha256": hashlib.sha256(b"input").hexdigest(),
                        "size_bytes": 5,
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "version",
        "root",
        "root_parent",
        "empty",
        "member",
        "case_alias",
        "order",
        "digest",
        "size_bool",
        "size_negative",
        "file_bound",
        "total_bound",
        "tree_bound",
        "namespace",
        "input_unknown",
        "input_size",
        "input_total",
        "count",
        "file_unknown",
        "directory_alias",
        "aggregate_count",
        "directory_count",
    ],
)
def test_resource_declaration_is_exact_and_bounded(tmp_path: Path, fault: str) -> None:
    value = _resources(tmp_path)
    execution = value["execution"]
    file = execution["files"][0]
    if fault == "unknown":
        value["unknown"] = True
    elif fault == "version":
        value["schema_version"] = True
    elif fault == "root":
        execution["root"] = "relative"
    elif fault == "root_parent":
        execution["root"] = str(tmp_path / ".." / "source")
    elif fault == "empty":
        execution["files"] = []
    elif fault == "member":
        file["path"] = "../tool.exe"
    elif fault == "case_alias":
        execution["files"].append({**file, "path": "TOOL.EXE"})
        execution["files"].sort(key=lambda item: item["path"])
    elif fault == "order":
        execution["files"].append({**file, "path": "a.exe"})
    elif fault == "digest":
        file["sha256"] = "a" * 63
    elif fault == "size_bool":
        file["size_bytes"] = True
    elif fault == "size_negative":
        file["size_bytes"] = -1
    elif fault == "file_bound":
        file["size_bytes"] = 512 * 1024**2 + 1
    elif fault == "total_bound":
        execution["files"] = [
            {**file, "path": f"{i:02}.exe", "size_bytes": 512 * 1024**2} for i in range(9)
        ]
    elif fault == "tree_bound":
        value["inputs"] *= 17
    elif fault == "namespace":
        execution["files"].append({**file, "path": "tool.exe/child"})
    elif fault == "input_unknown":
        value["inputs"][0]["unknown"] = True
    elif fault == "input_size":
        value["inputs"][0]["files"][0]["size_bytes"] = 16 * 1024**3 + 1
    elif fault == "input_total":
        value["inputs"][0]["files"] = [
            {**file, "path": f"{i}.bin", "size_bytes": 16 * 1024**3} for i in range(5)
        ]
    elif fault == "file_unknown":
        file["unknown"] = True
    elif fault == "directory_alias":
        execution["files"] = [{**file, "path": name} for name in ("A/x", "a/y")]
    elif fault == "aggregate_count":
        execution["files"] = [{**file, "path": f"{i:05}.exe"} for i in range(8193)]
        value["inputs"][0]["files"] = [{**file, "path": f"{i:05}.bin"} for i in range(8193)]
    elif fault == "directory_count":
        execution["files"] = [{**file, "path": f"{i:05}/a/tool.exe"} for i in range(8193)]
    else:
        execution["files"] = [{**file, "path": f"{i:05}.exe"} for i in range(16385)]
    with pytest.raises(ValueError):
        launch._validate_sealed_resources(value)


def test_valid_resource_declaration_is_copied(tmp_path: Path) -> None:
    value = _resources(tmp_path)
    original = copy.deepcopy(value)
    actual = launch._validate_sealed_resources(value)
    assert actual == original
    value["execution"]["files"][0]["sha256"] = "0" * 64
    assert actual == original


@pytest.mark.parametrize("fault", ["none", "hash", "size", "members", "copied", "cleanup"])
def test_guardian_verifies_held_bytes_and_owns_both_resource_kinds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from scripts import qualification_execution_files as execution
    from scripts import qualification_file_seals as seals

    events = []
    value = _resources(tmp_path)
    expected = {"tool.exe": b"tool", "weights.bin": b"input"}

    @contextmanager
    def retain(root: Path, members: tuple[str, ...]):
        events.append(("retain", root.name, members))
        try:
            yield members
        finally:
            events.append(("release", root.name))
            if fault == "cleanup" and root.name == "inputs":
                raise OSError("synthetic resource cleanup")

    def metadata(receipt):
        items = tuple(
            (name, hashlib.sha256(expected[name]).hexdigest(), len(expected[name]))
            for name in receipt
        )
        if receipt == ("tool.exe",):
            if fault == "hash":
                return (("tool.exe", "0" * 64, 4),)
            if fault == "size":
                return (("tool.exe", items[0][1], 5),)
            if fault == "members":
                return (*items, ("extra", "0" * 64, 0))
        return items

    def read(receipt, name, maximum):
        assert receipt == (name,) and maximum == len(expected[name])
        events.append(("read", name))
        return b"drift" if fault == "copied" else expected[name]

    @contextmanager
    def snapshot(contents):
        assert contents == {"tool.exe": b"tool"}
        events.append(("snapshot",))
        try:
            yield "snapshot"
        finally:
            events.append(("snapshot_release",))

    monkeypatch.setattr(seals, "retain_file_seals", retain)
    monkeypatch.setattr(seals, "sealed_file_metadata", metadata)
    monkeypatch.setattr(seals, "sealed_file_bytes", read)
    monkeypatch.setattr(execution, "owned_execution_files", snapshot)
    monkeypatch.setattr(execution, "_execution_files_for_consumer", lambda _: tmp_path / "fresh")
    stack = ExitStack()
    if fault not in {"none", "cleanup"}:
        with stack, pytest.raises(ValueError):
            launch._acquire_sealed_resources(value, stack)
        assert ("snapshot",) not in events
    else:
        actual = launch._acquire_sealed_resources(value, stack)
        assert actual == str(tmp_path / "fresh")
        assert not any("release" in event[0] for event in events)
        assert ("read", "weights.bin") not in events  # Large inputs are never copied.
        if fault == "cleanup":
            with pytest.raises(OSError, match="synthetic resource cleanup"):
                stack.close()
        else:
            stack.close()
        assert ("snapshot_release",) in events
        assert ("release", "inputs") in events


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "held",
        "partial",
        "source",
        "acquisition_deadline",
        "active",
        "query",
        "close",
        "release",
        "env",
    ],
)
def test_resources_survive_until_all_consumers_settle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from scripts import qualify_evidence_slice_zero as core

    events = []
    live = [False]
    clock = [0]

    def acquire(value, stack):
        events.append("acquire")
        assert value == _resources(tmp_path)
        if fault == "held":
            raise ValueError("synthetic held mismatch")
        live[0] = True

        def release():
            events.append("release")
            live[0] = False
            if fault == "release":
                invalid = OSError("PRIVATE_SENTINEL")
                invalid.winerror = 65536
                failure = OSError("PRIVATE_SENTINEL")
                failure.winerror = 32
                raise BaseExceptionGroup(
                    "PRIVATE_SENTINEL", [invalid, BaseExceptionGroup("PRIVATE_SENTINEL", [failure])]
                )

        stack.callback(release)
        if fault == "partial":
            raise ValueError("synthetic partial acquisition failure")
        if fault == "acquisition_deadline":
            clock[0] = 10
        return str(tmp_path / "fresh")

    class Kernel:
        def __init__(self):
            def exit_code(handle, code):
                code._obj.value = 0
                return 1

            self.api = SimpleNamespace(
                WaitForSingleObject=lambda *a: 0, GetExitCodeProcess=exit_code
            )

        def _api(self):
            return self.api

        def create_job(self, limits):
            assert live[0]
            events.append("job")
            return 10

        def create_process_suspended(self, command, environment, cwd, handles, flags, *, job):
            assert live[0]
            assert dict(environment)[launch.EXECUTION_ROOT_ENV] == str(tmp_path / "fresh")
            events.append("create")
            return SimpleNamespace(process_handle=20, thread_handle=21)

        def resume_thread(self, thread):
            assert live[0]
            events.append("resume")

        def terminate_job(self, job):
            assert live[0]
            events.append("terminate")

        def wait(self, process, timeout):
            assert live[0]
            events.append("wait")

        def query_job_active_process_count(self, job):
            assert live[0]
            events.append("query")
            if fault == "active":
                clock[0] = 40
                return 1
            if fault == "query":
                raise OSError("synthetic query failure")
            return 0

        def close_handle(self, handle):
            assert live[0]
            events.append(("close", handle))
            if fault == "close":
                raise OSError("synthetic handle cleanup")

    monkeypatch.setattr(launch, "_acquire_sealed_resources", acquire)

    def verify(*args, **kwargs):
        if fault == "source":
            raise ValueError("synthetic source mismatch")

    monkeypatch.setattr(launch, "_verify_resource_sources", verify)
    monkeypatch.setattr(
        launch, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda _: None)
    )
    monkeypatch.setattr(core, "_CtypesWindowsKernelV1", Kernel)
    if os.name != "nt":
        monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(get_osfhandle=lambda fd: fd))
    monkeypatch.delenv(launch.EXECUTION_ROOT_ENV, raising=False)
    if fault == "env":
        monkeypatch.setenv(launch.EXECUTION_ROOT_ENV.lower(), str(tmp_path / "unbound"))
    kernel_path = Path(core.__file__).resolve()
    plan = {
        "timeout_seconds": 10,
        "work": str(tmp_path),
        "plan_path": "unused",
        "cwd": str(tmp_path),
        "completion": str(tmp_path / "completion"),
        "kernel_path": str(kernel_path),
        "kernel_sha256": hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
        "sealed_resources": _resources(tmp_path),
        "sealed_resources_sources": {},
        "execution_root": str(tmp_path / "unbound"),
    }
    (tmp_path / "completion").write_text('{"normal_exit":true,"exit_code":0}')
    evidence = {}
    try:
        if fault == "none":
            assert (
                launch._owned_controller(
                    plan, launch.InterpreterBinding("base", "venv", "x", "y"), evidence
                )[2]
                is True
            )
            assert events[-1] == "release"
            assert evidence["cleanup_verified"] is True
        else:
            with pytest.raises(BaseExceptionGroup):
                launch._owned_controller(
                    plan, launch.InterpreterBinding("base", "venv", "x", "y"), evidence
                )
            assert evidence["cleanup_verified"] is False
            if fault in {"active", "query"}:
                assert live[0] and "release" not in events
                assert ("close", 10) in events  # Kill-on-close before hard guardian exit.
                assert len(launch._UNSETTLED_RESOURCES) == 1
            elif fault in {"env", "held", "partial", "source", "acquisition_deadline"}:
                assert "job" not in events
                assert "create" not in events and "resume" not in events
                assert events == (
                    []
                    if fault == "env"
                    else ["acquire"]
                    if fault == "held"
                    else ["acquire", "release"]
                )
            else:
                assert events[-1] == "release" and not live[0]
            if fault == "release":
                assert evidence == {
                    "cleanup_verified": False,
                    "failure_phase": "resource_release",
                    "failure_category": "cleanup_failed",
                    "resource_release_winerror": 32,
                }
                assert "PRIVATE_SENTINEL" not in json.dumps(evidence)
    finally:
        # Synthetic consumers are now stopped. Avoid leaking test-only callbacks.
        for stack in launch._UNSETTLED_RESOURCES:
            stack.close()
        launch._UNSETTLED_RESOURCES.clear()


def test_unsettled_guardian_records_failure_before_bypassing_generator_finalizers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.jsonl"
    events = []
    stack = ExitStack()
    stack.callback(lambda: events.append("release"))

    def owned(plan, binding, final):
        launch._UNSETTLED_RESOURCES.append(stack)
        raise RuntimeError("synthetic unproven settlement")

    class HardExit(BaseException):
        pass

    def hard_exit(code):
        assert code == 2
        assert json.loads(receipt.read_bytes().splitlines()[-1])["stage"] == "incomplete"
        assert not events
        raise HardExit

    monkeypatch.setattr(launch, "_owned_controller", owned)
    monkeypatch.setattr(launch, "_outside_jobs", lambda: True)
    monkeypatch.setattr(
        launch, "load_binding", lambda *a, **k: launch.InterpreterBinding("b", "v", "x", "y")
    )
    monkeypatch.setattr(launch.os, "_exit", hard_exit)
    raw = json.dumps(
        {
            "receipt": str(receipt),
            "binding": "unused",
            "binding_sha256": "unused",
            "launcher_sha256": hashlib.sha256(Path(launch.__file__).read_bytes()).hexdigest(),
        }
    ).encode()
    path = tmp_path / "plan"
    path.write_bytes(raw)
    try:
        with pytest.raises(HardExit):
            launch._guardian(path, hashlib.sha256(raw).hexdigest())
        assert "release" not in events
    finally:
        stack.close()
        launch._UNSETTLED_RESOURCES.clear()


@pytest.mark.parametrize("fault", ["none", "missing", "unknown", "digest", "origin"])
def test_resource_source_inventory_binds_exact_bytes_and_loaded_origins(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    from scripts import qualification_execution_files  # noqa: F401

    sources = launch._resource_sources()
    if fault == "missing":
        sources.pop("qualification_file_seals")
    elif fault == "unknown":
        sources["other"] = "0" * 64
    elif fault == "digest":
        sources["qualification_file_seals"] = "0" * 64
    elif fault == "origin":
        monkeypatch.setitem(
            sys.modules, "qualification_file_seals", SimpleNamespace(__file__="wrong-source.py")
        )
    if fault == "none":
        launch._verify_resource_sources(sources, imported=True)
    else:
        with pytest.raises(ValueError):
            launch._verify_resource_sources(sources, imported=fault == "origin")


@pytest.mark.parametrize("fault", ["none", "source"])
def test_guardian_binds_resource_summary_and_checks_sources_before_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    receipt = tmp_path / "receipt.jsonl"
    value = _resources(tmp_path)
    sources = launch._resource_sources()
    if fault == "source":
        sources["qualification_file_seals"] = "0" * 64
    called = []

    def owned(plan, binding, final):
        called.append(True)
        return b"synthetic", {"normal_exit": True, "exit_code": 0}, True

    monkeypatch.setattr(launch, "_owned_controller", owned)
    monkeypatch.setattr(launch, "_outside_jobs", lambda: True)
    monkeypatch.setattr(
        launch, "load_binding", lambda *a, **k: launch.InterpreterBinding("b", "v", "x", "y")
    )
    raw = json.dumps(
        {
            "receipt": str(receipt),
            "binding": "unused",
            "binding_sha256": "unused",
            "launcher_sha256": hashlib.sha256(Path(launch.__file__).read_bytes()).hexdigest(),
            "sealed_resources": value,
            "sealed_resources_sources": sources,
            "result": str(tmp_path / "result"),
        }
    ).encode()
    path = tmp_path / "plan"
    path.write_bytes(raw)
    assert launch._guardian(path, hashlib.sha256(raw).hexdigest()) == (
        2 if fault == "source" else 0
    )
    final = json.loads(receipt.read_bytes().splitlines()[-1])
    if fault == "source":
        assert not called and final["stage"] == "incomplete"
    else:
        assert final["stage"] == "complete"
        assert (
            final["sealed_resources_sha256"]
            == hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        )
        assert final["execution_file_count"] == final["input_file_count"] == 1
        assert final["input_tree_count"] == 1
        assert "root" not in final and str(tmp_path) not in json.dumps(final)


@pytest.mark.skipif(os.name != "nt", reason="Windows caller boundary, subprocess fully mocked")
@pytest.mark.parametrize("resources_enabled", [True, False])
def test_caller_serializes_declaration_and_sources_without_owning_resource_contexts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resources_enabled: bool,
) -> None:
    value = _resources(tmp_path)
    script = tmp_path / "controller.py"
    script.write_text("pass\n")
    receipt = tmp_path / "receipt.jsonl"
    called = []

    def popen(command, **kwargs):
        plan_path = Path(command[-2])
        raw = plan_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == command[-1]
        plan = json.loads(raw)
        assert ("sealed_resources" in plan) is resources_enabled
        if resources_enabled:
            assert plan["sealed_resources"] == value
            assert plan["sealed_resources_sources"] == launch._resource_sources()
        assert "execution_root" not in plan
        called.append(True)

        def wait(**kwargs):
            (Path(plan["work"]) / "result").write_bytes(b"synthetic")
            launch._record(
                receipt,
                {
                    "stage": "complete",
                    "exit_code": 0,
                    "stdout_sha256": hashlib.sha256(b"synthetic").hexdigest(),
                    "stdout_bytes": 9,
                },
            )

        return SimpleNamespace(wait=wait, returncode=0)

    monkeypatch.setattr(
        launch, "load_binding", lambda *a, **k: launch.InterpreterBinding("b", "v", "x", "y")
    )
    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(
        launch,
        "_acquire_sealed_resources",
        lambda *a: pytest.fail("caller acquired guardian resources"),
    )
    assert (
        launch.run_owned_controller(
            script,
            (),
            binding_path=tmp_path / "binding",
            binding_sha256="0" * 64,
            receipt=receipt,
            sealed_resources=value if resources_enabled else None,
        ).stdout
        == b"synthetic"
    )
    assert called == [True]

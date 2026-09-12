"""Tool execution evidence requires live files, real exit and complete Job cleanup."""

import os
from pathlib import Path
from runpy import run_path

import pytest

tools = run_path(str(Path(__file__).with_name("test_qualification_build_inputs.py")))["tools"]
pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows tool process ownership")


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "exit",
        "wait",
        "active",
        "close",
        "unobserved_child",
        "limit",
        "observed_child",
        "foreign_child",
        "child_exit",
        "observed_shell",
    ],
)
def test_process_failure_or_incomplete_cleanup_cannot_mint_completion(
    tmp_path, tools, monkeypatch, fault
):
    from scripts import qualification_tool_process as process
    from scripts import qualify_evidence_slice_zero as core
    from scripts.candidate_source_archive_oracle import _trusted_windows_directories
    from scripts.qualification_tool_environment import _tool_image_for_consumer

    helpers = run_path(str(Path.cwd() / "tests/test_qualify_evidence_slice_zero.py"))
    kernel = helpers["_FakeWindowsKernel"](core)
    image, digest, _ = _tool_image_for_consumer(tools, "build_python")
    kernel.opened[os.getpid()] = 205
    kernel.identities[205] = kernel._identity(os.getpid(), 1, 70, "runner.exe", "f" * 64)
    kernel.identities[201] = kernel._identity(
        42,
        os.getpid(),
        420,
        image.name,
        digest,
        parent_creation=70,
    )
    child = fault in {"observed_child", "foreign_child", "child_exit", "observed_shell"}
    if child:
        kernel.members.append(43)
        if fault == "observed_shell":
            import hashlib

            _, system = _trusted_windows_directories()
            shell = Path(system) / "cmd.exe"
            kernel.identities[203] = kernel._identity(
                43, 42, 430, shell.name, hashlib.sha256(shell.read_bytes()).hexdigest()
            )
        elif fault != "foreign_child":
            kernel.identities[203] = kernel._identity(43, 42, 430, image.name, digest)
    for name in ("SystemRoot", "SystemDrive", "WINDIR"):
        monkeypatch.setenv(name, str(tmp_path / "untrusted-ambient-root"))
    monkeypatch.setattr(
        process,
        "_job_accounting",
        lambda _kernel, _job: (
            2 if fault == "unobserved_child" or child else 1,
            kernel.active_count,
            1 if fault == "limit" else 0,
        ),
        raising=False,
    )
    if fault == "wait":
        kernel.fail.add("wait:201")
    elif fault == "active":
        kernel.active_count = 1
        clock = iter([0.0, 60.0, 60.0])
        monkeypatch.setattr(process, "monotonic", lambda: next(clock))
    elif fault == "close":
        kernel.fail_close.add(201)
    monkeypatch.setattr(core, "_CtypesWindowsKernelV1", lambda: kernel)
    monkeypatch.setattr(
        process,
        "_exit_code",
        lambda _kernel, _handle: (
            1 if fault == "exit" or (fault == "child_exit" and _handle == 203) else 0
        ),
    )
    if fault in {None, "observed_child", "observed_shell"}:
        receipt = process._run_tool(tools, "build_python", ("-c", "pass"), tmp_path)
        metadata = process.tool_invocation_metadata(receipt)
        assert metadata.exit_code == 0 and metadata.image_sha256 == digest
        assert len(metadata.command_sha256) == 64
        assert ("wait", 201, 0) in kernel.events
        assert metadata.process_count == (2 if child else 1)
        if child:
            assert ("identity", 203) in kernel.events
            assert ("wait", 203, 0) in kernel.events
            assert ("close", 203) in kernel.events
        assert ("close", 201) in kernel.events and ("close", 205) in kernel.events
        launched = next(row for row in kernel.events if row[0] == "create_process")
        assert launched[1][:4] == (str(image), "-I", "-S", "-B")
        environment = dict(launched[2])
        windows, _ = _trusted_windows_directories()
        assert environment["SystemRoot"] == environment["WINDIR"] == windows
        assert environment["SystemDrive"] == Path(windows).drive
        assert environment["TEMP"] == environment["TMP"] == environment["TMPDIR"] == str(tmp_path)
        assert not {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}.intersection(environment)
    else:
        with pytest.raises((ValueError, core._WindowsScenarioJobError, ExceptionGroup)):
            process._run_tool(tools, "build_python", ("-c", "pass"), tmp_path)
        assert any(row[0] == "terminate_job" for row in kernel.events)


@pytest.mark.parametrize("fault", ["owner", "role", "arguments", "timeout"])
def test_invalid_tool_authority_refuses_before_native_work(tmp_path, tools, monkeypatch, fault):
    from scripts import qualification_tool_process as process
    from scripts import qualify_evidence_slice_zero as core

    def forbidden():
        pytest.fail("invalid authority allocated native process resources")

    monkeypatch.setattr(core, "_CtypesWindowsKernelV1", forbidden)
    with pytest.raises((ValueError, TypeError)):
        process._run_tool(
            {"passed": True} if fault == "owner" else tools,
            "unavailable" if fault == "role" else "build_python",
            ["-c", "pass"] if fault == "arguments" else ("-c", "pass"),
            tmp_path,
            timeout_milliseconds=0 if fault == "timeout" else 60000,
        )


def test_completed_tool_receipt_cannot_be_supplied():
    from scripts.qualification_tool_process import (
        CompletedToolInvocationV1,
        tool_invocation_metadata,
    )

    with pytest.raises(TypeError):
        CompletedToolInvocationV1()
    with pytest.raises(ValueError, match="unregistered"):
        tool_invocation_metadata(object.__new__(CompletedToolInvocationV1))


@pytest.mark.parametrize("settles", [True, False])
def test_root_wait_and_job_quiescence_share_one_bounded_budget(monkeypatch, settles):
    from types import SimpleNamespace

    from scripts import qualification_tool_process as process

    now = [0.0]
    events = []

    class Kernel:
        def wait(self, handle, milliseconds):
            events.append(("wait", handle, milliseconds))
            assert milliseconds == 0 and now[0] >= 0.010

        def query_job_processes(self, job):
            events.append(("members", job))
            return (42, 43) if now[0] < 0.009 else (43,)

    def sleep(seconds):
        events.append(("sleep", seconds))
        now[0] += seconds

    monkeypatch.setattr(process, "monotonic", lambda: now[0], raising=False)
    monkeypatch.setattr(process, "sleep", sleep, raising=False)
    monkeypatch.setattr(process, "_exit_code", lambda kernel, handle: 0)
    monkeypatch.setattr(
        process,
        "_job_accounting",
        lambda kernel, job: (
            2,
            0 if settles and now[0] >= 0.010 else 2 if now[0] < 0.009 else 1,
            0,
        ),
    )
    owner = SimpleNamespace(
        _job_handle=101,
        _members={
            42: SimpleNamespace(process_handle=201),
            43: SimpleNamespace(process_handle=203),
        },
    )
    if settles:
        assert process._wait_for_tool_exit(Kernel(), 201, owner, 11) == 2
        assert 0.010 <= now[0] <= 0.011
    else:
        with pytest.raises(ValueError, match="active consumers"):
            process._wait_for_tool_exit(Kernel(), 201, owner, 11)
        assert now[0] == pytest.approx(0.011)
    assert events[0] == ("members", 101)
    assert sum(row[1] for row in events if row[0] == "sleep") <= 0.011

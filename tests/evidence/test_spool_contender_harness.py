"""Controlled failures for the test-only Task 5D contender ownership boundary."""

from __future__ import annotations

import ctypes
import io
import json
import os
import subprocess

import pytest
import spool_crash_worker as worker
import test_sqlite_spool as harness


class Pipe(io.StringIO):
    def __init__(self, value="", *, fail=False):
        super().__init__(value)
        self.fail = fail
        self.close_calls = 0
        self.write_calls = 0

    def write(self, value):
        self.write_calls += 1
        return super().write(value)

    def flush(self):
        if self.fail:
            raise OSError(22, "PRIVATE-CHILD-CONTENT")
        super().flush()

    def close(self):
        self.close_calls += 1
        super().close()


class Child:
    def __init__(self, output="", *, send_error=False, exit_code=0, timeout=False):
        self.stdin = Pipe(fail=send_error)
        self.stdout = Pipe(output)
        self.pid = 999999
        self.exit_code = exit_code
        self.timeout = timeout
        self.wait_calls = 0
        self.kill_calls = 0

    def poll(self):
        return None

    def wait(self, timeout):
        assert timeout == 10
        self.wait_calls += 1
        if self.timeout and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("PRIVATE-CHILD-CONTENT", timeout)
        return self.exit_code

    def kill(self):
        self.kill_calls += 1


class Function:
    def __init__(self, result=True):
        self.result = result
        self.calls = 0

    def __call__(self, *args):
        self.calls += 1
        return self.result


class Kernel:
    def __init__(self):
        self.CreateEventW = Function(123)
        self.SetEvent = Function()
        self.CloseHandle = Function()


def invoke(monkeypatch, tmp_path, children, *, launch_error=False):
    kernel = Kernel()
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel, raising=False)
    monkeypatch.setattr(harness, "task_5d_worker_launch_environment", lambda: ("python", {}))
    from hermes_realtime.evidence import storage_security

    monkeypatch.setattr(storage_security, "parse_root_marker", lambda value: "root")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / harness.EXPECTED_MANIFEST_NAMES["root_marker"]).write_bytes(b"marker")
    launches = []

    def launch(*args, **kwargs):
        assert kwargs["stderr"] == subprocess.DEVNULL
        assert kwargs["creationflags"] == (subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if launch_error and launches:
            raise RuntimeError("controlled launch failure")
        child = children[len(launches)]
        launches.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    return kernel, lambda: harness.assert_task_5d_single_winner_consent_resume(tmp_path, b"marker")


def frames(index, terminal=True):
    ready = {"contenderId": index, "protocolVersion": 1, "state": "ARMED"}
    result = {
        "contenderId": index,
        "protocolVersion": 1,
        "state": "TERMINAL",
        "disposition": "committed" if index == 0 else "faulted",
        "ownershipRefusal": index == 1,
    }
    return json.dumps(ready) + "\n" + (json.dumps(result) + "\n" if terminal else "")


def assert_closed(children, kernel):
    assert [child.wait_calls for child in children] == [1] * len(children)
    assert all(child.stdin.close_calls == 1 and child.stdout.close_calls == 1 for child in children)
    assert kernel.CloseHandle.calls == 1


def test_missing_terminal_preserves_primary_and_cleans_every_owner(monkeypatch, tmp_path, capsys):
    children = [Child(frames(0, False), send_error=True, exit_code=1), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(pytest.fail.Exception, match="terminal-0 exited before its protocol frame"):
        run()
    assert_closed(children, kernel)
    output = capsys.readouterr().out
    assert "PRIVATE-CHILD-CONTENT" not in output
    assert "999999" not in output
    report = json.loads(output.split("[task-5d-contender-cleanup] ")[1])
    assert report == {
        "children": 2,
        "exits": [{"category": "nonzero", "returncode": 1}, {"category": "zero", "returncode": 0}],
        "failures": ["send", "nonzero_exit"],
        "primaryFailure": True,
    }


def test_cleanup_failure_alone_fails(monkeypatch, tmp_path):
    children = [Child(frames(0), send_error=True), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(pytest.fail.Exception, match="contender cleanup failed"):
        run()
    assert_closed(children, kernel)


def test_partial_launch_preserves_failure_and_closes_owned_child(monkeypatch, tmp_path):
    children = [Child(frames(0))]
    kernel, run = invoke(monkeypatch, tmp_path, children, launch_error=True)
    with pytest.raises(RuntimeError, match="controlled launch failure"):
        run()
    assert_closed(children, kernel)


def test_timeout_reaps_and_closes_all_owners(monkeypatch, tmp_path, capsys):
    children = [Child(frames(0, False), timeout=True), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(pytest.fail.Exception, match="exited before its protocol frame"):
        run()
    assert children[0].kill_calls == 1
    assert children[0].wait_calls == 2
    assert children[1].wait_calls == 1
    assert all(c.stdin.close_calls == 1 and c.stdout.close_calls == 1 for c in children)
    assert kernel.CloseHandle.calls == 1
    assert '"wait_timeout"' in capsys.readouterr().out


@pytest.mark.parametrize("bad", [False, True], ids=["known-failure", "invalid-failure"])
def test_failure_frame_is_bounded_and_fail_closed(monkeypatch, tmp_path, capsys, bad):
    failure = {
        "contenderId": 0,
        "protocolVersion": 1,
        "state": "FAILED",
        "stage": "shared_start",
        "category": "runtime_error",
    }
    if bad:
        failure["stage"] = "PRIVATE-CHILD-CONTENT"
    children = [Child(frames(0, False) + json.dumps(failure) + "\n"), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(
        pytest.fail.Exception, match="invalid failure frame" if bad else "failed at shared_start"
    ):
        run()
    assert_closed(children, kernel)
    assert "PRIVATE-CHILD-CONTENT" not in capsys.readouterr().out


def test_worker_reports_stage_without_exception_content(monkeypatch, tmp_path, capsys):
    class Owned:
        closed = False

        def close(self):
            self.closed = True

    owned = Owned()
    monkeypatch.setattr(worker, "_make_spool", lambda *a, **k: owned)

    def fail(name):
        raise RuntimeError("PRIVATE-CHILD-CONTENT")

    monkeypatch.setattr(worker, "_wait_for_shared_start", fail)
    with pytest.raises(RuntimeError, match="PRIVATE-CHILD-CONTENT"):
        worker._run_marker_only_contender(tmp_path, 0, "private-event")
    assert owned.closed
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert reports == [
        {"contenderId": 0, "protocolVersion": 1, "state": "ARMED"},
        {
            "contenderId": 0,
            "protocolVersion": 1,
            "state": "FAILED",
            "stage": "shared_start",
            "category": "runtime_error",
        },
    ]


@pytest.mark.parametrize(
    "kind",
    [
        "release",
        "release_raise",
        "handle_close",
        "handle_raise",
        "poll",
        "pipe_close",
        "wait",
        "kill",
        "reap",
        "nonzero_exit",
    ],
)
def test_cleanup_errors_never_skip_later_owners(monkeypatch, tmp_path, capsys, kind):
    children = [Child(frames(0)), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)

    def fail(*a, **k):
        raise OSError(22, "PRIVATE-CHILD-CONTENT")

    if kind in {"release", "release_raise"}:
        # Only cleanup release fails; the initial release remains successful.
        calls = []

        def release(*args):
            calls.append(1)
            if len(calls) > 1 and kind == "release_raise":
                raise OSError(22, "PRIVATE-CHILD-CONTENT")
            return len(calls) == 1

        monkeypatch.setattr(kernel.SetEvent, "result", True)
        monkeypatch.setattr(kernel, "SetEvent", release)
    elif kind == "handle_close":
        kernel.CloseHandle.result = False
    elif kind == "handle_raise":
        monkeypatch.setattr(kernel.CloseHandle, "result", None)

        class CloseFailure(Function):
            def __call__(self, *a):
                self.calls += 1
                return fail()

        kernel.CloseHandle = CloseFailure()
    elif kind == "poll":
        monkeypatch.setattr(children[0], "poll", fail)
    elif kind == "pipe_close":
        monkeypatch.setattr(children[0].stdin, "close", fail)
    elif kind == "nonzero_exit":
        children[0].exit_code = 1
    else:
        children[0].timeout = True
        if kind == "wait":
            original = children[0].wait

            def wait(timeout):
                if children[0].wait_calls == 0:
                    children[0].wait_calls += 1
                    raise OSError(22, "PRIVATE-CHILD-CONTENT")
                return original(timeout)

            monkeypatch.setattr(children[0], "wait", wait)
        elif kind == "kill":
            monkeypatch.setattr(children[0], "kill", fail)
        elif kind == "reap":
            monkeypatch.setattr(children[0], "wait", fail)
    with pytest.raises(pytest.fail.Exception, match="contender cleanup failed"):
        run()
    assert children[1].wait_calls == 1
    assert children[1].stdin.closed and children[1].stdout.closed
    assert children[0].stdout.closed == (kind != "reap")
    assert kernel.CloseHandle.calls == 1
    report = json.loads(capsys.readouterr().out.split("[task-5d-contender-cleanup] ")[1])
    expected = {"release_raise": "release", "handle_raise": "handle_close", "poll": "send"}.get(
        kind, kind
    )
    assert expected in report["failures"]
    assert not report["primaryFailure"]
    if kind == "reap":
        assert report["failures"] == ["wait", "reap", "stdout_unreaped"]
        assert report["exits"] == [
            {"category": "unreaped", "returncode": None},
            {"category": "zero", "returncode": 0},
        ]
    assert "PRIVATE-CHILD-CONTENT" not in json.dumps(report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", "PRIVATE-CHILD-CONTENT"),
        ("contenderId", 1),
        ("contenderId", True),
        ("contenderId", False),
        ("protocolVersion", 2),
        ("protocolVersion", True),
        ("stage", []),
        ("category", {}),
        ("category", "PRIVATE-CHILD-CONTENT"),
    ],
)
def test_failure_schema_rejects_each_untrusted_field(monkeypatch, tmp_path, capsys, field, value):
    failure = {
        "contenderId": 0,
        "protocolVersion": 1,
        "state": "FAILED",
        "stage": "shared_start",
        "category": "runtime_error",
    }
    failure[field] = value
    children = [Child(frames(0, False) + json.dumps(failure) + "\n"), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(pytest.fail.Exception, match="invalid failure frame"):
        run()
    assert_closed(children, kernel)
    assert "PRIVATE-CHILD-CONTENT" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "error,category",
    [
        (OSError(22, "PRIVATE-CHILD-CONTENT"), "os_error"),
        (RuntimeError("PRIVATE-CHILD-CONTENT"), "runtime_error"),
        (ValueError("PRIVATE-CHILD-CONTENT"), "unexpected"),
    ],
)
def test_worker_categories_are_content_free(capsys, error, category):
    worker._emit_contender_failure(0, "setup", error)
    assert json.loads(capsys.readouterr().out) == {
        "contenderId": 0,
        "protocolVersion": 1,
        "state": "FAILED",
        "stage": "setup",
        "category": category,
    }


@pytest.mark.parametrize("error", [OSError(22, "private"), ValueError("private")])
def test_broken_diagnostic_pipe_cannot_mask_worker_failure(monkeypatch, error):
    def fail(payload):
        raise error

    monkeypatch.setattr(worker, "_emit_contender_frame", fail)
    worker._emit_contender_failure(0, "setup", RuntimeError("original"))


@pytest.mark.parametrize("primary", [False, True], ids=["cleanup-alone", "primary-and-cleanup"])
def test_worker_close_failure_preserves_primary_and_never_turns_green(
    monkeypatch, tmp_path, capsys, primary
):
    class Owned:
        def close(self):
            raise OSError(22, "PRIVATE-CLOSE")

    monkeypatch.setattr(worker, "_make_spool", lambda *a, **k: Owned())

    def fail(name):
        raise RuntimeError("PRIVATE-PRIMARY")

    if primary:
        monkeypatch.setattr(worker, "_wait_for_shared_start", fail)
    else:
        from types import SimpleNamespace

        from hermes_realtime.evidence.models import StoreDisposition

        monkeypatch.setattr(worker, "_wait_for_shared_start", lambda name: None)
        monkeypatch.setattr(
            Owned, "create_epoch", lambda *a: StoreDisposition.COMMITTED, raising=False
        )
        monkeypatch.setattr(
            Owned, "diagnostics", lambda *a: SimpleNamespace(sticky_fault=None), raising=False
        )
        monkeypatch.setattr(worker.sys, "stdin", io.StringIO("CLOSE\n"))
    with pytest.raises(
        RuntimeError if primary else OSError,
        match="PRIVATE-PRIMARY" if primary else "PRIVATE-CLOSE",
    ):
        worker._run_marker_only_contender(tmp_path, 0, "private-event")
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    expected_stages = [None, "shared_start", "close"] if primary else [None, None, "close"]
    assert [r.get("stage") for r in reports] == expected_stages
    assert "PRIVATE" not in json.dumps(reports)


def test_worker_setup_failure_reports_without_unowned_cleanup(monkeypatch, tmp_path, capsys):
    def fail(*a, **k):
        raise OSError(22, "PRIVATE-SETUP")

    monkeypatch.setattr(worker, "_make_spool", fail)
    with pytest.raises(OSError, match="PRIVATE-SETUP"):
        worker._run_marker_only_contender(tmp_path, 0, "private-event")
    assert json.loads(capsys.readouterr().out) == {
        "contenderId": 0,
        "protocolVersion": 1,
        "state": "FAILED",
        "stage": "setup",
        "category": "os_error",
    }


def test_exited_child_is_reaped_without_close_send(monkeypatch, tmp_path):
    children = [Child(frames(0, False), send_error=True, exit_code=1), Child(frames(1))]
    monkeypatch.setattr(children[0], "poll", lambda: 1)
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(pytest.fail.Exception, match="exited before its protocol frame"):
        run()
    assert_closed(children, kernel)
    assert children[0].stdin.write_calls == 0


def test_missing_pipe_is_not_a_cleanup_failure(monkeypatch, tmp_path, capsys):
    children = [Child(frames(0, False), exit_code=1), Child(frames(1))]
    children[0].stdin = None
    kernel, run = invoke(monkeypatch, tmp_path, children)
    with pytest.raises(pytest.fail.Exception, match="exited before its protocol frame"):
        run()
    assert children[0].stdout.closed
    assert children[1].stdin.closed and children[1].stdout.closed
    assert kernel.CloseHandle.calls == 1
    report = json.loads(capsys.readouterr().out.split("[task-5d-contender-cleanup] ")[1])
    assert report["failures"] == ["nonzero_exit"]


@pytest.mark.parametrize(
    "stage", ["armed", "create_epoch", "diagnostics", "terminal", "close_protocol"]
)
def test_worker_failure_stage_matches_actual_boundary(monkeypatch, tmp_path, stage):
    from types import SimpleNamespace

    from hermes_realtime.evidence.models import StoreDisposition

    reports = []

    def fail():
        raise RuntimeError("PRIVATE-CHILD-CONTENT")

    class Owned:
        closed = False

        def create_epoch(self, request):
            if stage == "create_epoch":
                fail()
            return StoreDisposition.COMMITTED

        def diagnostics(self):
            if stage == "diagnostics":
                fail()
            return SimpleNamespace(sticky_fault=None)

        def close(self):
            self.closed = True

    owned = Owned()
    monkeypatch.setattr(worker, "_make_spool", lambda *a, **k: owned)
    monkeypatch.setattr(worker, "_wait_for_shared_start", lambda name: None)
    monkeypatch.setattr(
        worker.sys, "stdin", io.StringIO("WRONG\n" if stage == "close_protocol" else "CLOSE\n")
    )

    def emit(payload):
        if payload["state"] == "ARMED" and stage == "armed":
            fail()
        if payload["state"] == "TERMINAL" and stage == "terminal":
            fail()
        reports.append(payload)

    monkeypatch.setattr(worker, "_emit_contender_frame", emit)
    with pytest.raises(RuntimeError):
        worker._run_marker_only_contender(tmp_path, 0, "private-event")
    assert owned.closed
    assert reports[-1] == {
        "contenderId": 0,
        "protocolVersion": 1,
        "state": "FAILED",
        "stage": stage,
        "category": "runtime_error",
    }
    assert "PRIVATE" not in json.dumps(reports)


@pytest.mark.parametrize("primary", [True, False], ids=["primary", "cleanup-only"])
@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_broken_parent_report_preserves_primary_or_fails_cleanup(
    monkeypatch, tmp_path, primary, error_type
):
    children = [Child(frames(0, not primary)), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)

    def broken_report(*args, **kwargs):
        raise error_type("PRIVATE-REPORT")

    monkeypatch.setattr(harness, "print", broken_report, raising=False)
    with pytest.raises(
        pytest.fail.Exception,
        match="terminal-0 exited before its protocol frame" if primary else "cleanup failed",
    ):
        run()
    assert_closed(children, kernel)


def test_unreaped_stdout_never_blocks_later_cleanup(monkeypatch, tmp_path, capsys):
    children = [Child(frames(0, False)), Child(frames(1))]
    kernel, run = invoke(monkeypatch, tmp_path, children)

    def unreapable(timeout):
        children[0].wait_calls += 1
        raise OSError(22, "PRIVATE-WAIT")

    def reader_locked_close():
        children[0].stdout.close_calls += 1
        raise AssertionError("simulated reader lock: close would block")

    monkeypatch.setattr(children[0], "wait", unreapable)
    monkeypatch.setattr(children[0].stdout, "close", reader_locked_close)
    with pytest.raises(pytest.fail.Exception, match="terminal-0 exited before its protocol frame"):
        run()
    assert children[0].stdout.close_calls == 0
    assert children[0].stdin.closed
    assert children[1].wait_calls == 1 and children[1].stdin.closed and children[1].stdout.closed
    assert kernel.CloseHandle.calls == 1
    report = json.loads(capsys.readouterr().out.split("[task-5d-contender-cleanup] ")[1])
    assert report["failures"] == ["wait", "reap", "stdout_unreaped"]

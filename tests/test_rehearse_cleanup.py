from __future__ import annotations

import ctypes
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "rehearse_desktop_mvp.py"
sys.path.insert(0, str(_PATH.parent))
rehearsal = importlib.import_module("rehearse_desktop_mvp")


def test_empty_cleanup_is_a_healthy_control(tmp_path, monkeypatch, capsys):
    owner = rehearsal.Processes(tmp_path)
    monkeypatch.setattr(rehearsal, "process_table", lambda: {})
    assert owner.survivors(SimpleNamespace(members=lambda: {os.getpid()})) == {}


@pytest.mark.parametrize(
    ("state", "missing", "expected"),
    [
        ("exited", False, {}),
        ("running_or_terminating", False, {"private-image.exe": 1}),
        ("unknown", False, {"<unknown>": 1}),
        ("identity_changed", False, {}),
        ("exited", True, {"<unknown>": 1}),
    ],
)
def test_job_only_cleanup_is_classified_at_refusal(
    tmp_path, monkeypatch, capsys, state, missing, expected
):
    owner = rehearsal.Processes(tmp_path)
    table = {} if missing else {101: (99, "private-image.exe")}
    monkeypatch.setattr(rehearsal, "process_table", lambda: table)
    monkeypatch.setattr(rehearsal, "_cleanup_process_state", lambda *args: state)
    assert owner.survivors(SimpleNamespace(_job=66, members=lambda: {os.getpid(), 101})) == expected
    record = json.loads(capsys.readouterr().out.removeprefix("[desktop-mvp-cleanup] "))
    observed = "unknown" if missing else state
    assert record["current"]["job_only"][observed] == 1
    assert record["current"]["tracked"] == dict.fromkeys(
        ("exited", "running_or_terminating", "identity_changed", "unknown"), 0
    )
    assert "private" not in json.dumps(record)
    assert "101" not in json.dumps(record)


def test_cleanup_retains_the_first_refusal(tmp_path, monkeypatch, capsys):
    owner = rehearsal.Processes(tmp_path)
    owner.seen[101] = ("private-image.exe", 42)
    monkeypatch.setattr(rehearsal, "process_table", lambda: {101: (99, "private-image.exe")})
    states = iter(["running_or_terminating", "exited"])
    monkeypatch.setattr(rehearsal, "_cleanup_process_state", lambda *args: next(states))
    assert owner.survivors(None) == {"private-image.exe": 1}
    first = owner.first_cleanup_refusal
    assert owner.survivors(None) == {}
    records = [
        json.loads(row.removeprefix("[desktop-mvp-cleanup] "))
        for row in capsys.readouterr().out.splitlines()
    ]
    assert owner.first_cleanup_refusal == first == records[0]["current"]
    assert records[1]["first_refusal"] == first
    assert first["tracked"]["running_or_terminating"] == 1


@pytest.mark.asyncio
async def test_close_and_emitted_summary_keep_first_cleanup_refusal(tmp_path, monkeypatch, capsys):
    owner = rehearsal.Processes(tmp_path)
    owner.seen[101] = ("private-image.exe", 42)
    monkeypatch.setattr(rehearsal, "process_table", lambda: {101: (99, "private-image.exe")})
    states = iter(["running_or_terminating", "exited"])
    monkeypatch.setattr(rehearsal, "_cleanup_process_state", lambda *args: next(states))
    containment = SimpleNamespace(_job=66, members=lambda: set())
    stops = []
    monkeypatch.setattr(owner, "stop_all", lambda *args: stops.append(args) or 10.0)
    monkeypatch.setattr(rehearsal, "listeners", lambda port, **kwargs: [])
    assert owner.survivors(None)
    first = owner.first_cleanup_refusal
    run = object.__new__(rehearsal.Rehearsal)
    run.browser = run.playwright = None
    run.containment = containment
    run.processes = owner
    run.stand_in = SimpleNamespace(close=AsyncMock())
    run.api_port, run.companion_port, run.host_port = 1, 2, 3
    record = await run.close()
    assert stops == [(containment,)]
    assert record["processes_left"] == {}
    assert record["first_cleanup_refusal"] == first
    rehearsal._emit({"name": "summary"} | record)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(rehearsal._PREFIX))
    assert summary["first_cleanup_refusal"] == first
    assert "private" not in json.dumps(summary)


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_refusal", [False, True])
async def test_public_summary_refuses_stop_error_even_after_healthy_query(
    tmp_path, monkeypatch, capsys, prior_refusal
):
    owner = rehearsal.Processes(tmp_path)
    if prior_refusal:
        owner.seen[101] = ("private.exe", 42)
        monkeypatch.setattr(rehearsal, "process_table", lambda: {101: (99, "private.exe")})
        monkeypatch.setattr(
            rehearsal, "_cleanup_process_state", lambda *args: "running_or_terminating"
        )
        assert owner.survivors(None)
    first = owner.first_cleanup_refusal
    queries = []

    def healthy_query():
        queries.append(True)
        return {}

    monkeypatch.setattr(rehearsal, "process_table", healthy_query)
    ports = []
    monkeypatch.setattr(rehearsal, "listeners", lambda port, **kwargs: ports.append(port) or [])

    def stop_error(*args):
        raise OSError("private path, process ID and exception detail")

    monkeypatch.setattr(owner, "stop_all", stop_error)
    run = object.__new__(rehearsal.Rehearsal)
    run.browser = run.playwright = run.containment = None
    run.processes = owner
    run.stand_in = SimpleNamespace(close=AsyncMock(), requests={})
    run.api_port, run.companion_port, run.host_port = 1, 2, 3
    run.home, run.logs_dir, run.state_dir = (tmp_path / name for name in ("home", "logs", "state"))
    run.records = [{"step": "1", "outcome": "as_expected"}]
    for name in (
        "setup",
        "gate",
        "start_and_talk",
        "typed",
        "delegate",
        "interrupt",
        "cancel",
        "reconnect",
        "restart",
        "delete",
        "reload_inputs",
        "cleanup",
    ):
        monkeypatch.setattr(run, name, AsyncMock())
    monkeypatch.setattr(rehearsal, "Rehearsal", lambda *args: run)
    monkeypatch.setattr(rehearsal, "Containment", lambda: None)
    monkeypatch.setattr(rehearsal, "_synthesize_clips", AsyncMock(return_value={}))
    assert await rehearsal._rehearse(tmp_path, "stand-in") == 1
    run.stand_in.close.assert_awaited_once()
    assert queries == [True]
    assert ports == [7880, 1, 2, 3]
    summary = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(rehearsal._PREFIX))
    assert summary["processes_left"] == {"<unknown>": 1}
    assert summary["cleanup_stop_failed"] is True
    assert summary["ports_listening"] == 0
    assert summary["first_cleanup_refusal"] is not None
    if prior_refusal:
        assert summary["first_cleanup_refusal"] == first
    else:
        assert summary["first_cleanup_refusal"]["stop_failed"] == 1
    assert "private" not in json.dumps(summary)


def test_unknown_cleanup_category_cannot_collide_with_an_image(tmp_path, monkeypatch, capsys):
    owner = rehearsal.Processes(tmp_path)
    states = iter(["running_or_terminating", "unknown"])
    monkeypatch.setattr(rehearsal, "_cleanup_process_state", lambda *args: next(states))
    containment = SimpleNamespace(_job=66, members=lambda: {101, 102})
    monkeypatch.setattr(
        rehearsal, "process_table", lambda: {101: (99, "unknown"), 102: (99, "other")}
    )
    assert owner.survivors(containment) == {"unknown": 1, "<unknown>": 1}


def test_absent_historical_identity_does_not_become_perpetually_unknown(
    tmp_path, monkeypatch, capsys
):
    owner = rehearsal.Processes(tmp_path)
    owner.seen[101] = ("private-image.exe", 42)
    monkeypatch.setattr(rehearsal, "process_table", lambda: {})
    assert owner.survivors(None) == {}
    assert owner.first_cleanup_refusal is None


def test_reused_tracked_pid_still_has_independent_job_authority(tmp_path, monkeypatch, capsys):
    owner = rehearsal.Processes(tmp_path)
    owner.seen[101] = ("old-private-image.exe", 42)
    monkeypatch.setattr(rehearsal, "process_table", lambda: {101: (99, "new-private-image.exe")})

    def state(pid, expected_start, job):
        assert (pid, expected_start, job) == (101, 42, 66)
        return "running_or_terminating"

    monkeypatch.setattr(rehearsal, "_cleanup_process_state", state)
    assert owner.survivors(SimpleNamespace(_job=66, members=lambda: {101})) == {
        "new-private-image.exe": 1
    }


@pytest.mark.parametrize("broken", ["snapshot", "bound"])
def test_cleanup_refuses_incomplete_observation(tmp_path, monkeypatch, capsys, broken):
    owner = rehearsal.Processes(tmp_path)
    if broken == "snapshot":

        def snapshot():
            raise OSError("private native details")

        monkeypatch.setattr(rehearsal, "process_table", snapshot)
        containment = None
    else:
        monkeypatch.setattr(rehearsal, "process_table", lambda: {})
        containment = SimpleNamespace(members=lambda: set(range(10000, 14097)))
    assert owner.survivors(containment) == {"<unknown>": 1}
    record = json.loads(capsys.readouterr().out.removeprefix("[desktop-mvp-cleanup] "))
    assert record["current"]["observation_failed"] == 1
    assert "private" not in json.dumps(record)


class _Kernel:
    def __init__(self, *, wait=258, start=42, member=True, opened=True, times=True, job_query=True):
        self.wait, self.start, self.member = wait, start, member
        self.opened, self.times = opened, times
        self.job_query = job_query
        self.closed = []
        self.access = None
        self.OpenProcess = self.open

    def open(self, access, inherit, pid):
        self.access = access
        assert inherit is False
        assert pid == 101
        return 55 if self.opened else 0

    def GetProcessTimes(self, handle, created, *unused):
        created._obj.value = self.start
        return self.times

    def IsProcessInJob(self, handle, job, member):
        assert job.value == 66
        member._obj.value = self.member
        return self.job_query

    def WaitForSingleObject(self, handle, timeout):
        assert handle.value == 55
        assert timeout == 0
        return self.wait

    def CloseHandle(self, handle):
        self.closed.append(handle.value)


@pytest.mark.parametrize(
    ("kwargs", "expected_start", "job", "expected"),
    [
        ({}, 42, None, "running_or_terminating"),
        ({"wait": 0}, 42, None, "exited"),
        ({"wait": 0xFFFFFFFF}, 42, None, "unknown"),
        ({"start": 43}, 42, None, "identity_changed"),
        ({"times": False}, 42, None, "unknown"),
        ({"opened": False}, 42, None, "unknown"),
        ({}, None, 66, "running_or_terminating"),
        ({"member": False}, None, 66, "identity_changed"),
        ({"start": 43}, 42, 66, "running_or_terminating"),
        ({"job_query": False}, None, 66, "unknown"),
        ({"wait": -1}, 42, None, "unknown"),
    ],
)
def test_cleanup_uses_one_identity_checked_native_handle(
    monkeypatch, kwargs, expected_start, job, expected
):
    kernel = _Kernel(**kwargs)

    # Functions on a real WinDLL allow assigning restype.
    def open_process(*args):
        return kernel.open(*args)

    kernel.OpenProcess = open_process
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kw: kernel, raising=False)
    assert rehearsal._cleanup_process_state(101, expected_start, job) == expected
    assert kernel.access == 0x00101000
    assert kernel.closed == ([55] if kernel.opened else [])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, 42),
        ({"wait": 0}, None),
        ({"wait": -1}, None),
        ({"times": False}, None),
        ({"opened": False}, None),
    ],
)
def test_started_at_uses_the_same_zero_time_handle_liveness(monkeypatch, kwargs, expected):
    kernel = _Kernel(**kwargs)

    def open_process(*args):
        return kernel.open(*args)

    kernel.OpenProcess = open_process
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kw: kernel, raising=False)
    assert rehearsal.started_at(101) == expected
    assert kernel.access == 0x00101000
    assert kernel.closed == ([55] if kernel.opened else [])


@pytest.mark.parametrize(
    ("now", "expected_waits"), [([10, 11, 13], [4000, 2000]), ([10, 11, 16], [4000, 0])]
)
def test_cleanup_settlement_shares_one_wait_budget_including_job_only(
    tmp_path, monkeypatch, capsys, now, expected_waits
):
    owner = rehearsal.Processes(tmp_path)
    owner.seen[101] = ("private.exe", 42)
    monkeypatch.setattr(owner, "alive", lambda: [])
    monkeypatch.setattr(
        rehearsal, "process_table", lambda: {101: (99, "private.exe"), 102: (99, "private.exe")}
    )
    ticks = iter(now)
    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: next(ticks))
    calls = []

    def state(pid, expected_start, job, wait_ms=0):
        calls.append((pid, expected_start, job, wait_ms))
        return "exited" if pid == 101 else "running_or_terminating"

    monkeypatch.setattr(rehearsal, "_cleanup_process_state", state)
    owner.stop_all(SimpleNamespace(_job=66, members=lambda: {102}))
    assert calls == [(101, 42, None, expected_waits[0]), (102, None, 66, expected_waits[1])]
    assert owner.first_cleanup_refusal is None
    record = json.loads(capsys.readouterr().out.removeprefix("[desktop-mvp-cleanup-settlement] "))
    assert record == {
        "wait_budget_ms": 5000,
        "candidates": 2,
        "exited": 1,
        "running_or_terminating": 1,
        "identity_changed": 0,
        "unknown": 0,
        "observation_failed": 0,
    }
    assert "private" not in json.dumps(record)


def test_cleanup_settlement_waits_on_the_identity_checked_handle(monkeypatch):
    kernel = _Kernel(wait=0)

    def open_process(*args):
        return kernel.open(*args)

    kernel.OpenProcess = open_process
    waits = []
    kernel.WaitForSingleObject = lambda handle, ms: waits.append((handle.value, ms)) or 0
    monkeypatch.setattr(ctypes, "WinDLL", lambda *args, **kw: kernel, raising=False)
    assert rehearsal._cleanup_process_state(101, 42, None, wait_ms=1234) == "exited"
    assert waits == [(55, 1234)]
    assert kernel.closed == [55]


@pytest.mark.parametrize("state", ["running_or_terminating", "unknown"])
def test_final_cleanup_requeries_new_job_child_after_settlement(
    tmp_path, monkeypatch, capsys, state
):
    owner = rehearsal.Processes(tmp_path)
    tables = iter(
        [{101: (99, "private.exe")}, {101: (99, "private.exe"), 202: (101, "new-private.exe")}]
    )
    members = iter([{101}, {101, 202}])
    monkeypatch.setattr(rehearsal, "process_table", lambda: next(tables))
    monkeypatch.setattr(
        rehearsal, "_cleanup_process_state", lambda pid, *args: "exited" if pid == 101 else state
    )
    containment = SimpleNamespace(_job=66, members=lambda: next(members))
    owner.stop_all(containment)
    assert owner.first_cleanup_refusal is None
    assert owner.survivors(containment) == (
        {"new-private.exe": 1} if state == "running_or_terminating" else {"<unknown>": 1}
    )
    assert owner.first_cleanup_refusal["job_only"][state] == 1


def test_cleanup_settlement_reports_native_failure_without_masking(tmp_path, monkeypatch, capsys):
    owner = rehearsal.Processes(tmp_path)

    def broken():
        raise OSError("private query detail")

    monkeypatch.setattr(rehearsal, "process_table", broken)
    with pytest.raises(OSError):
        owner.stop_all(None)
    record = json.loads(capsys.readouterr().out.removeprefix("[desktop-mvp-cleanup-settlement] "))
    assert record["observation_failed"] == 1
    assert owner.first_cleanup_refusal is None
    assert "private" not in json.dumps(record)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process handles")
def test_cleanup_native_wait_recognizes_held_exited_handle_even_exit_259():
    child = subprocess.Popen(
        [sys.executable, "-c", "import os; os._exit(259)"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        child.wait(timeout=20)
        assert child.returncode == 259
        # Popen keeps the exited process object/identity alive until its handle closes.
        assert rehearsal.started_at(child.pid) is None
        assert rehearsal._cleanup_process_state(child.pid, None, None) == "exited"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=20)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows Job and process handles")
def test_normal_hidden_owned_child_cleanup_with_populated_seen_is_clean(tmp_path):
    probe = tmp_path / "cleanup.py"
    probe.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, {str(_PATH.parent)!r})\n"
        "import rehearse_desktop_mvp as r\n"
        "job = r.Containment()\n"
        f"owner = r.Processes(Path({str(tmp_path)!r}))\n"
        "owner.spawn('tiny', [sys.executable, '-c', "
        "'import threading; threading.Event().wait()'], env=os.environ, log=Path(os.devnull))\n"
        "assert owner.seen\n"
        "owner.stop_all(job)\n"
        "assert owner.survivors(job) == {}\n"
        "print(json.dumps({'clean': True, 'seenPopulated': bool(owner.seen)}))\n",
        encoding="utf-8",
    )
    base = getattr(sys, "_base_executable", sys.executable)
    completed = subprocess.run(
        [base, str(probe)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.splitlines()[-1]) == {"clean": True, "seenPopulated": True}

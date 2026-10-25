from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
rehearsal = importlib.import_module("rehearse_desktop_mvp")

_LOOPBACK = "  TCP    127.0.0.1:7880    0.0.0.0:0    LISTENING    123\n"
_PUBLIC = "  TCP    0.0.0.0:7880    0.0.0.0:0    LISTENING    123\n"


def queries(monkeypatch, results):
    clock, calls = [0.0], []
    rows = iter(results)

    def run(argv, **options):
        calls.append((argv, options))
        code, output, elapsed = next(rows)
        clock[0] += elapsed
        if isinstance(code, BaseException):
            raise code
        return subprocess.CompletedProcess(argv, code, output)

    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rehearsal.subprocess, "run", run)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    return clock, calls


def test_nonzero_output_cannot_establish_absence(monkeypatch):
    """Mutation: restore stdout-only parsing of a failed query."""
    queries(monkeypatch, [(1, "", 0), (1, "", 0)])
    with pytest.raises(RuntimeError, match="listener_query_exit"):
        rehearsal.listeners(7880, deadline=10)


@pytest.mark.parametrize("failed_family", [0, 1])
def test_either_family_failure_refuses_even_plausible_output(monkeypatch, failed_family):
    """Mutation: accept a failed family or return the other family's partial result."""
    rows = [(0, _LOOPBACK, 0), (0, _LOOPBACK, 0)]
    rows[failed_family] = (1, _LOOPBACK, 0)
    queries(monkeypatch, rows)
    with pytest.raises(RuntimeError, match="listener_query_exit"):
        rehearsal.listeners(7880, deadline=10)


@pytest.mark.parametrize(
    ("error", "category"),
    [
        (subprocess.TimeoutExpired("private-command", 1, output="private-output"), "timeout"),
        (FileNotFoundError("private-path"), "launch"),
    ],
)
def test_query_errors_refuse_with_one_sanitized_marker(monkeypatch, capsys, error, category):
    queries(monkeypatch, [(0, _LOOPBACK, 0), (error, _PUBLIC, 0)])
    with pytest.raises(RuntimeError, match=f"listener_query_{category}"):
        rehearsal.listeners(7880, deadline=10)
    rows = capsys.readouterr().out.splitlines()
    assert len(rows) == 1
    record = json.loads(rows[0].removeprefix("[desktop-mvp-listener-refusal] "))
    assert record == {"category": f"listener_query_{category}", "completed_families": 1}


@pytest.mark.parametrize("deadline", [0, float("inf"), float("nan"), None, "10", True])
def test_expired_or_nonfinite_deadline_never_launches(monkeypatch, deadline):
    _, calls = queries(monkeypatch, [(0, "", 0), (0, "", 0)])
    with pytest.raises(RuntimeError, match="listener_query_deadline"):
        rehearsal.listeners(7880, deadline=deadline)
    assert calls == []


def test_one_deadline_bounds_both_families(monkeypatch):
    _, calls = queries(monkeypatch, [(0, _LOOPBACK, 3), (0, "", 1)])
    assert rehearsal.listeners(7880, deadline=10) == ["127.0.0.1"]
    assert [row[1]["timeout"] for row in calls] == [10, 7]
    assert [row[0][-1] for row in calls] == ["tcp", "tcpv6"]
    for _, options in calls:
        assert options["stdin"] is subprocess.DEVNULL
        assert options["stderr"] is subprocess.DEVNULL
        assert options["creationflags"] == 0x08000000


def test_late_success_does_not_establish_absence(monkeypatch):
    _, calls = queries(monkeypatch, [(0, "", 0), (0, "", 10)])
    with pytest.raises(RuntimeError, match="listener_query_deadline"):
        rehearsal.listeners(7880, deadline=10)
    assert len(calls) == 2


@pytest.mark.parametrize(
    "output,expected", [("", []), (_LOOPBACK, []), (_PUBLIC, [("fail", "api_not_loopback")])]
)
def test_successful_family_controls_keep_existing_verdicts(monkeypatch, output, expected):
    queries(monkeypatch, [(0, output, 0), (0, "", 0)])
    addresses = rehearsal.listeners(7880, deadline=10)
    if not output:
        assert addresses == []
    else:
        assert rehearsal.listener_verdict("api", addresses) == expected


def test_successful_ipv6_output_is_included(monkeypatch):
    output = "  TCP    [::1]:7880    [::]:0    LISTENING    123\n"
    queries(monkeypatch, [(0, _LOOPBACK, 0), (0, output, 0)])
    assert rehearsal.listeners(7880, deadline=10) == ["127.0.0.1", "[::1]"]


@pytest.mark.asyncio
async def test_cleanup_query_failure_stays_unknown_and_keeps_first_refusal(
    tmp_path, monkeypatch
):
    first = {"tracked": {"unknown": 1}}
    owner = SimpleNamespace(
        stop_all=lambda *a: 10.0, survivors=lambda *a: {},
        first_cleanup_refusal=first, cleanup_stop_failed=False,
    )
    run = object.__new__(rehearsal.Rehearsal)
    run.browser = run.playwright = run.containment = None
    run.processes = owner
    run.stand_in = SimpleNamespace(close=AsyncMock())
    run.api_port, run.companion_port, run.host_port = 1, 2, 3
    queries(monkeypatch, [(1, "", 0)])
    record = await run.close()
    assert record["ports_listening"] is None
    assert record["listener_query_failure"] == "listener_query_exit"
    assert record["first_cleanup_refusal"] == first


def test_preflight_query_failure_is_failure_not_missing_dependency(monkeypatch, capsys):
    monkeypatch.setattr(rehearsal, "take_rehearsal_lock", lambda: None)

    def preflight(model):
        raise rehearsal.ListenerQueryFailure("listener_query_exit")

    monkeypatch.setattr(rehearsal, "preflight", preflight)
    assert rehearsal.main(["--ollama-model", "synthetic"]) == 1
    record = json.loads(capsys.readouterr().out.removeprefix(rehearsal._PREFIX))
    assert record["outcome"] == "failed"
    assert record["category"] == "listener_query_exit"


@pytest.mark.asyncio
async def test_step_keeps_named_query_failure():
    run = object.__new__(rehearsal.Rehearsal)
    run.logs, run.records, run.page = [], [], None
    async with run.step("1", "synthetic"):
        raise rehearsal.ListenerQueryFailure("listener_query_exit")
    assert run.records[0]["outcome"] == "failed"
    assert run.records[0]["category"] == "listener_query_exit"


@pytest.mark.asyncio
async def test_cleanup_ports_share_stop_deadline(monkeypatch):
    run = object.__new__(rehearsal.Rehearsal)
    run.browser = run.playwright = run.containment = None
    run.processes = SimpleNamespace(
        stop_all=lambda *a: 10.0, survivors=lambda *a: {},
        first_cleanup_refusal=None, cleanup_stop_failed=False,
    )
    run.stand_in = SimpleNamespace(close=AsyncMock())
    run.api_port, run.companion_port, run.host_port = 1, 2, 3
    _, calls = queries(monkeypatch, [(0, "", 3), (0, "", 3), (0, "", 4)] + [(0, "", 0)] * 5)
    result = await run.close()
    assert result["ports_listening"] is None
    assert result["listener_query_failure"] == "listener_query_deadline"
    assert [options["timeout"] for _, options in calls] == [10, 7, 4]


def test_stop_returns_existing_wait_deadline(tmp_path, monkeypatch):
    owner = rehearsal.Processes(tmp_path)
    clock = [10.0]
    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: clock[0])

    def snapshot():
        clock[0] = 14.0
        return {}

    monkeypatch.setattr(rehearsal, "process_table", snapshot)
    assert owner.stop_all() == 15.0


@pytest.mark.asyncio
async def test_successful_empty_cleanup_control(monkeypatch):
    run = object.__new__(rehearsal.Rehearsal)
    run.browser = run.playwright = run.containment = None
    run.processes = SimpleNamespace(
        stop_all=lambda *a: 10.0, survivors=lambda *a: {},
        first_cleanup_refusal=None, cleanup_stop_failed=False,
    )
    run.stand_in = SimpleNamespace(close=AsyncMock())
    run.api_port, run.companion_port, run.host_port = 1, 2, 3
    _, calls = queries(monkeypatch, [(0, "", 0)] * 8)
    result = await run.close()
    assert result["ports_listening"] == 0
    assert result["listener_query_failure"] is None
    assert len(calls) == 8


@pytest.mark.asyncio
@pytest.mark.parametrize("ports_listening,expected", [(None, 1), (1, 1), (0, 0)])
async def test_overall_completion_requires_known_absence(
    tmp_path, monkeypatch, ports_listening, expected
):
    run = SimpleNamespace(
        home=tmp_path / "home", logs_dir=tmp_path / "logs", state_dir=tmp_path / "state",
        records=[{"step": "1", "outcome": "as_expected"}], stand_in=SimpleNamespace(requests={}),
        close=AsyncMock(return_value={"processes_left": {}, "ports_listening": ports_listening}),
    )
    for name in (
        "setup", "gate", "start_and_talk", "typed", "delegate", "interrupt", "cancel",
        "reconnect", "restart", "delete", "reload_inputs", "cleanup",
    ):
        setattr(run, name, AsyncMock())
    monkeypatch.setattr(rehearsal, "Rehearsal", lambda *a: run)
    monkeypatch.setattr(rehearsal, "Containment", lambda: None)
    monkeypatch.setattr(rehearsal, "_synthesize_clips", AsyncMock(return_value={}))
    assert await rehearsal._rehearse(tmp_path, "synthetic") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,budget", [("gateway", 240), ("livekit", 30)])
async def test_startup_queries_keep_the_health_deadline(tmp_path, monkeypatch, kind, budget):
    clock, observed = [100.0], []
    run = object.__new__(rehearsal.Rehearsal)
    run.home = run.logs_dir = tmp_path
    run.api_port, run.companion_port = 1, 2
    run.ready = {"gateway": False}
    run.processes = SimpleNamespace(spawn=lambda *a, **k: None, observe=lambda: None)
    run._tail = lambda *a: None
    monkeypatch.setattr(rehearsal.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rehearsal, "_isolated", lambda *a: {})

    def health(url):
        clock[0] += budget - 1
        return 200, b"OK"

    def listener(port, *, deadline):
        observed.append(deadline)
        clock[0] += 0.1
        return ["127.0.0.1"]

    monkeypatch.setattr(rehearsal, "_http", health)
    monkeypatch.setattr(rehearsal, "listeners", listener)
    monkeypatch.setattr(rehearsal.local_livekit, "verified_server", lambda: None)
    monkeypatch.setattr(rehearsal.local_livekit, "server_command", lambda *a: [])
    step = rehearsal.Step("1", "synthetic", [])
    if kind == "gateway":
        await run._start_gateway(step)
        assert run.ready["gateway"]
        assert len(observed) == 3
    else:
        assert await run._start_livekit(step)
        assert len(observed) == 1
    assert observed == [100 + budget] * len(observed)

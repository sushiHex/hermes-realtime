"""The single owner of a host-wide fault (#531, proposed ADR 0008 step 4).

``HostFatalExit`` ends the host once: it stops input, lets the saved conversation reach the
disk, prints one line with a fixed cause, and exits with a distinct status. These tests drive
the owner alone, with the process exit replaced by a recorder.
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest

from hermes_realtime.host_launcher import (
    HOST_FATAL_CAUSES,
    HOST_FATAL_EXIT_STATUS,
    HostFatalExit,
)

_PREFIX = "[host-fatal-exit] "


class _Recorder:
    """Record the order of the owner's steps, and what was on stdout when the exit came."""

    def __init__(self, capsys: pytest.CaptureFixture[str]) -> None:
        self.steps: list[str] = []
        self.statuses: list[int] = []
        self.stdout_at_exit: list[str] = []
        self.exited = threading.Event()
        self._capsys = capsys

    async def stop_input(self) -> None:
        self.steps.append("input")

    async def flush(self) -> None:
        self.steps.append("flush")

    def exit(self, status: int) -> None:
        self.stdout_at_exit.append(self._capsys.readouterr().out)
        self.steps.append("exit")
        self.statuses.append(status)
        self.exited.set()

    def owner(self, **overrides: object) -> HostFatalExit:
        options: dict[str, object] = {
            "stop_input": self.stop_input,
            "flush_conversation": self.flush,
            "exit_process": self.exit,
        }
        options.update(overrides)
        owner = HostFatalExit(**options)  # type: ignore[arg-type]
        owner.arm()
        return owner


async def _settled(owner: HostFatalExit) -> None:
    operation = owner._operation
    assert operation is not None
    await asyncio.wait_for(operation, timeout=5)


def _line(text: str) -> dict[str, object]:
    (line,) = text.splitlines()
    assert line.startswith(_PREFIX)
    document = json.loads(line.removeprefix(_PREFIX))
    assert isinstance(document, dict)
    return document


def test_the_unavailable_notice_has_one_source_and_promises_no_session_restart() -> None:
    from pathlib import Path

    from hermes_realtime.conversation.work_tools import WORK_CONTROL_UNAVAILABLE_NOTICE

    # The health behind it belongs to the host; restarting a session cannot clear it.
    assert WORK_CONTROL_UNAVAILABLE_NOTICE == "Task control is unavailable until the host restarts."
    root = Path(__file__).resolve().parents[1]
    package = root / "src" / "hermes_realtime"
    copies = {
        path.relative_to(package).as_posix(): path.read_text(encoding="utf-8").count(
            "Task control is unavailable"
        )
        for path in sorted(package.rglob("*.py"))
    }
    assert {name: count for name, count in copies.items() if count} == {
        "conversation/work_tools.py": 1
    }
    # The browser shows only sentences it knows; its list is bound to the source by equality.
    browser = (root / "web" / "src" / "main.ts").read_text(encoding="utf-8")
    assert browser.count("Task control is unavailable") == 1
    assert browser.count(f'"{WORK_CONTROL_UNAVAILABLE_NOTICE}"') == 1


def test_the_exit_status_is_distinct_from_every_other_host_status() -> None:
    assert HOST_FATAL_EXIT_STATUS == 75
    assert HOST_FATAL_EXIT_STATUS not in {0, 1, 2, 3, 4, 130}


@pytest.mark.asyncio
async def test_a_report_stops_input_then_flushes_then_prints_one_line_then_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder(capsys)
    owner = recorder.owner()

    owner.reporter("task_control")()
    await _settled(owner)

    assert recorder.steps == ["input", "flush", "exit"]
    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]
    # The one line was already written when the process exit was reached.
    assert recorder.stdout_at_exit == [
        '[host-fatal-exit] {"cause":"task_control","flush":"done","input":"done","version":1}\n'
    ]
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", sorted(HOST_FATAL_CAUSES))
async def test_each_fixed_cause_exits_once_however_often_it_is_reported(
    capsys: pytest.CaptureFixture[str], cause: str,
) -> None:
    recorder = _Recorder(capsys)
    owner = recorder.owner()
    report = owner.reporter(cause)

    report()
    report()
    for other in sorted(HOST_FATAL_CAUSES):
        owner.reporter(other)()
    await _settled(owner)
    report()
    await asyncio.sleep(0)

    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]
    assert recorder.steps == ["input", "flush", "exit"]
    # The first cause is the one named; later reports change nothing.
    assert _line(recorder.stdout_at_exit[0])["cause"] == cause
    assert owner.cause == cause


def test_a_cause_outside_the_fixed_set_is_refused_where_the_holder_is_built(
    capsys: pytest.CaptureFixture[str],
) -> None:
    owner = _Recorder(capsys).owner()
    for cause in ("", "dispatch failed: private text", "TASK_CONTROL", None, 7):
        with pytest.raises(ValueError, match="fixed causes"):
            owner.reporter(cause)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_report_before_arming_or_after_disarming_does_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder(capsys)
    owner = HostFatalExit(
        stop_input=recorder.stop_input, flush_conversation=recorder.flush,
        exit_process=recorder.exit,
    )
    # A failed start is reported by the start itself.
    owner.reporter("foreground_adapter")()
    assert owner._operation is None and owner.cause is None

    owner.arm()
    owner.disarm()
    # An orderly shutdown already owns the exit.
    owner.reporter("task_control")()
    await asyncio.sleep(0)

    assert owner._operation is None and owner.cause is None
    assert recorder.steps == [] and recorder.statuses == []
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_a_failed_or_hung_input_stop_does_not_prevent_the_flush_or_the_exit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for behaviour, outcome in (("raise", "failed"), ("hang", "timeout")):
        recorder = _Recorder(capsys)

        async def stop_input(behaviour: str = behaviour) -> None:
            if behaviour == "raise":
                raise RuntimeError("private detail that must not be printed")
            await asyncio.Event().wait()

        owner = recorder.owner(stop_input=stop_input, input_stop_seconds=0.05)
        owner.reporter("update_executor")()
        await _settled(owner)

        assert recorder.steps == ["flush", "exit"]
        assert recorder.stdout_at_exit == [
            '[host-fatal-exit] {"cause":"update_executor","flush":"done",'
            f'"input":"{outcome}","version":1}}\n'
        ]


@pytest.mark.asyncio
async def test_a_failed_flush_still_exits_and_prints_nothing_from_the_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder(capsys)

    async def flush() -> None:
        raise RuntimeError("C:/private/path and transcript text")

    owner = recorder.owner(flush_conversation=flush)
    owner.reporter("update_director")()
    await _settled(owner)

    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]
    (printed,) = recorder.stdout_at_exit
    assert _line(printed) == {
        "cause": "update_director", "flush": "failed", "input": "done", "version": 1,
    }
    assert "private" not in printed and "RuntimeError" not in printed


@pytest.mark.asyncio
async def test_a_flush_that_outlives_its_bound_is_reported_as_a_timeout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder(capsys)

    async def flush() -> None:
        await asyncio.Event().wait()

    owner = recorder.owner(flush_conversation=flush, flush_seconds=0.05)
    owner.reporter("task_control")()
    await _settled(owner)

    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]
    assert _line(recorder.stdout_at_exit[0])["flush"] == "timeout"


@pytest.mark.asyncio
async def test_a_host_without_a_saved_conversation_says_so_and_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder(capsys)
    owner = recorder.owner(flush_conversation=None)
    owner.reporter("task_control")()
    await _settled(owner)

    assert recorder.steps == ["input", "exit"]
    assert _line(recorder.stdout_at_exit[0])["flush"] == "absent"


def test_a_blocked_event_loop_is_ended_by_the_timer_thread(
    capsys: pytest.CaptureFixture[str],
) -> None:
    recorder = _Recorder(capsys)
    owner = recorder.owner(deadline_seconds=0.05)

    async def report_then_block_the_loop() -> None:
        owner.reporter("foreground_adapter")()
        # A blocking wait, deliberately: the loop never yields, so the exit task cannot run
        # and only the timer thread can end the process.
        assert recorder.exited.wait(5)
        assert recorder.steps == ["exit"]

    asyncio.run(report_then_block_the_loop())

    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]
    assert recorder.stdout_at_exit == [
        '[host-fatal-exit] {"cause":"foreground_adapter","flush":"timeout",'
        '"input":"timeout","version":1}\n'
    ]


def test_a_report_with_no_running_loop_still_exits(capsys: pytest.CaptureFixture[str]) -> None:
    recorder = _Recorder(capsys)
    owner = recorder.owner()

    owner.reporter("task_control")()

    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]
    assert _line(recorder.stdout_at_exit[0]) == {
        "cause": "task_control", "flush": "failed", "input": "failed", "version": 1,
    }


@pytest.mark.asyncio
async def test_the_exit_happens_even_when_the_evidence_line_cannot_be_printed(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    recorder = _Recorder(capsys)
    owner = recorder.owner()

    def broken_print(*_args: object, **_kwargs: object) -> None:
        raise OSError("stdout is gone")

    monkeypatch.setattr(builtins, "print", broken_print)
    owner.reporter("task_control")()
    operation = owner._operation
    assert operation is not None
    with pytest.raises(OSError, match="stdout is gone"):
        await asyncio.wait_for(operation, timeout=5)

    assert recorder.statuses == [HOST_FATAL_EXIT_STATUS]

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Win32 handle contract")

_CHECKPOINT_STARTUP_TIMEOUT_SECONDS = 10.0


def _read_line(fd: int) -> bytes:
    value = bytearray()
    while True:
        chunk = os.read(fd, 1)
        if not chunk:
            return bytes(value)
        value.extend(chunk)
        if chunk == b"\n":
            return bytes(value)


async def _read_checkpoint_startup_frame(fd: int) -> bytes:
    return await asyncio.wait_for(
        asyncio.to_thread(_read_line, fd),
        timeout=_CHECKPOINT_STARTUP_TIMEOUT_SECONDS,
    )


@pytest.mark.asyncio
async def test_checkpoint_startup_reader_allows_cold_host_import_budget() -> None:
    read_fd, write_fd = os.pipe()

    def write_after_cold_import_budget() -> None:
        time.sleep(2.1)
        os.write(write_fd, b"ready\n")

    writer = asyncio.create_task(asyncio.to_thread(write_after_cold_import_budget))
    try:
        assert await _read_checkpoint_startup_frame(read_fd) == b"ready\n"
    finally:
        await writer
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.asyncio
async def test_external_checkpoint_child_emits_three_ordered_acknowledged_barriers(
    tmp_path: Path,
) -> None:
    import msvcrt

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    write_handle = msvcrt.get_osfhandle(checkpoint_write_fd)
    resume_handle = msvcrt.get_osfhandle(resume_read_fd)
    os.set_handle_inheritable(write_handle, True)
    os.set_handle_inheritable(resume_handle, True)
    startup = subprocess.STARTUPINFO()
    startup.lpAttributeList = {"handle_list": [write_handle, resume_handle]}
    nonce = "e" * 64
    child = (
        "import asyncio,os,sys\n"
        "from hermes_realtime._qualification import _new_qualification_checkpoint_channel\n"
        "async def main():\n"
        " c=_new_qualification_checkpoint_channel(write_handle=int(sys.argv[1]),"
        "resume_handle=int(sys.argv[2]),nonce=sys.argv[3])\n"
        " assert not os.get_handle_inheritable(c.write_handle)\n"
        " assert not os.get_handle_inheritable(c.resume_handle)\n"
        " try:\n"
        "  for name in ('host_consent_active','host_response_completed_before_shutdown',"
        "'host_drain_started'):\n"
        "   await c.emit(name)\n"
        " finally:\n"
        "  c.close()\n"
        "asyncio.run(main())\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    environment["TMP"] = str(child_temp)
    environment["TEMP"] = str(child_temp)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", child, str(write_handle), str(resume_handle), nonce],
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env=environment,
            startupinfo=startup,
            stderr=subprocess.PIPE,
        )
        os.close(checkpoint_write_fd)
        checkpoint_write_fd = -1
        os.close(resume_read_fd)
        resume_read_fd = -1

        checkpoints = (
            "host_consent_active",
            "host_response_completed_before_shutdown",
            "host_drain_started",
        )
        for ordinal, checkpoint in enumerate(checkpoints, start=1):
            frame = await asyncio.wait_for(
                asyncio.to_thread(_read_line, checkpoint_read_fd),
                timeout=2.0,
            )
            assert frame == (
                b'{"checkpoint":"'
                + checkpoint.encode("ascii")
                + b'","nonce":"'
                + nonce.encode("ascii")
                + b'","ordinal":'
                + str(ordinal).encode("ascii")
                + b',"protocolVersion":1}\n'
            )
            os.write(
                resume_write_fd,
                b'{"nonce":"'
                + nonce.encode("ascii")
                + b'","protocolVersion":1,"resumeOrdinal":'
                + str(ordinal).encode("ascii")
                + b"}\n",
            )
        assert await asyncio.to_thread(process.wait, 5.0) == 0
        assert await asyncio.to_thread(_read_line, checkpoint_read_fd) == b""
        assert process.stderr is not None
        assert process.stderr.read() == b""
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_case",
    (
        "ack_eof",
        "duplicate_key",
        "trailing_bytes",
        "oversized",
        "wrong_nonce",
        "unknown_field",
    ),
)
async def test_external_cli_checkpoint_failures_exit_nonzero_and_reap(
    failure_case: str,
    tmp_path: Path,
) -> None:
    await _assert_external_cli_checkpoint_failure(failure_case, tmp_path)


@pytest.mark.asyncio
async def test_external_cli_checkpoint_failure_reaps_with_stderr_backpressure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _assert_external_cli_checkpoint_failure("ack_eof", tmp_path, stderr_bytes=64 * 1024)
    observation = _checkpoint_observation(capsys)
    assert set(observation) == {
        "version", "case", "pid", "completed", "killed", "reaped", "exit_code",
        "stderr_bytes", "events_ms",
    }
    assert observation["version"] == 1
    assert observation["completed"] is True
    assert observation["reaped"] is True
    assert observation["killed"] is False
    assert observation["stderr_bytes"] >= 64 * 1024
    assert observation["exit_code"] != 0
    assert type(observation["pid"]) is int
    events = observation["events_ms"]
    assert list(events) == [
        "spawn_started", "spawn_returned", "checkpoint_received", "failure_sent",
        "completion_entered", "completion_returned", "cleanup_entered", "cleanup_returned",
    ]
    assert list(events.values()) == sorted(events.values())
    assert events["spawn_started"] == 0


def _checkpoint_observation(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    prefix = "[checkpoint-child] "
    assert lines[0].startswith(prefix)
    return json.loads(lines[0][len(prefix):])


@pytest.mark.asyncio
async def test_external_cli_checkpoint_completion_timeout_propagates_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    children: list[subprocess.Popen[bytes]] = []

    def expire_completion(
        process: subprocess.Popen[bytes], *, timeout: float,
    ) -> tuple[None, bytes]:
        children.append(process)
        assert timeout == 5.0
        raise subprocess.TimeoutExpired("qualification-child", timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", expire_completion)
    with pytest.raises(subprocess.TimeoutExpired):
        await _assert_external_cli_checkpoint_failure("ack_eof", tmp_path)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stderr is not None and children[0].stderr.closed
    observation = _checkpoint_observation(capsys)
    assert observation["completed"] is False
    assert observation["reaped"] is True
    assert observation["stderr_bytes"] is None
    assert "completion_entered" in observation["events_ms"]
    assert "completion_returned" not in observation["events_ms"]
    assert "cleanup_returned" in observation["events_ms"]


async def _assert_external_cli_checkpoint_failure(
    failure_case: str,
    tmp_path: Path,
    *,
    stderr_bytes: int = 0,
) -> None:
    import msvcrt

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    write_handle = msvcrt.get_osfhandle(checkpoint_write_fd)
    resume_handle = msvcrt.get_osfhandle(resume_read_fd)
    os.set_handle_inheritable(write_handle, True)
    os.set_handle_inheritable(resume_handle, True)
    startup = subprocess.STARTUPINFO()
    startup.lpAttributeList = {"handle_list": [write_handle, resume_handle]}
    nonce = "6" * 64
    close_body = (
        f"sys.stderr.buffer.write(b'x' * {stderr_bytes}); sys.stderr.buffer.flush()"
        if stderr_bytes
        else "pass"
    )
    child = (
        "import asyncio,sys\n"
        "import hermes_realtime.host_launcher as h\n"
        "from hermes_realtime._qualification import _current_qualification_checkpoint_channel\n"
        "class L:\n"
        " def __init__(self,c): self.c=c\n"
        " async def start(self):\n"
        "  t=asyncio.create_task(self.c.emit('host_consent_active'))\n"
        "  t.add_done_callback(lambda x:x.exception())\n"
        "  await asyncio.sleep(0)\n"
        "  return 'https://127.0.0.1:8443/'\n"
        f" async def close(self): {close_body}\n"
        "def build(**kw):\n"
        " c=_current_qualification_checkpoint_channel(); assert c is not None; return L(c)\n"
        "h.build_local_host_launcher=build\n"
        "h._load_livekit_credentials=lambda **kw:('devkey','local-'+('x'*32))\n"
        "sys.argv=['hermes-realtime-host','--qualification-no-hermes-tasks',"
        "'--qualification-checkpoint-write-handle',sys.argv[1],"
        "'--qualification-checkpoint-resume-handle',sys.argv[2],"
        "'--qualification-checkpoint-nonce',sys.argv[3]]\n"
        "raise SystemExit(h.main())\n"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["HERMES_REALTIME_QUALIFICATION_CHILD"] = "1"
    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    environment["TMP"] = str(child_temp)
    environment["TEMP"] = str(child_temp)
    process: subprocess.Popen[bytes] | None = None
    started = time.perf_counter_ns()
    events_ms: dict[str, float] = {"spawn_started": 0.0}
    completed = False
    killed = False
    stderr_size: int | None = None

    def mark(event: str) -> None:
        events_ms[event] = round((time.perf_counter_ns() - started) / 1_000_000, 3)

    try:
        process = subprocess.Popen(
            [sys.executable, "-c", child, str(write_handle), str(resume_handle), nonce],
            close_fds=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            env=environment,
            startupinfo=startup,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        mark("spawn_returned")
        os.close(checkpoint_write_fd)
        checkpoint_write_fd = -1
        os.close(resume_read_fd)
        resume_read_fd = -1
        frame = await _read_checkpoint_startup_frame(checkpoint_read_fd)
        mark("checkpoint_received")
        assert frame.startswith(b'{"checkpoint":"host_consent_active"')

        if failure_case == "ack_eof":
            os.close(resume_write_fd)
            resume_write_fd = -1
        elif failure_case == "duplicate_key":
            os.write(
                resume_write_fd,
                b'{"nonce":"'
                + nonce.encode("ascii")
                + b'","nonce":"'
                + nonce.encode("ascii")
                + b'","protocolVersion":1,"resumeOrdinal":1}\n',
            )
        elif failure_case == "trailing_bytes":
            os.write(resume_write_fd, b'{}{}\n')
        elif failure_case == "oversized":
            os.write(resume_write_fd, (b"x" * 193) + b"\n")
        elif failure_case == "wrong_nonce":
            os.write(
                resume_write_fd,
                b'{"nonce":"'
                + (b"7" * 64)
                + b'","protocolVersion":1,"resumeOrdinal":1}\n',
            )
        else:
            os.write(
                resume_write_fd,
                b'{"extra":1,"nonce":"'
                + nonce.encode("ascii")
                + b'","protocolVersion":1,"resumeOrdinal":1}\n',
            )

        mark("failure_sent")
        # Drain while waiting: a full stderr pipe must not prevent the child from exiting.
        mark("completion_entered")
        _, stderr = await asyncio.to_thread(process.communicate, timeout=5.0)
        mark("completion_returned")
        stderr_size = len(stderr)
        assert process.returncode != 0
        assert process.poll() is not None
        assert await asyncio.to_thread(_read_line, checkpoint_read_fd) == b""
        assert stderr.startswith(b"x" * stderr_bytes)
        assert b"qualification checkpoint channel failed" in stderr
        completed = True
    finally:
        mark("cleanup_entered")
        try:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    killed = True
                    process.wait(timeout=5.0)
                if process.stderr is not None:
                    process.stderr.close()
            for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
                if fd >= 0:
                    os.close(fd)
            mark("cleanup_returned")
        finally:
            # Pytest retains this on failure; -s also exposes healthy comparison samples.
            # Only parent-observed timings and scalar process facts leave the fixture.
            print("[checkpoint-child] " + json.dumps({
                "version": 1,
                "case": failure_case,
                "pid": None if process is None else process.pid,
                "completed": completed,
                "killed": killed,
                "reaped": process is not None and process.poll() is not None,
                "exit_code": None if process is None else process.returncode,
                "stderr_bytes": stderr_size,
                "events_ms": events_ms,
            }))


@pytest.mark.asyncio
async def test_qualification_sigbreak_enters_owned_close_and_restores_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt
    import signal

    import hermes_realtime.host_launcher as host_launcher

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    args = host_launcher._build_argument_parser().parse_args(
        [
            "--qualification-no-hermes-tasks",
            "--qualification-checkpoint-write-handle",
            str(msvcrt.get_osfhandle(checkpoint_write_fd)),
            "--qualification-checkpoint-resume-handle",
            str(msvcrt.get_osfhandle(resume_read_fd)),
            "--qualification-checkpoint-nonce",
            "d" * 64,
        ]
    )
    previous_handler = object()
    signal_calls: list[tuple[object, object]] = []
    closed: list[bool] = []

    def install_signal(signum: object, handler: object) -> object:
        signal_calls.append((signum, handler))
        return previous_handler

    class Launcher:
        async def start(self) -> str:
            assert signal_calls and signal_calls[0][0] is signal.SIGBREAK
            handler = signal_calls[0][1]
            assert callable(handler)
            handler(signal.SIGBREAK, None)
            return "https://127.0.0.1:8443/"

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(signal, "signal", install_signal)
    monkeypatch.setattr(host_launcher, "build_local_host_launcher", lambda **_kwargs: Launcher())
    monkeypatch.setattr(
        host_launcher,
        "_load_livekit_credentials",
        lambda **_kwargs: ("devkey", "local-" + "x" * 32),
    )
    monkeypatch.setenv("HERMES_REALTIME_QUALIFICATION_CHILD", "1")
    try:
        run = host_launcher._run_host_cli(args)
        checkpoint_write_fd = -1
        resume_read_fd = -1
        await asyncio.wait_for(run, timeout=1.0)

        assert closed == [True]
        assert len(signal_calls) == 2
        assert signal_calls[1] == (signal.SIGBREAK, previous_handler)
    finally:
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_cli_checkpoint_failure_terminates_owned_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt
    import signal

    import hermes_realtime.host_launcher as host_launcher
    from hermes_realtime._qualification import _current_qualification_checkpoint_channel

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    args = host_launcher._build_argument_parser().parse_args(
        [
            "--qualification-no-hermes-tasks",
            "--qualification-checkpoint-write-handle",
            str(msvcrt.get_osfhandle(checkpoint_write_fd)),
            "--qualification-checkpoint-resume-handle",
            str(msvcrt.get_osfhandle(resume_read_fd)),
            "--qualification-checkpoint-nonce",
            "4" * 64,
        ]
    )
    captured: dict[str, object] = {}
    background_errors: list[BaseException | None] = []

    class Launcher:
        async def start(self) -> str:
            channel = captured["channel"]
            assert channel is not None
            emission = asyncio.create_task(
                channel.emit("host_response_completed_before_shutdown")  # type: ignore[union-attr]
            )
            emission.add_done_callback(lambda task: background_errors.append(task.exception()))
            await asyncio.sleep(0)
            return "https://127.0.0.1:8443/"

        async def close(self) -> None:
            captured["launcher_closed"] = True

    def build(**_kwargs: object) -> Launcher:
        captured["channel"] = _current_qualification_checkpoint_channel()
        return Launcher()

    monkeypatch.setattr(signal, "signal", lambda _signum, _handler: signal.SIG_DFL)
    monkeypatch.setattr(host_launcher, "build_local_host_launcher", build)
    monkeypatch.setattr(
        host_launcher,
        "_load_livekit_credentials",
        lambda **_kwargs: ("devkey", "local-" + "x" * 32),
    )
    monkeypatch.setenv("HERMES_REALTIME_QUALIFICATION_CHILD", "1")
    try:
        run = host_launcher._run_host_cli(args)
        checkpoint_write_fd = -1
        resume_read_fd = -1
        with pytest.raises(RuntimeError, match="checkpoint channel failed"):
            await asyncio.wait_for(run, timeout=1.0)
        assert captured["launcher_closed"] is True
        assert len(background_errors) == 1
        assert isinstance(background_errors[0], RuntimeError)
    finally:
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_cli_child_installs_channel_before_host_construction_and_closes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    import hermes_realtime.host_launcher as host_launcher
    from hermes_realtime._qualification import _current_qualification_checkpoint_channel

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    write_handle = msvcrt.get_osfhandle(checkpoint_write_fd)
    resume_handle = msvcrt.get_osfhandle(resume_read_fd)
    os.set_handle_inheritable(write_handle, True)
    os.set_handle_inheritable(resume_handle, True)
    args = host_launcher._build_argument_parser().parse_args(
        [
            "--qualification-no-hermes-tasks",
            "--qualification-checkpoint-write-handle",
            str(write_handle),
            "--qualification-checkpoint-resume-handle",
            str(resume_handle),
            "--qualification-checkpoint-nonce",
            "c" * 64,
        ]
    )
    captured: dict[str, object] = {}

    class Launcher:
        async def start(self) -> str:
            raise asyncio.CancelledError

        async def close(self) -> None:
            captured["launcher_closed"] = True

    def build(**_kwargs: object) -> Launcher:
        channel = _current_qualification_checkpoint_channel()
        captured["channel"] = channel
        captured["write_inheritable"] = os.get_handle_inheritable(write_handle)
        captured["resume_inheritable"] = os.get_handle_inheritable(resume_handle)
        return Launcher()

    monkeypatch.setattr(host_launcher, "build_local_host_launcher", build)
    monkeypatch.setattr(
        host_launcher,
        "_load_livekit_credentials",
        lambda **_kwargs: ("devkey", "local-" + "x" * 32),
    )
    monkeypatch.setenv("HERMES_REALTIME_QUALIFICATION_CHILD", "1")
    try:
        with pytest.raises(asyncio.CancelledError):
            await host_launcher._run_host_cli(args)
        checkpoint_write_fd = -1
        resume_read_fd = -1

        assert captured["channel"] is not None
        assert captured["write_inheritable"] is False
        assert captured["resume_inheritable"] is False
        assert captured["launcher_closed"] is True
        assert _current_qualification_checkpoint_channel() is None
        assert await asyncio.to_thread(os.read, checkpoint_read_fd, 1) == b""
    finally:
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_checkpoint_scope_is_absent_by_default_and_restores_after_construction() -> None:
    try:
        from hermes_realtime._qualification import (
            _current_qualification_checkpoint_channel,
            _new_qualification_checkpoint_channel,
            _qualification_checkpoint_channel_scope,
        )
    except ImportError:
        pytest.fail("RED bootstrap: expected private qualification checkpoint scope")

    import msvcrt

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    channel = _new_qualification_checkpoint_channel(
        write_handle=msvcrt.get_osfhandle(checkpoint_write_fd),
        resume_handle=msvcrt.get_osfhandle(resume_read_fd),
        nonce="b" * 64,
    )
    checkpoint_write_fd = -1
    resume_read_fd = -1
    try:
        assert _current_qualification_checkpoint_channel() is None
        with _qualification_checkpoint_channel_scope(channel):
            assert _current_qualification_checkpoint_channel() is channel
            with (
                pytest.raises(RuntimeError, match="already active"),
                _qualification_checkpoint_channel_scope(channel),
            ):
                pass
        assert _current_qualification_checkpoint_channel() is None
    finally:
        channel.close()
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


def test_checkpoint_constructor_closes_transferred_handles_on_partial_failure() -> None:
    import _winapi
    import msvcrt

    from hermes_realtime._qualification import _new_qualification_checkpoint_channel

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    write_handle = msvcrt.get_osfhandle(checkpoint_write_fd)
    invalid_resume_handle = msvcrt.get_osfhandle(resume_read_fd)
    os.close(resume_read_fd)
    resume_read_fd = -1
    checkpoint_write_fd = -1
    try:
        with pytest.raises(OSError):
            _new_qualification_checkpoint_channel(
                write_handle=write_handle,
                resume_handle=invalid_resume_handle,
                nonce="2" * 64,
            )
        with pytest.raises(BrokenPipeError):
            _winapi.PeekNamedPipe(msvcrt.get_osfhandle(checkpoint_read_fd))
    finally:
        with suppress(OSError):
            _winapi.CloseHandle(write_handle)
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_checkpoint_channel_uses_bounded_nonblocking_pipe_polling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    from hermes_realtime import _qualification as qualification

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    channel = qualification._new_qualification_checkpoint_channel(
        write_handle=msvcrt.get_osfhandle(checkpoint_write_fd),
        resume_handle=msvcrt.get_osfhandle(resume_read_fd),
        nonce="1" * 64,
    )
    checkpoint_write_fd = -1
    resume_read_fd = -1

    async def forbidden_to_thread(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("checkpoint transport must not strand worker threads")

    monkeypatch.setattr(qualification.asyncio, "to_thread", forbidden_to_thread)
    try:
        emission = asyncio.create_task(channel.emit("host_consent_active"))
        await asyncio.sleep(0.01)
        assert emission.done() is False
        frame = os.read(checkpoint_read_fd, 193)
        assert frame.endswith(b'"protocolVersion":1}\n')
        os.write(
            resume_write_fd,
            b'{"nonce":"'
            + (b"1" * 64)
            + b'","protocolVersion":1,"resumeOrdinal":1}\n',
        )
        await asyncio.wait_for(emission, timeout=1.0)
    finally:
        channel.close()
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_checkpoint_timeout_closes_pipes_and_notifies_terminal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import msvcrt

    from hermes_realtime import _qualification as qualification

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    failures: list[BaseException] = []
    monkeypatch.setattr(qualification, "_CHECKPOINT_TIMEOUT_SECONDS", 0.02)
    channel = qualification._new_qualification_checkpoint_channel(
        write_handle=msvcrt.get_osfhandle(checkpoint_write_fd),
        resume_handle=msvcrt.get_osfhandle(resume_read_fd),
        nonce="5" * 64,
        failure_callback=failures.append,
    )
    checkpoint_write_fd = -1
    resume_read_fd = -1
    try:
        emission = asyncio.create_task(channel.emit("host_consent_active"))
        await asyncio.sleep(0)
        assert _read_line(checkpoint_read_fd).endswith(b'"protocolVersion":1}\n')
        with pytest.raises(TimeoutError, match="timed out"):
            await asyncio.wait_for(emission, timeout=1.0)
        assert len(failures) == 1
        assert isinstance(failures[0], TimeoutError)
        assert _read_line(checkpoint_read_fd) == b""
    finally:
        channel.close()
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_checkpoint_failure_callback_is_one_shot_and_terminal() -> None:
    import msvcrt

    from hermes_realtime._qualification import _new_qualification_checkpoint_channel

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    failures: list[BaseException] = []
    channel = _new_qualification_checkpoint_channel(
        write_handle=msvcrt.get_osfhandle(checkpoint_write_fd),
        resume_handle=msvcrt.get_osfhandle(resume_read_fd),
        nonce="3" * 64,
        failure_callback=failures.append,
    )
    checkpoint_write_fd = -1
    resume_read_fd = -1
    try:
        with pytest.raises(RuntimeError, match="out of order"):
            await channel.emit("host_response_completed_before_shutdown")
        with pytest.raises(RuntimeError, match="unavailable"):
            await channel.emit("host_consent_active")
        assert len(failures) == 1
        assert str(failures[0]) == "qualification checkpoint is duplicate or out of order"
    finally:
        channel.close()
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_checkpoint_ack_rejects_json_booleans_for_integer_fields() -> None:
    import msvcrt

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    channel = None
    try:
        from hermes_realtime._qualification import _new_qualification_checkpoint_channel

        channel = _new_qualification_checkpoint_channel(
            write_handle=msvcrt.get_osfhandle(checkpoint_write_fd),
            resume_handle=msvcrt.get_osfhandle(resume_read_fd),
            nonce="f" * 64,
        )
        checkpoint_write_fd = -1
        resume_read_fd = -1
        emission = asyncio.create_task(channel.emit("host_consent_active"))
        assert await asyncio.to_thread(_read_line, checkpoint_read_fd)
        os.write(
            resume_write_fd,
            b'{"nonce":"'
            + (b"f" * 64)
            + b'","protocolVersion":true,"resumeOrdinal":true}\n',
        )
        with pytest.raises(RuntimeError, match="does not match"):
            await emission
        with pytest.raises(RuntimeError, match="unavailable"):
            await channel.emit("host_response_completed_before_shutdown")
    finally:
        if channel is not None:
            channel.close()
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)


@pytest.mark.asyncio
async def test_checkpoint_channel_writes_canonical_frame_and_requires_exact_ack() -> None:
    try:
        from hermes_realtime._qualification import _new_qualification_checkpoint_channel
    except ImportError:
        pytest.fail("RED bootstrap: expected private qualification checkpoint channel")

    import msvcrt

    checkpoint_read_fd, checkpoint_write_fd = os.pipe()
    resume_read_fd, resume_write_fd = os.pipe()
    channel = _new_qualification_checkpoint_channel(
        write_handle=msvcrt.get_osfhandle(checkpoint_write_fd),
        resume_handle=msvcrt.get_osfhandle(resume_read_fd),
        nonce="a" * 64,
    )
    checkpoint_write_fd = -1
    resume_read_fd = -1
    try:
        assert os.get_handle_inheritable(channel.write_handle) is False
        assert os.get_handle_inheritable(channel.resume_handle) is False

        emission = asyncio.create_task(channel.emit("host_consent_active"))
        frame = await asyncio.to_thread(os.read, checkpoint_read_fd, 193)
        assert frame == (
            b'{"checkpoint":"host_consent_active","nonce":"'
            + (b"a" * 64)
            + b'","ordinal":1,"protocolVersion":1}\n'
        )
        os.write(
            resume_write_fd,
            b'{"nonce":"' + (b"a" * 64) + b'","protocolVersion":1,"resumeOrdinal":1}\n',
        )
        await emission
    finally:
        channel.close()
        for fd in (checkpoint_read_fd, resume_write_fd, checkpoint_write_fd, resume_read_fd):
            if fd >= 0:
                os.close(fd)

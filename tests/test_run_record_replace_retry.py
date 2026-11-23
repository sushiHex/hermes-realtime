"""The run journal writer survives a transient foreign handle on Windows, within a bound."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from hermes_realtime.integration import run_record
from hermes_realtime.integration.run_record import write_run_record

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows open-handle denial")


def _temporaries(directory: Path) -> list[str]:
    return [item.name for item in directory.iterdir() if item.name.endswith(".tmp")]


@contextlib.contextmanager
def _held_open(path: Path, seconds: float) -> Iterator[None]:
    """Hold a plain read handle on ``path`` in a thread; always released and joined."""
    opened, release = threading.Event(), threading.Event()

    def hold() -> None:
        with path.open("rb"):
            opened.set()
            release.wait(seconds)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    try:
        assert opened.wait(5)
        yield
    finally:
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()


@windows_only
def test_replace_succeeds_after_a_transient_foreign_handle(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b'{"n": 0}')
    with _held_open(path, 0.05):
        write_run_record(path, b'{"n": 1}')
    assert path.read_bytes() == b'{"n": 1}'
    assert _temporaries(tmp_path) == []


@windows_only
def test_handle_outliving_the_bound_raises_keeps_old_record_and_leaves_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b'{"n": 0}')
    real, calls = os.replace, []

    def counting(source: object, destination: object) -> None:
        calls.append(1)
        real(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(run_record.os, "replace", counting)
    monkeypatch.setattr(run_record, "_REPLACE_ATTEMPTS", 3)
    monkeypatch.setattr(run_record, "_REPLACE_RETRY_SECONDS", 0.001)
    with _held_open(path, 30.0), pytest.raises(PermissionError):
        write_run_record(path, b'{"n": 1}')
    assert len(calls) == 3
    assert _temporaries(tmp_path) == []
    assert capsys.readouterr().out.splitlines() == [
        '[hermes-run-record-replace] {"category":"replace_denied","attempts":3}'
    ]
    assert path.read_bytes() == b'{"n": 0}'


def _replace_failing(
    monkeypatch: pytest.MonkeyPatch, failures: int, error: OSError
) -> tuple[list[int], list[float]]:
    real, calls, sleeps = os.replace, [], []

    def flaky(source: object, destination: object) -> None:
        calls.append(1)
        if len(calls) <= failures:
            raise error
        real(source, destination)  # type: ignore[arg-type]

    monkeypatch.setattr(run_record.os, "replace", flaky)
    monkeypatch.setattr(run_record.time, "sleep", sleeps.append)
    return calls, sleeps


def test_writer_retries_permission_error_then_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "journal.json"
    calls, sleeps = _replace_failing(monkeypatch, 3, PermissionError(13, "denied"))
    write_run_record(path, b"1")
    assert path.read_bytes() == b"1"
    assert len(calls) == 4
    assert capsys.readouterr().out == ""  # no marker on success
    assert sleeps == [run_record._REPLACE_RETRY_SECONDS] * 3
    assert _temporaries(tmp_path) == []


def test_writer_raises_original_error_after_bound_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "journal.json"
    error = PermissionError(13, "denied")
    calls, sleeps = _replace_failing(monkeypatch, 10**6, error)
    with pytest.raises(PermissionError) as raised:
        write_run_record(path, b"1")
    assert raised.value is error
    assert len(calls) == run_record._REPLACE_ATTEMPTS
    assert len(sleeps) == run_record._REPLACE_ATTEMPTS - 1
    assert not path.exists()
    assert _temporaries(tmp_path) == []
    out = capsys.readouterr().out
    assert out.splitlines() == [
        '[hermes-run-record-replace] {"category":"replace_denied","attempts":'
        + str(run_record._REPLACE_ATTEMPTS) + "}"
    ]
    assert "denied\"" not in out.replace('"replace_denied"', "") and str(tmp_path) not in out


@pytest.mark.parametrize("failure", [BrokenPipeError(32, "pipe"), OSError(9, "bad"), ValueError()])
def test_a_failed_evidence_write_never_replaces_the_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b"old")
    error = PermissionError(13, "denied")
    calls, _ = _replace_failing(monkeypatch, 10**6, error)

    def broken_print(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(run_record, "print", broken_print, raising=False)
    with pytest.raises(PermissionError) as raised:
        write_run_record(path, b"new")
    assert raised.value is error
    assert len(calls) == run_record._REPLACE_ATTEMPTS
    assert path.read_bytes() == b"old"
    assert _temporaries(tmp_path) == []


def test_writer_does_not_retry_other_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.json"
    calls, sleeps = _replace_failing(monkeypatch, 10**6, OSError(5, "io"))
    with pytest.raises(OSError):
        write_run_record(path, b"1")
    assert len(calls) == 1
    assert sleeps == []
    assert _temporaries(tmp_path) == []


@windows_only
@pytest.mark.timeout(60)
def test_atomic_writes_never_fail_beside_a_concurrent_plain_reader(tmp_path: Path) -> None:
    """200 real replaces while a plain reader opens the record in a loop: the writer never
    fails. A reader open denied during a replace is counted, not asserted: that reader-side
    transient belongs to the reader's contract, which this change does not alter."""
    path = tmp_path / "journal.json"
    documents = {json.dumps({"n": n}).encode() for n in range(200)}
    write_run_record(path, b'{"n": 0}')
    writer_errors: list[BaseException] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for n in range(200):
                if stop.is_set():
                    return
                write_run_record(path, json.dumps({"n": n}).encode())
        except BaseException as error:  # noqa: BLE001
            writer_errors.append(error)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    reads = denied = 0
    try:
        while thread.is_alive():
            try:
                raw = path.read_bytes()
            except PermissionError:
                denied += 1
                continue
            assert raw in documents
            reads += 1
    finally:
        stop.set()
        thread.join(timeout=30)
    assert not thread.is_alive()
    assert writer_errors == []
    assert reads > 0
    print(f"reads={reads} reader_denied={denied}")

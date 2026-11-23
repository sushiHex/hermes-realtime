"""The run journal survives a transient foreign handle: writer and harness reader both retry."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

import scripts.native_typed_journey as journey
from hermes_realtime.integration import run_record
from hermes_realtime.integration.run_record import write_run_record

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows open-handle denial")


def _temporaries(directory: Path) -> list[str]:
    return [item.name for item in directory.iterdir() if item.name.endswith(".tmp")]


def _hold_open(path: Path, seconds: float) -> threading.Thread:
    opened = threading.Event()

    def hold() -> None:
        with path.open("rb"):
            opened.set()
            time.sleep(seconds)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert opened.wait(5)
    return thread


@windows_only
def test_replace_succeeds_after_a_transient_foreign_handle(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b'{"n": 0}')
    holder = _hold_open(path, 0.05)
    write_run_record(path, b'{"n": 1}')
    holder.join(timeout=5)
    assert path.read_bytes() == b'{"n": 1}'
    assert _temporaries(tmp_path) == []


@windows_only
def test_handle_outliving_the_bound_raises_keeps_old_record_and_leaves_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    holder = _hold_open(path, 1.0)
    with pytest.raises(PermissionError):
        write_run_record(path, b'{"n": 1}')
    assert len(calls) == 3
    assert _temporaries(tmp_path) == []
    holder.join(timeout=5)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.json"
    calls, sleeps = _replace_failing(monkeypatch, 3, PermissionError(13, "denied"))
    write_run_record(path, b"1")
    assert path.read_bytes() == b"1"
    assert len(calls) == 4
    assert sleeps == [run_record._REPLACE_RETRY_SECONDS] * 3
    assert _temporaries(tmp_path) == []


def test_writer_raises_original_error_after_bound_and_removes_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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


def test_read_json_returns_value_and_missing_file_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(journey.time, "sleep", sleeps.append)
    path = tmp_path / "journal.json"
    path.write_bytes(b'{"a": [1, 2]}')
    assert journey.read_json(path) == {"a": [1, 2]}
    with pytest.raises(FileNotFoundError):
        journey.read_json(tmp_path / "missing.json")
    assert sleeps == []


def test_read_json_keeps_cap_and_json_error_mapping(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b"[" + b"0," * 40 + b"0]")
    with pytest.raises(ValueError, match="journey_file_bound"):
        journey.read_json(path, cap=16)
    path.write_bytes(b"{not json")
    with pytest.raises(ValueError, match="journey_json_invalid"):
        journey.read_json(path)


def _flaky_reader(
    monkeypatch: pytest.MonkeyPatch, failures: int, error: OSError
) -> tuple[list[int], list[float]]:
    calls: list[int] = []
    sleeps: list[float] = []
    real = journey._open_journal

    def opener(path: Path):  # type: ignore[no-untyped-def]
        calls.append(1)
        if len(calls) <= failures:
            raise error
        return real(path)

    monkeypatch.setattr(journey, "_open_journal", opener)
    monkeypatch.setattr(journey.time, "sleep", sleeps.append)
    return calls, sleeps


def test_reader_retries_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b"7")
    failures = 2
    calls, sleeps = _flaky_reader(monkeypatch, failures, PermissionError(13, "denied"))
    assert journey.read_json(path) == 7
    assert len(calls) == failures + 1
    assert sleeps == [journey._OPEN_RETRY_SECONDS] * failures


def test_reader_raises_persistent_permission_error_after_bounded_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b"7")
    error = PermissionError(13, "denied")
    calls, _ = _flaky_reader(monkeypatch, 10**6, error)
    with pytest.raises(PermissionError) as raised:
        journey.read_json(path)
    assert raised.value is error
    assert len(calls) == journey._OPEN_ATTEMPTS


def test_reader_does_not_retry_other_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.json"
    path.write_bytes(b"7")
    calls, sleeps = _flaky_reader(monkeypatch, 10**6, OSError(5, "io"))
    with pytest.raises(OSError):
        journey.read_json(path)
    assert len(calls) == 1
    assert sleeps == []


@windows_only
@pytest.mark.timeout(60)
def test_concurrent_reads_and_atomic_writes_never_fail(tmp_path: Path) -> None:
    path = tmp_path / "journal.json"
    documents = {json.dumps({"n": n}).encode() for n in range(200)}
    write_run_record(path, b'{"n": 0}')
    writer_errors: list[BaseException] = []

    def writer() -> None:
        try:
            for n in range(200):
                write_run_record(path, json.dumps({"n": n}).encode())
        except BaseException as error:  # noqa: BLE001
            writer_errors.append(error)

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    reads = 0
    while thread.is_alive():
        assert json.dumps(journey.read_json(path)).encode() in documents
        reads += 1
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert writer_errors == []
    assert reads > 0

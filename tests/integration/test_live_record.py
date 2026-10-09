"""A test may read a record while its writer replaces it, without reading a refusal as a result."""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from pathlib import Path

import pytest

from hermes_realtime.integration.run_record import read_run_record, write_run_record
from tests.support import live_record
from tests.support.live_record import DENIED_ATTEMPTS, read_live_record


def _deny_opens(
    monkeypatch: pytest.MonkeyPatch, target: Path, refusals: int, error: OSError
) -> list[int]:
    """Refuse the first ``refusals`` opens of ``target``, as a replace in flight does."""
    opens = [0]
    real_open = Path.open

    def open_(self: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self == target:
            opens[0] += 1
            if opens[0] <= refusals:
                raise error
        return real_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "open", open_)
    return opens


def _sharing_violation() -> PermissionError:
    return PermissionError(13, "sharing violation")


def test_the_unguarded_read_fails_on_a_refusal_the_live_read_waits_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tail.json"
    write_run_record(path, b"record")
    opens = _deny_opens(monkeypatch, path, 2, _sharing_violation())

    with pytest.raises(PermissionError):
        read_run_record(path, 1 << 20)
    assert read_live_record(path) == b"record"
    assert opens == [3]


def test_a_refusal_that_never_clears_raises_after_a_bounded_number_of_opens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tail.json"
    write_run_record(path, b"record")
    monkeypatch.setattr(live_record, "DENIED_PAUSE_SECONDS", 0.0)
    opens = _deny_opens(monkeypatch, path, 10**6, _sharing_violation())

    with pytest.raises(PermissionError):
        read_live_record(path)
    assert opens == [DENIED_ATTEMPTS]


def test_only_a_refusal_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tail.json"
    write_run_record(path, b"record")
    opens = _deny_opens(monkeypatch, path, 1, OSError(5, "some other failure"))

    with pytest.raises(OSError, match="some other failure") as raised:
        read_live_record(path)
    assert type(raised.value) is OSError
    assert opens == [1]


def test_a_missing_record_is_absent_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "tail.json"
    opens = _deny_opens(monkeypatch, path, 0, _sharing_violation())

    assert read_live_record(path) is None
    assert opens == [1]


@pytest.mark.skipif(sys.platform != "win32", reason="a sharing violation is a Windows refusal")
def test_a_real_sharing_violation_clears_when_the_other_handle_closes(tmp_path: Path) -> None:
    path = tmp_path / "tail.json"
    write_run_record(path, b"record")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    generic_read, open_existing, no_sharing = 0x80000000, 3, 0
    handle = kernel32.CreateFileW(str(path), generic_read, no_sharing, None, open_existing, 0, None)
    assert handle not in (None, wintypes.HANDLE(-1).value)

    release = threading.Timer(0.05, kernel32.CloseHandle, args=(wintypes.HANDLE(handle),))
    try:
        with pytest.raises(PermissionError):
            read_run_record(path, 1 << 20)
    except BaseException:
        kernel32.CloseHandle(wintypes.HANDLE(handle))
        raise
    release.start()
    assert read_live_record(path) == b"record"
    release.join()

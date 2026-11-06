"""Actual SDK lifecycle contracts with fake FFI functions; no native initialization."""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import subprocess
import sys
import weakref
from pathlib import Path

import pytest


class _Function:
    """A callable supporting the ctypes signature attributes used by the SDK."""

    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


class _Library:
    def __init__(self):
        self.disposals = 0
        self.drops = 0
        self.late_drops = 0
        self.livekit_ffi_initialize = _Function(lambda *_: None)
        self.livekit_ffi_request = _Function(lambda *_: 0)
        self.livekit_ffi_drop_handle = _Function(self.drop)
        self.livekit_ffi_dispose = _Function(self.dispose)

    def drop(self, handle):
        self.drops += 1
        self.late_drops += int(self.disposals > 0)
        return handle.value != 999999

    def dispose(self):
        self.disposals += 1


def _observe_sdk_lifecycle(sdk_root: Path) -> dict[str, object]:
    """Child-only observation; this function is never called by a parent test."""
    sys.path.insert(0, str(sdk_root))
    dll_attempts = 0

    def no_dll(*_args, **_kwargs):
        nonlocal dll_attempts
        dll_attempts += 1
        raise AssertionError("SDK lifecycle contract must not load a native library")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ctypes, "CDLL", no_dll)
        sdk = importlib.import_module("livekit.rtc._ffi_client")
        source = Path(sdk.__file__).resolve()
        assert source.is_relative_to(sdk_root)
        assert sdk.FfiClient._instance is None
        monkeypatch.setattr(sdk, "_live_handles", weakref.WeakSet(), raising=False)
        monkeypatch.setattr(sdk.FfiClient, "_instance", None)
        monkeypatch.setattr(sdk.FfiClient, "_ffi_disposed", False, raising=False)
        library = _Library()
        monkeypatch.setattr(sdk, "get_ffi_lib", lambda: library)
        callbacks = []

        def register(callback, *args, **kwargs):
            callbacks.append((callback, args, kwargs))
            return callback

        monkeypatch.setattr(atexit, "register", register)
        client = sdk.FfiClient()
        monkeypatch.setattr(sdk.FfiClient, "_instance", client)
        assert len(callbacks) == 1
        callback, args, kwargs = callbacks[0]
        handles = []

        def handle(value):
            item = sdk.FfiHandle(value)
            handles.append(item)
            return item

        try:
            invalid = handle(999999)
            invalid_refused = False
            try:
                invalid.dispose()
            except AssertionError:
                invalid_refused = True
            active_drop_calls = library.drops
            first, second = handle(1), handle(2)
            assert not first.disposed and not second.disposed
            callback(*args, **kwargs)
            consumed = sum(item.disposed for item in (first, second))
            first_disposals = library.disposals
            drops_during_global = library.drops - active_drop_calls
            first.dispose()
            second.dispose()
            late = handle(3)
            late.dispose()
            late.__del__()
            callback(*args, **kwargs)
            return {
                "outcome": "observed",
                "sdk_version": importlib.metadata.version("livekit"),
                "ffi_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "live_handles_consumed": consumed,
                "first_global_disposals": first_disposals,
                "drops_during_global_disposal": drops_during_global,
                "late_handle_disposed": late.disposed,
                "late_drops": library.late_drops,
                "global_disposals": library.disposals,
                "active_invalid_handle_refused": invalid_refused,
                "active_drop_calls": active_drop_calls,
                "dll_load_attempts": dll_attempts,
                "native_calls": 0,
            }
        finally:
            # Never register the fake SDK callback at real interpreter exit.
            for item in handles:
                item.mark_consumed()


@pytest.fixture(scope="module")
def sdk_lifecycle():
    # Metadata discovery does not import or patch the parent's SDK classes.
    sdk_root = Path(importlib.metadata.distribution("livekit").locate_file("")).resolve()
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
         "--sdk-probe", str(sdk_root)],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, "isolated SDK lifecycle observation failed"
    assert len(result.stdout) <= 4096
    observation = json.loads(result.stdout)
    assert observation["outcome"] == "observed"
    assert observation["dll_load_attempts"] == observation["native_calls"] == 0
    return observation


def test_global_disposal_consumes_live_handles(sdk_lifecycle):
    assert sdk_lifecycle["live_handles_consumed"] == 2
    assert sdk_lifecycle["first_global_disposals"] == 1
    assert sdk_lifecycle["drops_during_global_disposal"] == 0


def test_handle_created_after_global_disposal_does_not_drop(sdk_lifecycle):
    assert sdk_lifecycle["late_handle_disposed"]
    assert sdk_lifecycle["late_drops"] == 0


def test_global_disposal_callback_is_idempotent(sdk_lifecycle):
    assert sdk_lifecycle["global_disposals"] == 1


def test_active_runtime_invalid_handle_drop_still_refuses(sdk_lifecycle):
    assert sdk_lifecycle["active_invalid_handle_refused"]
    assert sdk_lifecycle["active_drop_calls"] == 1


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--sdk-probe":
        raise SystemExit(2)
    try:
        print(json.dumps(_observe_sdk_lifecycle(Path(sys.argv[2]).resolve(strict=True))))
    except Exception as error:
        print(json.dumps({"outcome": "refused", "failure_kind": type(error).__name__}))
        raise SystemExit(2) from None

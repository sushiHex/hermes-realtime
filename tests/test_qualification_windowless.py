"""Observe the real venv redirector without displaying a test on the input desktop."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
import uuid
from ctypes import wintypes
from pathlib import Path
from typing import Any

import pytest

from scripts import qualification_launch as launch

_CHILD = """
import ctypes as c,json,os,sys,time
from pathlib import Path
k=c.WinDLL('kernel32',use_last_error=True);u=c.WinDLL('user32',use_last_error=True)
k.GetConsoleWindow.restype=c.c_void_p;k.GetCurrentThreadId.restype=c.c_uint32
u.GetThreadDesktop.argtypes=[c.c_uint32];u.GetThreadDesktop.restype=c.c_void_p
u.GetUserObjectInformationW.argtypes=[c.c_void_p,c.c_int,c.c_void_p,c.c_uint32,c.POINTER(c.c_uint32)]
u.IsWindowVisible.argtypes=[c.c_void_p]
name=c.create_unicode_buffer(256);needed=c.c_uint32()
desktop=u.GetThreadDesktop(k.GetCurrentThreadId())
if not u.GetUserObjectInformationW(desktop,2,name,c.sizeof(name),c.byref(needed)):
    raise OSError('child_desktop_unavailable')
console=k.GetConsoleWindow()
p=Path(sys.argv[1]); temporary=p.with_suffix('.tmp')
temporary.write_text(json.dumps({'pid':os.getpid(),'desktop':name.value,
    'console':bool(console),'visible':bool(u.IsWindowVisible(console)) if console else False,
    'executable':sys.executable}))
temporary.replace(p)
until=time.monotonic()+10
while not p.with_suffix('.finish').exists():
    if time.monotonic()>=until: raise TimeoutError('observer_absent')
    time.sleep(.01)
"""


def _private_desktop() -> tuple[Any, str, str, Any]:
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.CreateDesktopW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
    ]
    user.CreateDesktopW.restype = wintypes.HANDLE
    user.CloseDesktop.argtypes = [wintypes.HANDLE]
    user.CloseDesktop.restype = wintypes.BOOL
    user.GetProcessWindowStation.restype = wintypes.HANDLE
    user.GetUserObjectInformationW.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    user.GetUserObjectInformationW.restype = wintypes.BOOL
    station, needed = ctypes.create_unicode_buffer(256), wintypes.DWORD()
    if not user.GetUserObjectInformationW(
        user.GetProcessWindowStation(), 2, station, ctypes.sizeof(station), ctypes.byref(needed)
    ):
        raise OSError("test_window_station_unavailable")
    name = "qualification-" + uuid.uuid4().hex
    # No SWITCHDESKTOP access, and this test never calls SwitchDesktop.
    desktop = user.CreateDesktopW(name, None, None, 0, 0xFF, None)
    if not desktop:
        pytest.fail("Private desktop unavailable; windowless chain was not exercised")
    return user, station.value, name, desktop


def _close_owned(kernel: Any, job: Any, root: Any, primary: BaseException | None) -> bool:
    failures: list[BaseException] = []
    try:
        kernel.terminate_job(job)
        until = time.monotonic() + 5
        while kernel.query_job_active_process_count(job):
            if time.monotonic() >= until:
                raise TimeoutError("test_owned_cleanup_unproved")
            time.sleep(0.01)
    except BaseException as error:
        failures.append(error)
    finally:
        handles = (() if root is None else (root.thread_handle, root.process_handle)) + (job,)
        for handle in handles:
            try:
                kernel.close_handle(handle)
            except BaseException as error:
                failures.append(error)
    if failures:
        if primary is not None:
            failures.insert(0, primary)
        raise BaseExceptionGroup("Private test ownership cleanup failed", failures)
    return True


@pytest.mark.parametrize("failure", ["terminate", "query", "close"])
def test_private_probe_closes_every_handle_on_cleanup_failure(failure: str) -> None:
    from types import SimpleNamespace

    class Kernel:
        def __init__(self) -> None:
            self.closed: list[int] = []

        def terminate_job(self, job: int) -> None:
            if failure == "terminate":
                raise OSError("terminate_refused")

        def query_job_active_process_count(self, job: int) -> int:
            if failure == "query":
                raise OSError("query_refused")
            return 0

        def close_handle(self, handle: int) -> None:
            self.closed.append(handle)
            if failure == "close" and handle == 1:
                raise OSError("close_refused")

    kernel, primary = Kernel(), RuntimeError("original_refusal")
    with pytest.raises(BaseExceptionGroup) as observed:
        _close_owned(kernel, 3, SimpleNamespace(thread_handle=1, process_handle=2), primary)
    assert kernel.closed == [1, 2, 3]
    assert observed.value.exceptions[0] is primary


class _DesktopAPI:
    """Only isolate the desktop and set the same redirected stdio as Popen."""

    def __init__(self, api: Any, desktop: str, handles: tuple[int, int, int]) -> None:
        self.api, self.desktop, self.handles = api, desktop, handles

    def __getattr__(self, key: str) -> Any:
        return getattr(self.api, key)

    def CreateProcessW(self, *args: Any) -> Any:
        startup = args[8]._obj.StartupInfo
        startup.lpDesktop = self.desktop
        startup.dwFlags = 0x100  # STARTF_USESTDHANDLES, no window-hiding hint
        startup.hStdInput, startup.hStdOutput, startup.hStdError = self.handles
        return self.api.CreateProcessW(*args)


def _observe_bound_chain(tmp_path: Path) -> None:
    import msvcrt

    from scripts import qualify_evidence_slice_zero as core

    if sys.prefix == sys.base_prefix:
        pytest.skip("Actual venv redirector observation requires a venv interpreter")
    kernel = core._CtypesWindowsKernelV1()
    api = kernel._api()
    user, station, name, desktop = _private_desktop()
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.EnumDesktopWindows.argtypes = [wintypes.HANDLE, callback_type, wintypes.LPARAM]
    user.EnumDesktopWindows.restype = wintypes.BOOL
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    job = root = None
    cleanup = False
    closed = False

    try:
        job = kernel.create_job(core._WindowsJobLimitsV1(6, 1024**3, 2 * 1024**3))
        with open(os.devnull, "r+b") as null:
            os.set_inheritable(null.fileno(), True)
            null_handle = msvcrt.get_osfhandle(null.fileno())
            kernel._kernel32 = _DesktopAPI(
                api, station + "\\" + name, (null_handle, null_handle, null_handle)
            )
            environment = tuple(
                sorted(os.environ.items(), key=lambda item: (item[0].casefold(), item[0]))
            )
            marker = tmp_path / "child.json"
            try:
                root = kernel.create_process_suspended(
                    (sys.executable, "-B", "-c", _CHILD, str(marker)),
                    environment,
                    str(tmp_path),
                    (null_handle,),
                    launch.WINDOWLESS_BREAKAWAY_FLAGS | 0x4 | 0x400 | 0x80000,
                    job=job,
                )
            except OSError as error:
                if getattr(error, "winerror", None) == 5:
                    pytest.skip("Explicit breakaway refused; windowless chain was not exercised")
                raise
            kernel.resume_thread(root.thread_handle)
            deadline = time.monotonic() + 10
            while not marker.exists():
                if time.monotonic() >= deadline:
                    pytest.fail("Harmless venv child did not report; observation incomplete")
                time.sleep(0.01)
            child = json.loads(marker.read_bytes())
            assert child["desktop"] == name, "test_desktop_isolation_failed"
            assert Path(child["executable"]).resolve() == Path(sys.executable).resolve()
            assert child["pid"] != root.pid, "venv_redirector_child_not_observed"
            visible = []

            @callback_type
            def observe(hwnd: int, unused: int) -> bool:
                visible.append(bool(user.IsWindowVisible(hwnd)))
                return True

            ctypes.set_last_error(0)
            enumerated = user.EnumDesktopWindows(desktop, observe, 0)
            if not enumerated and ctypes.get_last_error() != 0:
                raise OSError("test_private_window_observation_failed")
            assert child["console"] is False, "windowless_venv_child_has_console"
            assert child["visible"] is False and not any(visible), "private_window_visible"
            marker.with_suffix(".finish").touch()
            kernel.wait(root.process_handle, 5000)
    finally:
        try:
            if job is not None:
                cleanup = _close_owned(kernel, job, root, sys.exception())
        finally:
            closed = bool(user.CloseDesktop(desktop))
            print(
                "[qualification-windowless-test] "
                + json.dumps({"owned_job_empty": cleanup, "private_desktop_closed": closed})
            )
        assert cleanup and closed, "test_cleanup_incomplete"


@pytest.mark.skipif(os.name != "nt", reason="native Windows private-desktop observation")
def test_bound_venv_caller_chain_has_no_console_on_private_desktop(tmp_path: Path) -> None:
    import msvcrt

    from scripts import qualify_evidence_slice_zero as core

    # Give the real venv chain a console to inherit, wholly off the input desktop.
    kernel = core._CtypesWindowsKernelV1()
    api = kernel._api()
    user, station, name, desktop = _private_desktop()
    job = root = None
    cleanup = closed = False
    program = (
        "import ctypes as c,runpy,sys\nfrom pathlib import Path\nimport pytest\n"
        "k=c.WinDLL('kernel32'); k.GetConsoleWindow.restype=c.c_void_p\n"
        "assert k.GetConsoleWindow(), 'private_parent_console_absent'\n"
        "try:\n"
        f" runpy.run_path({str(Path(__file__).resolve())!r})['_observe_bound_chain']("
        f"Path({str(tmp_path)!r}))\n"
        "except pytest.skip.Exception:\n print('private_parent_observation_unexercised')\n"
    )
    try:
        job = kernel.create_job(core._WindowsJobLimitsV1(8, 1024**3, 2 * 1024**3))
        info = core._JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1()
        assert api.QueryInformationJobObject(
            ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info), None
        )
        # Only this owned test Job permits the inner observation to break away
        # into its own atomic, noninherited kill-on-close Job.
        info.BasicLimitInformation.LimitFlags |= 0x800
        assert api.SetInformationJobObject(
            ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info)
        )
        with open(os.devnull, "r+b") as null, (tmp_path / "parent-output").open("x+b") as out:
            for stream in (null, out):
                os.set_inheritable(stream.fileno(), True)
            null_handle, output_handle = (
                msvcrt.get_osfhandle(stream.fileno()) for stream in (null, out)
            )
            kernel._kernel32 = _DesktopAPI(
                api, station + "\\" + name, (null_handle, output_handle, output_handle)
            )
            environment = tuple(
                sorted(os.environ.items(), key=lambda item: (item[0].casefold(), item[0]))
            )
            root = kernel.create_process_suspended(
                (sys.executable, "-B", "-c", program),
                environment,
                str(Path(__file__).resolve().parents[1]),
                (null_handle, output_handle),
                0x4 | 0x10 | 0x400 | 0x80000,  # suspended, private NEW_CONSOLE, Unicode, JOB_LIST
                job=job,
            )
            kernel.resume_thread(root.thread_handle)
            kernel.wait(root.process_handle, 30000)
            api.GetExitCodeProcess.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint32),
            ]
            api.GetExitCodeProcess.restype = ctypes.c_int
            exit_code = ctypes.c_uint32()
            if not api.GetExitCodeProcess(
                ctypes.c_void_p(root.process_handle), ctypes.byref(exit_code)
            ):
                pytest.fail("Private console-parent exit evidence unavailable")
            out.seek(0)
            retained = out.read(65537)
            assert len(retained) <= 65536, "private_parent_output_bound"
            if b"private_parent_observation_unexercised" in retained:
                pytest.skip("Private redirector observation refused; not native coverage")
            if b'"owned_job_empty": true, "private_desktop_closed": true' not in retained:
                pytest.fail("Private child observation lacks cleanup evidence")
            if exit_code.value != 0:
                if b"windowless_venv_child_has_console" in retained:
                    pytest.fail("windowless_venv_child_has_console")
                pytest.fail("Private console-parent observation failed; output retained locally")
    finally:
        try:
            if job is not None:
                cleanup = _close_owned(kernel, job, root, sys.exception())
        finally:
            closed = bool(user.CloseDesktop(desktop))
            print(
                "[qualification-windowless-parent-test] "
                + json.dumps({"owned_job_empty": cleanup, "private_desktop_closed": closed})
            )
        assert cleanup and closed, "test_parent_cleanup_incomplete"

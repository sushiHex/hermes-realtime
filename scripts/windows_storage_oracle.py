"""Read bounded fixture security through Windows handles, without candidate code.

API authorities: Microsoft's GetSecurityInfo, GetFileInformationByHandle,
GetFinalPathNameByHandleW, and FindFirstStreamW contracts. Only ordinary allow
ACEs for the current user, SYSTEM, Administrators and owner identities are accepted. This narrow
fixture profile does not implement a general Windows access-check evaluator.
"""

from __future__ import annotations

import ctypes as c
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import cache
from itertools import islice
from pathlib import Path
from typing import Any

from scripts.equivalence_process import _require


class _FileInfo(c.Structure):
    _fields_ = (
        ("attributes", c.c_uint32),
        ("times", c.c_uint32 * 6),
        ("volume", c.c_uint32),
        ("size_high", c.c_uint32),
        ("size_low", c.c_uint32),
        ("links", c.c_uint32),
        ("index_high", c.c_uint32),
        ("index_low", c.c_uint32),
    )


class _Stream(c.Structure):
    _fields_ = (("size", c.c_int64), ("name", c.c_wchar * 296))


@cache
def _api() -> tuple[Any, Any]:
    _require(
        os.name == "nt" and c.sizeof(c.c_void_p) == 8, "storage oracle requires 64-bit Windows"
    )
    kernel, security = (
        c.WinDLL("kernel32", use_last_error=True),
        c.WinDLL("advapi32", use_last_error=True),
    )
    pointer, word, text = c.c_void_p, c.c_uint32, c.c_wchar_p
    signatures = (
        (kernel, "CreateFileW", pointer, [text, word, word, pointer, word, word, pointer]),
        (kernel, "CloseHandle", c.c_int, [pointer]),
        (kernel, "LocalFree", pointer, [pointer]),
        (kernel, "GetFileInformationByHandle", c.c_int, [pointer, pointer]),
        (kernel, "GetFinalPathNameByHandleW", word, [pointer, text, word, word]),
        (kernel, "GetVolumePathNameW", c.c_int, [text, text, word]),
        (kernel, "GetDriveTypeW", word, [text]),
        (kernel, "FindFirstStreamW", pointer, [text, c.c_int, pointer, word]),
        (kernel, "FindNextStreamW", c.c_int, [pointer, pointer]),
        (kernel, "FindClose", c.c_int, [pointer]),
        (kernel, "WaitForSingleObject", word, [pointer, word]),
        (
            security,
            "GetSecurityInfo",
            word,
            [pointer, c.c_int, word, pointer, pointer, pointer, pointer, pointer],
        ),
        (security, "GetAclInformation", c.c_int, [pointer, pointer, word, c.c_int]),
        (security, "GetAce", c.c_int, [pointer, word, pointer]),
        (security, "ConvertSidToStringSidW", c.c_int, [pointer, pointer]),
        (security, "OpenProcessToken", c.c_int, [pointer, word, pointer]),
        (security, "GetTokenInformation", c.c_int, [pointer, c.c_int, pointer, word, pointer]),
    )
    for library, name, result, arguments in signatures:
        function = getattr(library, name)
        function.restype, function.argtypes = result, arguments
    return kernel, security


def _sid_text(sid: Any) -> str:
    kernel, security = _api()
    value = c.c_wchar_p()
    _require(
        bool(security.ConvertSidToStringSidW(sid, c.byref(value))), "storage SID is unreadable"
    )
    try:
        result = value.value
        _require(type(result) is str and 0 < len(result) < 256, "storage SID exceeds its bound")
        return str(result)
    finally:
        _require(kernel.LocalFree(value) is None, "storage SID buffer did not close")


@cache
def _user_sid() -> str:
    kernel, security = _api()
    token = c.c_void_p()
    _require(
        bool(security.OpenProcessToken(c.c_void_p(-1), 8, c.byref(token))),
        "storage token is unavailable",
    )
    try:
        size = c.c_uint32()
        security.GetTokenInformation(token, 1, None, 0, c.byref(size))
        _require(0 < size.value <= 65536, "storage token exceeds its bound")
        buffer = c.create_string_buffer(size.value)
        _require(
            bool(security.GetTokenInformation(token, 1, buffer, size, c.byref(size))),
            "storage token is unreadable",
        )
        return _sid_text(c.c_void_p.from_buffer(buffer))
    finally:
        _require(bool(kernel.CloseHandle(token)), "storage token did not close")


def _permissions(handle: int) -> None:
    kernel, security = _api()
    owner, dacl, descriptor = c.c_void_p(), c.c_void_p(), c.c_void_p()
    _require(
        security.GetSecurityInfo(
            handle, 1, 5, c.byref(owner), None, c.byref(dacl), None, c.byref(descriptor)
        )
        == 0,
        "storage security descriptor is unreadable",
    )
    try:
        _require(bool(owner.value) and bool(dacl.value), "storage owner or DACL is absent")
        permitted = {_user_sid(), "S-1-5-18", "S-1-5-32-544"}
        _require(_sid_text(owner) in permitted, "storage owner is outside the fixture authority")
        info = (c.c_uint32 * 3)()
        _require(
            bool(security.GetAclInformation(dacl, info, c.sizeof(info), 2)),
            "storage ACL is unreadable",
        )
        _require(0 < info[0] <= 64 and 8 <= info[1] <= 65536, "storage ACL exceeds its bound")
        for index in range(info[0]):
            ace = c.c_void_p()
            _require(bool(security.GetAce(dacl, index, c.byref(ace))), "storage ACE is unreadable")
            address, start = int(ace.value or 0), int(dacl.value or 0)
            _require(start + 8 <= address <= start + info[1] - 4, "storage ACE escaped its ACL")
            header = c.string_at(address, 4)
            length = int.from_bytes(header[2:4], "little")
            _require(
                header[0] == 0 and 16 <= length <= start + info[1] - address,
                "storage ACE is outside the bounded allow profile",
            )
            sid_header = c.string_at(address + 8, 8)
            _require(
                sid_header[0] == 1 and sid_header[1] <= 15 and 16 + 4 * sid_header[1] <= length,
                "storage ACE SID exceeds its bound",
            )
            sid = _sid_text(address + 8)
            _require(
                sid in permitted
                or sid == "S-1-3-4"  # OWNER RIGHTS means the already-checked object owner.
                or (sid == "S-1-3-0" and bool(header[1] & 8)),  # Inherit-only CREATOR OWNER.
                "storage DACL grants another principal",
            )
    finally:
        _require(kernel.LocalFree(descriptor) is None, "storage security descriptor did not close")


def _streams(path: Path, *, directory: bool) -> None:
    kernel, _ = _api()
    data = _Stream()
    c.set_last_error(0)
    search = kernel.FindFirstStreamW(str(path), 0, c.byref(data), 0)
    if search in (None, c.c_void_p(-1).value):
        _require(directory and c.get_last_error() == 38, "storage stream inventory is unavailable")
        return
    try:
        _require(not directory and data.name == "::$DATA", "storage has an alternate stream")
        c.set_last_error(0)
        found = kernel.FindNextStreamW(search, c.byref(data))
        _require(not found and c.get_last_error() == 38, "storage has extra or unreadable streams")
    finally:
        _require(bool(kernel.FindClose(search)), "storage stream search did not close")


@contextmanager
def _retain(path: Path, *, directory: bool, exclusive: bool = False) -> Iterator[_FileInfo]:
    kernel, _ = _api()
    access = 0x80020080 if exclusive else 0x20080
    handle = kernel.CreateFileW(str(path), access, 0 if exclusive else 3, None, 3, 0x02200000, None)
    _require(
        handle not in (None, c.c_void_p(-1).value),
        "storage ownership was not released" if exclusive else "storage path could not be retained",
    )
    try:
        info = _FileInfo()
        _require(
            bool(kernel.GetFileInformationByHandle(handle, c.byref(info))),
            "storage file identity is unreadable",
        )
        _require(
            not info.attributes & 0x400 and bool(info.attributes & 0x10) == directory,
            "storage path is redirected or has the wrong kind",
        )
        if not directory:
            _require(
                info.links == 1 and info.size_high == 0 and info.size_low <= 4 * 1024**2,
                "storage file escapes its link or size bound",
            )
        final = c.create_unicode_buffer(32768)
        length = kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        _require(0 < length < len(final), "storage final path exceeds its bound")
        _require(
            final.value.removeprefix("\\\\?\\").casefold() == str(path.absolute()).casefold(),
            "storage path traverses redirected ancestry",
        )
        _permissions(handle)
        _streams(path, directory=directory)
        yield info
    finally:
        _require(bool(kernel.CloseHandle(handle)), "storage retained file did not close")


def audit_storage_release(root: Path, process_handle: int, names: tuple[str, ...]) -> None:
    """Prove release before process exit can supply it; never mutate the store."""
    kernel, _ = _api()
    _require(
        type(names) is tuple
        and 1 <= len(names) <= 13
        and all(
            type(name) is str and Path(name).name == name and name not in {".", ".."}
            for name in names
        )
        and len(set(names)) == len(names),
        "storage release inventory differs",
    )

    def require_alive() -> None:
        _require(
            kernel.WaitForSingleObject(process_handle, 0) == 258,  # WAIT_TIMEOUT
            "storage worker exited before live ownership observation",
        )

    require_alive()
    with ExitStack() as owned:
        parent = owned.enter_context(_retain(root, directory=True, exclusive=True))
        for name in names:
            child = owned.enter_context(_retain(root / name, directory=False, exclusive=True))
            _require(child.volume == parent.volume, "released storage changed volume")
        # Share mode zero excludes every existing read/write/delete handle,
        # including SQLite's connection and the sentinel's byte-range lease.
        # The retained worker is still blocked on its private exit handshake.
        require_alive()


def audit_drain_release(root: Path, process_handle: int) -> None:
    audit_storage_release(root, process_handle, ("capture-v1.owner", "capture-v1.sqlite3"))


@contextmanager
def audit_storage(root: Path, allowed: frozenset[str]) -> Iterator[tuple[Path, ...]]:
    """Hold the root and exact leaves while the independent observer reads them."""
    kernel, _ = _api()
    volume = c.create_unicode_buffer(32768)
    _require(
        bool(kernel.GetVolumePathNameW(str(root), volume, len(volume)))
        and kernel.GetDriveTypeW(volume) == 3,
        "storage volume is not fixed local",
    )
    with ExitStack() as owned:
        parent = owned.enter_context(_retain(root, directory=True))
        with os.scandir(root) as entries:
            paths = tuple(root / entry.name for entry in islice(entries, len(allowed) + 1))
        _require(
            len(paths) <= len(allowed) and all(p.name in allowed for p in paths),
            "storage inventory differs",
        )
        for path in paths:
            child = owned.enter_context(_retain(path, directory=False))
            _require(child.volume == parent.volume, "storage file changed volume")
        yield paths

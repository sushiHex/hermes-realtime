"""Retain immutable Windows file bytes and their ancestry for a bounded consumer.

This protects the selected file objects, not the directory namespace. Installed
execution additionally requires OS-enforced read-only import/resource locations.
The trusted owner must retain this context until every consumer has stopped.
Only a local fixed NTFS volume supplies this version's file-identity authority.
"""

from __future__ import annotations

import ctypes as c
import hashlib
import os
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

from scripts.qualify_hermes_v020_pluginmanager import _windows_archive_component_key
from scripts.windows_storage_oracle import _api, _FileInfo, _identity, _streams

_MAX_FILES = 16384
_MAX_FILE_BYTES = 16 * 1024**3
_MAX_TOTAL_BYTES = 64 * 1024**3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative_windows_member(value: str) -> str:
    """Ordinary Windows names for tool trees; v1 artifact paths remain stricter."""
    _require(
        type(value) is str and bool(value) and not value.startswith("/") and "\\" not in value,
        "file member is not a relative POSIX path",
    )
    for component in value.split("/"):
        _windows_archive_component_key(component)
    return value


class RetainedFileSealsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("file seals are created by their retained owner only")


@dataclass(frozen=True, slots=True)
class _File:
    handle: int
    identity: tuple[int, int]
    digest: str
    size: int


@dataclass(frozen=True, slots=True)
class _Seals:
    files: dict[str, _File]
    lock: threading.RLock


_LIVE: WeakKeyDictionary[RetainedFileSealsV1, _Seals] = WeakKeyDictionary()


def _active(receipt: RetainedFileSealsV1) -> _Seals:
    if type(receipt) is not RetainedFileSealsV1:
        raise TypeError("file seal capability type differs")
    _require(receipt in _LIVE, "file seal capability is closed or unregistered")
    return _LIVE[receipt]


def _info(handle: int) -> _FileInfo:
    kernel, _ = _api()
    info = _FileInfo()
    _require(
        bool(kernel.GetFileInformationByHandle(handle, c.byref(info))),
        "sealed file identity is unreadable",
    )
    return info


@contextmanager
def _retain(path: Path, *, directory: bool) -> Iterator[tuple[int, _FileInfo]]:
    kernel, _ = _api()
    handle = kernel.CreateFileW(
        str(path),
        0x80 if directory else 0x80020080,
        3 if directory else 1,
        None,
        3,
        0x02200000,
        None,
    )
    _require(handle not in (None, c.c_void_p(-1).value), "file seal could not be retained")
    try:
        info = _info(handle)
        _require(
            not info.attributes & 0x400 and bool(info.attributes & 0x10) == directory,
            "file seal is redirected or has the wrong kind",
        )
        if not directory:
            _require(info.links == 1, "file seal rejects a hard link")
            _require(
                0 <= (info.size_high << 32) | info.size_low <= _MAX_FILE_BYTES,
                "file seal exceeds its size bound",
            )
        final = c.create_unicode_buffer(32768)
        length = kernel.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        _require(0 < length < len(final), "file seal path exceeds its bound")
        _require(
            final.value.removeprefix("\\\\?\\").casefold() == str(path).casefold(),
            "file seal ancestry is redirected",
        )
        _streams(path, directory=directory)
        yield handle, info
        current = _info(handle)
        _require(
            _identity(current) == _identity(info)
            and current.attributes == info.attributes
            and current.links == info.links
            and (current.size_high, current.size_low) == (info.size_high, info.size_low),
            "retained file identity changed",
        )
    finally:
        _require(bool(kernel.CloseHandle(handle)), "file seal handle did not close")


def _chunks(handle: int, size: int) -> Iterator[bytes]:
    kernel, _ = _api()
    kernel.SetFilePointerEx.argtypes = [c.c_void_p, c.c_int64, c.c_void_p, c.c_uint32]
    kernel.SetFilePointerEx.restype = c.c_int
    kernel.ReadFile.argtypes = [c.c_void_p, c.c_void_p, c.c_uint32, c.c_void_p, c.c_void_p]
    kernel.ReadFile.restype = c.c_int
    _require(bool(kernel.SetFilePointerEx(handle, 0, None, 0)), "sealed file seek failed")
    buffer, read = c.create_string_buffer(1024**2), c.c_uint32()
    remaining = size
    while remaining:
        count = min(remaining, len(buffer))
        _require(
            bool(kernel.ReadFile(handle, buffer, count, c.byref(read), None))
            and 0 < read.value <= count,
            "sealed file read failed",
        )
        remaining -= read.value
        yield buffer.raw[: read.value]


@contextmanager
def retain_file_seals(root: Path, members: tuple[str, ...]) -> Iterator[RetainedFileSealsV1]:
    _require(os.name == "nt" and c.sizeof(c.c_void_p) == 8, "file seals require 64-bit Windows")
    _require(type(root) is type(Path()) and root.is_absolute(), "file seal root must be absolute")
    _require(
        type(members) is tuple
        and 0 < len(members) <= _MAX_FILES
        and all(type(name) is str for name in members)
        and tuple(sorted(members)) == members
        and len({name.casefold() for name in members}) == len(members),
        "file seal member set is not bounded, sorted and unique",
    )
    for name in members:
        _relative_windows_member(name)
    lexical = root.absolute()
    _require(
        ".." not in lexical.parts and not str(lexical).startswith("\\\\"),
        "file seal root is indirect",
    )
    parents = tuple(reversed(lexical.parents)) + (lexical,)
    kernel, _ = _api()
    volume = c.create_unicode_buffer(32768)
    _require(
        bool(kernel.GetVolumePathNameW(str(lexical), volume, len(volume)))
        and kernel.GetDriveTypeW(volume.value) == 3,
        "file seals require a local fixed volume",
    )
    files: dict[str, _File] = {}
    identities: set[tuple[int, int]] = set()
    with ExitStack() as owned:
        retained: set[Path] = set()
        for parent in parents:
            handle, _ = owned.enter_context(_retain(parent, directory=True))
            if parent == lexical:
                filesystem = c.create_unicode_buffer(32)
                _require(
                    bool(
                        kernel.GetVolumeInformationByHandleW(
                            handle,
                            None,
                            0,
                            None,
                            None,
                            None,
                            filesystem,
                            len(filesystem),
                        )
                    )
                    and filesystem.value == "NTFS",
                    "file seals require independently observed NTFS",
                )
            retained.add(parent)
        total = 0
        for name in members:
            path = lexical.joinpath(*name.split("/"))
            for parent in reversed(path.parents):
                if parent not in retained:
                    owned.enter_context(_retain(parent, directory=True))
                    retained.add(parent)
            handle, info = owned.enter_context(_retain(path, directory=False))
            identity = _identity(info)
            _require(identity not in identities, "file seal roles alias one identity")
            identities.add(identity)
            size = (info.size_high << 32) | info.size_low
            total += size
            _require(total <= _MAX_TOTAL_BYTES, "file seal aggregate exceeds its bound")
            digest = hashlib.sha256()
            for chunk in _chunks(handle, size):
                digest.update(chunk)
            files[name] = _File(handle, identity, digest.hexdigest(), size)
        receipt = object.__new__(RetainedFileSealsV1)
        _LIVE[receipt] = _Seals(files, threading.RLock())
        try:
            yield receipt
        finally:
            state = _LIVE[receipt]
            with state.lock:
                del _LIVE[receipt]


def sealed_file_metadata(receipt: RetainedFileSealsV1) -> tuple[tuple[str, str, int], ...]:
    """Private relative-member inventory; not a public qualification summary."""
    state = _active(receipt)
    with state.lock:
        _active(receipt)
        return tuple((name, item.digest, item.size) for name, item in state.files.items())


def sealed_file_bytes(receipt: RetainedFileSealsV1, member: str, maximum: int) -> bytes:
    state = _active(receipt)
    with state.lock:
        _active(receipt)
        _require(type(member) is str and member in state.files, "file seal member is absent")
        item = state.files[member]
        _require(
            type(maximum) is int and 0 <= item.size <= maximum <= _MAX_FILE_BYTES,
            "file read bound differs",
        )
        raw = b"".join(_chunks(item.handle, item.size))
        _require(hashlib.sha256(raw).hexdigest() == item.digest, "sealed file digest changed")
        return raw

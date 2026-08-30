#!/usr/bin/env python3
"""Private generic source-archive validation and retained Windows extraction."""

from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import stat
import tarfile
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceArchivePolicyV1:
    """Closed caller-supplied archive and owned-workspace naming limits."""

    prefix: str
    workspace_child: str
    owner_marker: str
    max_members: int
    max_file_bytes: int
    max_tree_bytes: int
    error_label: str

    def __post_init__(self) -> None:
        for text_label, text_value in (
            ("prefix", self.prefix),
            ("workspace_child", self.workspace_child),
            ("owner_marker", self.owner_marker),
        ):
            if type(text_value) is not str or text_value in {"", ".", ".."} or any(
                character in text_value for character in ("/", "\\", "\x00")
            ):
                raise ValueError(f"source archive policy {text_label} is invalid")
            _windows_component_key(text_value)
        for limit_label, limit_value in (
            ("max_members", self.max_members),
            ("max_file_bytes", self.max_file_bytes),
            ("max_tree_bytes", self.max_tree_bytes),
        ):
            if type(limit_value) is not int or limit_value <= 0:
                raise ValueError(
                    f"source archive policy {limit_label} must be a positive exact integer"
                )
        if self.max_tree_bytes < self.max_file_bytes:
            raise ValueError("source archive policy tree limit cannot be smaller than file limit")
        if type(self.error_label) is not str or not self.error_label.strip():
            raise ValueError("source archive policy error_label is invalid")


_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


def _exact_int(value: object, *, label: str = "native identity") -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _normal_final_path(value: str) -> str:
    value = (
        value.removeprefix("\\\\?\\UNC\\")
        if value.startswith("\\\\?\\UNC\\")
        else value.removeprefix("\\\\?\\")
    )
    if value.startswith("UNC\\"):
        value = "\\\\" + value[4:]
    return os.path.normcase(os.path.normpath(value))


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("attributes", wintypes.DWORD), ("creation_low", wintypes.DWORD),
        ("creation_high", wintypes.DWORD), ("access_low", wintypes.DWORD),
        ("access_high", wintypes.DWORD), ("write_low", wintypes.DWORD),
        ("write_high", wintypes.DWORD), ("volume_serial", wintypes.DWORD),
        ("size_high", wintypes.DWORD), ("size_low", wintypes.DWORD),
        ("links", wintypes.DWORD), ("file_index_high", wintypes.DWORD),
        ("file_index_low", wintypes.DWORD),
    ]


class _UNICODE_STRING(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.USHORT),
        ("maximum_length", wintypes.USHORT),
        ("buffer", wintypes.LPWSTR),
    ]


class _OBJECT_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.ULONG), ("root_directory", wintypes.HANDLE),
        ("object_name", ctypes.POINTER(_UNICODE_STRING)), ("attributes", wintypes.ULONG),
        ("security_descriptor", wintypes.LPVOID), ("security_quality_of_service", wintypes.LPVOID),
    ]


class _IO_STATUS_BLOCK(ctypes.Structure):
    _fields_ = [("status", ctypes.c_long), ("information", ctypes.c_size_t)]


class _FILE_DISPOSITION_INFO(ctypes.Structure):
    _fields_ = [("delete_file", wintypes.BOOL)]


def _native_handle_info(handle: int) -> tuple[int, int, str, int, int]:
    if os.name != "nt":
        raise RuntimeError("native handle identity is Windows-only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    info = _BY_HANDLE_FILE_INFORMATION()
    if not kernel.GetFileInformationByHandle(wintypes.HANDLE(handle), ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    needed = kernel.GetFinalPathNameByHandleW(wintypes.HANDLE(handle), None, 0, 0)
    if not needed:
        raise ctypes.WinError(ctypes.get_last_error())
    target = ctypes.create_unicode_buffer(needed + 1)
    written = kernel.GetFinalPathNameByHandleW(wintypes.HANDLE(handle), target, len(target), 0)
    if not written or written >= len(target):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(info.volume_serial), (int(info.file_index_high) << 32) | int(info.file_index_low),
        _normal_final_path(target.value), int(info.attributes),
        (int(info.size_high) << 32) | int(info.size_low),
    )


def _windows_component_key(component: str) -> str:
    if (
        not component
        or component[-1] in {" ", "."}
        or any(ord(character) < 32 or character in '<>:"|?*' for character in component)
        or component.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_STEMS
    ):
        raise ValueError("archive member has a Win32-unsafe component")
    return component.casefold()


def validated_archive_members(
    archive: tarfile.TarFile, policy: SourceArchivePolicyV1
) -> tuple[tarfile.TarInfo, ...]:
    """Return safe ordinary members under one policy-bound archive root."""
    prefix = policy.prefix + "/"
    seen: set[str] = set()
    regular: list[tarfile.TarInfo] = []
    files: set[str] = set()
    directories: set[str] = {_windows_component_key(policy.prefix)}
    root_seen = False
    members = archive.getmembers()
    if not 1 <= len(members) <= policy.max_members:
        raise ValueError(f"{policy.error_label} has an invalid member count")
    declared_bytes = 0
    for info in members:
        name = info.name
        if not name or "\\" in name or name.startswith("/") or ":" in name:
            raise ValueError(f"{policy.error_label} has an unsafe member name")
        is_directory = info.type == tarfile.DIRTYPE
        is_regular = info.type in {tarfile.REGTYPE, tarfile.AREGTYPE}
        if not (is_directory or is_regular) or info.issparse():
            raise ValueError(
                f"{policy.error_label} contains a non-ordinary regular or directory member"
            )
        if is_directory and info.size != 0:
            raise ValueError(f"{policy.error_label} member type is inconsistent")
        normalized = name
        parts = normalized.split("/")
        if not normalized or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"{policy.error_label} has an unsafe member name")
        try:
            component_keys = tuple(_windows_component_key(part) for part in parts)
        except ValueError:
            raise ValueError(
                f"{policy.error_label} has a Win32-unsafe member component"
            ) from None
        if normalized != policy.prefix and not normalized.startswith(prefix):
            raise ValueError(f"{policy.error_label} has an unexpected prefix")
        if type(info.size) is not int or info.size < 0:
            raise ValueError(f"{policy.error_label} member size is invalid")
        if is_regular:
            declared_bytes += info.size
            if info.size > policy.max_file_bytes or declared_bytes > policy.max_tree_bytes:
                raise ValueError(f"{policy.error_label} is oversized")
        folded = "/".join(component_keys)
        if folded in seen:
            raise ValueError(f"{policy.error_label} has a duplicate or case collision")
        seen.add(folded)
        if normalized == policy.prefix:
            if not is_directory:
                raise ValueError(f"{policy.error_label} root is not a directory")
            root_seen = True
            continue
        for index in range(1, len(component_keys)):
            directories.add("/".join(component_keys[:index]))
        if is_directory:
            directories.add(folded)
        else:
            files.add(folded)
            regular.append(info)
    if not root_seen or not regular or files & directories:
        raise ValueError(f"{policy.error_label} does not contain one safe source tree")
    return tuple(regular)


class WindowsSourceWorkspaceAuthorityV1:
    """One private, handle-bound source tree; it is never pathname-published."""

    _GENERIC_READ = 0x80000000
    _GENERIC_WRITE = 0x40000000
    _DELETE = 0x00010000
    _FILE_LIST_DIRECTORY = 0x0001
    _FILE_ADD_FILE = 0x0002
    _FILE_ADD_SUBDIRECTORY = 0x0004
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _FILE_SHARE_READ = 0x00000001
    _FILE_CREATE = 2
    _NT_FILE_OPEN = 1
    _OPEN_EXISTING = 3
    _FILE_DIRECTORY_FILE = 0x00000001
    _FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
    _SYNCHRONIZE = 0x00100000
    _OBJ_CASE_INSENSITIVE = 0x40
    _FILE_DISPOSITION_INFO = 4
    _INVALID = ctypes.c_void_p(-1).value if os.name == "nt" else -1

    def __init__(
        self,
        workspace: Path,
        policy: SourceArchivePolicyV1,
        watcher_factory: Callable[[Any, int], Any],
    ) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("native source extraction requires 64-bit Windows")
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self._bind()
        self._closed = False
        self._watch_stopped = False
        self._handles: dict[str, int] = {}
        self._seals: dict[str, tuple[object, ...]] = {}
        self._workspace = workspace
        self._policy = policy
        self._watcher_factory = watcher_factory
        self._workspace_handle = self._open_workspace(workspace)
        self._root_name = f"{policy.workspace_child}.{secrets.token_hex(16)}.tmp"
        try:
            self._handles["."] = self._create(
                self._workspace_handle,
                self._root_name,
                directory=True,
                write=False,
                asynchronous=True,
            )
        except BaseException:
            self._close_handle(self._workspace_handle)
            self._workspace_handle = 0
            raise
        self.root = workspace / self._root_name / policy.prefix
        self.inventory: dict[str, tuple[object, ...]] = {}
        self._watcher = self._new_watcher()

    def _bind(self) -> None:
        self._kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._kernel.CreateFileW.restype = wintypes.HANDLE
        self._kernel.WriteFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._kernel.WriteFile.restype = wintypes.BOOL
        self._kernel.FlushFileBuffers.argtypes = [wintypes.HANDLE]
        self._kernel.FlushFileBuffers.restype = wintypes.BOOL
        self._kernel.SetFilePointerEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.DWORD,
        ]
        self._kernel.SetFilePointerEx.restype = wintypes.BOOL
        self._kernel.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        self._kernel.ReadFile.restype = wintypes.BOOL
        self._kernel.SetFileInformationByHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        self._kernel.SetFileInformationByHandle.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._ntdll.NtCreateFile.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            ctypes.POINTER(_OBJECT_ATTRIBUTES),
            ctypes.POINTER(_IO_STATUS_BLOCK),
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.ULONG,
        ]
        self._ntdll.NtCreateFile.restype = ctypes.c_long

    @staticmethod
    def _native_name(name: str) -> tuple[ctypes.Array[ctypes.c_wchar], _UNICODE_STRING]:
        if type(name) is not str or not name or "\\" in name or "/" in name or "\x00" in name:
            raise ValueError("owned native child name is invalid")
        buffer = ctypes.create_unicode_buffer(name)
        size = len(name.encode("utf-16-le"))
        return buffer, _UNICODE_STRING(size, size, ctypes.cast(buffer, wintypes.LPWSTR))

    def _open_workspace(self, workspace: Path) -> int:
        raw = self._kernel.CreateFileW(
            str(workspace),
            self._GENERIC_READ,
            self._FILE_SHARE_READ | 0x00000006,
            None,
            self._OPEN_EXISTING,
            0x02000000 | 0x00200000,
            None,
        )
        handle = int(ctypes.cast(raw, ctypes.c_void_p).value or 0) if raw else 0
        if not handle or handle == self._INVALID:
            raise OSError(ctypes.get_last_error(), "cannot retain source workspace")
        try:
            _volume, _file_id, final, attributes, _size = _native_handle_info(handle)
            if (
                attributes & 0x400
                or not attributes & 0x10
                or final != _normal_final_path(str(workspace.resolve(strict=True)))
            ):
                raise RuntimeError("source workspace retained identity is invalid")
            return handle
        except BaseException:
            self._close_handle(handle)
            raise

    def _create(
        self, parent: int, name: str, *, directory: bool, write: bool, asynchronous: bool = False
    ) -> int:
        buffer, unicode = self._native_name(name)
        attributes = _OBJECT_ATTRIBUTES(
            ctypes.sizeof(_OBJECT_ATTRIBUTES),
            wintypes.HANDLE(parent),
            ctypes.pointer(unicode),
            self._OBJ_CASE_INSENSITIVE,
            None,
            None,
        )
        iosb = _IO_STATUS_BLOCK()
        result = wintypes.HANDLE()
        access = self._GENERIC_READ | self._DELETE | (0 if asynchronous else self._SYNCHRONIZE)
        if directory:
            access |= self._FILE_ADD_FILE | self._FILE_ADD_SUBDIRECTORY
        elif write:
            access |= self._GENERIC_WRITE
        status = int(
            self._ntdll.NtCreateFile(
                ctypes.byref(result),
                access,
                ctypes.byref(attributes),
                ctypes.byref(iosb),
                None,
                self._FILE_ATTRIBUTE_NORMAL,
                self._FILE_SHARE_READ,
                self._FILE_CREATE,
                (self._FILE_DIRECTORY_FILE if directory else 0)
                | (0 if asynchronous else self._FILE_SYNCHRONOUS_IO_NONALERT),
                None,
                0,
            )
        )
        del buffer
        handle = int(ctypes.cast(result, ctypes.c_void_p).value or 0)
        if status != 0 or not handle or handle == self._INVALID:
            raise OSError(status, "NtCreateFile could not atomically create owned source entry")
        return handle

    def _new_watcher(self) -> Any:
        return self._watcher_factory(self._kernel, self._handles["."])

    def _expected_path(self, relative: str) -> str:
        path = self._workspace / self._root_name
        if relative not in {"", "."}:
            path = path.joinpath(*relative.rstrip("/").split("/"))
        return _normal_final_path(str(path.resolve(strict=False)))

    def _read_digest(self, handle: int, size: int) -> str:
        position = ctypes.c_longlong()
        if not self._kernel.SetFilePointerEx(
            handle, ctypes.c_longlong(0), ctypes.byref(position), 0
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        digest = hashlib.sha256()
        read = 0
        buffer = ctypes.create_string_buffer(1024 * 1024)
        while read < size:
            count = wintypes.DWORD()
            if not self._kernel.ReadFile(
                handle, buffer, min(len(buffer), size - read), ctypes.byref(count), None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if count.value == 0:
                raise RuntimeError("owned source file ended before its sealed size")
            read += int(count.value)
            digest.update(buffer.raw[: count.value])
        count = wintypes.DWORD()
        if not self._kernel.ReadFile(handle, buffer, 1, ctypes.byref(count), None):
            raise ctypes.WinError(ctypes.get_last_error())
        if count.value:
            raise RuntimeError("owned source file exceeded its sealed size")
        return digest.hexdigest()

    def _seal(
        self, relative: str, handle: int, *, file_payload: bytes | None = None
    ) -> tuple[object, ...]:
        volume, file_id, final, attributes, size = _native_handle_info(handle)
        if final != self._expected_path(relative) or attributes & 0x400:
            raise RuntimeError("owned source native identity or type diverged")
        if file_payload is None:
            if not attributes & 0x10:
                raise RuntimeError("owned source directory native type diverged")
            return ("dir", volume, file_id, final, attributes)
        digest = hashlib.sha256(file_payload).hexdigest()
        if (
            attributes & 0x10
            or size != len(file_payload)
            or self._read_digest(handle, size) != digest
        ):
            raise RuntimeError("owned source file native payload diverged")
        return ("file", volume, file_id, size, digest, final, attributes)

    def _write_file(self, relative: str, parent: int, name: str, payload: bytes) -> None:
        handle = self._create(parent, name, directory=False, write=True)
        try:
            raw = ctypes.create_string_buffer(payload)
            offset = 0
            while offset < len(payload):
                count = wintypes.DWORD()
                view = ctypes.byref(raw, offset)
                if not self._kernel.WriteFile(
                    handle, view, len(payload) - offset, ctypes.byref(count), None
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if count.value == 0:
                    raise RuntimeError("owned source file write made no progress")
                offset += int(count.value)
            if not self._kernel.FlushFileBuffers(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self._handles[relative] = handle
            self._seals[relative] = self._seal(relative, handle, file_payload=payload)
        except BaseException:
            if relative not in self._handles:
                self._handles[relative] = handle
                self._seals[relative] = ("indeterminate",)
            raise

    def _mkdir(self, relative: str, parent: int, name: str) -> int:
        handle = self._create(parent, name, directory=True, write=False)
        self._handles[relative] = handle
        self._seals[relative] = self._seal(relative, handle)
        return handle

    def populate(self, files: tuple[tuple[str, bytes], ...]) -> None:
        self._seals["."] = self._seal(".", self._handles["."])
        self._write_file(
            self._policy.owner_marker,
            self._handles["."],
            self._policy.owner_marker,
            b"v1\n",
        )
        directories: dict[str, int] = {"": self._handles["."]}
        for name, payload in files:
            parts = name.split("/")
            parent_key = ""
            for index, component in enumerate(parts[:-1]):
                directory = "/".join(parts[: index + 1]) + "/"
                if directory not in directories:
                    directories[directory] = self._mkdir(
                        directory, directories[parent_key], component
                    )
                parent_key = directory
            self._write_file(name, directories[parent_key], parts[-1], payload)
        self._arm_watcher()
        # Windows can deliver this root's completed construction notifications to
        # the first arm. Drain that arm before the authoritative steady-state arm;
        # it never authorizes an unknown entry, because every generated handle is
        # sealed before the drain and the second arm is attached to the same root.
        self._stop_watcher()
        self._arm_watcher()
        self.inventory = dict(self._seals)
        self.assert_unchanged()

    def _filesystem_shape(self) -> dict[str, str]:
        root = self._workspace / self._root_name
        shape = {".": "dir"}
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for name in (*directory_names, *file_names):
                if ":" in name:
                    raise RuntimeError("owned source contains an alternate data stream spelling")
                candidate = current_path / name
                metadata = candidate.lstat()
                if candidate.is_symlink() or (
                    getattr(metadata, "st_file_attributes", 0) & 0x00000400
                ):
                    raise RuntimeError("owned source contains a link or reparse point")
            relative_root = current_path.relative_to(root)
            prefix = "" if str(relative_root) == "." else relative_root.as_posix() + "/"
            for name in directory_names:
                candidate = current_path / name
                if not stat.S_ISDIR(candidate.lstat().st_mode):
                    raise RuntimeError("owned source directory shape changed")
                shape[prefix + name + "/"] = "dir"
            for name in file_names:
                candidate = current_path / name
                if not stat.S_ISREG(candidate.lstat().st_mode):
                    raise RuntimeError("owned source file shape changed")
                shape[prefix + name] = "file"
        return shape

    def assert_unchanged(self) -> None:
        if self._closed or self._watch_stopped:
            raise RuntimeError("owned source authority is closed")
        if self._watcher._changed.is_set():
            raise RuntimeError("owned source directory changed during qualification")
        current_shape = self._filesystem_shape()
        expected_shape = {relative: str(seal[0]) for relative, seal in self._seals.items()}
        if current_shape != expected_shape:
            raise RuntimeError("owned source exact filesystem shape changed")
        if self._watcher._changed.is_set():
            raise RuntimeError("owned source directory changed during qualification")
        for relative, handle in self._handles.items():
            seal = self._seals.get(relative)
            if seal is None or seal[0] == "indeterminate":
                raise RuntimeError("owned source has indeterminate native authority")
            payload = None
            if seal[0] == "file":
                payload = self._read_exact_payload(handle, _exact_int(seal[3]))
            if self._seal(relative, handle, file_payload=payload) != seal:
                raise RuntimeError("owned source retained native identity changed")
        if self.inventory != self._seals:
            raise RuntimeError("owned source exact retained inventory changed")

    def _read_exact_payload(self, handle: int, size: int) -> bytes:
        position = ctypes.c_longlong()
        if not self._kernel.SetFilePointerEx(
            handle, ctypes.c_longlong(0), ctypes.byref(position), 0
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        result = bytearray()
        buffer = ctypes.create_string_buffer(1024 * 1024)
        while len(result) < size:
            count = wintypes.DWORD()
            if not self._kernel.ReadFile(
                handle, buffer, min(len(buffer), size - len(result)), ctypes.byref(count), None
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            if count.value == 0:
                raise RuntimeError("owned source payload read ended early")
            result.extend(buffer.raw[: count.value])
        return bytes(result)

    def _arm_watcher(self) -> None:
        self._watcher = self._new_watcher()
        self._watcher._ignore_known_notifications = frozenset(
            relative.rstrip("/").replace("/", "\\").casefold() for relative in self._seals
        )
        self._watcher._start_watcher(
            self._workspace / self._root_name, None, retained_handle=self._handles["."]
        )
        self._watch_stopped = False

    def _stop_watcher(self) -> None:
        if self._watch_stopped:
            return
        events: list[int] = []
        for watcher in self._watcher._watchers:
            values = watcher.get("events")
            if isinstance(values, list):
                events.extend(_exact_int(value, label="owned watcher event") for value in values)
        self._watcher._handles = events
        self._watcher.close(validate=False)
        self._watch_stopped = True

    def _close_handle(self, handle: int) -> None:
        if not self._kernel.CloseHandle(wintypes.HANDLE(handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def _dispose_and_close(self, relative: str) -> None:
        handle = self._handles[relative]
        seal = self._seals[relative]
        if seal[0] == "indeterminate":
            raise RuntimeError("cannot dispose indeterminate owned source authority")
        payload = (
            self._read_exact_payload(handle, _exact_int(seal[3])) if seal[0] == "file" else None
        )
        if self._seal(relative, handle, file_payload=payload) != seal:
            raise RuntimeError("owned source changed before handle-bound cleanup")
        disposition = _FILE_DISPOSITION_INFO(True)
        if not self._kernel.SetFileInformationByHandle(
            handle,
            self._FILE_DISPOSITION_INFO,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        self._close_handle(handle)
        del self._handles[relative]

    def _close_remaining(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for relative, handle in reversed(tuple(self._handles.items())):
            try:
                self._close_handle(handle)
            except BaseException as error:
                failures.append(error)
            else:
                del self._handles[relative]
        if self._workspace_handle:
            try:
                self._close_handle(self._workspace_handle)
            except BaseException as error:
                failures.append(error)
            else:
                self._workspace_handle = 0
        return failures

    def close(self, *, cleanup: bool = True) -> None:
        if self._closed:
            return
        failures: list[BaseException] = []
        if cleanup:
            try:
                self.assert_unchanged()
            except BaseException as error:
                failures.append(error)
        try:
            self._stop_watcher()
        except BaseException as error:
            failures.append(error)
        if self._watcher._changed.is_set():
            failures.append(RuntimeError("owned source directory changed during watcher shutdown"))
        if cleanup and not failures:
            ordered = sorted(
                self._handles,
                key=lambda item: (item.count("/"), not item.endswith("/"), item),
                reverse=True,
            )
            try:
                for relative in ordered:
                    self._dispose_and_close(relative)
            except BaseException as error:
                failures.append(error)
        failures.extend(self._close_remaining())
        self._closed = True
        if failures:
            raise BaseExceptionGroup("owned source extraction finalization failed", failures)

def archive_member_validator(
    policy: SourceArchivePolicyV1,
) -> Callable[[tarfile.TarFile], tuple[tarfile.TarInfo, ...]]:
    """Bind a frozen policy while retaining the legacy one-argument validator shape."""

    def validate(archive: tarfile.TarFile) -> tuple[tarfile.TarInfo, ...]:
        return validated_archive_members(archive, policy)

    return validate

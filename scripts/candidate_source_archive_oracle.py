# ruff: noqa: E501
"""Candidate-bound, in-memory Git source archive authority.

Ordinary pins attest the selected ``git.exe`` file object under the protected
Program Files runtime; they do not attest dependent DLL/helper closure or broader
runtime provenance. The owned-tool entry point instead requires the live owner of
its complete admitted distribution tree. Both use the existing native Git child
owner and fixed sanitized environment. Neither isolates a hostile controller or
OS, proves an independent build, or supplies durable-journal recovery. The opaque
archive prevents ordinary callers from substituting bytes or checkout authority.
"""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import os
import re
import stat
import subprocess
import tarfile
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

try:
    from task13_artifact_orchestrator import CandidateIdentityV1
except ModuleNotFoundError as error:
    if error.name != "task13_artifact_orchestrator":
        raise
    from scripts.task13_artifact_orchestrator import CandidateIdentityV1

if TYPE_CHECKING:
    from scripts.qualification_tool_distributions import ToolDistributionMetadataV1
    from scripts.qualification_tool_environment import ImmutableToolEnvironmentV1


_MAX_ARCHIVE_BYTES = 15 * 1024 * 1024
_MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
_MAX_STDERR_BYTES = 64 * 1024
_MAX_MEMBER_COUNT = 4096
_MAX_TREE_DEPTH = 64
_MAX_TREE_OBJECTS = 2048
_MAX_TREE_BYTES = 4 * 1024 * 1024
_MAX_TREE_ENTRIES = 8192
_MAX_VISIBLE_MEMBERS = 4096
_MAX_BLOB_BYTES = 4 * 1024 * 1024
_MAX_AGGREGATE_BLOB_BYTES = 12 * 1024 * 1024
_MAX_CANDIDATE_TREE_BYTES = 12 * 1024 * 1024
_PREFIX = "hermes-realtime-0.0.3"
_OID = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 1
_OPEN_EXISTING = 3
_FILE_ATTRIBUTE_NORMAL = 0x80
_FILE_FLAG_OPEN_REPARSE_POINT = 0x00200000
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_INVALID_HANDLE = ctypes.c_void_p(-1).value if os.name == "nt" else -1
_PATH_TYPE = type(Path())


class CandidateSourceArchiveError(RuntimeError):
    """The candidate or immutable archive authority failed closed."""


@dataclass(frozen=True, slots=True)
class GitExecutablePinV1:
    """Exact, caller-supplied identity for one Windows Git executable."""

    path: Path
    sha256: str
    version: str
    link_count: int
    max_bytes: int = _MAX_EXECUTABLE_BYTES

    def __post_init__(self) -> None:
        if type(self.path) is not _PATH_TYPE:
            raise TypeError("Git executable path must be an exact Path")
        lexical = str(self.path)
        if not PureWindowsPath(lexical).is_absolute() or not lexical.lower().endswith(".exe"):
            raise ValueError("Git executable path must be an absolute Windows .exe path")
        if type(self.sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise ValueError("Git executable SHA-256 must be lowercase hexadecimal")
        if (
            type(self.version) is not str
            or not self.version
            or any(ord(character) < 0x20 or ord(character) > 0x7E for character in self.version)
        ):
            raise ValueError("Git executable version must be strict printable ASCII")
        if type(self.link_count) is not int or self.link_count <= 0:
            raise ValueError("Git executable link count must be an exact positive integer")
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes <= _MAX_EXECUTABLE_BYTES:
            raise ValueError("Git executable size cap is invalid")
        _validate_public_git_location(self.path)


@dataclass(frozen=True, slots=True)
class CandidateArchiveMemberV1:
    path: str
    kind: str
    git_mode: str
    tar_mode: int
    size: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class CandidateSourceArchiveMetadataV1:
    candidate_head_oid: str
    candidate_tree_oid: str
    canonical_baseline_oid: str
    canonical_diff_sha256: str
    prefix: str
    archive_sha256: str
    archive_bytes: int
    manifest: tuple[CandidateArchiveMemberV1, ...]


@dataclass(frozen=True, slots=True)
class _VerifiedArchiveRecord:
    identity: CandidateIdentityV1
    metadata: CandidateSourceArchiveMetadataV1
    archive: bytes


class VerifiedCandidateSourceArchiveV1:
    """Opaque token; ordinary construction never mints archive authority."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("VerifiedCandidateSourceArchiveV1 tokens are factory-minted only")


_RECORDS: WeakKeyDictionary[VerifiedCandidateSourceArchiveV1, _VerifiedArchiveRecord] = (
    WeakKeyDictionary()
)
_TOOL_CAPTURES: WeakKeyDictionary[
    VerifiedCandidateSourceArchiveV1, tuple[ToolDistributionMetadataV1, ...]
] = WeakKeyDictionary()


def verified_candidate_source_archive_metadata(
    token: VerifiedCandidateSourceArchiveV1,
) -> CandidateSourceArchiveMetadataV1:
    if type(token) is not VerifiedCandidateSourceArchiveV1:
        raise TypeError("archive token type is invalid")
    try:
        return _RECORDS[token].metadata
    except KeyError as error:
        raise CandidateSourceArchiveError("archive token is unregistered or closed") from error


def _archive_bytes_for_consumer(
    token: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1
) -> bytes:
    """Private later-consumer seam; it intentionally exposes no path authority."""
    if (
        type(token) is not VerifiedCandidateSourceArchiveV1
        or type(identity) is not CandidateIdentityV1
    ):
        raise TypeError("archive consumer authority types are invalid")
    try:
        record = _RECORDS[token]
    except KeyError as error:
        raise CandidateSourceArchiveError("archive token is unregistered or closed") from error
    if record.identity != identity:
        raise CandidateSourceArchiveError("archive token candidate identity differs")
    return record.archive


def _archive_tool_capture_for_consumer(
    token: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1,
) -> tuple[ToolDistributionMetadataV1, ...]:
    """Completed capture provenance; does not keep removed tool trees executable."""
    _archive_bytes_for_consumer(token, identity)
    if token not in _TOOL_CAPTURES:
        raise CandidateSourceArchiveError("archive has no complete tool capture provenance")
    return _TOOL_CAPTURES[token]


class _BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("attributes", wintypes.DWORD),
        ("creation_low", wintypes.DWORD),
        ("creation_high", wintypes.DWORD),
        ("access_low", wintypes.DWORD),
        ("access_high", wintypes.DWORD),
        ("write_low", wintypes.DWORD),
        ("write_high", wintypes.DWORD),
        ("volume_serial", wintypes.DWORD),
        ("size_high", wintypes.DWORD),
        ("size_low", wintypes.DWORD),
        ("links", wintypes.DWORD),
        ("file_index_high", wintypes.DWORD),
        ("file_index_low", wintypes.DWORD),
    ]


@dataclass(frozen=True, slots=True)
class _GitExecutableHandleSealV1:
    """Complete immutable observation of a deny-write retained file handle."""

    volume_serial: int
    file_id: int
    attributes: int
    size: int
    links: int
    final_path: str
    digest: str


def _normal_final_path(value: str) -> str:
    value = (
        value.removeprefix("\\\\?\\UNC\\")
        if value.startswith("\\\\?\\UNC\\")
        else value.removeprefix("\\\\?\\")
    )
    if value.startswith("UNC\\"):
        value = "\\\\" + value[4:]
    return os.path.normcase(os.path.normpath(value))


def _known_program_files() -> Path:
    """Obtain Program Files from the native known-folder authority, never env."""
    if os.name != "nt":
        raise CandidateSourceArchiveError("Git executable authority requires Windows")
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    ole32 = ctypes.WinDLL("ole32", use_last_error=True)
    folder = (ctypes.c_byte * 16).from_buffer_copy(
        bytes.fromhex("b6635e90bfc14e49b29c65b732d3d21a")
    )
    value = wintypes.LPWSTR()
    method = shell32.SHGetKnownFolderPath
    method.argtypes = [
        ctypes.POINTER(ctypes.c_byte),
        wintypes.DWORD,
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.LPWSTR),
    ]
    method.restype = ctypes.c_long
    result = method(folder, 0, None, ctypes.byref(value))
    if result != 0 or not value.value:
        raise CandidateSourceArchiveError("could not resolve native Program Files authority")
    try:
        return Path(value.value)
    finally:
        ole32.CoTaskMemFree(value)


def _attributes(path: Path) -> int:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    method = kernel.GetFileAttributesW
    method.argtypes = [wintypes.LPCWSTR]
    method.restype = wintypes.DWORD
    result = int(method(str(path)))
    if result == 0xFFFFFFFF:
        raise CandidateSourceArchiveError("could not inspect native path component")
    return result


def _drive_type(root: str) -> int:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    drive_type = kernel.GetDriveTypeW
    drive_type.argtypes = [wintypes.LPCWSTR]
    drive_type.restype = wintypes.UINT
    return int(drive_type(root))


def _require_local_nonreparse_path(path: Path) -> None:
    windows = PureWindowsPath(str(path))
    if (
        not windows.is_absolute()
        or str(windows).startswith("\\\\")
        or not windows.drive.endswith(":")
    ):
        raise CandidateSourceArchiveError("path must be a local fixed-drive absolute path")
    root = windows.drive + "\\"
    if _drive_type(root) != _DRIVE_FIXED:  # Lexical C: is not authority.
        raise CandidateSourceArchiveError("path must be on a native fixed drive")
    current = Path(root)
    for component in windows.parts[1:]:
        current /= component
        if _attributes(current) & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise CandidateSourceArchiveError("path contains a reparse-mediated component")


def _validate_public_git_location(path: Path) -> None:
    """Public pins are only for the selected local Program Files git.exe file."""
    lexical = str(path)
    windows = PureWindowsPath(lexical)
    if (
        lexical != os.path.normpath(lexical)
        or any(component in {".", ".."} for component in windows.parts)
    ):
        raise ValueError("Git executable path must be canonical and traversal-free")
    program_files = _known_program_files()
    _require_local_nonreparse_path(program_files)
    _require_local_nonreparse_path(path)
    normalized_path = PureWindowsPath(_normal_final_path(lexical))
    normalized_program_files = PureWindowsPath(_normal_final_path(str(program_files)))
    try:
        normalized_path.relative_to(normalized_program_files)
    except ValueError as error:
        raise ValueError("Git executable must be under native Program Files") from error
    if path.name.casefold() != "git.exe":
        raise ValueError("Git executable must name git.exe")


def _trusted_windows_directories() -> tuple[str, str]:
    if os.name != "nt":
        raise CandidateSourceArchiveError("trusted Windows directories require Windows")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    values: list[str] = []
    for name in ("GetWindowsDirectoryW", "GetSystemDirectoryW"):
        function = getattr(kernel, name)
        function.argtypes = [wintypes.LPWSTR, wintypes.UINT]
        function.restype = wintypes.UINT
        buffer = ctypes.create_unicode_buffer(32768)
        size = int(function(buffer, len(buffer)))
        if not size or size >= len(buffer):
            raise CandidateSourceArchiveError("could not resolve trusted Windows directory")
        values.append(buffer.value)
    return values[0], values[1]


_CREATE_SUSPENDED = 0x00000004
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_STARTF_USESTDHANDLES = 0x00000100
_HANDLE_FLAG_INHERIT = 0x00000001
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 258
_STILL_ACTIVE = 259
_ERROR_BROKEN_PIPE = 109
_ERROR_NOT_FOUND = 1168
_ERROR_OPERATION_ABORTED = 995
_CHILD_TIMEOUT_SECONDS = 30.0
_CLEANUP_TIMEOUT_SECONDS = 5.0
_WAIT_POLL_MS = 50
_DRIVE_FIXED = 3


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", wintypes.DWORD),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t),
        ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD),
    ]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "read_operation_count",
            "write_operation_count",
            "other_operation_count",
            "read_transfer_count",
            "write_transfer_count",
            "other_transfer_count",
        )
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("io_info", _IO_COUNTERS),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("total_user_time", ctypes.c_int64),
        ("total_kernel_time", ctypes.c_int64),
        ("period_user_time", ctypes.c_int64),
        ("period_kernel_time", ctypes.c_int64),
        ("page_faults", wintypes.DWORD),
        ("total_processes", wintypes.DWORD),
        ("active_processes", wintypes.DWORD),
        ("terminated_processes", wintypes.DWORD),
    ]


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _STARTUPINFO(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEX(ctypes.Structure):
    _fields_ = [("startup_info", _STARTUPINFO), ("attribute_list", wintypes.LPVOID)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("process", wintypes.HANDLE),
        ("thread", wintypes.HANDLE),
        ("pid", wintypes.DWORD),
        ("tid", wintypes.DWORD),
    ]


class _NativeWin32GitKernel:
    """Minimal ctypes-only API boundary for one retained Git child."""

    def __init__(self) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise CandidateSourceArchiveError("native Git child ownership requires 64-bit Windows")
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self._cancelled_reads: set[int] = set()
        self._bind()

    @staticmethod
    def _value(raw: Any) -> int:
        value = int(ctypes.cast(raw, ctypes.c_void_p).value or 0)
        if not value or value == _INVALID_HANDLE:
            raise CandidateSourceArchiveError("native Git child handle creation failed")
        return value

    def _call(self, ok: Any, label: str) -> None:
        if not ok:
            raise CandidateSourceArchiveError(f"{label} (Win32 error {ctypes.get_last_error()})")

    def _bind(self) -> None:
        api = self.api
        api.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        api.SetInformationJobObject.restype = wintypes.BOOL
        api.CreatePipe.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(wintypes.HANDLE),
            ctypes.POINTER(_SECURITY_ATTRIBUTES),
            wintypes.DWORD,
        ]
        api.CreatePipe.restype = wintypes.BOOL
        api.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
        api.SetHandleInformation.restype = wintypes.BOOL
        api.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        api.CreateFileW.restype = wintypes.HANDLE
        api.InitializeProcThreadAttributeList.argtypes = [
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        api.InitializeProcThreadAttributeList.restype = wintypes.BOOL
        api.UpdateProcThreadAttribute.argtypes = [
            wintypes.LPVOID,
            ctypes.c_size_t,
            ctypes.c_size_t,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        api.UpdateProcThreadAttribute.restype = wintypes.BOOL
        api.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
        api.DeleteProcThreadAttributeList.restype = None
        api.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            wintypes.LPVOID,
            ctypes.POINTER(_PROCESS_INFORMATION),
        ]
        api.CreateProcessW.restype = wintypes.BOOL
        api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        api.AssignProcessToJobObject.restype = wintypes.BOOL
        api.ResumeThread.argtypes = [wintypes.HANDLE]
        api.ResumeThread.restype = wintypes.DWORD
        api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.TerminateJobObject.restype = wintypes.BOOL
        api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.TerminateProcess.restype = wintypes.BOOL
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetExitCodeProcess.restype = wintypes.BOOL
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.LPVOID,
        ]
        api.QueryInformationJobObject.restype = wintypes.BOOL
        api.ReadFile.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
        ]
        api.ReadFile.restype = wintypes.BOOL
        api.CancelIoEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
        api.CancelIoEx.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        api.CloseHandle.restype = wintypes.BOOL

    def create_job(self, active_limit: int) -> int:
        job = self._value(self.api.CreateJobObjectW(None, None))
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.basic_limit_information.limit_flags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        )
        info.basic_limit_information.active_process_limit = active_limit
        try:
            self._call(
                self.api.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)),
                "could not configure Git Job Object",
            )
        except BaseException as primary:
            try:
                self.close(job)
            except BaseException as close_error:
                raise BaseExceptionGroup(
                    "Git Job setup and handle close both failed",
                    [primary, close_error],
                ) from None
            raise
        return job

    def create_stdio(self) -> tuple[int, int, int, int, int, int]:
        attributes = _SECURITY_ATTRIBUTES(ctypes.sizeof(_SECURITY_ATTRIBUTES), None, True)
        stdin = self._value(
            self.api.CreateFileW(
                "NUL", _GENERIC_READ, 3, None, _OPEN_EXISTING, _FILE_ATTRIBUTE_NORMAL, None
            )
        )
        out_read, out_write, err_read, err_write = (
            wintypes.HANDLE(),
            wintypes.HANDLE(),
            wintypes.HANDLE(),
            wintypes.HANDLE(),
        )
        try:
            self._call(
                self.api.SetHandleInformation(stdin, _HANDLE_FLAG_INHERIT, _HANDLE_FLAG_INHERIT),
                "could not make Git stdin NUL inheritable",
            )
            self._call(
                self.api.CreatePipe(
                    ctypes.byref(out_read), ctypes.byref(out_write), ctypes.byref(attributes), 0
                ),
                "could not create Git stdout pipe",
            )
            self._call(
                self.api.CreatePipe(
                    ctypes.byref(err_read), ctypes.byref(err_write), ctypes.byref(attributes), 0
                ),
                "could not create Git stderr pipe",
            )
            for handle in (out_read, err_read):
                self._call(
                    self.api.SetHandleInformation(handle, _HANDLE_FLAG_INHERIT, 0),
                    "could not restrict Git pipe inheritance",
                )
            return (
                stdin,
                self._value(out_read),
                self._value(out_write),
                self._value(err_read),
                self._value(err_write),
                0,
            )
        except BaseException as primary:
            cleanup: list[BaseException] = []
            for raw in (
                stdin,
                int(out_read.value or 0),
                int(out_write.value or 0),
                int(err_read.value or 0),
                int(err_write.value or 0),
            ):
                if raw:
                    try:
                        self.close(raw)
                    except BaseException as close_error:
                        cleanup.append(close_error)
            if cleanup:
                raise BaseExceptionGroup(
                    "Git stdio setup and partial-handle cleanup failed",
                    [primary, *cleanup],
                ) from None
            raise

    def create_suspended(
        self,
        application: str,
        command: tuple[str, ...],
        root: str,
        environment: dict[str, str],
        handles: tuple[int, ...],
    ) -> tuple[int, int]:
        size = ctypes.c_size_t()
        self.api.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        attributes = (ctypes.c_byte * size.value)()
        if not self.api.InitializeProcThreadAttributeList(
            ctypes.byref(attributes), 1, 0, ctypes.byref(size)
        ):
            raise CandidateSourceArchiveError("could not initialize Git inherited-handle list")
        inherited = (ctypes.c_void_p * len(handles))(*handles)
        try:
            self._call(
                self.api.UpdateProcThreadAttribute(
                    ctypes.byref(attributes),
                    0,
                    0x00020002,
                    ctypes.byref(inherited),
                    ctypes.sizeof(inherited),
                    None,
                    None,
                ),
                "could not configure exact Git inherited-handle allowlist",
            )
            startup = _STARTUPINFOEX()
            startup.startup_info.cb = ctypes.sizeof(startup)
            startup.startup_info.dwFlags = _STARTF_USESTDHANDLES
            (
                startup.startup_info.hStdInput,
                startup.startup_info.hStdOutput,
                startup.startup_info.hStdError,
            ) = handles
            startup.attribute_list = ctypes.cast(ctypes.byref(attributes), wintypes.LPVOID)
            process = _PROCESS_INFORMATION()
            block = (
                "".join(
                    f"{name}={value}\0"
                    for name, value in sorted(
                        environment.items(), key=lambda item: item[0].casefold()
                    )
                )
                + "\0"
            )
            command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
            self._call(
                self.api.CreateProcessW(
                    application,
                    command_line,
                    None,
                    None,
                    True,
                    _CREATE_SUSPENDED | _CREATE_UNICODE_ENVIRONMENT | _EXTENDED_STARTUPINFO_PRESENT,
                    ctypes.c_wchar_p(block),
                    root,
                    ctypes.byref(startup),
                    ctypes.byref(process),
                ),
                "could not launch retained Git executable suspended",
            )
            process_raw = int(ctypes.cast(process.process, ctypes.c_void_p).value or 0)
            thread_raw = int(ctypes.cast(process.thread, ctypes.c_void_p).value or 0)
            def valid(handle: int) -> bool:
                return bool(handle and handle != _INVALID_HANDLE)

            if not valid(process_raw) or not valid(thread_raw):
                failures: list[BaseException] = [
                    CandidateSourceArchiveError("kernel returned invalid CreateProcessW handles")
                ]
                for handle in (thread_raw, process_raw):
                    if valid(handle):
                        try:
                            self.close(handle)
                        except BaseException as error:
                            failures.append(error)
                if len(failures) > 1:
                    raise BaseExceptionGroup("CreateProcessW handle cleanup failures", failures)
                raise failures[0]
            return process_raw, thread_raw
        finally:
            self.api.DeleteProcThreadAttributeList(ctypes.byref(attributes))

    def assign(self, job: int, process: int) -> None:
        self._call(
            self.api.AssignProcessToJobObject(job, process),
            "could not assign suspended Git child to Job Object",
        )

    def resume(self, thread: int) -> None:
        if self.api.ResumeThread(thread) == 0xFFFFFFFF:
            raise CandidateSourceArchiveError("could not resume Job-owned Git child")

    def terminate(self, job: int) -> None:
        self._call(self.api.TerminateJobObject(job, 1), "could not terminate Git Job Object")

    def terminate_process(self, process: int) -> None:
        self._call(
            self.api.TerminateProcess(process, 1), "could not terminate unassigned Git child"
        )

    def exit_code(self, process: int) -> int:
        code = wintypes.DWORD()
        self._call(
            self.api.GetExitCodeProcess(process, ctypes.byref(code)),
            "could not obtain Git child exit code",
        )
        return int(code.value)

    def wait(self, process: int, timeout_ms: int) -> int:
        return int(self.api.WaitForSingleObject(process, timeout_ms))

    def active(self, job: int) -> int:
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        self._call(
            self.api.QueryInformationJobObject(
                job, 1, ctypes.byref(info), ctypes.sizeof(info), None
            ),
            "could not prove Git Job active-process count",
        )
        return int(info.active_processes)

    def read_bounded(self, handle: int, limit: int) -> bytes:
        parts: list[bytes] = []
        total = 0
        buffer = ctypes.create_string_buffer(65536)
        while True:
            read = wintypes.DWORD()
            if not self.api.ReadFile(handle, buffer, len(buffer), ctypes.byref(read), None):
                error = ctypes.get_last_error()
                if error == _ERROR_BROKEN_PIPE:
                    return b"".join(parts)
                if error == _ERROR_OPERATION_ABORTED and handle in self._cancelled_reads:
                    return b"".join(parts)
                raise CandidateSourceArchiveError("could not read bounded Git pipe")
            if not read.value:
                return b"".join(parts)
            total += int(read.value)
            if total > limit:
                raise CandidateSourceArchiveError("Git child exceeded a bounded output limit")
            parts.append(buffer.raw[: read.value])

    def cancel(self, handle: int) -> None:
        if not self.api.CancelIoEx(handle, None) and ctypes.get_last_error() != _ERROR_NOT_FOUND:
            raise CandidateSourceArchiveError("could not cancel bounded Git pipe read")
        self._cancelled_reads.add(handle)

    def close(self, handle: int) -> None:
        self._call(self.api.CloseHandle(handle), "could not close retained Git child handle")


class _NativeGitChildOwner:
    """Own a Git child from suspended creation through proven Job quiescence."""

    def __init__(
        self, kernel: object, application: str, root: Path, environment: dict[str, str]
    ) -> None:
        self._kernel, self._application, self._root, self._environment = (
            kernel,
            application,
            str(root),
            environment,
        )

    def run(self, command: tuple[str, ...], *, stdout_limit: int) -> bytes:
        """Run only a Job-owned child and preserve every cleanup failure."""
        kernel: Any = self._kernel
        job = process = thread = None
        stdout_read = stderr_read = None
        parent_handles: list[int] = []
        closed_handles: set[int] = set()
        readers: list[threading.Thread] = []
        reader_errors: list[BaseException] = []
        reader_lock = threading.Lock()
        reader_event = threading.Event()
        assigned = False
        readers_ended = False
        primary: BaseException | None = None
        result: bytes | None = None

        def close_once(handle: int) -> None:
            if handle and handle not in closed_handles:
                closed_handles.add(handle)
                kernel.close(handle)

        def first_reader_error() -> BaseException | None:
            if not reader_event.is_set():
                return None
            with reader_lock:
                return reader_errors[0] if reader_errors else None

        def wait_until_exit(deadline: float, label: str, *, watch_readers: bool = True) -> None:
            assert process is not None
            while True:
                reader_error = first_reader_error()
                if watch_readers and reader_error is not None:
                    raise reader_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CandidateSourceArchiveError(f"Git child {label} wait timed out")
                status = kernel.wait(process, max(1, min(_WAIT_POLL_MS, int(remaining * 1000))))
                if status == _WAIT_TIMEOUT:
                    continue
                if status != _WAIT_OBJECT_0:
                    raise CandidateSourceArchiveError(
                        f"Git child {label} wait returned unexpected status {status}"
                    )
                return

        def wait_for_success(deadline: float) -> None:
            wait_until_exit(deadline, "process")
            assert process is not None
            code = kernel.exit_code(process)
            if code == _STILL_ACTIVE:
                raise CandidateSourceArchiveError("signaled Git child remained active")
            if code != 0:
                raise CandidateSourceArchiveError("Git child failed")

        def prove_zero_active(deadline: float) -> None:
            assert job is not None
            while True:
                if kernel.active(job) == 0:
                    return
                if time.monotonic() >= deadline:
                    raise CandidateSourceArchiveError("Git Job active-process count is not zero")
                time.sleep(0.01)

        def join_readers(deadline: float) -> list[BaseException]:
            nonlocal readers_ended
            for reader in readers:
                reader.join(max(0.0, deadline - time.monotonic()))
            readers_ended = not any(reader.is_alive() for reader in readers)
            if readers_ended:
                return []
            return [CandidateSourceArchiveError("Git reader thread outlived child")]

        try:
            job = kernel.create_job(4)
            stdin, stdout_read, stdout_write, stderr_read, stderr_write, extra = kernel.create_stdio()
            parent_handles = [stdin, stdout_read, stdout_write, stderr_read, stderr_write, extra]
            process, thread = kernel.create_suspended(
                self._application,
                command,
                self._root,
                self._environment,
                (stdin, stdout_write, stderr_write),
            )
            # Close every inherited parent-side handle immediately after creation.
            for handle in (stdin, stdout_write, stderr_write, extra):
                close_once(handle)
            kernel.assign(job, process)
            assigned = True
            kernel.resume(thread)
            output: list[bytes] = []
            stderr: list[bytes] = []

            def reader(handle: int, limit: int, sink: list[bytes]) -> None:
                try:
                    sink.append(kernel.read_bounded(handle, limit))
                except BaseException as error:
                    with reader_lock:
                        reader_errors.append(error)
                    reader_event.set()

            readers = [
                threading.Thread(target=reader, args=(stdout_read, stdout_limit, output)),
                threading.Thread(target=reader, args=(stderr_read, _MAX_STDERR_BYTES, stderr)),
            ]
            for reader_thread in readers:
                reader_thread.start()
            deadline = time.monotonic() + _CHILD_TIMEOUT_SECONDS
            wait_for_success(deadline)
            prove_zero_active(deadline)
            reader_cleanup = join_readers(deadline)
            if reader_cleanup:
                raise reader_cleanup[0]
            reader_error = first_reader_error()
            if reader_error is not None:
                raise reader_error
            result = b"".join(output[:1])
        except BaseException as error:
            primary = error

        cleanup: list[BaseException] = []
        cleanup_deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
        if primary is not None and process is not None:
            try:
                if assigned:
                    assert job is not None
                    kernel.terminate(job)
                else:
                    kernel.terminate_process(process)
            except BaseException as error:
                cleanup.append(error)
            try:
                wait_until_exit(cleanup_deadline, "cleanup", watch_readers=False)
            except BaseException as error:
                cleanup.append(error)
        if job is not None:
            try:
                prove_zero_active(cleanup_deadline)
            except BaseException as error:
                cleanup.append(error)
        join_readers(cleanup_deadline)
        if not readers_ended:
            cancel = getattr(kernel, "cancel", None)
            if callable(cancel):
                for handle in (stdout_read, stderr_read):
                    if handle is not None:
                        try:
                            cancel(handle)
                        except BaseException as error:
                            cleanup.append(error)
            cancellation_deadline = time.monotonic() + _CLEANUP_TIMEOUT_SECONDS
            join_readers(cancellation_deadline)
            if not readers_ended:
                cleanup.append(CandidateSourceArchiveError("Git reader thread outlived child"))
        for handle in parent_handles:
            if handle in {stdout_read, stderr_read} and not readers_ended:
                continue
            try:
                close_once(handle)
            except BaseException as error:
                cleanup.append(error)
        for handle in (thread, process):
            if handle is not None:
                try:
                    close_once(handle)
                except BaseException as error:
                    cleanup.append(error)
        if job is not None:
            try:
                close_once(job)
            except BaseException as error:
                cleanup.append(error)
        if primary is not None and cleanup:
            raise BaseExceptionGroup("Git child primary and cleanup failures", [primary, *cleanup])
        if cleanup:
            raise BaseExceptionGroup("Git child cleanup failures", cleanup)
        if primary is not None:
            raise primary
        assert result is not None
        return result


class _RetainedFileV1:
    """A deny-write/delete handle with the exact Win32 seal primitives bound."""

    def __init__(self, path: Path) -> None:
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.GetFileInformationByHandle.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_BY_HANDLE_FILE_INFORMATION)
        ]
        kernel.GetFileInformationByHandle.restype = wintypes.BOOL
        kernel.GetFinalPathNameByHandleW.argtypes = [
            wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD
        ]
        kernel.GetFinalPathNameByHandleW.restype = wintypes.DWORD
        kernel.SetFilePointerEx.argtypes = [
            wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), wintypes.DWORD
        ]
        kernel.SetFilePointerEx.restype = wintypes.BOOL
        kernel.ReadFile.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID
        ]
        kernel.ReadFile.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        self._kernel = kernel
        # FILE_SHARE_READ only: aliases cannot acquire WRITE or DELETE access.
        raw = kernel.CreateFileW(
            str(path),
            _GENERIC_READ,
            _FILE_SHARE_READ,
            None,
            _OPEN_EXISTING,
            _FILE_ATTRIBUTE_NORMAL | _FILE_FLAG_OPEN_REPARSE_POINT,
            None,
        )
        self._handle = int(ctypes.cast(raw, ctypes.c_void_p).value or 0)
        if not self._handle or self._handle == _INVALID_HANDLE:
            raise CandidateSourceArchiveError("could not retain Git executable handle")

    def _file_information(self, handle: int) -> _BY_HANDLE_FILE_INFORMATION:
        info = _BY_HANDLE_FILE_INFORMATION()
        if not self._kernel.GetFileInformationByHandle(wintypes.HANDLE(handle), ctypes.byref(info)):
            raise CandidateSourceArchiveError("could not inspect retained Git executable handle")
        return info

    def _final_path(self, handle: int) -> str:
        buffer = ctypes.create_unicode_buffer(32768)
        size = int(
            self._kernel.GetFinalPathNameByHandleW(wintypes.HANDLE(handle), buffer, len(buffer), 0)
        )
        if not size or size >= len(buffer):
            raise CandidateSourceArchiveError("could not resolve retained Git executable final path")
        return _normal_final_path(buffer.value)

    def _reset_handle_position(self, handle: int) -> None:
        if not self._kernel.SetFilePointerEx(wintypes.HANDLE(handle), 0, None, 0):
            raise CandidateSourceArchiveError("could not reset retained Git executable handle")

    def _digest_handle_exact(self, handle: int, size: int, maximum: int) -> str:
        if size < 0 or size > maximum:
            raise CandidateSourceArchiveError("retained Git executable size exceeds sealed bound")
        digest = hashlib.sha256()
        remaining = size
        buffer = ctypes.create_string_buffer(65536)
        self._reset_handle_position(handle)
        try:
            while remaining:
                request = min(len(buffer), remaining)
                read = wintypes.DWORD()
                if not self._kernel.ReadFile(
                    wintypes.HANDLE(handle), buffer, request, ctypes.byref(read), None
                ):
                    raise CandidateSourceArchiveError("could not read retained Git executable handle")
                if not read.value or read.value > request:
                    raise CandidateSourceArchiveError("retained Git executable handle read was not exact")
                digest.update(buffer.raw[: read.value])
                remaining -= int(read.value)
            return digest.hexdigest()
        finally:
            self._reset_handle_position(handle)

    def close(self) -> None:
        if self._handle:
            if not self._kernel.CloseHandle(wintypes.HANDLE(self._handle)):
                raise CandidateSourceArchiveError("could not close retained Git executable handle")
            self._handle = 0


@dataclass(frozen=True, slots=True)
class _OwnedGitFileBindingV1:
    path: Path
    sha256: str
    version: str
    link_count: int = 1
    max_bytes: int = _MAX_EXECUTABLE_BYTES


class _RetainedGitExecutableV1(_RetainedFileV1):
    def __init__(self, pin: GitExecutablePinV1) -> None:
        _validate_public_git_location(pin.path)
        self._owned_tools: ImmutableToolEnvironmentV1 | None = None
        self._initialize_pin(pin)

    @classmethod
    def from_owned_tools(cls, tools: ImmutableToolEnvironmentV1) -> _RetainedGitExecutableV1:
        from scripts.qualification_tool_environment import _tool_image_for_consumer

        path, digest, version = _tool_image_for_consumer(tools, "git")
        retained = object.__new__(cls)
        retained._owned_tools = tools
        retained._initialize_pin(_OwnedGitFileBindingV1(path, digest, "git version " + version))
        return retained

    def _initialize_pin(self, pin: GitExecutablePinV1 | _OwnedGitFileBindingV1) -> None:
        self.executable = str(pin.path)
        self._pin = pin
        self._max_seal_bytes = pin.max_bytes
        super().__init__(pin.path)
        try:
            seal = self._seal_from_handle(self._handle)
            self._require_pin_seal(seal)
            self._seal = seal
        except BaseException as primary:
            try:
                self.close()
            except BaseException as close_error:
                raise BaseExceptionGroup(
                    "retained Git executable setup cleanup failed", [primary, close_error]
                ) from None
            raise

    def _seal_from_handle(self, handle: int) -> _GitExecutableHandleSealV1:
        info = self._file_information(handle)
        attributes = int(info.attributes)
        if attributes & (_FILE_ATTRIBUTE_DIRECTORY | _FILE_ATTRIBUTE_REPARSE_POINT):
            raise CandidateSourceArchiveError("retained Git executable is directory or reparse point")
        size = (int(info.size_high) << 32) | int(info.size_low)
        return _GitExecutableHandleSealV1(
            int(info.volume_serial),
            (int(info.file_index_high) << 32) | int(info.file_index_low),
            attributes,
            size,
            int(info.links),
            self._final_path(handle),
            self._digest_handle_exact(handle, size, self._max_seal_bytes),
        )

    def _require_pin_seal(self, seal: _GitExecutableHandleSealV1) -> None:
        if (
            seal.final_path != _normal_final_path(str(self._pin.path))
            or seal.size > self._pin.max_bytes
            or seal.links != self._pin.link_count
            or not hmac.compare_digest(seal.digest, self._pin.sha256)
        ):
            raise CandidateSourceArchiveError("Git executable retained handle seal differs from pin")

    def _assert_complete_seal(
        self, observed: _GitExecutableHandleSealV1, label: str
    ) -> None:
        if observed != self._seal:
            raise CandidateSourceArchiveError(f"{label} Git executable handle seal changed")

    def assert_sealed(self) -> None:
        if self._owned_tools is not None:
            from scripts.qualification_tool_environment import _tool_image_for_consumer

            path, digest, version = _tool_image_for_consumer(self._owned_tools, "git")
            if (path, digest, "git version " + version) != (self._pin.path, self._pin.sha256, self._pin.version):
                raise CandidateSourceArchiveError("owned Git distribution binding changed")
        self._assert_complete_seal(self._seal_from_handle(self._handle), "retained")
        fresh = _RetainedFileV1(self._pin.path)
        failure: BaseException | None = None
        try:
            self._assert_complete_seal(self._seal_from_handle(fresh._handle), "fresh")
        except BaseException as error:
            failure = error
        try:
            fresh.close()
        except BaseException as close_error:
            if failure is not None:
                raise BaseExceptionGroup(
                    "fresh Git executable seal and close both failed", [failure, close_error]
                ) from None
            raise
        if failure is not None:
            raise failure


def _retained_git_executable_for_test(path: Path) -> _RetainedGitExecutableV1:
    """Private disposable-file seam for handle-seal REDs; never validates Program Files."""
    if path.name.casefold() != "git.exe":
        raise CandidateSourceArchiveError("disposable retained file must be git.exe")
    retained = object.__new__(_RetainedGitExecutableV1)
    retained._owned_tools = None
    _RetainedFileV1.__init__(retained, path)
    try:
        info = retained._file_information(retained._handle)
        size = (int(info.size_high) << 32) | int(info.size_low)
        retained.executable = str(path)
        retained._max_seal_bytes = _MAX_EXECUTABLE_BYTES
        retained._pin = type(
            "_DisposableGitExecutablePinV1",
            (),
            {
                "path": path,
                "sha256": retained._digest_handle_exact(retained._handle, size, _MAX_EXECUTABLE_BYTES),
                "link_count": int(info.links),
                "max_bytes": _MAX_EXECUTABLE_BYTES,
            },
        )()
        seal = retained._seal_from_handle(retained._handle)
        retained._require_pin_seal(seal)
        retained._seal = seal
        return retained
    except BaseException:
        retained.close()
        raise


def _retained_disposable_file_for_test(path: Path) -> _RetainedFileV1:
    """Private test seam: only caller-created disposable files, never Program Files."""
    if path.name.casefold() != "git.exe":
        raise CandidateSourceArchiveError("disposable retained file must be git.exe")
    primary = _RetainedFileV1(path)
    aliases: list[_RetainedFileV1] = []
    try:
        # Private fixture only; it never scans Program Files.
        for sibling in path.parent.iterdir():
            if sibling != path and os.path.samefile(path, sibling):
                aliases.append(_RetainedFileV1(sibling))
    except BaseException:
        for alias in aliases:
            alias.close()
        primary.close()
        raise
    original_close = primary.close

    def close_all() -> None:
        failures: list[BaseException] = []
        for retained in [*aliases, primary]:
            try:
                (original_close if retained is primary else retained.close)()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("could not close retained disposable aliases", failures)

    primary.close = close_all  # type: ignore[method-assign]
    return primary


def _clean_git_environment(executable: str) -> dict[str, str]:
    windows, system = _trusted_windows_directories()
    return {
        "SystemRoot": windows,
        "WINDIR": windows,
        "COMSPEC": str(Path(system) / "cmd.exe"),
        "PATH": os.pathsep.join((str(Path(executable).parent), system)),
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "NUL",
        "GIT_CONFIG_COUNT": "0",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
    }


class _GitExecutor:
    def __init__(self, retained: _RetainedGitExecutableV1, root: Path) -> None:
        self._retained = retained
        self._root = root
        self._root_text = str(root)

    def run(self, *arguments: str, stdout_limit: int = _MAX_ARCHIVE_BYTES) -> bytes:
        self._retained.assert_sealed()
        fixed = (
            "-c",
            f"safe.directory={self._root_text}",
            "-c",
            "core.attributesFile=NUL",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.hooksPath=NUL",
            "-c",
            "maintenance.auto=false",
            "-c",
            "gc.auto=0",
            "--no-replace-objects",
            "-C",
            self._root_text,
        )
        command = (self._retained.executable, *fixed, *arguments)
        primary: BaseException | None = None
        result: bytes | None = None
        try:
            result = _NativeGitChildOwner(
                _NativeWin32GitKernel(),
                self._retained.executable,
                self._root,
                _clean_git_environment(self._retained.executable),
            ).run(command, stdout_limit=stdout_limit)
        except BaseException as error:
            primary = error
        try:
            self._retained.assert_sealed()
        except BaseException as seal_error:
            if primary is not None:
                raise BaseExceptionGroup(
                    "Git child and retained executable revalidation both failed",
                    [primary, seal_error],
                ) from None
            raise
        if primary is not None:
            raise primary
        assert result is not None
        return result


def _text(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise CandidateSourceArchiveError(f"{label} is not strict UTF-8") from error


def _oid(value: bytes, label: str) -> str:
    try:
        text = value.decode("ascii", "strict")
    except UnicodeDecodeError as error:
        raise CandidateSourceArchiveError(f"{label} is not ASCII") from error
    if _OID.fullmatch(text) is None:
        raise CandidateSourceArchiveError(f"{label} is not a full SHA-1 OID")
    return text


def _git_line(executor: _GitExecutor, *arguments: str, label: str) -> str:
    result = executor.run(*arguments, stdout_limit=4096)
    if not result.endswith(b"\n") or result.count(b"\n") != 1:
        raise CandidateSourceArchiveError(f"{label} did not produce one canonical line")
    return _oid(result[:-1], label)


def _validate_supplied_identity(supplied: CandidateIdentityV1) -> None:
    """Reject attacker-controlled revision syntax before any Git child exists."""
    if type(supplied) is not CandidateIdentityV1:
        raise TypeError("candidate identity must be exact CandidateIdentityV1")
    values = (
        supplied.candidate_head_oid,
        supplied.candidate_tree_oid,
        supplied.canonical_baseline_oid,
    )
    if any(type(value) is not str or _OID.fullmatch(value) is None for value in values):
        raise CandidateSourceArchiveError(
            "candidate identity OIDs must be lowercase full SHA-1 values"
        )
    if (
        type(supplied.canonical_diff_sha256) is not str
        or _SHA256.fullmatch(supplied.canonical_diff_sha256) is None
    ):
        raise CandidateSourceArchiveError("candidate identity diff must be lowercase SHA-256")


def _require_root_and_identity(
    executor: _GitExecutor, root: Path, supplied: CandidateIdentityV1
) -> None:
    if (
        type(supplied) is not CandidateIdentityV1
        or not root.is_absolute()
        or root != root.resolve(strict=True)
    ):
        raise CandidateSourceArchiveError("candidate root or identity type is invalid")
    object_format = executor.run("rev-parse", "--show-object-format", stdout_limit=32)
    if object_format != b"sha1\n":
        raise CandidateSourceArchiveError("candidate object format must be exact sha1")
    top = executor.run("rev-parse", "--show-toplevel", stdout_limit=32768)
    if not top.endswith(b"\n") or _normal_final_path(
        _text(top[:-1], "checkout root")
    ) != _normal_final_path(str(root)):
        raise CandidateSourceArchiveError("candidate must be the exact checkout root")
    status = executor.run(
        "status", "--porcelain=v1", "-z", "--untracked-files=no", stdout_limit=1024
    )
    if status:
        raise CandidateSourceArchiveError(f"candidate has tracked index/worktree drift: {status!r}")
    head = _git_line(executor, "rev-parse", "--verify", "HEAD^{commit}", label="candidate head")
    tree = _git_line(executor, "rev-parse", "--verify", "HEAD^{tree}", label="candidate tree")
    base = _git_line(
        executor,
        "rev-parse",
        "--verify",
        f"{supplied.canonical_baseline_oid}^{{commit}}",
        label="baseline",
    )
    diff = executor.run(
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        "--no-ext-diff",
        f"{base}..{head}",
    )
    captured = CandidateIdentityV1(head, tree, base, hashlib.sha256(diff).hexdigest())
    if captured != supplied:
        raise CandidateSourceArchiveError("candidate identity drifted")


def _git_object(
    executor: _GitExecutor, kind: str, oid: str, *, stdout_limit: int = _MAX_ARCHIVE_BYTES
) -> bytes:
    data = executor.run("cat-file", kind, oid, stdout_limit=stdout_limit)
    digest = hashlib.sha1(
        kind.encode("ascii") + b" " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()
    if digest != oid:
        raise CandidateSourceArchiveError(f"Git {kind} object hash differs from OID")
    return data


def _safe_path(path: str) -> None:
    if not path or path.startswith("/") or "\\" in path or ":" in path or len(path) > 1024:
        raise CandidateSourceArchiveError("committed path is unsafe")
    seen: set[str] = set()
    for part in path.split("/"):
        if (
            not part
            or part in {".", ".."}
            or part[-1:] in {" ", "."}
            or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
        ):
            raise CandidateSourceArchiveError("committed path has an unsafe Windows component")
        stem = part.casefold().split(".", 1)[0]
        if stem in {"con", "prn", "aux", "nul"} or stem in {f"com{i}" for i in range(1, 10)} | {
            f"lpt{i}" for i in range(1, 10)
        }:
            raise CandidateSourceArchiveError("committed path has a reserved Windows component")
        seen.add(part.casefold())


def _walk_tree(
    executor: _GitExecutor, tree_oid: str
) -> tuple[dict[str, tuple[str, bytes]], set[str]]:
    """Read candidate objects under a closed producer-only traversal budget."""
    files: dict[str, tuple[str, bytes]] = {}
    directories: set[str] = {""}
    tree_oids: set[str] = set()
    tree_bytes = entries = aggregate_blobs = 0

    def walk(oid: str, prefix: str, depth: int) -> None:
        nonlocal tree_bytes, entries, aggregate_blobs
        if depth > _MAX_TREE_DEPTH or oid in tree_oids or len(tree_oids) >= _MAX_TREE_OBJECTS:
            raise CandidateSourceArchiveError("candidate tree recursion/object budget exceeded")
        tree_oids.add(oid)
        remaining_trees = _MAX_TREE_BYTES - tree_bytes
        if remaining_trees <= 0:
            raise CandidateSourceArchiveError("candidate tree byte budget exceeded")
        data = _git_object(executor, "tree", oid, stdout_limit=remaining_trees)
        tree_bytes += len(data)
        if tree_bytes > _MAX_TREE_BYTES:
            raise CandidateSourceArchiveError("candidate tree byte budget exceeded")
        position = 0
        local: set[str] = set()
        while position < len(data):
            entries += 1
            if entries > _MAX_TREE_ENTRIES:
                raise CandidateSourceArchiveError("candidate tree entry budget exceeded")
            space, nul = data.find(b" ", position), data.find(b"\0", position)
            if space < position or nul < space or nul + 21 > len(data):
                raise CandidateSourceArchiveError("Git tree is malformed")
            mode, raw_name = data[position:space], data[space + 1 : nul]
            name = _text(raw_name, "committed path")
            if "/" in name:
                raise CandidateSourceArchiveError("Git tree entry contains a slash")
            full = prefix + name
            _safe_path(full)
            folded = name.casefold()
            if folded in local:
                raise CandidateSourceArchiveError("Git tree has a case collision")
            local.add(folded)
            object_oid = _oid(data[nul + 1 : nul + 21].hex().encode("ascii"), "tree object")
            position = nul + 21
            if mode == b"40000":
                directories.add(full)
                walk(object_oid, full + "/", depth + 1)
            elif mode in {b"100644", b"100755"}:
                if full in files:
                    raise CandidateSourceArchiveError("Git tree has duplicate path")
                remaining_blobs = min(_MAX_BLOB_BYTES, _MAX_AGGREGATE_BLOB_BYTES - aggregate_blobs)
                if remaining_blobs <= 0:
                    raise CandidateSourceArchiveError("candidate aggregate blob budget exceeded")
                blob = _git_object(executor, "blob", object_oid, stdout_limit=remaining_blobs)
                aggregate_blobs += len(blob)
                if len(blob) > _MAX_BLOB_BYTES or aggregate_blobs > _MAX_AGGREGATE_BLOB_BYTES:
                    raise CandidateSourceArchiveError("candidate blob budget exceeded")
                files[full] = (mode.decode("ascii"), blob)
            elif mode in {b"120000", b"160000"}:
                raise CandidateSourceArchiveError("Git tree contains symlink or gitlink")
            else:
                raise CandidateSourceArchiveError("Git tree has noncanonical mode")
        if position != len(data):
            raise CandidateSourceArchiveError("Git tree parsing did not consume all bytes")

    walk(tree_oid, "", 0)
    if tree_bytes + aggregate_blobs > _MAX_CANDIDATE_TREE_BYTES:
        raise CandidateSourceArchiveError("candidate tree aggregate byte budget exceeded")
    return files, directories


def _attribute(executor: _GitExecutor, path: str) -> tuple[str, str]:
    answer = executor.run(
        "check-attr",
        "--cached",
        "-z",
        "export-ignore",
        "export-subst",
        "--",
        path,
        stdout_limit=8192,
    )
    fields = answer.split(b"\0")
    if len(fields) != 7 or fields[-1] != b"":
        raise CandidateSourceArchiveError("committed attribute output is malformed")
    if (
        _text(fields[0], "attribute path") != path
        or _text(fields[1], "attribute name") != "export-ignore"
        or _text(fields[3], "attribute path") != path
        or _text(fields[4], "attribute name") != "export-subst"
    ):
        raise CandidateSourceArchiveError("committed attribute output is unexpected")
    return _text(fields[2], "attribute value"), _text(fields[5], "attribute value")


def _visible_manifest(
    executor: _GitExecutor, tree: str
) -> tuple[tuple[CandidateArchiveMemberV1, ...], dict[str, bytes]]:
    files, directories = _walk_tree(executor, tree)
    ignored: set[str] = set()
    for path in sorted({*files, *(item for item in directories if item)}):
        export_ignore, export_subst = _attribute(executor, path)
        if export_ignore not in {"set", "unset", "unspecified"} or export_subst not in {
            "set",
            "unset",
            "unspecified",
        }:
            raise CandidateSourceArchiveError("committed archive attributes are ambiguous")
        if export_subst == "set":
            raise CandidateSourceArchiveError("committed export-subst is forbidden")
        if export_ignore == "set":
            ignored.add(path)

    def included(path: str) -> bool:
        return not any(path == omitted or path.startswith(omitted + "/") for omitted in ignored)

    manifest: list[CandidateArchiveMemberV1] = [
        CandidateArchiveMemberV1("", "dir", "40000", 0o775, 0, None)
    ]
    payloads: dict[str, bytes] = {}
    visible_paths = sorted(
        {*(item for item in directories if item), *files},
        key=lambda path: path + ("/" if path in directories else ""),
    )
    for path in visible_paths:
        if not included(path):
            continue
        if path in directories:
            manifest.append(CandidateArchiveMemberV1(path, "dir", "40000", 0o775, 0, None))
            continue
        mode, data = files[path]
        tar_mode = 0o664 if mode == "100644" else 0o775
        manifest.append(
            CandidateArchiveMemberV1(path, "file", mode, tar_mode, len(data), hashlib.sha256(data).hexdigest())
        )
        payloads[path] = data
    for member in manifest:
        _canonical_ustar_name_prefix(member)
    if len(manifest) > _MAX_VISIBLE_MEMBERS:
        raise CandidateSourceArchiveError("candidate visible member budget exceeded")
    return tuple(manifest), payloads


def _pax_comment(oid: str) -> bytes:
    body = f"comment={oid}\n".encode("ascii")
    length = len(body) + 3
    while True:
        record = str(length).encode("ascii") + b" " + body
        if len(record) == length:
            return record
        length = len(record)


def _canonical_tar_number(raw: bytes, value: int, width: int, label: str) -> None:
    expected = f"{value:0{width - 1}o}".encode("ascii") + b"\0"
    if raw != expected:
        raise CandidateSourceArchiveError(f"archive {label} header is not canonical")


def _canonical_ustar_name_prefix(member: CandidateArchiveMemberV1) -> tuple[bytes, bytes]:
    """Return the one accepted USTAR spelling for a visible member path."""
    full_text = _PREFIX + (f"/{member.path}" if member.path else "") + (
        "/" if member.kind == "dir" else ""
    )
    try:
        full = full_text.encode("ascii", "strict")
    except UnicodeEncodeError as error:
        raise CandidateSourceArchiveError("archive path is not ASCII USTAR-representable") from error
    if len(full) <= 100:
        return full + b"\0" * (100 - len(full)), b"\0" * 155
    for split in range(len(full) - 1, -1, -1):
        if full[split : split + 1] != b"/":
            continue
        prefix, name = full[:split], full[split + 1 :]
        if prefix and len(prefix) <= 155 and 1 <= len(name) <= 100:
            return name + b"\0" * (100 - len(name)), prefix + b"\0" * (155 - len(prefix))
    raise CandidateSourceArchiveError("archive path is not USTAR-representable")


def _validate_raw_member_header(
    header: bytes, member: CandidateArchiveMemberV1, timestamp: int
) -> None:
    if len(header) != 512:
        raise CandidateSourceArchiveError("archive member header is truncated")
    stored = header[148:156].rstrip(b"\0 ")
    try:
        checksum = int(stored, 8)
    except ValueError as error:
        raise CandidateSourceArchiveError("archive member checksum is malformed") from error
    calculated = sum(header[:148]) + (32 * 8) + sum(header[156:])
    if checksum != calculated or header[148:156] != f"{calculated:07o}".encode("ascii") + b"\0":
        raise CandidateSourceArchiveError("archive member checksum differs or is noncanonical")
    name, prefix = _canonical_ustar_name_prefix(member)
    if header[:100] != name or header[345:500] != prefix:
        raise CandidateSourceArchiveError("archive member name/prefix header is not canonical")
    if header[156:157] != (b"5" if member.kind == "dir" else b"0"):
        raise CandidateSourceArchiveError("archive member type header is not canonical")
    _canonical_tar_number(header[100:108], member.tar_mode, 8, "mode")
    _canonical_tar_number(header[108:116], 0, 8, "uid")
    _canonical_tar_number(header[116:124], 0, 8, "gid")
    _canonical_tar_number(header[124:136], member.size, 12, "size")
    _canonical_tar_number(header[136:148], timestamp, 12, "mtime")
    if (
        header[157:257] != b"\0" * 100
        or header[257:265] != b"ustar\x0000"
        or header[265:297] != b"root\0" + b"\0" * 27
        or header[297:329] != b"root\0" + b"\0" * 27
        or header[329:345] != b"0" * 7 + b"\0" + b"0" * 7 + b"\0"
        or header[500:512] != b"\0" * 12
    ):
        raise CandidateSourceArchiveError("archive member ustar fields are not canonical")


def _validate_global_pax_header(header: bytes, head: str, timestamp: int) -> None:
    payload = _pax_comment(head)
    calculated = sum(header[:148]) + (32 * 8) + sum(header[156:])
    if (
        header[:100] != b"pax_global_header" + b"\0" * 83
        or header[100:108] != b"0000666\0"
        or header[108:116] != b"0000000\0"
        or header[116:124] != b"0000000\0"
        or header[124:136] != f"{len(payload):011o}".encode("ascii") + b"\0"
        or header[136:148] != f"{timestamp:011o}".encode("ascii") + b"\0"
        or header[148:156] != f"{calculated:07o}".encode("ascii") + b"\0"
        or header[156:157] != b"g"
        or header[157:257] != b"\0" * 100
        or header[257:265] != b"ustar\x0000"
        or header[265:297] != b"root\0" + b"\0" * 27
        or header[297:329] != b"root\0" + b"\0" * 27
        or header[329:345] != b"0" * 7 + b"\0" + b"0" * 7 + b"\0"
        or header[345:512] != b"\0" * 167
    ):
        raise CandidateSourceArchiveError("archive lacks exact leading Git global PAX comment")


def _parse_raw_tar(
    data: bytes,
    head: str,
    commit_timestamp: int,
    manifest: tuple[CandidateArchiveMemberV1, ...],
) -> None:
    if len(data) % 512 or len(data) < 1536:
        raise CandidateSourceArchiveError("archive has invalid raw tar framing")
    offset = 0
    header = data[:512]
    _validate_global_pax_header(header, head, commit_timestamp)
    pax_size = len(_pax_comment(head))
    if data[512 : 512 + pax_size] != _pax_comment(head):
        raise CandidateSourceArchiveError("archive lacks exact leading Git global PAX comment")
    pax_payload_end = 512 + pax_size
    offset = 512 + ((pax_size + 511) // 512) * 512
    if any(data[pax_payload_end:offset]):
        raise CandidateSourceArchiveError("archive PAX payload padding is nonzero")
    for member in manifest:
        header = data[offset : offset + 512]
        if len(header) != 512 or header == b"\0" * 512:
            raise CandidateSourceArchiveError("archive raw member sequence differs from manifest")
        if header[156:157] in {b"g", b"x", b"L", b"K", b"S"}:
            raise CandidateSourceArchiveError("archive contains an unsafe tar extension")
        _validate_raw_member_header(header, member, commit_timestamp)
        payload_end = offset + 512 + member.size
        next_offset = offset + 512 + ((member.size + 511) // 512) * 512
        if next_offset > len(data):
            raise CandidateSourceArchiveError("archive raw member sequence differs from manifest")
        if any(data[payload_end:next_offset]):
            raise CandidateSourceArchiveError("archive payload padding is nonzero")
        offset = next_offset
    if data[offset : offset + 512] != b"\0" * 512:
        raise CandidateSourceArchiveError("archive raw member sequence differs from manifest")
    if data[offset + 512 : offset + 1024] != b"\0" * 512 or any(data[offset + 1024 :]):
        raise CandidateSourceArchiveError("archive has invalid tar termination")


def _validate_tar(
    data: bytes,
    identity: CandidateIdentityV1,
    manifest: tuple[CandidateArchiveMemberV1, ...],
    payloads: dict[str, bytes],
    commit_timestamp: int,
) -> None:
    _parse_raw_tar(data, identity.candidate_head_oid, commit_timestamp, manifest)
    expected = {member.path: member for member in manifest}
    observed: dict[str, tarfile.TarInfo] = {}
    try:
        archive = tarfile.open(fileobj=BytesIO(data), mode="r:")  # noqa: SIM115
    except tarfile.TarError as error:
        raise CandidateSourceArchiveError("archive is not a readable tar") from error
    with archive:
        members = archive.getmembers()
        if not 1 <= len(members) <= _MAX_MEMBER_COUNT:
            raise CandidateSourceArchiveError("archive member count is invalid")
        for info in members:
            name = info.name.rstrip("/")
            if not name.startswith(_PREFIX) or (
                name != _PREFIX and not name.startswith(_PREFIX + "/")
            ):
                raise CandidateSourceArchiveError("archive member escapes exact prefix")
            relative = name[len(_PREFIX) :].lstrip("/")
            if relative in observed or relative.casefold() in {
                item.casefold() for item in observed
            }:
                raise CandidateSourceArchiveError("archive has duplicate or case-colliding members")
            if not (info.isdir() or info.isreg()) or info.issparse() or info.linkname:
                raise CandidateSourceArchiveError(
                    "archive contains link, device, sparse, or nonordinary member"
                )
            if (
                info.uid != 0
                or info.gid != 0
                or info.uname != "root"
                or info.gname != "root"
                or info.mtime != commit_timestamp
            ):
                raise CandidateSourceArchiveError("archive member metadata drifted")
            observed[relative] = info
        if set(observed) != set(expected):
            raise CandidateSourceArchiveError(
                "archive members differ from committed visible manifest"
            )
        for relative, member in expected.items():
            info = observed[relative]
            if (
                (member.kind == "dir") != info.isdir()
                or info.mode != member.tar_mode
                or info.size != member.size
            ):
                raise CandidateSourceArchiveError("archive member kind, mode, or size differs")
            if member.kind == "file":
                extracted = archive.extractfile(info)
                if extracted is None or extracted.read() != payloads[relative]:
                    raise CandidateSourceArchiveError(
                        "archive file payload differs from committed blob"
                    )


def _commit_timestamp(commit: bytes, expected_tree: str) -> int:
    tree: str | None = None
    timestamp: int | None = None
    for line in commit.split(b"\n\n", 1)[0].splitlines():
        if line.startswith(b"tree "):
            tree = _oid(line[5:], "commit tree")
        if line.startswith(b"committer "):
            parts = line.rsplit(b" ", 2)
            if len(parts) == 3 and parts[1].isdigit():
                timestamp = int(parts[1])
    if tree != expected_tree or timestamp is None:
        raise CandidateSourceArchiveError(
            "commit tree or timestamp differs from candidate identity"
        )
    return timestamp


def _refuse_info_attributes(executor: _GitExecutor, metadata: _GitMetadataAuthority) -> None:
    path = executor.run(
        "rev-parse", "--path-format=absolute", "--git-path", "info/attributes", stdout_limit=32768
    )
    if path.count(b"\n") != 1 or not path.endswith(b"\n"):
        raise CandidateSourceArchiveError("could not resolve one absolute Git info attributes path")
    info = Path(_text(path[:-1], "Git info attributes"))
    metadata.assert_info_attributes(info)
    if info.exists() and info.read_bytes():
        raise CandidateSourceArchiveError("Git info attributes must be empty")


class _GitMetadataAuthority:
    """Pinned local Git metadata roots, including the standard linked-worktree form."""

    def __init__(self, executor: _GitExecutor, root: Path) -> None:
        self._executor = executor
        self._root = root
        self.git_dir = self._path("--git-dir")
        self.common_dir = self._path("--git-common-dir")
        self.objects = self._path("--git-path", "objects")
        self.config = self._path("--git-path", "config")
        for item in (self.git_dir, self.common_dir, self.objects, self.config):
            _require_local_nonreparse_path(item)
        if self.objects != self.common_dir / "objects" or self.config != self.common_dir / "config":
            raise CandidateSourceArchiveError("Git object or config routing is not standard")
        dot_git = root / ".git"
        if dot_git.is_dir():
            if self.git_dir != dot_git or self.common_dir != self.git_dir:
                raise CandidateSourceArchiveError("unexpected non-linked Git metadata routing")
        elif dot_git.is_file():
            raw = dot_git.read_bytes()
            expected = b"gitdir: " + self.git_dir.as_posix().encode("utf-8") + b"\n"
            if (
                raw != expected
                or self.git_dir.parent.name != "worktrees"
                or self.git_dir.parent.parent != self.common_dir
            ):
                raise CandidateSourceArchiveError("linked worktree gitfile routing is not standard")
        else:
            raise CandidateSourceArchiveError("candidate lacks a standard Git metadata entry")
        self._controls = (
            self.config,
            self.git_dir / "HEAD",
            self.git_dir / "index",
            self.git_dir / "commondir",
            self.common_dir / "packed-refs",
            self.common_dir / "info" / "attributes",
            self.git_dir / "info" / "attributes",
            self.common_dir / "objects" / "info" / "alternates",
            self.common_dir / "info" / "grafts",
            self.git_dir / "shallow",
        )
        self._seal = self._snapshot()
        self._reject_routing_files()

    def _path(self, *arguments: str) -> Path:
        value = self._executor.run(
            "rev-parse", "--path-format=absolute", *arguments, stdout_limit=32768
        )
        if value.count(b"\n") != 1 or not value.endswith(b"\n"):
            raise CandidateSourceArchiveError("Git metadata path is not one canonical line")
        path = Path(_text(value[:-1], "Git metadata path"))
        if not PureWindowsPath(str(path)).is_absolute():
            raise CandidateSourceArchiveError("Git metadata path is not absolute")
        return path

    @staticmethod
    def _lstat(path: Path) -> os.stat_result | None:
        """Inspect controls without following a link or Windows reparse point."""
        try:
            observed = os.lstat(path)
        except FileNotFoundError:
            return None
        attributes = int(getattr(observed, "st_file_attributes", 0))
        if stat.S_ISLNK(observed.st_mode) or attributes & _FILE_ATTRIBUTE_REPARSE_POINT:
            raise CandidateSourceArchiveError("Git metadata control is link or reparse-mediated")
        return observed

    @staticmethod
    def _identity(observed: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            observed.st_dev,
            observed.st_ino,
            observed.st_mode,
            observed.st_size,
            observed.st_mtime_ns,
        )

    def _snapshot(self) -> tuple[tuple[object, ...], ...]:
        """Seal each control or the exact existing parent that proves its absence.

        This is deliberately cooperative same-SID protection: lstat identities and
        bytes catch replacement/creation between checks, but cannot defeat a peer
        with equal authority racing every filesystem operation.
        """
        result: list[tuple[object, ...]] = []
        for control in self._controls:
            observed = self._lstat(control)
            if observed is not None:
                if not stat.S_ISREG(observed.st_mode):
                    raise CandidateSourceArchiveError("Git metadata control is not a regular file")
                result.append(
                    (
                        str(control),
                        self._identity(observed),
                        hashlib.sha256(control.read_bytes()).hexdigest(),
                    )
                )
                continue
            parent = control.parent
            parent_observed = self._lstat(parent)
            while parent_observed is None and parent != parent.parent:
                parent = parent.parent
                parent_observed = self._lstat(parent)
            if parent_observed is None or not stat.S_ISDIR(parent_observed.st_mode):
                raise CandidateSourceArchiveError("Git metadata control lacks a stable existing parent")
            _require_local_nonreparse_path(parent)
            result.append((str(control), None, str(parent), self._identity(parent_observed)))
        return tuple(result)

    def _reject_routing_files(self) -> None:
        for item in (
            self.common_dir / "objects" / "info" / "alternates",
            self.common_dir / "info" / "grafts",
            self.git_dir / "shallow",
        ):
            if self._lstat(item) is not None and item.read_bytes().strip():
                raise CandidateSourceArchiveError(
                    "Git metadata alternate, graft, or shallow routing is forbidden"
                )

    def assert_info_attributes(self, path: Path) -> None:
        if not any(
            _normal_final_path(str(path)).startswith(_normal_final_path(str(root)) + os.sep)
            or _normal_final_path(str(path)) == _normal_final_path(str(root))
            for root in (self.git_dir, self.common_dir)
        ):
            raise CandidateSourceArchiveError("Git info attributes escaped metadata authority")
        _require_local_nonreparse_path(path.parent)

    def revalidate(self) -> None:
        try:
            current = _GitMetadataAuthority(self._executor, self._root)
        except CandidateSourceArchiveError as error:
            raise CandidateSourceArchiveError("Git metadata authority drifted") from error
        if (
            current.git_dir,
            current.common_dir,
            current.objects,
            current.config,
            current._seal,
        ) != (self.git_dir, self.common_dir, self.objects, self.config, self._seal):
            raise CandidateSourceArchiveError("Git metadata authority drifted")


def _verify_reachable_objects(executor: _GitExecutor, head: str) -> None:
    """Require Git's bounded connectivity check in addition to manual SHA-1 reads."""

    if executor.run(
        "fsck",
        "--connectivity-only",
        "--no-dangling",
        "--no-progress",
        "--",
        head,
        stdout_limit=_MAX_STDERR_BYTES,
    ):
        raise CandidateSourceArchiveError(
            "Git object connectivity check produced unexpected output"
        )


def _verify_retained_version(executor: _GitExecutor, pin: GitExecutablePinV1 | _OwnedGitFileBindingV1) -> None:
    observed = executor.run("--version", stdout_limit=4096)
    expected = pin.version.encode("ascii", "strict") + b"\n"
    if observed != expected:
        raise CandidateSourceArchiveError("retained Git executable version differs from pin")


def _require_equal_repeated_archives(first: bytes, second: bytes) -> bytes:
    """Tiny private repeat boundary: no token can exist for unequal captures."""
    if not hmac.compare_digest(first, second):
        raise CandidateSourceArchiveError("repeated Git archives differ")
    return first


def capture_candidate_source_archive(
    checkout_root: Path, identity: CandidateIdentityV1, git_pin: GitExecutablePinV1
) -> VerifiedCandidateSourceArchiveV1:
    """Mint opaque authority only for two equal validated in-memory commit archives."""
    if (
        type(checkout_root) is not _PATH_TYPE
        or type(identity) is not CandidateIdentityV1
        or type(git_pin) is not GitExecutablePinV1
    ):
        raise TypeError("candidate archive inputs require exact V1 types")
    _validate_supplied_identity(identity)
    retained = _RetainedGitExecutableV1(git_pin)
    return _capture_retained_source_archive(checkout_root, identity, git_pin, retained)


def capture_candidate_source_archive_from_tools(
    checkout_root: Path,
    identity: CandidateIdentityV1,
    tools: ImmutableToolEnvironmentV1,
) -> VerifiedCandidateSourceArchiveV1:
    """Capture through the existing Git owner while its complete tool tree is sealed."""
    if type(checkout_root) is not _PATH_TYPE or type(identity) is not CandidateIdentityV1:
        raise TypeError("owned-tool archive inputs require exact V1 types")
    _validate_supplied_identity(identity)
    from scripts.qualification_tool_environment import tool_environment_metadata

    distributions = tool_environment_metadata(tools)
    retained = _RetainedGitExecutableV1.from_owned_tools(tools)
    captured = _capture_retained_source_archive(checkout_root, identity, retained._pin, retained)
    if tool_environment_metadata(tools) != distributions:
        raise CandidateSourceArchiveError("archive tool capture environment changed")
    _TOOL_CAPTURES[captured] = distributions
    return captured


def _capture_retained_source_archive(
    checkout_root: Path,
    identity: CandidateIdentityV1,
    git_pin: GitExecutablePinV1 | _OwnedGitFileBindingV1,
    retained: _RetainedGitExecutableV1,
) -> VerifiedCandidateSourceArchiveV1:
    minted: VerifiedCandidateSourceArchiveV1 | None = None
    failure: BaseException | None = None
    try:
        executor = _GitExecutor(retained, checkout_root)
        _verify_retained_version(executor, git_pin)
        _require_root_and_identity(executor, checkout_root, identity)
        metadata_authority = _GitMetadataAuthority(executor, checkout_root)
        _refuse_info_attributes(executor, metadata_authority)
        _verify_reachable_objects(executor, identity.candidate_head_oid)
        commit = _git_object(executor, "commit", identity.candidate_head_oid)
        timestamp = _commit_timestamp(commit, identity.candidate_tree_oid)
        manifest, payloads = _visible_manifest(executor, identity.candidate_tree_oid)
        archives: list[bytes] = []
        for _ in range(2):
            metadata_authority.revalidate()
            _require_root_and_identity(executor, checkout_root, identity)
            _refuse_info_attributes(executor, metadata_authority)
            archive = executor.run(
                "archive", "--format=tar", f"--prefix={_PREFIX}/", identity.candidate_head_oid
            )
            if not archive:
                raise CandidateSourceArchiveError("Git archive output is empty")
            _validate_tar(archive, identity, manifest, payloads, timestamp)
            archives.append(archive)
            metadata_authority.revalidate()
            _require_root_and_identity(executor, checkout_root, identity)
            _refuse_info_attributes(executor, metadata_authority)
        archive = _require_equal_repeated_archives(archives[0], archives[1])
        metadata_authority.revalidate()
        _require_root_and_identity(executor, checkout_root, identity)
        _refuse_info_attributes(executor, metadata_authority)
        metadata = CandidateSourceArchiveMetadataV1(
            identity.candidate_head_oid,
            identity.candidate_tree_oid,
            identity.canonical_baseline_oid,
            identity.canonical_diff_sha256,
            _PREFIX,
            hashlib.sha256(archive).hexdigest(),
            len(archive),
            manifest,
        )
        minted = object.__new__(VerifiedCandidateSourceArchiveV1)
        _RECORDS[minted] = _VerifiedArchiveRecord(identity, metadata, archive)
    except BaseException as error:
        failure = error
    try:
        retained.close()
    except BaseException as close_error:
        if failure is None:
            failure = close_error
        else:
            failure = BaseExceptionGroup(
                "candidate archive capture and retained handle close both failed",
                [failure, close_error],
            )
    if failure is not None:
        raise failure
    assert minted is not None
    return minted

"""Own an immutable Windows snapshot of selected executable/import file bytes.

This is a filesystem capability, not tool authentication, installation evidence,
or a process owner. A trusted consumer must stop its entire owned process tree
before leaving the context. No existing directory is adopted or modified.
"""

from __future__ import annotations

import ctypes as c
import hashlib
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from weakref import WeakKeyDictionary

from scripts.qualification_file_seals import (
    RetainedFileSealsV1,
    _info,
    _relative_windows_member,
    _retain,
    retain_file_seals,
    sealed_file_metadata,
)
from scripts.windows_storage_oracle import _api, _identity, _user_sid

_MAX_FILES = 16384
_MAX_DIRECTORIES = 16384
_MAX_ENTRIES = _MAX_FILES + _MAX_DIRECTORIES
_MAX_FILE_BYTES = 512 * 1024**2
_MAX_TOTAL_BYTES = 4 * 1024**3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ImmutableExecutionFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("execution file snapshots are owner-minted only")


@dataclass(frozen=True, slots=True)
class _Snapshot:
    root: Path
    seals: RetainedFileSealsV1
    members: tuple[str, ...]
    directories: tuple[str, ...]
    controls: dict[str, _SecurityControl]


@dataclass(frozen=True, slots=True)
class _SecurityFact:
    owner_sha256: str
    protected: bool
    dacl_sha256: str
    dacl_size: int


@dataclass(frozen=True, slots=True)
class _SecurityControl:
    handle: int
    identity: tuple[int, int]
    directory: bool
    frozen: _SecurityFact


_LIVE: WeakKeyDictionary[ImmutableExecutionFilesV1, _Snapshot] = WeakKeyDictionary()


def _snapshot(receipt: ImmutableExecutionFilesV1) -> _Snapshot:
    if type(receipt) is not ImmutableExecutionFilesV1:
        raise TypeError("execution file capability type differs")
    _require(receipt in _LIVE, "execution file capability is closed or unregistered")
    value = _LIVE[receipt]
    sealed_file_metadata(value.seals)
    _validate_snapshot(value)
    return value


class _Acl(c.Structure):
    _fields_ = [
        ("revision", c.c_ubyte),
        ("reserved", c.c_ubyte),
        ("size", c.c_ushort),
        ("ace_count", c.c_ushort),
        ("reserved_two", c.c_ushort),
    ]


def _security_sddl(*, directory: bool, frozen: bool) -> str:
    deny_control = "(D;;WDWO;;;OW)(D;;WDWO;;;WD)" if frozen else ""
    deny_namespace = "(D;;0x46;;;WD)" if directory and frozen else ""
    return (
        f"D:P{deny_control}{deny_namespace}"
        f"(A;;FA;;;{_user_sid()})(A;;FA;;;SY)(A;;FA;;;BA)"
    )


def _descriptor_dacl(sddl: str) -> tuple[c.c_void_p, c.c_void_p]:
    kernel, security = _api()
    pointer, word = c.c_void_p, c.c_uint32
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        c.c_wchar_p,
        word,
        pointer,
        pointer,
    ]
    security.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = c.c_int
    security.GetSecurityDescriptorDacl.argtypes = [pointer, pointer, pointer, pointer]
    security.GetSecurityDescriptorDacl.restype = c.c_int
    security.SetSecurityInfo.argtypes = [pointer, c.c_int, word, pointer, pointer, pointer, pointer]
    security.SetSecurityInfo.restype = word
    descriptor, dacl = pointer(), pointer()
    present, defaulted = c.c_int(), c.c_int()
    _require(
        bool(
            security.ConvertStringSecurityDescriptorToSecurityDescriptorW(
                sddl,
                1,
                c.byref(descriptor),
                None,
            )
        ),
        "execution security descriptor is unavailable",
    )
    valid = bool(
        security.GetSecurityDescriptorDacl(
            descriptor,
            c.byref(present),
            c.byref(dacl),
            c.byref(defaulted),
        )
    ) and bool(present.value) and bool(dacl.value)
    if not valid:
        _require(kernel.LocalFree(descriptor) is None, "execution descriptor did not close")
        _require(False, "execution DACL is absent")
    return descriptor, dacl


def _set_security(handle: int, sddl: str) -> _SecurityFact:
    """Apply a protected DACL through a controller handle opened before freezing."""
    kernel, security = _api()
    pointer, word = c.c_void_p, c.c_uint32
    security.SetSecurityInfo.argtypes = [pointer, c.c_int, word, pointer, pointer, pointer, pointer]
    security.SetSecurityInfo.restype = word
    descriptor, dacl = _descriptor_dacl(sddl)
    try:
        acl_size = int(c.cast(dacl, c.POINTER(_Acl)).contents.size)
        _require(
            c.sizeof(_Acl) <= acl_size <= 65535,
            "execution DACL exceeds bounds",
        )
        expected = hashlib.sha256(c.string_at(dacl, acl_size)).hexdigest()
        _require(
            security.SetSecurityInfo(handle, 1, 0x80000004, None, None, dacl, None) == 0,
            "execution DACL could not be applied",
        )
    finally:
        _require(kernel.LocalFree(descriptor) is None, "execution descriptor did not close")
    fact = _security_fact(handle)
    _require(
        fact.protected and fact.dacl_size == acl_size and fact.dacl_sha256 == expected,
        "execution DACL differs",
    )
    return fact


def _security_fact(handle: int) -> _SecurityFact:
    """Read only the private owner/protected-DACL facts needed for equality."""
    kernel, security = _api()
    pointer, word = c.c_void_p, c.c_uint32
    security.GetSecurityInfo.argtypes = [
        pointer,
        c.c_int,
        word,
        pointer,
        pointer,
        pointer,
        pointer,
        pointer,
    ]
    security.GetSecurityInfo.restype = word
    security.GetSecurityDescriptorControl.argtypes = [pointer, pointer, pointer]
    security.GetSecurityDescriptorControl.restype = c.c_int
    security.IsValidSid.argtypes = [pointer]
    security.IsValidSid.restype = c.c_int
    security.GetLengthSid.argtypes = [pointer]
    security.GetLengthSid.restype = word
    owner, dacl, descriptor = pointer(), pointer(), pointer()
    result = security.GetSecurityInfo(
        handle,
        1,
        0x5,
        c.byref(owner),
        None,
        c.byref(dacl),
        None,
        c.byref(descriptor),
    )
    try:
        _require(
            result == 0
            and bool(owner.value)
            and bool(dacl.value)
            and bool(descriptor.value),
            "execution security facts are unavailable",
        )
        control, revision = c.c_ushort(), word()
        _require(
            bool(
                security.GetSecurityDescriptorControl(
                    descriptor, c.byref(control), c.byref(revision)
                )
            )
            and bool(security.IsValidSid(owner)),
            "execution security facts are invalid",
        )
        owner_size = int(security.GetLengthSid(owner))
        acl_size = int(c.cast(dacl, c.POINTER(_Acl)).contents.size)
        _require(
            0 < owner_size <= 68 and c.sizeof(_Acl) <= acl_size <= 65535,
            "execution security facts exceed bounds",
        )
        return _SecurityFact(
            hashlib.sha256(c.string_at(owner, owner_size)).hexdigest(),
            bool(control.value & 0x1000),
            hashlib.sha256(c.string_at(dacl, acl_size)).hexdigest(),
            acl_size,
        )
    finally:
        if descriptor.value:
            _require(
                kernel.LocalFree(descriptor) is None,
                "execution security facts did not close",
            )


def _open_security_control(path: Path, *, directory: bool) -> tuple[int, tuple[int, int]]:
    kernel, _ = _api()
    handle = kernel.CreateFileW(str(path), 0x60080, 3, None, 3, 0x02200000, None)
    _require(handle not in (None, c.c_void_p(-1).value), "execution security handle failed")
    try:
        flags = c.c_uint32()
        kernel.GetHandleInformation.argtypes = [c.c_void_p, c.c_void_p]
        kernel.GetHandleInformation.restype = c.c_int
        info = _info(handle)
        _require(
            not info.attributes & 0x400
            and bool(info.attributes & 0x10) == directory
            and bool(kernel.GetHandleInformation(handle, c.byref(flags)))
            and not flags.value & 1,
            "execution security handle differs",
        )
        return int(handle), _identity(info)
    except BaseException:
        _require(bool(kernel.CloseHandle(handle)), "execution security handle did not close")
        raise


def _security(path: Path, *, directory: bool, frozen: bool) -> None:
    """Apply the writable DACL used by a separate, already-owned workspace."""
    _require(not frozen, "frozen execution security requires a retained controller handle")
    kernel, _ = _api()
    handle, _ = _open_security_control(path, directory=directory)
    primary: BaseException | None = None
    try:
        _set_security(handle, _security_sddl(directory=directory, frozen=False))
    except BaseException as error:
        primary = error
    close_error: BaseException | None = None
    try:
        _require(bool(kernel.CloseHandle(handle)), "execution security handle did not close")
    except BaseException as error:
        close_error = error
    if primary is not None and close_error is not None:
        raise BaseExceptionGroup(
            "execution security and handle cleanup failed", [primary, close_error]
        )
    if primary is not None:
        raise primary
    if close_error is not None:
        raise close_error


def _namespace(snapshot: _Snapshot) -> None:
    expected = {
        **{name: True for name in snapshot.directories},
        **{name: False for name in snapshot.members},
    }
    observed: set[str] = set()
    pending = [("", snapshot.root)]
    while pending:
        parent_name, parent = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                _require(len(observed) < len(expected), "execution snapshot namespace differs")
                name = f"{parent_name}/{entry.name}" if parent_name else entry.name
                _require(
                    name in expected and name not in observed,
                    "execution snapshot namespace differs",
                )
                info = entry.stat(follow_symlinks=False)
                directory = stat.S_ISDIR(info.st_mode)
                _require(
                    not getattr(info, "st_file_attributes", 0) & 0x400
                    and (directory or stat.S_ISREG(info.st_mode))
                    and directory == expected[name],
                    "execution snapshot namespace differs",
                )
                observed.add(name)
                if directory:
                    pending.append((name, Path(entry.path)))
    _require(observed == set(expected), "execution snapshot namespace differs")


def _validate_snapshot(snapshot: _Snapshot) -> None:
    _namespace(snapshot)
    kernel, _ = _api()
    for control in snapshot.controls.values():
        flags = c.c_uint32()
        info = _info(control.handle)
        _require(
            _identity(info) == control.identity
            and not info.attributes & 0x400
            and bool(info.attributes & 0x10) == control.directory
            and bool(kernel.GetHandleInformation(control.handle, c.byref(flags)))
            and not flags.value & 1,
            "execution snapshot identity differs",
        )
        _require(
            _security_fact(control.handle) == control.frozen,
            "execution snapshot security differs",
        )


@contextmanager
def owned_execution_files(contents: dict[str, bytes]) -> Iterator[ImmutableExecutionFilesV1]:
    """Copy a bounded selection into a fresh snapshot, never authorize its execution."""
    _require(os.name == "nt", "execution snapshots require Windows")
    _require(
        type(contents) is dict and 0 < len(contents) <= _MAX_FILES,
        "execution snapshot file set exceeds its bound",
    )
    _require(all(type(name) is str for name in contents), "execution snapshot names differ")
    # Freeze the caller's mapping before validation and filesystem work. Each
    # value is already required to be immutable bytes.
    contents = dict(contents)
    members = tuple(sorted(contents))
    directories: set[str] = set()
    total = 0
    for name in members:
        _relative_windows_member(name)
        raw = contents[name]
        _require(
            type(raw) is bytes and len(raw) <= _MAX_FILE_BYTES,
            "execution snapshot file bytes exceed bounds",
        )
        total += len(raw)
        for relative_parent in PurePosixPath(name).parents:
            if str(relative_parent) != ".":
                directories.add(str(relative_parent))
                _require(
                    len(directories) <= _MAX_DIRECTORIES,
                    "execution snapshot directory set exceeds its bound",
                )
    _require(total <= _MAX_TOTAL_BYTES, "execution snapshot aggregate exceeds bounds")
    _require(
        len(members) + len(directories) <= _MAX_ENTRIES,
        "execution snapshot namespace exceeds its bound",
    )
    folded = {name.casefold() for name in members}
    _require(
        len(folded) == len(members)
        and len({name.casefold() for name in directories}) == len(directories)
        and not folded.intersection(name.casefold() for name in directories),
        "execution snapshot namespace is ambiguous",
    )
    parent = Path(tempfile.gettempdir()).resolve(strict=True)
    root = Path(tempfile.mkdtemp(prefix="hermes-execution-files-", dir=parent)).resolve(strict=True)
    _require(root.parent == parent, "execution snapshot escaped its owned parent")
    opened: dict[str, tuple[int, tuple[int, int], bool]] = {}
    controls: dict[str, _SecurityControl] = {}
    root_identity: tuple[int, int] | None = None
    primary: BaseException | None = None
    primary_traceback = None
    try:
        with _retain(root, directory=True) as (_, initial):
            root_identity = _identity(initial)
        handle, identity = _open_security_control(root, directory=True)
        opened[""] = (handle, identity, True)
        _set_security(handle, _security_sddl(directory=True, frozen=False))
        for name in sorted(directories, key=lambda item: (item.count("/"), item)):
            path = root.joinpath(*name.split("/"))
            path.mkdir()
            handle, identity = _open_security_control(path, directory=True)
            opened[name] = (handle, identity, True)
            _set_security(handle, _security_sddl(directory=True, frozen=False))
        for name in members:
            path = root.joinpath(*name.split("/"))
            with path.open("xb") as output:
                output.write(contents[name])
            handle, identity = _open_security_control(path, directory=False)
            opened[name] = (handle, identity, False)
            _set_security(handle, _security_sddl(directory=False, frozen=False))
        for name, (handle, identity, directory) in opened.items():
            frozen = _set_security(handle, _security_sddl(directory=directory, frozen=True))
            controls[name] = _SecurityControl(
                handle,
                identity,
                directory,
                frozen,
            )
        with retain_file_seals(root, members) as seals:
            _require(
                sealed_file_metadata(seals)
                == tuple(
                    (name, hashlib.sha256(contents[name]).hexdigest(), len(contents[name]))
                    for name in members
                ),
                "execution snapshot copied bytes differ",
            )
            receipt = object.__new__(ImmutableExecutionFilesV1)
            value = _Snapshot(root, seals, members, tuple(sorted(directories)), controls)
            _validate_snapshot(value)
            _LIVE[receipt] = value
            try:
                yield receipt
                _validate_snapshot(value)
            finally:
                del _LIVE[receipt]
    except BaseException as error:
        primary = error
        primary_traceback = error.__traceback__

    failures: list[BaseException] = []
    safe_target = root.parent == parent
    if "" in opened:
        try:
            safe_target = (
                safe_target
                and root.resolve(strict=True) == root
                and _identity(_info(opened[""][0])) == opened[""][1]
            )
            _require(safe_target, "execution cleanup identity differs")
        except BaseException as error:
            safe_target = False
            failures.append(error)
    elif root_identity is not None:
        try:
            with _retain(root, directory=True) as (_, current):
                safe_target = (
                    safe_target
                    and root.resolve(strict=True) == root
                    and _identity(current) == root_identity
                )
            _require(safe_target, "execution cleanup identity differs")
        except BaseException as error:
            safe_target = False
            failures.append(error)
    else:
        safe_target = False
        failures.append(ValueError("execution cleanup root handle is absent"))
    for handle, _, directory in opened.values():
        try:
            _set_security(handle, _security_sddl(directory=directory, frozen=False))
        except BaseException as error:
            failures.append(error)
    kernel, _ = _api()
    for name, (handle, _, _) in reversed(tuple(opened.items())):
        try:
            _require(bool(kernel.CloseHandle(handle)), "execution security handle did not close")
        except BaseException as error:
            failures.append(error)
        else:
            del opened[name]
    if safe_target:
        try:
            shutil.rmtree(root)
            _require(not root.exists(), "execution snapshot cleanup is incomplete")
        except BaseException as error:
            failures.append(error)
    if primary is not None and failures:
        raise BaseExceptionGroup(
            "execution snapshot primary and cleanup failures", [primary, *failures]
        )
    if failures:
        raise BaseExceptionGroup("execution snapshot cleanup failures", failures)
    if primary is not None:
        raise primary.with_traceback(primary_traceback)


def execution_file_metadata(
    receipt: ImmutableExecutionFilesV1,
) -> tuple[tuple[str, str, int], ...]:
    """Private relative-member inventory, not a public report or installed receipt."""
    metadata: tuple[tuple[str, str, int], ...] = sealed_file_metadata(_snapshot(receipt).seals)
    return metadata


def _execution_files_for_consumer(receipt: ImmutableExecutionFilesV1) -> Path:
    return _snapshot(receipt).root

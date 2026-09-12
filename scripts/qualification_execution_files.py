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


_LIVE: WeakKeyDictionary[ImmutableExecutionFilesV1, _Snapshot] = WeakKeyDictionary()


def _snapshot(receipt: ImmutableExecutionFilesV1) -> _Snapshot:
    if type(receipt) is not ImmutableExecutionFilesV1:
        raise TypeError("execution file capability type differs")
    _require(receipt in _LIVE, "execution file capability is closed or unregistered")
    value = _LIVE[receipt]
    sealed_file_metadata(value.seals)
    return value


def _security(path: Path, *, directory: bool, frozen: bool) -> None:
    """Set a protected, noninheriting ACL on one newly owned object only."""
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
    deny = "(D;;0x46;;;WD)" if directory and frozen else ""
    sddl = f"D:P{deny}(A;;FA;;;{_user_sid()})(A;;FA;;;SY)(A;;FA;;;BA)"
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
    try:
        _require(
            bool(
                security.GetSecurityDescriptorDacl(
                    descriptor,
                    c.byref(present),
                    c.byref(dacl),
                    c.byref(defaulted),
                )
            )
            and bool(present.value)
            and bool(dacl.value),
            "execution DACL is absent",
        )
        handle = kernel.CreateFileW(str(path), 0x60080, 3, None, 3, 0x02200000, None)
        _require(handle not in (None, c.c_void_p(-1).value), "execution security handle failed")
        try:
            info = _info(handle)
            _require(
                not info.attributes & 0x400 and bool(info.attributes & 0x10) == directory,
                "execution security target is indirect",
            )
            _require(
                security.SetSecurityInfo(handle, 1, 0x80000004, None, None, dacl, None) == 0,
                "execution DACL could not be applied",
            )
        finally:
            _require(bool(kernel.CloseHandle(handle)), "execution security handle did not close")
    finally:
        _require(kernel.LocalFree(descriptor) is None, "execution descriptor did not close")


@contextmanager
def owned_execution_files(contents: dict[str, bytes]) -> Iterator[ImmutableExecutionFilesV1]:
    """Copy a bounded selection into a fresh snapshot, never authorize its execution."""
    _require(os.name == "nt", "execution snapshots require Windows")
    _require(
        type(contents) is dict and 0 < len(contents) <= 16384,
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
            type(raw) is bytes and len(raw) <= 512 * 1024**2,
            "execution snapshot file bytes exceed bounds",
        )
        total += len(raw)
        for relative_parent in PurePosixPath(name).parents:
            if str(relative_parent) != ".":
                directories.add(str(relative_parent))
    _require(total <= 4 * 1024**3, "execution snapshot aggregate exceeds bounds")
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
    created: list[Path] = [root]
    with _retain(root, directory=True) as (_, initial):
        root_identity = _identity(initial)
    try:
        _security(root, directory=True, frozen=False)
        for name in sorted(directories, key=lambda item: (item.count("/"), item)):
            path = root.joinpath(*name.split("/"))
            path.mkdir()
            created.append(path)
            _security(path, directory=True, frozen=False)
        for name in members:
            path = root.joinpath(*name.split("/"))
            with path.open("xb") as output:
                output.write(contents[name])
            _security(path, directory=False, frozen=False)
        for path in created:
            _security(path, directory=True, frozen=True)
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
            _LIVE[receipt] = _Snapshot(root, seals)
            try:
                yield receipt
            finally:
                del _LIVE[receipt]
    finally:
        # Only this fresh tree is disposable; caller-selected existing paths are
        # never cleanup targets. Check its resolved location and retained identity.
        _require(
            root.resolve(strict=True) == root and root.parent == parent,
            "execution cleanup target moved",
        )
        with _retain(root, directory=True) as (_, current):
            _require(_identity(current) == root_identity, "execution cleanup identity differs")
            for path in created:
                _security(path, directory=True, frozen=False)
        shutil.rmtree(root)
        _require(not root.exists(), "execution snapshot cleanup is incomplete")


def execution_file_metadata(
    receipt: ImmutableExecutionFilesV1,
) -> tuple[tuple[str, str, int], ...]:
    """Private relative-member inventory, not a public report or installed receipt."""
    return sealed_file_metadata(_snapshot(receipt).seals)


def _execution_files_for_consumer(receipt: ImmutableExecutionFilesV1) -> Path:
    return _snapshot(receipt).root

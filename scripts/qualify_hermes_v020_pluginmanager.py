#!/usr/bin/env python3
# ruff: noqa: E501, E701, E702, SIM115
"""Qualify hermes-realtime through Hermes v0.20's real PluginManager."""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import re
import secrets
import stat
import tarfile
import threading
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, NoReturn, Protocol, cast

if TYPE_CHECKING:
    from scripts.source_archive_authority import (
        WindowsSourceWorkspaceAuthorityV1 as _WindowsSourceWorkspaceAuthorityV1,
    )

try:
    _source_archive_authority = importlib.import_module("scripts.source_archive_authority")
except ModuleNotFoundError:
    _source_archive_authority = importlib.import_module("source_archive_authority")
SourceArchivePolicyV1 = _source_archive_authority.SourceArchivePolicyV1
WindowsSourceWorkspaceAuthorityV1 = _source_archive_authority.WindowsSourceWorkspaceAuthorityV1
archive_member_validator = _source_archive_authority.archive_member_validator

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

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

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("internal", ctypes.c_size_t),
            ("internal_high", ctypes.c_size_t),
            ("offset", wintypes.DWORD),
            ("offset_high", wintypes.DWORD),
            ("event", wintypes.HANDLE),
        ]

    class _UNICODE_STRING(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.USHORT),
            ("maximum_length", wintypes.USHORT),
            ("buffer", wintypes.LPWSTR),
        ]

    class _OBJECT_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("length", wintypes.ULONG),
            ("root_directory", wintypes.HANDLE),
            ("object_name", ctypes.POINTER(_UNICODE_STRING)),
            ("attributes", wintypes.ULONG),
            ("security_descriptor", wintypes.LPVOID),
            ("security_quality_of_service", wintypes.LPVOID),
        ]

    class _IO_STATUS_BLOCK(ctypes.Structure):
        _fields_ = [("status", ctypes.c_long), ("information", ctypes.c_size_t)]

    class _FILE_DISPOSITION_INFO(ctypes.Structure):
        _fields_ = [("delete_file", wintypes.BOOL)]


def _normal_final_path(value: str) -> str:
    value = (
        value.removeprefix("\\\\?\\UNC\\")
        if value.startswith("\\\\?\\UNC\\")
        else value.removeprefix("\\\\?\\")
    )
    if value.startswith("UNC\\"):
        value = "\\\\" + value[4:]
    return os.path.normcase(os.path.normpath(value))


def _native_handle_info(handle: int) -> tuple[int, int, str, int, int]:
    """Return volume/file identity, exact final path, attributes and size."""
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
    file_id = (int(info.file_index_high) << 32) | int(info.file_index_low)
    size = (int(info.size_high) << 32) | int(info.size_low)
    return (
        int(info.volume_serial),
        file_id,
        _normal_final_path(target.value),
        int(info.attributes),
        size,
    )


_PRIVATE_RESULT_KEYS = {
    "activeProfileUnchanged",
    "defaultRootUnchanged",
    "disabledEntryPointDiscovered",
    "disabledImportOrRegistration",
    "distributionVersion",
    "enabledEntryPointDiscovered",
    "evidenceRootUnchanged",
    "exactConfigDelta",
    "pluginContextRegistrationObserved",
}
_RAW_IDENTIFIER_FRAGMENTS = ("/", "\\", ":", "users", "profile", "evidence", "cwd")
_MAX_SNAPSHOT_FILE_BYTES = 16 * 1024 * 1024
_MAX_SNAPSHOT_TREE_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_ARCHIVE_MEMBERS = 100_000
_MAX_HERMES_SOURCE_ARCHIVE_BYTES = 256 * 1024 * 1024


def _identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _exact_int(value: object, *, label: str = "native identity") -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _reject_link_or_reparse(path: Path, metadata: os.stat_result) -> None:
    if path.is_symlink() or (getattr(metadata, "st_file_attributes", 0) & 0x00000400):
        raise ValueError("snapshot tree contains a link or reparse point")


def canonical_child_result(value: object) -> bytes:
    validated = _validate_private_result(value)
    payload = (
        json.dumps(
            validated, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    if len(payload) > 4096:
        raise ValueError("private child result is oversized")
    return payload


def parse_child_result(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 4096:
        raise ValueError("private child result is absent or oversized")
    try:
        value = json.loads(
            payload.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ValueError())
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("private child result is malformed") from error
    validated = _validate_private_result(value)
    if canonical_child_result(validated) != payload:
        raise ValueError("private child result is not canonical")
    return validated


def _validate_private_result(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _PRIVATE_RESULT_KEYS:
        raise ValueError("private child result has an invalid closed shape")
    if value["distributionVersion"] != "0.0.3":
        raise ValueError("private child candidate version is invalid")
    if type(value["distributionVersion"]) is not str or any(
        fragment in value["distributionVersion"].casefold()
        for fragment in _RAW_IDENTIFIER_FRAGMENTS
    ):
        raise ValueError("private child result contains a raw identifier")
    expected_true = _PRIVATE_RESULT_KEYS - {"distributionVersion", "disabledImportOrRegistration"}
    if any(value[key] is not True for key in expected_true):
        raise ValueError("private child result did not prove every required predicate")
    if value["disabledImportOrRegistration"] is not False:
        raise ValueError("disabled sweep imported or registered the plugin")
    return value


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def snapshot_regular_tree(root: Path) -> dict[str, tuple[object, ...]]:
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("snapshot root must be an existing absolute directory")
    root_lstat = root.lstat()
    _reject_link_or_reparse(root, root_lstat)
    root_metadata = root.stat()
    if not stat.S_ISDIR(root_metadata.st_mode) or _identity(root_lstat) != _identity(root_metadata):
        raise ValueError("snapshot root must be an existing absolute directory")
    root_identity = _identity(root_metadata)

    def directory_seal(path: Path) -> tuple[object, ...]:
        if os.name != "nt":
            metadata = path.stat()
            return ("dir", metadata.st_dev, metadata.st_ino)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel.CreateFileW(
            str(path), 0x80000000, 0x00000001, None, 3, 0x02000000 | 0x00200000, None
        )
        value = int(ctypes.cast(handle, ctypes.c_void_p).value or 0) if handle else 0
        if not value or value == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            volume, file_id, final, attributes, _size = _native_handle_info(value)
            if not attributes & 0x10 or attributes & 0x400:
                raise ValueError("snapshot directory is not a non-reparse directory")
            return ("dir", volume, file_id, final, attributes)
        finally:
            if not kernel.CloseHandle(wintypes.HANDLE(value)):
                raise ctypes.WinError(ctypes.get_last_error())

    snapshot: dict[str, tuple[object, ...]] = {".": directory_seal(root)}
    directory_identities: dict[Path, tuple[int, int]] = {root: root_identity}
    seen: set[str] = set()
    tree_bytes = 0
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        expected_current = directory_identities.get(current_path)
        current_lstat = current_path.lstat()
        _reject_link_or_reparse(current_path, current_lstat)
        current_stat = current_path.stat()
        if (
            expected_current is None
            or _identity(current_lstat) != expected_current
            or _identity(current_stat) != expected_current
        ):
            raise ValueError("snapshot directory identity changed")
        for name in [*directory_names, *file_names]:
            if ":" in name:
                raise ValueError("snapshot tree contains an alternate data stream spelling")
            candidate = current_path / name
            metadata = candidate.lstat()
            if candidate.is_symlink() or (getattr(metadata, "st_file_attributes", 0) & 0x00000400):
                raise ValueError("snapshot tree contains a link or reparse point")
        for name in directory_names:
            candidate = current_path / name
            relative = candidate.relative_to(root).as_posix() + "/"
            folded = relative.casefold()
            if folded in seen:
                raise ValueError("snapshot tree contains a case-insensitive duplicate")
            seen.add(folded)
            metadata = candidate.stat()
            if not stat.S_ISDIR(metadata.st_mode) or _identity(candidate.lstat()) != _identity(
                metadata
            ):
                raise ValueError("snapshot directory identity changed")
            directory_identities[candidate] = _identity(metadata)
            snapshot[relative] = directory_seal(candidate)
        for name in file_names:
            candidate = current_path / name
            if not candidate.is_file():
                raise ValueError("snapshot tree contains a non-regular file")
            relative = candidate.relative_to(root).as_posix()
            folded = relative.casefold()
            if folded in seen:
                raise ValueError("snapshot tree contains a case-insensitive duplicate")
            seen.add(folded)
            before_path = candidate.lstat()
            _reject_link_or_reparse(candidate, before_path)
            digest = hashlib.sha256()
            size = 0
            with candidate.open("rb") as retained:
                before = os.fstat(retained.fileno())
                if not stat.S_ISREG(before.st_mode) or _identity(before) != _identity(before_path):
                    raise ValueError("snapshot file identity changed before retained read")
                while chunk := retained.read(1024 * 1024):
                    size += len(chunk)
                    tree_bytes += len(chunk)
                    if size > _MAX_SNAPSHOT_FILE_BYTES or tree_bytes > _MAX_SNAPSHOT_TREE_BYTES:
                        raise ValueError("snapshot tree contains an oversized file set")
                    digest.update(chunk)
                after = os.fstat(retained.fileno())
                native = (
                    _native_handle_info(msvcrt.get_osfhandle(retained.fileno()))
                    if os.name == "nt"
                    else None
                )
            pathname_lstat = candidate.lstat()
            _reject_link_or_reparse(candidate, pathname_lstat)
            pathname_stat = candidate.stat()
            if (
                _identity(after) != _identity(before)
                or after.st_size != before.st_size
                or size != before.st_size
                or _identity(pathname_lstat) != _identity(before)
                or _identity(pathname_stat) != _identity(before)
            ):
                raise ValueError("snapshot file identity or size changed during retained read")
            if native is None:
                snapshot[relative] = (
                    "file",
                    before.st_dev,
                    before.st_ino,
                    size,
                    digest.hexdigest(),
                )
            else:
                volume, file_id, final, attributes, native_size = native
                if (
                    attributes & (0x10 | 0x400)
                    or native_size != size
                    or final != _normal_final_path(str(candidate.resolve(strict=True)))
                ):
                    raise ValueError("snapshot file native identity or type changed")
                snapshot[relative] = (
                    "file",
                    volume,
                    file_id,
                    size,
                    digest.hexdigest(),
                    final,
                    attributes,
                )
    for directory, expected in directory_identities.items():
        metadata = directory.lstat()
        _reject_link_or_reparse(directory, metadata)
        if _identity(metadata) != expected or _identity(directory.stat()) != expected:
            raise ValueError("snapshot directory identity changed before completion")
    final_root = root.lstat()
    _reject_link_or_reparse(root, final_root)
    if _identity(final_root) != root_identity or _identity(root.stat()) != root_identity:
        raise ValueError("snapshot root identity changed before completion")
    return dict(sorted(snapshot.items()))


def snapshot_protected_path(root: Path) -> dict[str, tuple[object, ...]]:
    """Snapshot a protected tree or its exact absent leaf and existing parent."""
    if root.exists() or root.is_symlink():
        return snapshot_regular_tree(root)
    ancestor = root
    leaf: list[str] = []
    while not ancestor.exists() and not ancestor.is_symlink():
        leaf.append(ancestor.name)
        ancestor = ancestor.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise ValueError("absent protected path has an unsafe parent")
    metadata = ancestor.stat()
    return {
        ".absent": (
            "absent",
            str(ancestor.resolve(strict=True)),
            metadata.st_dev,
            metadata.st_ino,
            *reversed(leaf),
        )
    }


def _snapshot_single_file(
    path: Path, *, maximum_bytes: int = _MAX_SNAPSHOT_FILE_BYTES
) -> tuple[object, ...]:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("single file snapshot bound is invalid")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        size = 0
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError("single file snapshot exceeds its bound")
            digest.update(chunk)
        if os.name == "nt":
            volume, file_id, final, attributes, native_size = _native_handle_info(
                msvcrt.get_osfhandle(stream.fileno())
            )
            if (
                attributes & (0x10 | 0x400)
                or native_size != size
                or final != _normal_final_path(str(path.resolve(strict=True)))
            ):
                raise ValueError("single file snapshot native identity changed")
            return ("file", volume, file_id, size, digest.hexdigest(), final, attributes)
        metadata = os.fstat(stream.fileno())
        return ("file", metadata.st_dev, metadata.st_ino, size, digest.hexdigest())


def require_config_only_delta(
    before: Mapping[str, tuple[object, ...]], after: Mapping[str, tuple[object, ...]]
) -> None:
    changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
    if changed != {"config.yaml"} or "config.yaml" not in after:
        raise RuntimeError(
            f"Hermes enable did not produce an exact config-only delta: {sorted(changed)!r}"
        )


def require_tree_unchanged(
    before: Mapping[str, tuple[object, ...]], after: Mapping[str, tuple[object, ...]]
) -> None:
    if dict(before) != dict(after):
        raise RuntimeError("protected tree was not unchanged")


class _Job(Protocol):
    def launch(
        self, command: tuple[str, ...], *, cwd: Path, environment: Mapping[str, str], role: str
    ) -> object: ...
    def wait_for_exit(
        self, child: object, *, timeout_seconds: int, output_limit: int
    ) -> tuple[int, bytes]: ...
    def force_finalize(self, *, timeout_seconds: int) -> object: ...


class ParentRequest(NamedTuple):
    hermes_source_archive: Path
    candidate_wheel: Path
    wheelhouse: Path
    requirements: Path
    constraints: Path
    workspace: Path
    profile: Path
    default_root: Path
    active_profile: Path
    evidence_root: Path
    output: Path
    build_python: Path
    hermes_source_archive_sha256: str
    wheelhouse_manifest: Path
    wheelhouse_manifest_sha256: str


_SOURCE_ARCHIVE_PREFIX = "hermes-agent-v0.20.0"
_SOURCE_WORKSPACE_CHILD = ".pluginmanager-hermes-v020-source"
_SOURCE_OWNER_MARKER = ".pluginmanager-source-owner"
_SOURCE_ARCHIVE_POLICY = SourceArchivePolicyV1(
    prefix=_SOURCE_ARCHIVE_PREFIX,
    workspace_child=_SOURCE_WORKSPACE_CHILD,
    owner_marker=_SOURCE_OWNER_MARKER,
    max_members=_MAX_SOURCE_ARCHIVE_MEMBERS,
    max_file_bytes=_MAX_SNAPSHOT_FILE_BYTES,
    max_tree_bytes=_MAX_SNAPSHOT_TREE_BYTES,
    error_label="Hermes source archive",
)
_validated_archive_members = archive_member_validator(_SOURCE_ARCHIVE_POLICY)
_ExtractedSourceV1 = WindowsSourceWorkspaceAuthorityV1
_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_WHEELHOUSE_COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9._-]+\Z")
_WHEELHOUSE_MANIFEST_KEYS = frozenset(
    {
        "schemaVersion",
        "purpose",
        "pythonVersion",
        "platform",
        "requirements",
        "constraints",
        "wheels",
    }
)
_WHEELHOUSE_ENTRY_KEYS = frozenset({"role", "relativePath", "basename", "sha256", "bytes"})


def _require_sha256(value: object, *, label: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} digest is invalid")
    return value


def _strict_json_object(payload: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} has duplicate keys")
            value[key] = item
        return value

    if type(payload) is not bytes or not payload:
        raise ValueError(f"{label} is absent")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    canonical = (
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    if canonical != payload:
        raise ValueError(f"{label} is not canonical")
    return value


def _wheelhouse_relative_path(value: object, *, label: str) -> tuple[str, tuple[str, ...]]:
    if type(value) is not str or not value or len(value) > 512 or "\\" in value:
        raise ValueError(f"wheelhouse {label} path is invalid")
    parts = tuple(value.split("/"))
    if any(
        part in {"", ".", ".."}
        or len(part) > 256
        or _WHEELHOUSE_COMPONENT_PATTERN.fullmatch(part) is None
        for part in parts
    ):
        raise ValueError(f"wheelhouse {label} path is invalid")
    return value, parts


def _validate_wheelhouse_entry(
    value: object, *, role: str
) -> tuple[str, tuple[str, ...], str, int, str]:
    if type(value) is not dict or set(value) != _WHEELHOUSE_ENTRY_KEYS:
        raise ValueError("wheelhouse manifest entry has an invalid closed shape")
    if value["role"] != role:
        raise ValueError("wheelhouse manifest entry role is invalid")
    relative, parts = _wheelhouse_relative_path(value["relativePath"], label=role)
    if len(parts) != 1:
        raise ValueError("wheelhouse manifest nested paths are not allowed by this harness")
    basename = value["basename"]
    if (
        type(basename) is not str
        or basename != parts[-1]
        or len(basename) > 256
        or _WHEELHOUSE_COMPONENT_PATTERN.fullmatch(basename) is None
        or (role == "wheel" and not basename.casefold().endswith(".whl"))
    ):
        raise ValueError("wheelhouse manifest basename is invalid")
    try:
        _windows_archive_component_key(basename)
    except ValueError as error:
        raise ValueError("wheelhouse manifest basename is not Win32-safe") from error
    size = value["bytes"]
    if type(size) is not int or not 1 <= size <= 9223372036854775807:
        raise ValueError("wheelhouse manifest bytes is invalid")
    digest = _require_sha256(value["sha256"], label="wheelhouse manifest entry")
    return relative, parts, basename, size, digest


def _verify_bound_wheelhouse_closure(
    request: ParentRequest, manifest_payload: bytes | None = None
) -> dict[str, object]:
    """Recompute the parent-manifest closure; no arbitrary wheelhouse is trusted."""

    manifest_path = request.wheelhouse_manifest
    expected_manifest_digest = _require_sha256(
        request.wheelhouse_manifest_sha256, label="PluginManager wheelhouse manifest"
    )
    if (
        not isinstance(manifest_path, Path)
        or not manifest_path.is_absolute()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or manifest_path.resolve(strict=True).is_relative_to(
            request.wheelhouse.resolve(strict=True)
        )
    ):
        raise ValueError("PluginManager wheelhouse manifest is invalid")
    if manifest_payload is None:
        manifest_payload = manifest_path.read_bytes()
    if type(manifest_payload) is not bytes:
        raise ValueError("PluginManager wheelhouse manifest payload is invalid")
    if hashlib.sha256(manifest_payload).hexdigest() != expected_manifest_digest:
        raise ValueError("PluginManager wheelhouse manifest digest changed")
    manifest = _strict_json_object(manifest_payload, label="wheelhouse manifest")
    if set(manifest) != _WHEELHOUSE_MANIFEST_KEYS:
        raise ValueError("wheelhouse manifest has an invalid closed shape")
    if type(manifest["schemaVersion"]) is not int or manifest["schemaVersion"] != 1:
        raise ValueError("wheelhouse manifest schemaVersion is invalid")
    if manifest["purpose"] != "hermes_v020_pluginmanager_runtime":
        raise ValueError("wheelhouse manifest purpose is invalid")
    # The standalone v1 harness historically used the minor version. Complete
    # input binding additionally requires the admitted interpreter's full patch
    # version; accept that canonical spelling here without weakening its check.
    python_version = manifest["pythonVersion"]
    if (
        type(python_version) is not str
        or re.fullmatch(r"3\.11(?:\.(?:0|[1-9][0-9]*))?", python_version) is None
    ):
        raise ValueError("wheelhouse manifest Python pin is invalid")
    if manifest["platform"] != "windows_amd64":
        raise ValueError("wheelhouse manifest platform pin is invalid")
    requirements = _validate_wheelhouse_entry(manifest["requirements"], role="requirements")
    constraints = _validate_wheelhouse_entry(manifest["constraints"], role="constraints")
    wheels_value = manifest["wheels"]
    if type(wheels_value) is not list or not 1 <= len(wheels_value) <= 4096:
        raise ValueError("wheelhouse manifest wheels is invalid")
    wheels = tuple(_validate_wheelhouse_entry(item, role="wheel") for item in wheels_value)
    entries = (requirements, constraints, *wheels)
    folded: set[str] = set()
    if len({entry[0] for entry in entries}) != len(entries):
        raise ValueError("wheelhouse manifest contains duplicate paths")
    for entry in entries:
        key = entry[0].casefold()
        if key in folded:
            raise ValueError("wheelhouse manifest contains case-colliding paths")
        folded.add(key)
    wheelhouse = request.wheelhouse.resolve(strict=True)
    for consumed in (request.candidate_wheel, request.requirements, request.constraints):
        try:
            consumed_path = consumed.resolve(strict=True)
        except OSError as error:
            raise ValueError("wheelhouse consumed path is missing") from error
        if consumed_path.parent != wheelhouse:
            raise ValueError("wheelhouse consumed path is not an exact direct child")
    inventory = snapshot_regular_tree(wheelhouse)
    expected_inventory = {"."}
    for relative, parts, _basename, _size, _digest in entries:
        expected_inventory.add(relative)
        expected_inventory.update("/".join(parts[:index]) + "/" for index in range(1, len(parts)))
    if set(inventory) != expected_inventory:
        raise ValueError("wheelhouse closure has missing or extra directory entries")
    resolved_entries: list[Path] = []
    for relative, parts, basename, size, digest in entries:
        path = wheelhouse.joinpath(*parts)
        if path.name != basename or path.resolve(strict=True) != wheelhouse.joinpath(*parts):
            raise ValueError("wheelhouse manifest path is confused")
        seal = inventory.get(relative)
        if seal is None or seal[0] != "file" or seal[3] != size or seal[4] != digest:
            raise ValueError("wheelhouse closure file does not match its manifest entry")
        resolved_entries.append(path)
    requirements_path, constraints_path, *wheel_paths = resolved_entries
    if requirements_path != request.requirements or constraints_path != request.constraints:
        raise ValueError("wheelhouse manifest does not bind requirements and constraints paths")
    candidate_relative = request.candidate_wheel.relative_to(wheelhouse).as_posix()
    candidate_matches = [entry for entry in wheels if entry[0] == candidate_relative]
    if (
        len(candidate_matches) != 1
        or len(wheel_paths) != len(wheels)
        or request.candidate_wheel not in wheel_paths
    ):
        raise ValueError("wheelhouse manifest does not bind the candidate wheel exactly once")
    if request.candidate_wheel.suffix.casefold() != ".whl":
        raise ValueError("wheelhouse candidate is not a wheel")
    return {
        "manifest": manifest_path,
        "candidate": request.candidate_wheel,
        "requirements": request.requirements,
        "constraints": request.constraints,
        "wheels": tuple(wheel_paths),
    }


def _windows_archive_component_key(component: str) -> str:
    """Return the Win32 collision key for one already-separated archive component."""

    if (
        not component
        or component[-1] in {" ", "."}
        or any(ord(character) < 32 or character in '<>:"|?*' for character in component)
        or component.casefold().split(".", 1)[0] in _WINDOWS_RESERVED_STEMS
    ):
        raise ValueError("Hermes source archive has a Win32-unsafe member component")
    return component.casefold()


def _verify_bound_source_archive(request: ParentRequest, payload: bytes) -> None:
    expected = request.hermes_source_archive_sha256
    if (
        type(payload) is not bytes
        or not payload
        or len(payload) > _MAX_HERMES_SOURCE_ARCHIVE_BYTES
        or type(expected) is not str
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("PluginManager Hermes source archive digest is invalid")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("PluginManager Hermes source archive digest changed")


class WindowsRetainedImmutableInputs:
    """Retained Windows sharing and directory-change boundary for one governed run."""

    _GENERIC_READ = 0x80000000
    _FILE_LIST_DIRECTORY = 0x0001
    _SHARE_READ = 0x00000001
    _OPEN_EXISTING = 3
    _BACKUP_SEMANTICS = 0x02000000
    _OPEN_REPARSE_POINT = 0x00200000
    _OVERLAPPED_FLAG = 0x40000000
    _NOTIFY_FILTER = 0x0000001F
    _ERROR_IO_PENDING = 997
    _ERROR_OPERATION_ABORTED = 995
    _ERROR_NOT_FOUND = 1168
    _WAIT_OBJECT_0 = 0
    _WAIT_ABANDONED_0 = 0x80
    _WAIT_TIMEOUT = 0x102
    _WAIT_FAILED = 0xFFFFFFFF
    _INVALID = ctypes.c_void_p(-1).value if os.name == "nt" else -1

    def __init__(self, request: ParentRequest) -> None:
        if os.name != "nt":
            raise RuntimeError("retained immutable inputs are Windows-only")
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._handles: list[int] = []
        self._watchers: list[dict[str, object]] = []
        self._watch_lock = threading.Lock()
        self._changed = threading.Event()
        self._closed = False
        self._inventories: dict[Path, dict[str, tuple[object, ...]]] = {}
        self._file_seals: dict[Path, tuple[int, int, int, str]] = {}
        self._retained: dict[Path, tuple[int, tuple[object, ...]]] = {}
        self._ignore_known_notifications: frozenset[str] = frozenset()
        self._request = request
        self._bind()

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
        self._kernel.ReadDirectoryChangesW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            wintypes.LPVOID,
            wintypes.LPVOID,
        ]
        self._kernel.ReadDirectoryChangesW.restype = wintypes.BOOL
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
            ctypes.POINTER(_OVERLAPPED),
        ]
        self._kernel.ReadFile.restype = wintypes.BOOL
        self._kernel.CancelIoEx.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
        self._kernel.CancelIoEx.restype = wintypes.BOOL
        self._kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel.CloseHandle.restype = wintypes.BOOL
        self._kernel.CreateEventW.argtypes = [
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._kernel.CreateEventW.restype = wintypes.HANDLE
        self._kernel.GetOverlappedResult.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_OVERLAPPED),
            ctypes.POINTER(wintypes.DWORD),
            wintypes.BOOL,
        ]
        self._kernel.GetOverlappedResult.restype = wintypes.BOOL
        self._kernel.ResetEvent.argtypes = [wintypes.HANDLE]
        self._kernel.ResetEvent.restype = wintypes.BOOL
        self._kernel.WaitForMultipleObjects.argtypes = [
            wintypes.DWORD,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self._kernel.WaitForMultipleObjects.restype = wintypes.DWORD
        self._kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self._kernel.WaitForSingleObject.restype = wintypes.DWORD

    def _open(self, path: Path, access: int, flags: int) -> int:
        handle = self._kernel.CreateFileW(
            str(path), access, self._SHARE_READ, None, self._OPEN_EXISTING, flags, None
        )
        value = int(ctypes.cast(handle, ctypes.c_void_p).value or 0) if handle else 0
        if not value or value == self._INVALID:
            raise OSError(ctypes.get_last_error(), f"cannot retain immutable input: {path}")
        self._handles.append(value)
        return value

    def _validate_handle(self, path: Path, handle: int, seal: tuple[object, ...]) -> None:
        volume, file_id, final, attributes, size = _native_handle_info(handle)
        expected_final = _normal_final_path(str(path.resolve(strict=True)))
        if (volume, file_id) != (seal[1], seal[2]) or final != expected_final:
            raise ValueError("retained handle does not bind the snapshotted pathname")
        if seal[0] == "dir":
            if (
                not attributes & 0x10
                or attributes & 0x400
                or len(seal) < 5
                or (seal[3], seal[4]) != (final, attributes)
            ):
                raise ValueError("retained directory native type or final path changed")
        else:
            if (
                attributes & (0x10 | 0x400)
                or size != seal[3]
                or len(seal) < 7
                or (seal[5], seal[6]) != (final, attributes)
            ):
                raise ValueError("retained file native type, size, or final path changed")
            digest = hashlib.sha256()
            position = ctypes.c_longlong()
            if not self._kernel.SetFilePointerEx(
                handle, ctypes.c_longlong(0), ctypes.byref(position), 0
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(1024 * 1024)
            count = wintypes.DWORD()
            while True:
                if not self._kernel.ReadFile(
                    handle, buffer, len(buffer), ctypes.byref(count), None
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if count.value == 0:
                    break
                digest.update(buffer.raw[: count.value])
            if digest.hexdigest() != seal[4]:
                raise ValueError("retained file content does not match snapshot")

    def _retain(self, path: Path, seal: tuple[object, ...]) -> int:
        flags = self._OPEN_REPARSE_POINT | (self._BACKUP_SEMANTICS if seal[0] == "dir" else 0)
        handle = self._open(path, self._GENERIC_READ, flags)
        self._validate_handle(path, handle, seal)
        self._retained[path] = (handle, seal)
        return handle

    def read_retained_bytes(
        self, path: Path, *, maximum_bytes: int = _MAX_SNAPSHOT_FILE_BYTES
    ) -> bytes:
        """Read and digest the exact bytes consumed through the retained file handle."""
        if self._closed:
            raise RuntimeError("retained immutable inputs are closed")
        resolved = path.resolve(strict=True)
        retained = self._retained.get(resolved)
        if retained is None or retained[1][0] != "file":
            raise ValueError("requested file is not retained authority")
        handle, seal = retained
        expected_size = _exact_int(seal[3])
        expected_digest = seal[4]
        if (
            type(maximum_bytes) is not int
            or maximum_bytes <= 0
            or type(expected_digest) is not str
            or expected_size > maximum_bytes
        ):
            raise ValueError("retained file seal is invalid")
        self._validate_handle(resolved, handle, seal)
        position = ctypes.c_longlong()
        if not self._kernel.SetFilePointerEx(
            handle, ctypes.c_longlong(0), ctypes.byref(position), 0
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        payload = bytearray()
        digest = hashlib.sha256()
        buffer = ctypes.create_string_buffer(1024 * 1024)
        count = wintypes.DWORD()
        while True:
            if not self._kernel.ReadFile(handle, buffer, len(buffer), ctypes.byref(count), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if count.value == 0:
                break
            chunk = buffer.raw[: count.value]
            payload.extend(chunk)
            digest.update(chunk)
            if len(payload) > expected_size:
                raise ValueError("retained file exceeded its sealed size while consumed")
        if len(payload) != expected_size or digest.hexdigest() != expected_digest:
            raise ValueError("retained file bytes consumed do not match their seal")
        self._validate_handle(resolved, handle, seal)
        return bytes(payload)

    @staticmethod
    def _absent_anchor(path: Path) -> tuple[Path, str]:
        missing: list[str] = []
        anchor = path
        while not anchor.exists() and not anchor.is_symlink():
            missing.append(anchor.name)
            anchor = anchor.parent
        if anchor.is_symlink() or not anchor.is_dir() or not missing:
            raise ValueError("absent protected root has an unsafe parent")
        return anchor.resolve(strict=True), "\\".join(reversed(missing)).casefold()

    def __enter__(self) -> WindowsRetainedImmutableInputs:
        try:
            trees = (self._request.wheelhouse,)
            protected = (
                self._request.default_root,
                self._request.active_profile,
                self._request.evidence_root,
            )
            existing_roots: list[Path] = []
            absent: list[tuple[Path, str]] = []
            for root in (*trees, *protected):
                if root.exists():
                    resolved = root.resolve(strict=True)
                    if resolved not in self._inventories:
                        self._inventories[resolved] = snapshot_regular_tree(resolved)
                        existing_roots.append(resolved)
                else:
                    absent.append(self._absent_anchor(root))
            # Lock all regular consumed/protected files before any child can launch.
            seen_files: set[tuple[int, int]] = set()
            seen_dirs: set[tuple[int, int]] = set()
            for root, inventory in self._inventories.items():
                for relative, seal in inventory.items():
                    path = root if relative == "." else root / relative.rstrip("/")
                    if type(seal[1]) is not int or type(seal[2]) is not int:
                        raise ValueError("retained inventory identity is invalid")
                    identity = (seal[1], seal[2])
                    if seal[0] == "file" and identity not in seen_files:
                        self._retain(path, seal)
                        seen_files.add(identity)
                    elif seal[0] == "dir" and identity not in seen_dirs:
                        self._retain(path, seal)
                        seen_dirs.add(identity)
            for path in (
                self._request.hermes_source_archive,
                self._request.candidate_wheel,
                self._request.requirements,
                self._request.constraints,
                self._request.build_python,
                self._request.wheelhouse_manifest,
            ):
                resolved = path.resolve(strict=True)
                owner = next(
                    (root for root in self._inventories if resolved.is_relative_to(root)), None
                )
                consumed_seal: tuple[object, ...] | None
                if owner is None:
                    consumed_seal = _snapshot_single_file(
                        resolved,
                        maximum_bytes=(
                            _MAX_HERMES_SOURCE_ARCHIVE_BYTES
                            if path == self._request.hermes_source_archive
                            else _MAX_SNAPSHOT_FILE_BYTES
                        ),
                    )
                else:
                    relative = resolved.relative_to(owner).as_posix()
                    consumed_seal = self._inventories[owner].get(relative)
                if consumed_seal is None or consumed_seal[0] != "file":
                    raise ValueError("consumed file is absent from retained inventory")
                identity = (_exact_int(consumed_seal[1]), _exact_int(consumed_seal[2]))
                if identity not in seen_files:
                    self._retain(resolved, consumed_seal)
                    seen_files.add(identity)
            watched: set[tuple[int, int, str | None]] = set()
            for root in existing_roots:
                seal = self._inventories[root]["."]
                watch_key: tuple[int, int, str | None] = (
                    _exact_int(seal[1]),
                    _exact_int(seal[2]),
                    None,
                )
                if watch_key not in watched:
                    self._start_watcher(root, None)
                    watched.add(watch_key)
            for anchor, prefix in absent:
                anchor_inventory = snapshot_regular_tree(anchor)
                anchor_seal = anchor_inventory["."]
                absent_key: tuple[int, int, str | None] = (
                    _exact_int(anchor_seal[1]),
                    _exact_int(anchor_seal[2]),
                    prefix,
                )
                if absent_key not in watched:
                    if (_exact_int(anchor_seal[1]), _exact_int(anchor_seal[2])) not in seen_dirs:
                        self._retain(anchor, anchor_seal)
                        seen_dirs.add((_exact_int(anchor_seal[1]), _exact_int(anchor_seal[2])))
                    self._start_watcher(anchor, prefix)
                    watched.add(absent_key)
            self.assert_unchanged()
            return self
        except BaseException as primary:
            try:
                self.close(validate=False)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "retained input acquisition and cleanup failed", [primary, cleanup]
                ) from None
            raise

    def _start_watcher(
        self, root: Path, prefix: str | None, *, retained_handle: int | None = None
    ) -> None:
        handle = (
            self._open(
                root,
                self._FILE_LIST_DIRECTORY,
                self._BACKUP_SEMANTICS | self._OPEN_REPARSE_POINT | self._OVERLAPPED_FLAG,
            )
            if retained_handle is None
            else retained_handle
        )
        if retained_handle is not None and retained_handle not in self._handles:
            raise RuntimeError("directory watcher requires a retained exact directory handle")
        buffers = [bytearray(64 * 1024), bytearray(64 * 1024)]
        events: list[int] = []
        overlapped: list[_OVERLAPPED] = []
        for _ in range(2):
            raw = self._kernel.CreateEventW(None, True, False, None)
            event = int(ctypes.cast(raw, ctypes.c_void_p).value or 0) if raw else 0
            if not event:
                raise ctypes.WinError(ctypes.get_last_error())
            self._handles.append(event)
            events.append(event)
            value = _OVERLAPPED()
            value.event = wintypes.HANDLE(event)
            overlapped.append(value)
        watcher: dict[str, object] = {
            "handle": handle,
            "buffers": buffers,
            "events": events,
            "overlapped": overlapped,
            "prefix": prefix,
            "pending": [False, False],
            "started": False,
            "terminal": threading.Event(),
            "errors": [],
        }
        self._watchers.append(watcher)
        # Both authoritative arms occur in the entering thread before launch/entry.
        self._post_read(watcher, 0)
        self._post_read(watcher, 1)
        thread = threading.Thread(target=self._watch, args=(watcher,), daemon=True)
        watcher["thread"] = thread
        thread.start()
        watcher["started"] = True

    def _post_read(self, watcher: dict[str, object], index: int) -> None:
        buffers = watcher["buffers"]
        overlapped = watcher["overlapped"]
        assert isinstance(buffers, list) and isinstance(overlapped, list)
        buffer = buffers[index]
        ov = overlapped[index]
        pending = watcher["pending"]
        assert isinstance(pending, list)
        if pending[index] is not False:
            raise RuntimeError("retained directory journal slot was already pending")
        view = (ctypes.c_ubyte * len(buffer)).from_buffer(buffer)
        ctypes.set_last_error(0)
        ok = self._kernel.ReadDirectoryChangesW(
            watcher["handle"],
            view,
            len(buffer),
            True,
            self._NOTIFY_FILTER,
            None,
            ctypes.byref(ov),
            None,
        )
        error = ctypes.get_last_error()
        if not ok and error != self._ERROR_IO_PENDING:
            raise OSError(error, "cannot authoritatively arm retained directory journal")
        pending[index] = True

    def _parse_records(self, buffer: bytearray, count: int, prefix: str | None) -> None:
        if count <= 0 or count > len(buffer):
            raise RuntimeError("retained directory journal overflow or zero record")
        offset = 0
        while True:
            if offset + 12 > count:
                raise RuntimeError("retained directory journal record is malformed")
            next_offset = int.from_bytes(buffer[offset : offset + 4], "little")
            name_bytes = int.from_bytes(buffer[offset + 8 : offset + 12], "little")
            end = offset + 12 + name_bytes
            if (
                name_bytes == 0
                or name_bytes % 2
                or end > count
                or (
                    next_offset
                    and (next_offset < 12 or next_offset % 4 or offset + next_offset > count)
                )
            ):
                raise RuntimeError("retained directory journal record is malformed")
            try:
                name = bytes(buffer[offset + 12 : end]).decode("utf-16-le").casefold()
            except UnicodeDecodeError as error:
                raise RuntimeError("retained directory journal record is malformed") from error
            allowed: frozenset[str] = getattr(self, "_ignore_known_notifications", frozenset())
            if name not in allowed and (
                prefix is None
                or name == prefix
                or name.startswith(prefix + "\\")
                or prefix.startswith(name + "\\")
            ):
                self._changed.set()
            if next_offset == 0:
                if end != count:
                    raise RuntimeError("retained directory journal record has trailing bytes")
                return
            offset += next_offset

    def _watch(self, watcher: dict[str, object]) -> None:
        terminal = watcher["terminal"]
        errors = watcher["errors"]
        assert isinstance(terminal, threading.Event) and isinstance(errors, list)
        try:
            while True:
                events = watcher["events"]
                overlapped = watcher["overlapped"]
                buffers = watcher["buffers"]
                pending = watcher["pending"]
                assert (
                    isinstance(events, list)
                    and isinstance(overlapped, list)
                    and isinstance(buffers, list)
                    and isinstance(pending, list)
                )
                if (
                    len(events) != 2
                    or len(pending) != 2
                    or not all(type(value) is bool for value in pending)
                ):
                    raise RuntimeError("retained directory journal slot state is invalid")
                if not any(pending):
                    if self._closed:
                        return
                    raise RuntimeError("retained directory journal has no outstanding read")
                event_array = (wintypes.HANDLE * 2)(*(wintypes.HANDLE(value) for value in events))
                result = int(self._kernel.WaitForMultipleObjects(2, event_array, False, 0xFFFFFFFF))
                if result == self._WAIT_FAILED:
                    raise ctypes.WinError(ctypes.get_last_error())
                if result in (
                    self._WAIT_TIMEOUT,
                    self._WAIT_ABANDONED_0,
                    self._WAIT_ABANDONED_0 + 1,
                ):
                    raise RuntimeError("retained directory journal wait failed closed")
                if result not in (self._WAIT_OBJECT_0, self._WAIT_OBJECT_0 + 1):
                    raise RuntimeError(
                        "retained directory journal wait returned an ambiguous state"
                    )
                index = result - self._WAIT_OBJECT_0
                if pending[index] is not True:
                    raise RuntimeError("retained directory journal completed a nonpending slot")
                count = wintypes.DWORD()
                if not self._kernel.GetOverlappedResult(
                    watcher["handle"], ctypes.byref(overlapped[index]), ctypes.byref(count), False
                ):
                    error = ctypes.get_last_error()
                    if self._closed and error == self._ERROR_OPERATION_ABORTED:
                        pending[index] = False
                        if not self._kernel.ResetEvent(events[index]):
                            raise ctypes.WinError(ctypes.get_last_error())
                        continue
                    raise OSError(error, "retained directory journal completion failed")
                pending[index] = False
                with self._watch_lock:
                    if self._closed:
                        if not self._kernel.ResetEvent(events[index]):
                            raise ctypes.WinError(ctypes.get_last_error())
                        self._parse_records(
                            buffers[index],
                            int(count.value),
                            watcher["prefix"] if isinstance(watcher["prefix"], str) else None,
                        )
                        continue
                    if not self._kernel.ResetEvent(events[index]):
                        raise ctypes.WinError(ctypes.get_last_error())
                    overlapped[index] = _OVERLAPPED()
                    overlapped[index].event = wintypes.HANDLE(events[index])
                    self._post_read(watcher, index)  # rearm before parsing; peer stayed pending
                self._parse_records(
                    buffers[index],
                    int(count.value),
                    watcher["prefix"] if isinstance(watcher["prefix"], str) else None,
                )
        except BaseException as error:
            errors.append(error)
            self._changed.set()
        finally:
            terminal.set()

    def assert_unchanged(self) -> None:
        if self._changed.is_set():
            raise RuntimeError("retained input directory changed during qualification")
        for root, inventory in self._inventories.items():
            if snapshot_regular_tree(root) != inventory:
                raise RuntimeError("retained immutable input changed during qualification")
        for path, (handle, seal) in self._retained.items():
            try:
                self._validate_handle(path, handle, seal)
            except BaseException as error:
                raise RuntimeError(
                    "retained handle or pathname diverged during qualification"
                ) from error

    def close(self, *, validate: bool = True) -> None:
        if self._closed and not self._handles:
            return
        failures: list[BaseException] = []
        validation_failed = False
        if validate:
            try:
                self.assert_unchanged()
            except BaseException as error:
                failures.append(error)
                validation_failed = True
        with self._watch_lock:
            self._closed = True
            for watcher in self._watchers:
                terminal = watcher["terminal"]
                assert isinstance(terminal, threading.Event)
                pending = watcher["pending"]
                assert isinstance(pending, list)
                if not any(value is True for value in pending):
                    continue
                handle = _exact_int(watcher["handle"], label="watcher handle")
                ctypes.set_last_error(0)
                if not self._kernel.CancelIoEx(handle, None):
                    cancel_error = ctypes.get_last_error()
                    if cancel_error != self._ERROR_NOT_FOUND:
                        failures.append(
                            OSError(
                                cancel_error, "CancelIoEx failed for retained directory journal"
                            )
                        )

        live_handles: set[int] = set()
        for watcher in self._watchers:
            thread = watcher.get("thread")
            started = watcher.get("started") is True
            if started and isinstance(thread, threading.Thread):
                try:
                    thread.join(timeout=5)
                except BaseException as error:
                    failures.append(error)
            errors = watcher["errors"]
            assert isinstance(errors, list)
            failures.extend(error for error in errors if isinstance(error, BaseException))
            alive = started
            if started and isinstance(thread, threading.Thread):
                try:
                    alive = thread.is_alive()
                except BaseException as error:
                    failures.append(error)
                    alive = True
            if alive:
                live_handles.add(_exact_int(watcher["handle"], label="watcher handle"))
                events = watcher["events"]
                assert isinstance(events, list)
                live_handles.update(
                    _exact_int(value, label="watcher event handle") for value in events
                )
                failures.append(
                    RuntimeError("retained directory watcher did not reach terminal state")
                )
            else:
                pending = watcher["pending"]
                overlapped = watcher["overlapped"]
                events = watcher["events"]
                assert (
                    isinstance(pending, list)
                    and isinstance(overlapped, list)
                    and isinstance(events, list)
                )
                handle = _exact_int(watcher["handle"], label="watcher handle")
                for index in range(2):
                    if pending[index] is not True:
                        continue
                    wait_result = int(self._kernel.WaitForSingleObject(events[index], 5000))
                    if wait_result != self._WAIT_OBJECT_0:
                        failures.append(
                            RuntimeError(
                                "retained directory journal cancellation did not reach a bounded terminal event"
                            )
                        )
                        continue
                    count = wintypes.DWORD()
                    ctypes.set_last_error(0)
                    completed = self._kernel.GetOverlappedResult(
                        handle, ctypes.byref(overlapped[index]), ctypes.byref(count), False
                    )
                    completion_error = ctypes.get_last_error()
                    if not completed and completion_error != self._ERROR_OPERATION_ABORTED:
                        failures.append(
                            OSError(
                                completion_error,
                                "retained directory journal completion was not accounted",
                            )
                        )
                    else:
                        pending[index] = False
                if any(value is True for value in pending):
                    live_handles.add(handle)
                    live_handles.update(
                        _exact_int(value, label="watcher event handle") for value in events
                    )
        if validate and not validation_failed:
            try:
                self.assert_unchanged()
            except BaseException as error:
                failures.append(error)
        retained_after_close: set[int] = set(live_handles)
        for handle in reversed(self._handles):
            if handle in live_handles:
                continue
            if not self._kernel.CloseHandle(handle):
                failures.append(ctypes.WinError(ctypes.get_last_error()))
                retained_after_close.add(handle)
        self._handles = [handle for handle in self._handles if handle in retained_after_close]
        self._retained = {
            path: (handle, seal)
            for path, (handle, seal) in self._retained.items()
            if handle in retained_after_close
        }
        if failures:
            raise BaseExceptionGroup("retained input finalization failed", failures)

    def __exit__(self, kind: object, error: object, traceback: object) -> None:
        try:
            self.close(validate=True)
        except BaseException as close_error:
            if isinstance(error, BaseException):
                raise BaseExceptionGroup(
                    "qualification body and retained cleanup failed", [error, close_error]
                ) from None
            else:
                raise


def validate_parent_request(value: ParentRequest) -> ParentRequest:
    if type(value) is not ParentRequest:
        raise TypeError("PluginManager request must be parent-constructed")
    directories = (
        value.wheelhouse,
        value.workspace,
        value.profile,
        value.default_root,
        value.active_profile,
    )
    files = (
        value.hermes_source_archive,
        value.candidate_wheel,
        value.requirements,
        value.constraints,
        value.build_python,
        value.wheelhouse_manifest,
    )
    if any(
        not item.is_absolute() or item.is_symlink() or not item.is_dir() for item in directories
    ):
        raise ValueError("PluginManager request directory is invalid")
    if not value.evidence_root.is_absolute() or value.evidence_root.is_symlink():
        raise ValueError("PluginManager evidence root is invalid")
    if value.evidence_root.exists() and not value.evidence_root.is_dir():
        raise ValueError("PluginManager evidence root is invalid")
    if any(not item.is_absolute() or item.is_symlink() or not item.is_file() for item in files):
        raise ValueError("PluginManager request file is invalid")
    if value.hermes_source_archive.suffix.casefold() != ".tar":
        raise ValueError("PluginManager Hermes source archive is invalid")
    if (
        not value.output.is_absolute()
        or not value.output.parent.is_dir()
        or value.output.exists()
        or value.output.is_symlink()
    ):
        raise ValueError("PluginManager bounded output is invalid")
    resolved = tuple(item.resolve(strict=True) for item in directories)
    wheelhouse, workspace, profile, default_root, active_profile = resolved
    evidence_root = value.evidence_root.resolve(strict=False)
    non_active = (wheelhouse, workspace, profile, default_root, evidence_root)
    if len(set(non_active)) != len(non_active) or (
        active_profile != default_root and active_profile in set(non_active)
    ):
        raise ValueError("PluginManager request roots alias")
    if active_profile != default_root and active_profile.parent != default_root / "profiles":
        raise ValueError("PluginManager active profile is outside the default root")
    expected_profile = (value.workspace / "profiles" / "gate-profile").resolve(strict=False)
    if value.profile.resolve(strict=True) != expected_profile:
        raise ValueError("PluginManager profile is not the marker-owned gate-profile")
    cwd = (value.workspace / "child-cwd").resolve(strict=False)
    forbidden = (
        Path.cwd().resolve(strict=True),
        value.profile.resolve(),
        value.default_root.resolve(),
        value.active_profile.resolve(),
        value.evidence_root.resolve(),
    )
    if any(cwd == item or cwd.is_relative_to(item) for item in forbidden):
        raise ValueError("PluginManager child CWD is not isolated")
    _verify_bound_wheelhouse_closure(value)
    return value


def _new_source_workspace_watcher(
    kernel: object, root_handle: int
) -> WindowsRetainedImmutableInputs:
    """Adapt the harness's retained journal to the generic owned-tree authority."""

    watcher = WindowsRetainedImmutableInputs.__new__(WindowsRetainedImmutableInputs)
    watcher._kernel = cast(Any, kernel)
    watcher._handles = [root_handle]
    watcher._watchers = []
    watcher._watch_lock = threading.Lock()
    watcher._changed = threading.Event()
    watcher._closed = False
    watcher._inventories = {}
    watcher._file_seals = {}
    watcher._retained = {}
    watcher._ignore_known_notifications = frozenset()
    watcher._bind()
    return watcher


def _extract_official_source_archive(
    request: ParentRequest, payload: bytes
) -> _WindowsSourceWorkspaceAuthorityV1:
    """Parse exact retained bytes, then create and retain a private native source root."""
    if type(payload) is not bytes or not payload or len(payload) > _MAX_HERMES_SOURCE_ARCHIVE_BYTES:
        raise ValueError("Hermes source archive payload is invalid")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        members = _validated_archive_members(archive)
        extracted: list[tuple[str, bytes]] = []
        extracted_bytes = 0
        for info in members:
            source = archive.extractfile(info)
            if source is None:
                raise ValueError("Hermes source archive regular member is unreadable")
            with source:
                member_payload = source.read(info.size + 1)
            extracted_bytes += len(member_payload)
            if len(member_payload) != info.size or extracted_bytes > _MAX_SNAPSHOT_TREE_BYTES:
                raise ValueError("Hermes source archive member payload is invalid")
            extracted.append((info.name, member_payload))
    owned: _WindowsSourceWorkspaceAuthorityV1 = _ExtractedSourceV1(
        request.workspace,
        _SOURCE_ARCHIVE_POLICY,
        _new_source_workspace_watcher,
    )
    try:
        owned.populate(tuple(extracted))
        return owned
    except BaseException as primary:
        try:
            owned.close(cleanup=True)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "Hermes source extraction and handle-bound cleanup failed", [primary, cleanup]
            ) from None
        raise


def build_child_environment(
    *,
    base: Mapping[str, str],
    scripts: Path,
    profile: Path,
    temporary: Path,
    home_proxy: Path,
) -> dict[str, str]:
    if type(base) is not dict or any(
        type(k) is not str or type(v) is not str for k, v in base.items()
    ):
        raise TypeError("child base environment must be an exact string mapping")
    required = (
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "USERNAME",
    )
    if any(not base.get(key) for key in required):
        raise ValueError("child environment lacks Windows identity")
    if (
        not temporary.is_absolute()
        or not temporary.is_dir()
        or not home_proxy.is_absolute()
        or not home_proxy.is_dir()
        or home_proxy in {profile, temporary}
    ):
        raise ValueError("child disposable directory is invalid")
    keep = {key: base[key] for key in required}
    keep.update(
        {
            "PATH": str(scripts) + os.pathsep + str(Path(base["SYSTEMROOT"]) / "System32"),
            "HERMES_HOME": str(profile.resolve(strict=True)),
            "HOME": str(home_proxy.resolve(strict=True)),
            "LOCALAPPDATA": str(home_proxy.resolve(strict=True)),
            "USERPROFILE": str(home_proxy.resolve(strict=True)),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_PROGRESS_BAR": "off",
            "PIP_NO_COLOR": "1",
            "TEMP": str(temporary.resolve(strict=True)),
            "TMP": str(temporary.resolve(strict=True)),
        }
    )
    return keep


def _canonical_stage(payload: bytes, keys: set[str]) -> dict[str, object]:
    if type(payload) is not bytes or not payload or len(payload) > 4096:
        raise ValueError("PluginManager stage output is absent or oversized")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("PluginManager stage output is malformed") from error
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if canonical != payload or type(value) is not dict or set(value) != keys:
        raise ValueError("PluginManager stage output is noncanonical or not closed")
    if any(
        type(item) is str and any(token in item.casefold() for token in _RAW_IDENTIFIER_FRAGMENTS)
        for item in value.values()
    ):
        raise ValueError("PluginManager stage output leaked a raw identifier")
    return value


_DISABLED_CHILD = r"""import hashlib,importlib.metadata,json,pathlib,sys
source=pathlib.Path(sys.argv[1]).resolve(strict=True); profile=pathlib.Path(sys.argv[2]).resolve(strict=True); packages=pathlib.Path(sys.argv[3]).resolve(strict=True)
if any(n=="hermes_cli" or n.startswith("hermes_cli.") for n in sys.modules): raise RuntimeError("preloaded hermes_cli")
stdin_closed=sys.stdin is None or sys.stdin.read(1)==""
if not stdin_closed: raise RuntimeError("stdin not EOF")
sys.path.insert(0,str(packages)); sys.path.insert(0,str(source))
if sys.path.count(str(source))!=1: raise RuntimeError("source insertion")
from hermes_cli import plugins as hp
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home
for obj in (hp,pathlib.Path(sys.modules[hp.PluginContext.__module__].__file__),pathlib.Path(sys.modules[hp.PluginManager.__module__].__file__)):
 p=pathlib.Path(obj.__file__) if hasattr(obj,"__file__") else obj
 if not p.resolve(strict=True).is_relative_to(source): raise RuntimeError("official origin")
if pathlib.Path(get_hermes_home()).resolve()!=profile or pathlib.Path(get_config_path()).resolve()!=profile/"config.yaml": raise RuntimeError("profile resolution")
points=[p for p in importlib.metadata.entry_points().select(group="hermes_agent.plugins") if p.name=="hermes-realtime"]
if len(points)!=1 or points[0].value!="hermes_realtime.hermes_plugin": raise RuntimeError("entrypoint")
calls=[]
def observe(frame,event,arg):
 if event=="call" and frame.f_code.co_name=="register" and frame.f_globals.get("__name__")=="hermes_realtime.hermes_plugin": calls.append(frame)
sys.setprofile(observe)
try: m=hp.PluginManager(); m.discover_and_load(force=True)
finally: sys.setprofile(None)
loaded=m._plugins.get("hermes-realtime")
ok=loaded is not None and loaded.manifest.source=="entrypoint" and loaded.enabled is False and loaded.module is None and isinstance(loaded.error,str) and bool(loaded.error)
side=any(n=="hermes_realtime" or n.startswith("hermes_realtime.") for n in sys.modules) or bool(calls)
sys.stdout.buffer.write(json.dumps({"discovered":bool(ok),"importOrRegistration":bool(side),"sourceOriginSha256":hashlib.sha256(str(source).encode()).hexdigest()},sort_keys=True,separators=(",",":")).encode()+b"\n")"""

_ENABLE_CHILD = r"""import contextlib,copy,hashlib,json,os,pathlib,sys
source=pathlib.Path(sys.argv[1]).resolve(strict=True); profile=pathlib.Path(sys.argv[2]).resolve(strict=True); packages=pathlib.Path(sys.argv[3]).resolve(strict=True); config=profile/"config.yaml"
if any(n=="hermes_cli" or n.startswith("hermes_cli.") for n in sys.modules): raise RuntimeError("preloaded hermes_cli")
stdin_closed=sys.stdin is None or sys.stdin.read(1)==""
if not stdin_closed: raise RuntimeError("stdin not EOF")
def tree():
 result={}; seen=set()
 for base,dirs,files in os.walk(profile,followlinks=False):
  for name in dirs+files:
   p=pathlib.Path(base)/name
   if ":" in name or p.is_symlink(): raise RuntimeError("unsafe profile tree")
  for name in files:
   p=pathlib.Path(base)/name; rel=p.relative_to(profile).as_posix(); folded=rel.casefold()
   if folded in seen: raise RuntimeError("profile case collision")
   seen.add(folded); result[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
  for name in dirs:
   p=pathlib.Path(base)/name; rel=p.relative_to(profile).as_posix()+"/"; folded=rel.casefold()
   if folded in seen: raise RuntimeError("profile case collision")
   seen.add(folded); s=p.stat(); result[rel]=["directory",s.st_dev,s.st_ino]
 return result
tree_before=tree(); sys.path.insert(0,str(packages)); sys.path.insert(0,str(source)); import yaml
if sys.path.count(str(source))!=1: raise RuntimeError("source insertion")
before=yaml.safe_load(config.read_text(encoding="utf-8")) if config.exists() else {}
if type(before) is not dict or type(before.get("plugins",{})) is not dict: raise RuntimeError("plugins shape")
plugins_before=before.get("plugins",{}); enabled_before=plugins_before.get("enabled",[]); entries_before=plugins_before.get("entries",{})
if type(enabled_before) is not list or any(type(v) is not str for v in enabled_before) or "hermes-realtime" in enabled_before: raise RuntimeError("enabled shape")
if type(entries_before) is not dict or "hermes-realtime" in entries_before: raise RuntimeError("entries shape")
frozen=copy.deepcopy(before)
sys.argv=["hermes","plugins","enable","hermes-realtime","--no-allow-tool-override"]
from hermes_cli import main as hm
from hermes_cli import plugins as hp
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home
for obj in (hm,hp,pathlib.Path(sys.modules[hp.PluginContext.__module__].__file__),pathlib.Path(sys.modules[hp.PluginManager.__module__].__file__)):
 p=pathlib.Path(obj.__file__) if hasattr(obj,"__file__") else obj
 if not p.resolve(strict=True).is_relative_to(source): raise RuntimeError("official enable origin")
if pathlib.Path(get_hermes_home()).resolve()!=profile or pathlib.Path(get_config_path()).resolve()!=config: raise RuntimeError("profile resolution")
try:
 with contextlib.redirect_stdout(sys.stderr): result=hm.main()
except SystemExit as exc: result=exc.code
if result not in (None,0): raise RuntimeError("enable exit")
after=yaml.safe_load(config.read_text(encoding="utf-8")) or {}; expected=copy.deepcopy(frozen); expected["_config_version"]=33; plugins=expected.setdefault("plugins",{}); old=list(plugins.get("enabled",[])); plugins["enabled"]=old+["hermes-realtime"]; plugins["disabled"]=[]; plugins.setdefault("entries",{}).setdefault("hermes-realtime",{})["allow_tool_override"]=False
tree_after=tree(); changed={name for name in set(tree_before)|set(tree_after) if tree_before.get(name)!=tree_after.get(name)}
sys.stdout.buffer.write(json.dumps({"argvExact":sys.argv==["hermes","plugins","enable","hermes-realtime","--no-allow-tool-override"],"exactConfigDelta":after==expected and old.count("hermes-realtime")==0 and changed=={"config.yaml"},"stdinClosed":stdin_closed},sort_keys=True,separators=(",",":")).encode()+b"\n")"""

_ENABLED_CHILD = r"""import hashlib,importlib.metadata,json,pathlib,sys
source=pathlib.Path(sys.argv[1]).resolve(strict=True); profile=pathlib.Path(sys.argv[2]).resolve(strict=True); packages=pathlib.Path(sys.argv[3]).resolve(strict=True)
expected_dist_info=json.loads(sys.argv[4])
if any(n=="hermes_cli" or n.startswith("hermes_cli.") for n in sys.modules): raise RuntimeError("preloaded hermes_cli")
stdin_closed=sys.stdin is None or sys.stdin.read(1)==""
if not stdin_closed: raise RuntimeError("stdin not EOF")
sys.path.insert(0,str(packages)); sys.path.insert(0,str(source))
if sys.path.count(str(source))!=1: raise RuntimeError("source insertion")
from hermes_cli import plugins as hp
from hermes_cli.config import get_config_path
from hermes_constants import get_hermes_home
for obj in (hp,pathlib.Path(sys.modules[hp.PluginContext.__module__].__file__),pathlib.Path(sys.modules[hp.PluginManager.__module__].__file__)):
 p=pathlib.Path(obj.__file__) if hasattr(obj,"__file__") else obj
 if not p.resolve(strict=True).is_relative_to(source): raise RuntimeError("official enabled origin")
if pathlib.Path(get_hermes_home()).resolve()!=profile or pathlib.Path(get_config_path()).resolve()!=profile/"config.yaml": raise RuntimeError("profile resolution")
points=[p for p in importlib.metadata.entry_points().select(group="hermes_agent.plugins") if p.name=="hermes-realtime"]
if len(points)!=1 or points[0].value!="hermes_realtime.hermes_plugin": raise RuntimeError("entrypoint")
dist=importlib.metadata.distribution("hermes-realtime"); calls=[]
dist_module=pathlib.Path(dist.locate_file("hermes_realtime/hermes_plugin.py")).resolve(strict=True)
dist_metadata=pathlib.Path(dist.locate_file("hermes_realtime-0.0.3.dist-info/METADATA")).resolve(strict=True)
dist_entrypoints=pathlib.Path(dist.locate_file("hermes_realtime-0.0.3.dist-info/entry_points.txt")).resolve(strict=True)
if not all(p.is_relative_to(packages) for p in (dist_module,dist_metadata,dist_entrypoints)): raise RuntimeError("distribution origin")
if dist.version!="0.0.3" or len(points)!=1 or points[0] not in dist.entry_points: raise RuntimeError("distribution ownership")
dist_info_files=sorted(str(item).replace("\\","/") for item in (dist.files or ()))
if not expected_dist_info or len(expected_dist_info)!=len(set(expected_dist_info)): raise RuntimeError("expected dist-info inventory")
if any(item not in dist_info_files for item in expected_dist_info): raise RuntimeError("missing candidate dist-info member")
resolved_dist_info={item:pathlib.Path(dist.locate_file(item)).resolve(strict=True) for item in expected_dist_info}
if any(not path.is_relative_to(dist_metadata.parent) for path in resolved_dist_info.values()): raise RuntimeError("dist-info inventory origin")
dist_inventory={item:hashlib.sha256(path.read_bytes()).hexdigest() for item,path in resolved_dist_info.items()}
def observe(frame,event,arg):
 if event=="call" and frame.f_code.co_name=="register" and frame.f_globals.get("__name__")=="hermes_realtime.hermes_plugin": calls.append((frame.f_code,frame.f_locals.get("context"),frame.f_back.f_code if frame.f_back else None,frame.f_back.f_locals.get("self") if frame.f_back else None))
sys.setprofile(observe)
try: m=hp.PluginManager(); m.discover_and_load(force=True)
finally: sys.setprofile(None)
loaded=m._plugins.get("hermes-realtime"); module=loaded.module if loaded else None; origin=pathlib.Path(module.__file__).resolve(strict=True) if module else pathlib.Path("missing")
ok=loaded is not None and type(loaded).__module__==hp.__name__ and loaded.enabled is True and loaded.error is None and loaded.module is module and origin==dist_module
registered=ok and len(calls)==1 and calls[0][0] is module.register.__code__ and type(calls[0][1]) is hp.PluginContext and calls[0][3] is m and pathlib.Path(calls[0][2].co_filename).resolve(strict=True).is_relative_to(source) and module.get_dispatcher() is not None and module.get_runtime() is not None
sys.stdout.buffer.write(json.dumps({"contextObserved":bool(registered),"discovered":bool(ok),"distributionVersion":dist.version,"distEntryPointsContentSha256":hashlib.sha256(dist_entrypoints.read_bytes()).hexdigest(),"distInfoInventorySha256":hashlib.sha256(json.dumps(dist_inventory,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"distMetadataContentSha256":hashlib.sha256(dist_metadata.read_bytes()).hexdigest(),"distOriginSha256":hashlib.sha256(str(dist_metadata.parent).encode()).hexdigest(),"moduleContentSha256":hashlib.sha256(origin.read_bytes()).hexdigest(),"moduleOriginSha256":hashlib.sha256(str(origin).encode()).hexdigest()},sort_keys=True,separators=(",",":")).encode()+b"\n")"""


def _qualify_governed_request_body(
    request: ParentRequest,
    *,
    job: _Job,
    base_environment: Mapping[str, str] | None = None,
    revalidate_inputs: Callable[[], None],
    source_archive_payload: bytes,
    wheelhouse_manifest_payload: bytes,
    venv_name: str | None = None,
) -> dict[str, object]:
    request = validate_parent_request(request)
    revalidate_inputs()
    _verify_bound_wheelhouse_closure(request, wheelhouse_manifest_payload)
    _verify_bound_source_archive(request, source_archive_payload)
    extracted_source = _extract_official_source_archive(request, source_archive_payload)
    hermes_source = extracted_source.root

    def revalidate_all_inputs() -> None:
        revalidate_inputs()
        _verify_bound_wheelhouse_closure(request, wheelhouse_manifest_payload)
        extracted_source.assert_unchanged()

    revalidate_all_inputs()
    before = {
        name: snapshot_protected_path(root)
        for name, root in (
            ("default", request.default_root),
            ("active", request.active_profile),
            ("evidence", request.evidence_root),
        )
    }
    venv = request.workspace / (venv_name or ("pluginmanager-venv-" + secrets.token_hex(16)))
    cwd = request.workspace / "child-cwd"
    temporary = request.workspace / "temp"
    home_proxy = request.workspace / "home-proxy"
    cwd.mkdir(exist_ok=False)
    temporary.mkdir(exist_ok=False)
    home_proxy.mkdir(exist_ok=False)
    home_proxy_before = snapshot_protected_path(home_proxy)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    base = dict(os.environ if base_environment is None else base_environment)
    environment = build_child_environment(
        base=base,
        scripts=python.parent,
        profile=request.profile,
        temporary=temporary,
        home_proxy=home_proxy,
    )
    command_hash = hashlib.sha256()

    def run(command: tuple[str, ...]) -> bytes:
        revalidate_all_inputs()
        stage_before = {
            name: snapshot_protected_path(root)
            for name, root in (
                ("default", request.default_root),
                ("active", request.active_profile),
                ("evidence", request.evidence_root),
            )
        }
        stage_home_before = snapshot_protected_path(home_proxy)
        failures: list[BaseException] = []
        output = b""
        try:
            command_hash.update(json.dumps(command, separators=(",", ":")).encode("utf-8"))
            child = job.launch(
                command, cwd=cwd, environment=environment, role="hermes_pluginmanager_root"
            )
            code, output = job.wait_for_exit(child, timeout_seconds=300, output_limit=4096)
            if type(code) is not int or code != 0:
                raise RuntimeError("PluginManager governed child failed")
        except BaseException as error:
            failures.append(error)
        try:
            stage_after = {
                name: snapshot_protected_path(root)
                for name, root in (
                    ("default", request.default_root),
                    ("active", request.active_profile),
                    ("evidence", request.evidence_root),
                )
            }
            stage_home_after = snapshot_protected_path(home_proxy)
            if (
                stage_before != stage_after
                or stage_after != before
                or stage_home_before != stage_home_after
                or stage_home_after != home_proxy_before
            ):
                raise RuntimeError("protected root changed during PluginManager child")
        except BaseException as error:
            failures.append(error)
        try:
            revalidate_all_inputs()
        except BaseException as error:
            failures.append(error)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "PluginManager child and post-stage validation failed", failures
            )
        return output

    installer_owned_dist_info = {"RECORD", "INSTALLER", "REQUESTED", "direct_url.json"}
    with zipfile.ZipFile(request.candidate_wheel) as candidate_archive:
        candidate_names = candidate_archive.namelist()
        metadata_names = [name for name in candidate_names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in candidate_names if name.endswith(".dist-info/WHEEL")]
        entrypoint_names = [
            name for name in candidate_names if name.endswith(".dist-info/entry_points.txt")
        ]
        if (
            len(metadata_names) != 1
            or len(wheel_names) != 1
            or len(entrypoint_names) != 1
            or "hermes_realtime/hermes_plugin.py" not in candidate_names
        ):
            raise ValueError("candidate distribution provenance is incomplete")
        candidate_dist_info_name = metadata_names[0].split("/", 1)[0]
        expected_immutable_dist_info = sorted(
            name
            for name in candidate_names
            if name.startswith(candidate_dist_info_name + "/")
            and not name.endswith("/")
            and name.rsplit("/", 1)[-1] not in installer_owned_dist_info
        )
        if not expected_immutable_dist_info:
            raise ValueError("candidate immutable dist-info inventory is empty")

    try:
        run((str(request.build_python), "-m", "venv", str(venv)))
        run(
            (
                str(python),
                "-I",
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
                "--require-hashes",
                "--find-links",
                str(request.wheelhouse),
                "-r",
                str(request.requirements),
                "-c",
                str(request.constraints),
                "--no-deps",
            )
        )
        disabled = _canonical_stage(
            run(
                (
                    str(python),
                    "-I",
                    "-c",
                    _DISABLED_CHILD,
                    str(hermes_source),
                    str(request.profile),
                    str(venv / "Lib" / "site-packages"),
                )
            ),
            {"discovered", "importOrRegistration", "sourceOriginSha256"},
        )
        enabled_config = _canonical_stage(
            run(
                (
                    str(python),
                    "-I",
                    "-c",
                    _ENABLE_CHILD,
                    str(hermes_source),
                    str(request.profile),
                    str(venv / "Lib" / "site-packages"),
                )
            ),
            {"argvExact", "exactConfigDelta", "stdinClosed"},
        )
        enabled = _canonical_stage(
            run(
                (
                    str(python),
                    "-I",
                    "-c",
                    _ENABLED_CHILD,
                    str(hermes_source),
                    str(request.profile),
                    str(venv / "Lib" / "site-packages"),
                    json.dumps(expected_immutable_dist_info, separators=(",", ":")),
                )
            ),
            {
                "contextObserved",
                "discovered",
                "distributionVersion",
                "distEntryPointsContentSha256",
                "distInfoInventorySha256",
                "distMetadataContentSha256",
                "distOriginSha256",
                "moduleContentSha256",
                "moduleOriginSha256",
            },
        )
        if disabled["discovered"] is not True or disabled["importOrRegistration"] is not False:
            raise ValueError("disabled PluginManager state is not exact")
        expected_source_origin = hashlib.sha256(
            str(hermes_source.resolve(strict=True)).encode()
        ).hexdigest()
        packages = (venv / "Lib" / "site-packages").resolve(strict=False)
        expected_module = (packages / "hermes_realtime" / "hermes_plugin.py").resolve(strict=False)
        expected_module_origin = hashlib.sha256(str(expected_module).encode()).hexdigest()
        with zipfile.ZipFile(request.candidate_wheel) as candidate_archive:
            expected_module_content = hashlib.sha256(
                candidate_archive.read("hermes_realtime/hermes_plugin.py")
            ).hexdigest()
            metadata_names = [
                name
                for name in candidate_archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            entrypoint_names = [
                name
                for name in candidate_archive.namelist()
                if name.endswith(".dist-info/entry_points.txt")
            ]
            if len(metadata_names) != 1 or len(entrypoint_names) != 1:
                raise ValueError("candidate distribution provenance is incomplete")
            expected_metadata_content = hashlib.sha256(
                candidate_archive.read(metadata_names[0])
            ).hexdigest()
            expected_entrypoints_content = hashlib.sha256(
                candidate_archive.read(entrypoint_names[0])
            ).hexdigest()
            dist_info_name = metadata_names[0].split("/", 1)[0]
            expected_dist_info = (packages / dist_info_name).resolve(strict=False)
            expected_dist_origin = hashlib.sha256(str(expected_dist_info).encode()).hexdigest()
            dist_info_inventory = {
                name: hashlib.sha256(candidate_archive.read(name)).hexdigest()
                for name in expected_immutable_dist_info
            }
            expected_dist_inventory = hashlib.sha256(
                json.dumps(dist_info_inventory, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
        if (
            disabled["sourceOriginSha256"] != expected_source_origin
            or enabled["moduleOriginSha256"] != expected_module_origin
            or enabled["moduleContentSha256"] != expected_module_content
            or enabled["distOriginSha256"] != expected_dist_origin
            or enabled["distMetadataContentSha256"] != expected_metadata_content
            or enabled["distEntryPointsContentSha256"] != expected_entrypoints_content
            or enabled["distInfoInventorySha256"] != expected_dist_inventory
        ):
            raise ValueError("PluginManager parent-recomputed origin is invalid")
        after = {
            name: snapshot_protected_path(root)
            for name, root in (
                ("default", request.default_root),
                ("active", request.active_profile),
                ("evidence", request.evidence_root),
            )
        }
        revalidate_inputs()
        result = {
            "activeProfileUnchanged": before["active"] == after["active"],
            "defaultRootUnchanged": before["default"] == after["default"],
            "disabledEntryPointDiscovered": disabled["discovered"] is True,
            "disabledImportOrRegistration": False,
            "distributionVersion": enabled["distributionVersion"],
            "enabledEntryPointDiscovered": enabled["discovered"] is True,
            "evidenceRootUnchanged": before["evidence"] == after["evidence"],
            "exactConfigDelta": enabled_config["argvExact"] is True
            and enabled_config["stdinClosed"] is True
            and enabled_config["exactConfigDelta"] is True,
            "pluginContextRegistrationObserved": enabled["contextObserved"] is True,
        }
        payload = canonical_child_result(result)
        revalidate_inputs()
        temporary = request.output.with_name(
            request.output.name + "." + secrets.token_hex(16) + ".tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, request.output)
        finally:
            if temporary.exists():
                temporary.unlink()
        reopened = request.output.read_bytes()
        parsed = parse_child_result(reopened)
        revalidate_inputs()
        facts: dict[str, object] = {
            "commandSha256": command_hash.hexdigest(),
            "outputSha256": hashlib.sha256(reopened).hexdigest(),
            "result": parsed,
        }
    except BaseException as primary:
        try:
            extracted_source.close(cleanup=True)
        except BaseException as cleanup:
            raise BaseExceptionGroup(
                "PluginManager primary and native source finalization failed", [primary, cleanup]
            ) from None
        raise
    try:
        extracted_source.close(cleanup=True)
    except BaseException as cleanup:
        raise BaseExceptionGroup(
            "PluginManager native source finalization failed", [cleanup]
        ) from None
    return facts


def _qualify_governed_request_unretained(
    request: ParentRequest,
    *,
    job: _Job,
    base_environment: Mapping[str, str] | None = None,
    revalidate_inputs: Callable[[], None],
    source_archive_payload: bytes,
    wheelhouse_manifest_payload: bytes,
    venv_name: str | None = None,
) -> dict[str, object]:
    return _qualify_governed_request_body(
        request,
        job=job,
        base_environment=base_environment,
        revalidate_inputs=revalidate_inputs,
        source_archive_payload=source_archive_payload,
        wheelhouse_manifest_payload=wheelhouse_manifest_payload,
        venv_name=venv_name,
    )


def _qualify_governed_request(
    request: ParentRequest,
    *,
    job: _Job,
    base_environment: Mapping[str, str] | None = None,
    revalidate_inputs: Callable[[], None],
    venv_name: str | None = None,
) -> dict[str, object]:
    try:
        request = validate_parent_request(request)
        if os.name != "nt":
            raise RuntimeError("PluginManager governed qualification is Windows-only")
        with WindowsRetainedImmutableInputs(request) as retained:

            def revalidate_retained() -> None:
                revalidate_inputs()
                retained.assert_unchanged()

            source_archive_payload = retained.read_retained_bytes(
                request.hermes_source_archive,
                maximum_bytes=_MAX_HERMES_SOURCE_ARCHIVE_BYTES,
            )
            wheelhouse_manifest_payload = retained.read_retained_bytes(request.wheelhouse_manifest)
            facts = _qualify_governed_request_unretained(
                request,
                job=job,
                base_environment=base_environment,
                revalidate_inputs=revalidate_retained,
                source_archive_payload=source_archive_payload,
                wheelhouse_manifest_payload=wheelhouse_manifest_payload,
                venv_name=venv_name,
            )
            retained.assert_unchanged()
    except BaseException as primary_error:
        try:
            job.force_finalize(timeout_seconds=30)
        except BaseException as finalization_error:
            raise BaseExceptionGroup(
                "PluginManager qualification and Job finalization both failed",
                [primary_error, finalization_error],
            ) from None
        raise
    job.force_finalize(timeout_seconds=30)
    return facts


def main() -> int:
    fail(
        "PluginManager qualification is a private parent-bound API; "
        "Task13 must supply the validated closure and revalidation callback"
    )


if __name__ == "__main__":
    raise SystemExit(main())

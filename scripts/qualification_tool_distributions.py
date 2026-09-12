"""Admit complete tool distributions against governed release digests before parsing.

These immutable bytes establish distribution provenance, not an installed or
executing tool environment. Consumers must own and seal the complete materialized
namespace and bind their actual invocations separately.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts.qualification_file_seals import _relative_windows_member

_MAX_MEMBERS = 16384
_MAX_MEMBER = 128 * 1024**2
_MAX_EXPANDED = 512 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _ToolPolicy:
    version: str
    distribution: str
    executable: str
    format: str
    sha256: str
    size: int


# The published assets were independently read through the respective official
# repositories. Changing a digest or selected distribution requires source review.
# https://github.com/git-for-windows/git/releases/tag/v2.55.0.windows.5
# https://github.com/astral-sh/uv/releases/tag/0.11.28
# https://github.com/astral-sh/python-build-standalone/releases/tag/20260901
_TOOLS = {
    "git": _ToolPolicy(
        "2.55.0.windows.5",
        "MinGit-2.55.0.5-64-bit.zip",
        "mingw64/bin/git.exe",
        "zip",
        "56d7b226b7693196cfc71fef26568f536c4a021ab6c37ff2db4287bed908e96e",
        38989688,
    ),
    "uv": _ToolPolicy(
        "0.11.28",
        "uv-x86_64-pc-windows-msvc.zip",
        "uv.exe",
        "zip",
        "0a23463216d09c6a72ff80ef5dc5a795f07dc1575cb84d24596c2f124a441b7b",
        25568726,
    ),
    "build_python": _ToolPolicy(
        "3.11.16",
        "cpython-3.11.16+20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz",
        "python/python.exe",
        "tar.gz",
        "06cbe479e039f5b9cb5640c286d790074d63f549f92a32d599a3748293bd4510",
        25189257,
    ),
}


@dataclass(frozen=True, slots=True)
class ToolDistributionMetadataV1:
    role: str
    version: str
    distribution_sha256: str
    file_count: int
    expanded_bytes: int


class AdmittedToolDistributionV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("tool distributions are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Distribution:
    policy: _ToolPolicy
    metadata: ToolDistributionMetadataV1
    files: tuple[tuple[str, bytes], ...]


_ADMITTED: WeakKeyDictionary[AdmittedToolDistributionV1, _Distribution] = WeakKeyDictionary()


def _inspect_archive_members(raw: bytes, format: str) -> tuple[tuple[str, bytes], ...]:
    """Read one bounded ordinary-file archive namespace without selecting a tool."""
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    explicit: set[str] = set()
    spelling: dict[str, str] = {}
    expanded = 0

    def add(name: str, payload: bytes | None) -> None:
        nonlocal expanded
        _require(len(explicit) < _MAX_MEMBERS, "tool distribution member bound exceeded")
        _relative_windows_member(name)
        parts = name.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            key = prefix.casefold()
            _require(
                spelling.setdefault(key, prefix) == prefix,
                "tool distribution member spelling is ambiguous",
            )
        _require(name not in explicit, "tool distribution repeats a member")
        explicit.add(name)
        parents = {"/".join(parts[:index]) for index in range(1, len(parts))}
        _require(not parents.intersection(files), "tool distribution namespace collides")
        directories.update(parents)
        if payload is None:
            _require(name not in files, "tool distribution directory collides")
            directories.add(name)
        else:
            _require(
                name not in directories and len(payload) <= _MAX_MEMBER,
                "tool distribution file differs or exceeds its bound",
            )
            expanded += len(payload)
            _require(expanded <= _MAX_EXPANDED, "tool distribution expanded bound exceeded")
            files[name] = payload

    if format == "zip":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            _require(len(archive.infolist()) <= _MAX_MEMBERS, "tool member bound exceeded")
            for member in archive.infolist():
                directory = member.is_dir()
                kind = stat.S_IFMT(member.external_attr >> 16)
                _require(
                    not member.flag_bits & 1
                    and member.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    and kind in ({0, stat.S_IFDIR} if directory else {0, stat.S_IFREG})
                    and 0 <= member.file_size <= _MAX_MEMBER,
                    "tool ZIP member is not a bounded ordinary file",
                )
                if directory:
                    _require(member.file_size == 0, "tool directory carries data")
                    add(member.filename.removesuffix("/"), None)
                else:
                    with archive.open(member) as stream:
                        payload = stream.read(_MAX_MEMBER + 1)
                    _require(len(payload) == member.file_size, "tool ZIP member size differs")
                    add(member.filename, payload)
    else:
        _require(format == "tar.gz", "tool distribution format differs")
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
            expanded_tar = compressed.read(_MAX_EXPANDED + 1)
        _require(len(expanded_tar) <= _MAX_EXPANDED, "tool tar stream exceeds its bound")
        with tarfile.open(fileobj=io.BytesIO(expanded_tar), mode="r:") as tar_archive:
            for tar_member in tar_archive:
                _require(
                    (tar_member.isfile() or tar_member.isdir())
                    and not tar_member.sparse
                    and not tar_member.mode & 0o7000
                    and 0 <= tar_member.size <= _MAX_MEMBER,
                    "tool tar member is not a bounded ordinary file",
                )
                if tar_member.isdir():
                    _require(tar_member.size == 0, "tool directory carries data")
                    add(tar_member.name.removesuffix("/"), None)
                else:
                    tar_stream = tar_archive.extractfile(tar_member)
                    _require(tar_stream is not None, "tool tar member is unreadable")
                    assert tar_stream is not None
                    with tar_stream:
                        payload = tar_stream.read(_MAX_MEMBER + 1)
                    _require(len(payload) == tar_member.size, "tool tar member size differs")
                    add(tar_member.name, payload)
    return tuple(sorted(files.items()))


def _inspect(raw: bytes, policy: _ToolPolicy) -> tuple[tuple[str, bytes], ...]:
    files = _inspect_archive_members(raw, policy.format)
    _require(policy.executable in dict(files), "selected tool image is absent")
    return files


def admit_tool_distribution(role: str, raw: bytes) -> AdmittedToolDistributionV1:
    """No caller-selected version, origin, digest or member can authorize a tool."""
    _require(type(role) is str and role in _TOOLS, "tool distribution role is unavailable")
    policy = _TOOLS[role]
    _require(
        type(raw) is bytes
        and len(raw) == policy.size
        and hashlib.sha256(raw).hexdigest() == policy.sha256,
        "tool distribution differs from its governed release digest",
    )
    files = _inspect(raw, policy)
    metadata = ToolDistributionMetadataV1(
        role,
        policy.version,
        policy.sha256,
        len(files),
        sum(len(payload) for _, payload in files),
    )
    receipt = object.__new__(AdmittedToolDistributionV1)
    _ADMITTED[receipt] = _Distribution(policy, metadata, files)
    return receipt


def _distribution(receipt: AdmittedToolDistributionV1) -> _Distribution:
    if type(receipt) is not AdmittedToolDistributionV1:
        raise TypeError("tool distribution capability type differs")
    _require(receipt in _ADMITTED, "tool distribution capability is unregistered")
    value = _ADMITTED[receipt]
    _require(
        _TOOLS.get(value.metadata.role) == value.policy,
        "tool distribution governing policy changed",
    )
    return value


def tool_distribution_metadata(receipt: AdmittedToolDistributionV1) -> ToolDistributionMetadataV1:
    return _distribution(receipt).metadata


def _tool_distribution_files(receipt: AdmittedToolDistributionV1) -> tuple[tuple[str, bytes], ...]:
    return _distribution(receipt).files

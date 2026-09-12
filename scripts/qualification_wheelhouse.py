"""Inspect a closed, offline wheel set without executing it or claiming installation.

This returns byte-derived metadata, not provenance or execution authority. The
consumer must retain authenticated input bytes and independently own installation.
Resolved requirement files contain exact pins and one hash per wheel; installer
directives, URLs, editable sources and unresolved markers are outside this format.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import re
import stat
import zipfile
from collections import deque
from dataclasses import dataclass
from email.parser import BytesParser
from typing import cast
from weakref import WeakKeyDictionary

from packaging.markers import Environment
from packaging.metadata import Metadata, parse_email
from packaging.requirements import Requirement
from packaging.tags import Tag, compatible_tags, cpython_tags, parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from scripts.qualification_file_seals import _relative_windows_member
from scripts.qualify_evidence_slice_zero import _safe_posix_path

_PIN = re.compile(
    r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.!+_-]*)"
    r"(?: --hash=sha256:([0-9a-f]{64}))?"
)
_MAX_WHEEL_BYTES = 4 * 1024**3
_MAX_EXPANDED_BYTES = 16 * 1024**3
_MAX_METADATA_BYTES = 4 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class WheelDistributionV1:
    """Private wheel/member inventory; no execution or installation capability."""

    name: str
    version: str
    sha256: str
    requires: tuple[str, ...]
    members: tuple[tuple[str, str, int], ...]
    extras: tuple[str, ...]


class AuthenticatedLinuxWheelTargetV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Linux wheel targets are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _LinuxWheelRecipeBinding:
    candidate_commit: str
    candidate_tree: str
    source_archive_sha256: str
    direct_wheel: tuple[str, str, int]
    manifest_sha256: str
    source_lock_sha256: str
    requirements_sha256: str
    constraints_sha256: str
    wheels: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True, slots=True)
class _LinuxTarget:
    environment: Environment
    tags: frozenset[Tag]
    recipe: _LinuxWheelRecipeBinding
    image: tuple[str, str, tuple[str, ...], tuple[str, ...]]


_LINUX_TARGETS: WeakKeyDictionary[AuthenticatedLinuxWheelTargetV1, _LinuxTarget] = (
    WeakKeyDictionary()
)


def _mint_authenticated_linux_target(
    environment: Environment,
    tags: set[Tag],
    recipe: _LinuxWheelRecipeBinding,
    image: tuple[str, str, tuple[str, ...], tuple[str, ...]],
) -> AuthenticatedLinuxWheelTargetV1:
    _require(
        environment["sys_platform"] == "linux"
        and environment["platform_machine"] == "x86_64"
        and bool(tags),
        "authenticated Linux wheel target differs",
    )
    receipt = object.__new__(AuthenticatedLinuxWheelTargetV1)
    _LINUX_TARGETS[receipt] = _LinuxTarget(
        cast(Environment, dict(environment)),
        frozenset(tags),
        recipe,
        image,
    )
    return receipt


def _authenticated_linux_target_for_consumer(
    receipt: AuthenticatedLinuxWheelTargetV1,
) -> tuple[Environment, set[Tag]]:
    if type(receipt) is not AuthenticatedLinuxWheelTargetV1:
        raise TypeError("authenticated Linux wheel target type differs")
    _require(receipt in _LINUX_TARGETS, "authenticated Linux wheel target is unregistered")
    value = _LINUX_TARGETS[receipt]
    return cast(Environment, dict(value.environment)), set(value.tags)


def _authenticated_linux_binding_for_consumer(
    receipt: AuthenticatedLinuxWheelTargetV1,
) -> tuple[_LinuxWheelRecipeBinding, tuple[str, str, tuple[str, ...], tuple[str, ...]]]:
    _authenticated_linux_target_for_consumer(receipt)
    value = _LINUX_TARGETS[receipt]
    return value.recipe, value.image


def _pins(raw: bytes, *, hashed: bool) -> dict[str, tuple[Version, str | None]]:
    _require(
        type(raw) is bytes and len(raw) <= _MAX_METADATA_BYTES, "wheel pins exceed their bound"
    )
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise ValueError("wheel pins are not ASCII") from error
    _require("\r" not in text and "\x00" not in text, "wheel pins are not canonical text")
    result: dict[str, tuple[Version, str | None]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _PIN.fullmatch(line)
        _require(match is not None, "wheel pins contain an unresolved or indirect requirement")
        assert match is not None
        name, version, digest = match.groups()
        name = canonicalize_name(name)
        _require(
            name not in result and (digest is not None) == hashed,
            "wheel pin identity or hash differs",
        )
        result[name] = (Version(version), digest)
    return result


def _target(python_version: str, platform: str) -> tuple[Environment, set[Tag]]:
    version = Version(python_version)
    _require(
        len(version.release) == 3 and version.release[:2] == (3, 11) and not version.is_prerelease,
        "wheel target Python is outside the qualified CPython 3.11 boundary",
    )
    _require(platform in {"windows_amd64", "linux_x86_64"}, "wheel target platform is unavailable")
    windows = platform == "windows_amd64"
    # Linux native wheels require a genuine executor's ABI/platform inventory;
    # this preliminary inspector conservatively admits generic linux_x86_64 only.
    platforms = ["win_amd64" if windows else "linux_x86_64"]
    tags = set(cpython_tags((3, 11), abis=["cp311"], platforms=platforms))
    tags.update(compatible_tags((3, 11), interpreter="cp311", platforms=platforms))
    environment: Environment = {
        "implementation_name": "cpython",
        "implementation_version": str(version),
        "os_name": "nt" if windows else "posix",
        "platform_machine": "AMD64" if windows else "x86_64",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_version": "",
        "platform_system": "Windows" if windows else "Linux",
        "python_full_version": str(version),
        "python_version": "3.11",
        "sys_platform": "win32" if windows else "linux",
    }
    return environment, tags


def _metadata(raw: bytes, info: str, members: dict[str, tuple[str, int]]) -> Metadata:
    parsed, unparsed = parse_email(raw)
    _require(not unparsed, "wheel metadata contains invalid or unknown fields")
    legacy = parsed.get("metadata_version") in {"2.1", "2.2", "2.3"}
    licenses = parsed.get("license_files", [])
    _require(
        type(licenses) is list
        and len(licenses) <= 64
        and all(type(name) is str for name in licenses)
        and len({name.casefold() for name in licenses}) == len(licenses),
        "wheel license file references are ambiguous or unbounded",
    )
    for name in licenses:
        _safe_posix_path(name, label="wheel license member")
        locations: tuple[str, ...] = (info + "/licenses/" + name,)
        if legacy:
            locations += (info + "/" + name,)
        _require(
            sum(location in members for location in locations) == 1,
            "wheel license file is absent or ambiguous",
        )
    if legacy:
        # Setuptools published this extension before Core Metadata 2.4. Validate
        # its legacy dist-info or early Hatch licenses layout separately; retain every byte
        # and all other metadata validation, including dependency declarations.
        # https://peps.python.org/pep-0639/appendix-license-survey/#setuptools-and-wheel
        parsed.pop("license_files", None)
    return Metadata.from_raw(parsed, validate=True)


def _inspect(
    name: str,
    raw: bytes,
    environment: Environment,
    target_tags: set[Tag],
    *,
    site_processing: bool = True,
) -> WheelDistributionV1:
    _require(
        type(name) is str and "/" not in name and "\\" not in name,
        "wheel identity must be a basename",
    )
    distribution, version, _, tags = parse_wheel_filename(name)
    _require(bool(tags & target_tags), "wheel target tags are incompatible")
    _require(
        type(raw) is bytes and 0 < len(raw) <= _MAX_WHEEL_BYTES, "wheel bytes exceed their bound"
    )
    metadata_files: dict[str, bytes] = {}
    members: dict[str, tuple[str, int]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            _require(0 < len(entries) <= 65536, "wheel member count exceeds its bound")
            _require(
                sum(entry.file_size for entry in entries) <= _MAX_EXPANDED_BYTES,
                "wheel expansion exceeds its bound",
            )
            seen: set[str] = set()
            for entry in entries:
                path = entry.filename
                _relative_windows_member(path[:-1] if entry.is_dir() else path)
                _require(path.casefold() not in seen, "wheel member aliases another entry")
                seen.add(path.casefold())
                _require(
                    not entry.flag_bits & 1
                    and stat.S_IFMT(entry.external_attr >> 16) in {0, stat.S_IFREG, stat.S_IFDIR},
                    "wheel member is encrypted or indirect",
                )
                if entry.is_dir():
                    _require(entry.file_size == 0, "wheel directory contains bytes")
                    continue
                _require(
                    not site_processing or not path.lower().endswith(".pth"),
                    "wheel startup hooks require separate reviewed authority",
                )
                digest = hashlib.sha256()
                size = 0
                with archive.open(entry) as stream:
                    while chunk := stream.read(1024**2):
                        size += len(chunk)
                        _require(size <= entry.file_size, "wheel member expansion differs")
                        digest.update(chunk)
                _require(size == entry.file_size, "wheel member size differs")
                members[path] = (digest.hexdigest(), size)
                if path.count("/") == 1 and path.endswith(
                    (".dist-info/METADATA", ".dist-info/WHEEL", ".dist-info/RECORD")
                ):
                    _require(size <= _MAX_METADATA_BYTES, "wheel metadata exceeds its bound")
                    metadata_files[path] = archive.read(entry)
    except (zipfile.BadZipFile, RuntimeError, OSError) as error:
        raise ValueError("wheel archive is unreadable") from error
    roots = {
        path.split("/", 1)[0] for path in members if path.split("/", 1)[0].endswith(".dist-info")
    }
    _require(len(roots) == 1, "wheel metadata identity is ambiguous")
    info = next(iter(roots))
    _require(
        set(metadata_files) == {f"{info}/{part}" for part in ("METADATA", "WHEEL", "RECORD")},
        "wheel required metadata is absent",
    )
    try:
        metadata = _metadata(metadata_files[f"{info}/METADATA"], info, members)
        wheel = BytesParser().parsebytes(metadata_files[f"{info}/WHEEL"])
        record = list(
            csv.reader(io.StringIO(metadata_files[f"{info}/RECORD"].decode("utf-8")), strict=True)
        )
    except (ValueError, ExceptionGroup, UnicodeError, csv.Error) as error:
        raise ValueError("wheel metadata is invalid") from error
    _require(
        canonicalize_name(metadata.name) == distribution and metadata.version == version,
        "wheel metadata differs from filename",
    )
    info_name = info.removesuffix(".dist-info").rsplit("-", 1)
    _require(
        len(info_name) == 2
        and canonicalize_name(info_name[0]) == distribution
        and Version(info_name[1]) == version,
        "wheel metadata directory identity differs",
    )
    _require(
        not wheel.defects
        and not wheel.is_multipart()
        and wheel.get_all("Wheel-Version") == ["1.0"]
        and wheel.get_all("Root-Is-Purelib") in (["true"], ["false"]),
        "wheel format differs",
    )
    declared_tags = set().union(*(parse_tag(tag) for tag in wheel.get_all("Tag", [])))
    _require(declared_tags == tags, "wheel header tags differ from filename")
    _require(
        metadata.requires_python is None
        or metadata.requires_python.contains(environment["python_full_version"]),
        "wheel Requires-Python is incompatible",
    )
    _require(
        len(record) == len(members)
        and all(len(row) == 3 for row in record)
        and len({row[0] for row in record}) == len(record)
        and {row[0] for row in record} == set(members),
        "wheel RECORD closure differs",
    )
    for path, record_digest, record_size in record:
        actual_digest, actual_size = members[path]
        encoded = base64.urlsafe_b64encode(bytes.fromhex(actual_digest)).decode().rstrip("=")
        expected = ("", "") if path == f"{info}/RECORD" else ("sha256=" + encoded, str(actual_size))
        _require((record_digest, record_size) == expected, "wheel RECORD bytes differ")
    return WheelDistributionV1(
        str(distribution),
        str(version),
        hashlib.sha256(raw).hexdigest(),
        tuple(str(item) for item in metadata.requires_dist or ()),
        tuple((path, digest, size) for path, (digest, size) in sorted(members.items())),
        tuple(str(extra) for extra in metadata.provides_extra or ()),
    )


def _verify_installation_namespace(distributions: list[WheelDistributionV1]) -> None:
    """Refuse overwrites after wheel purelib/platlib relocation, before installation.

    Other data schemes need a genuinely observed target installation layout;
    this file inspector does not silently infer those destinations.
    """
    files: set[str] = set()
    directories: set[str] = set()
    for distribution in distributions:
        metadata = next(
            path.split("/", 1)[0]
            for path, _, _ in distribution.members
            if path.endswith(".dist-info/METADATA")
        )
        data = metadata.removesuffix(".dist-info") + ".data"
        for path, _, _ in distribution.members:
            parts = path.split("/")
            if parts[0].endswith(".data"):
                _require(
                    parts[0] == data and len(parts) >= 3 and parts[1] in {"purelib", "platlib"},
                    "wheel installation scheme requires separate target authority",
                )
                parts = parts[2:]
            key = "/".join(parts).casefold()
            parents = {"/".join(parts[:i]).casefold() for i in range(1, len(parts))}
            _require(
                key not in files and key not in directories and not parents.intersection(files),
                "wheel installation namespace is ambiguous",
            )
            files.add(key)
            directories.update(parents)


def _inspect_wheelhouse_for_target(
    *,
    requirements: bytes,
    constraints: bytes,
    wheels: dict[str, bytes],
    environment: Environment,
    tags: set[Tag],
    root_extras: dict[str, tuple[str, ...]] | None = None,
    roots: tuple[str, ...] | None = None,
    site_processing: bool = True,
) -> tuple[WheelDistributionV1, ...]:
    """Inspect with a target derived by an owning verifier, never mint authority."""
    _require(type(site_processing) is bool, "wheel site processing profile differs")
    _require(type(wheels) is dict and 0 < len(wheels) <= 2048, "wheel set exceeds its bound")
    _require(type(environment) is dict and type(tags) is set and bool(tags), "wheel target differs")
    pins, restrictions = _pins(requirements, hashed=True), _pins(constraints, hashed=False)
    inspected = [
        _inspect(name, raw, environment, tags, site_processing=site_processing)
        for name, raw in wheels.items()
    ]
    _verify_installation_namespace(inspected)
    by_name = {item.name: item for item in inspected}
    _require(
        len(by_name) == len(inspected) and set(by_name) == set(pins),
        "wheel pinned distribution set differs",
    )
    for name, item in by_name.items():
        _require(
            (Version(item.version), item.sha256) == pins[name],
            "wheel pin version or digest differs",
        )
    for name, (version, _) in restrictions.items():
        _require(
            name in by_name and Version(by_name[name].version) == version,
            "wheel constraint differs",
        )
    _require(
        sum(len(item.requires) for item in inspected) <= 16384
        and all(len(item.extras) <= 64 for item in inspected),
        "wheel dependency graph exceeds its bound",
    )
    requested = {} if root_extras is None else root_extras
    _require(
        type(requested) is dict and set(requested) <= set(by_name),
        "wheel requested extra root is absent",
    )
    active: dict[str, set[str]] = {name: set() for name in by_name}
    for name, values in requested.items():
        _require(
            type(values) is tuple
            and len(values) <= 64
            and all(type(value) is str for value in values)
            and tuple(sorted(set(values))) == values
            and set(values) <= set(by_name[name].extras),
            "wheel requested extra is unknown or ambiguous",
        )
        active[name].update(values)
    _require(
        roots is None
        or (
            type(roots) is tuple
            and bool(roots)
            and all(type(name) is str for name in roots)
            and tuple(sorted(set(roots))) == roots
            and set(roots) <= set(by_name)
            and set(requested) <= set(roots)
        ),
        "wheel dependency roots are missing or ambiguous",
    )
    reachable = set(by_name) if roots is None else set(roots)
    pending = reachable.copy()
    queue = deque(sorted(pending))
    while queue:
        name = queue.popleft()
        pending.remove(name)
        item = by_name[name]
        for text in item.requires:
            requirement = Requirement(text)
            _require(requirement.url is None, "wheel dependency URL is forbidden")
            if requirement.marker is not None:
                _require(
                    not any(
                        field in str(requirement.marker)
                        for field in ("platform_release", "platform_version")
                    ),
                    "wheel dependency needs a genuine target OS release inventory",
                )
                if not any(
                    requirement.marker.evaluate({**environment, "extra": extra})
                    for extra in ("", *sorted(active[name]))
                ):
                    continue
            dependency = by_name.get(canonicalize_name(requirement.name))
            _require(
                dependency is not None and requirement.specifier.contains(dependency.version),
                "wheel dependency closure is incomplete or incompatible",
            )
            assert dependency is not None
            extras = {str(canonicalize_name(extra)) for extra in requirement.extras}
            _require(extras <= set(dependency.extras), "wheel dependency extra is unavailable")
            added = extras - active[dependency.name]
            if added or dependency.name not in reachable:
                active[dependency.name].update(added)
                reachable.add(dependency.name)
                if dependency.name not in pending:
                    queue.append(dependency.name)
                    pending.add(dependency.name)
    _require(reachable == set(by_name), "wheel set contains an unrelated distribution")
    return tuple(by_name[name] for name in sorted(by_name))


def inspect_wheelhouse_files(
    *,
    requirements: bytes,
    constraints: bytes,
    wheels: dict[str, bytes],
    python_version: str,
    platform: str,
    root_extras: dict[str, tuple[str, ...]] | None = None,
    roots: tuple[str, ...] | None = None,
    site_processing: bool = True,
) -> tuple[WheelDistributionV1, ...]:
    """Derive a closed wheel inventory; no supplied data can mint install evidence."""
    environment, tags = _target(python_version, platform)
    return _inspect_wheelhouse_for_target(
        requirements=requirements,
        constraints=constraints,
        wheels=wheels,
        environment=environment,
        tags=tags,
        root_extras=root_extras,
        roots=roots,
        site_processing=site_processing,
    )

"""Bind retained dependency files to their purpose and genuine candidate bytes.

The candidate's source lock governs third-party wheel bytes. This does not prove
installation, publisher signatures, or the separate genuine Linux execution prerequisite.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
import tomllib
from dataclasses import dataclass
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from packaging.utils import canonicalize_name, parse_wheel_filename

from scripts import candidate_source_archive_oracle as archives
from scripts import qualify_evidence_slice_zero as core
from scripts.qualification_candidate_files import (
    BoundCandidateFilesV1,
    _candidate_files_for_consumer,
    candidate_file_metadata,
)
from scripts.qualification_file_seals import RetainedFileSealsV1, sealed_file_bytes
from scripts.qualification_wheelhouse import WheelDistributionV1, inspect_wheelhouse_files
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _include_candidate_wheel(
    wheels: dict[str, bytes],
    requirements: bytes,
    seals: RetainedFileSealsV1,
    path: str,
    digest: str,
) -> bytes:
    if not any(parse_wheel_filename(name)[0] == "hermes-realtime" for name in wheels):
        raw = sealed_file_bytes(seals, path, 16 * 1024**2)
        return _include_candidate_wheel_bytes(
            wheels, requirements, "hermes_realtime-0.0.3-py3-none-any.whl", raw, digest
        )
    return requirements


def _include_candidate_wheel_bytes(
    wheels: dict[str, bytes],
    requirements: bytes,
    basename: str,
    raw: bytes,
    digest: str,
) -> bytes:
    """Add the genuine candidate to a verified runtime recipe when it is separate."""
    if any(parse_wheel_filename(name)[0] == "hermes-realtime" for name in wheels):
        return requirements
    name, version, _, _ = parse_wheel_filename(basename)
    _require(
        name == "hermes-realtime"
        and hashlib.sha256(raw).hexdigest() == digest
        and digest == digest.lower(),
        "dependency candidate wheel differs from its genuine source binding",
    )
    wheels[basename] = raw
    return requirements + (
        f"\n{name}=={version} --hash=sha256:{digest}\n"
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class WheelhouseFileMetadataV1:
    purpose: str
    python_version: str
    platform: str
    distributions: tuple[WheelDistributionV1, ...]


@dataclass(frozen=True, slots=True)
class DependencyFileMetadataV1:
    qualification_input_sha256: str
    source_lock_sha256: str
    wheelhouses: tuple[WheelhouseFileMetadataV1, ...]


class BoundDependencyFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("dependency file bindings are verifier-minted only")


class BoundDependencyPurposeV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("dependency purpose bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    files: RetainedQualificationInputFilesV1
    candidate: BoundCandidateFilesV1
    metadata: DependencyFileMetadataV1


_BINDINGS: WeakKeyDictionary[BoundDependencyFilesV1, _Binding] = WeakKeyDictionary()
_PURPOSE_BINDINGS: WeakKeyDictionary[BoundDependencyPurposeV1, _Binding] = WeakKeyDictionary()


def _source_locked_wheels(
    candidate: BoundCandidateFilesV1,
) -> tuple[str, frozenset[tuple[str, str, str, str, int]]]:
    binding = _candidate_files_for_consumer(candidate)
    return _archive_locked_wheels(binding.archive, binding.identity)


def _archive_locked_wheels(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
) -> tuple[str, frozenset[tuple[str, str, str, str, int]]]:
    source = archives.verified_candidate_source_archive_metadata(archive)
    payload = archives._archive_bytes_for_consumer(archive, identity)
    entries = [item for item in source.manifest if item.path == "uv.lock"]
    _require(
        len(entries) == 1 and entries[0].kind == "file" and 0 < entries[0].size <= 4 * 1024**2,
        "dependency source lock is unavailable",
    )
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as contents:
        stream = contents.extractfile(source.prefix + "/uv.lock")
        _require(stream is not None, "dependency source lock is unreadable")
        assert stream is not None
        with stream:
            raw = stream.read(4 * 1024**2 + 1)
    digest = hashlib.sha256(raw).hexdigest()
    _require(
        (len(raw), digest) == (entries[0].size, entries[0].sha256),
        "dependency source lock differs from its candidate blob",
    )
    lock = tomllib.loads(raw.decode("utf-8"))
    _require(
        type(lock.get("version")) is int
        and lock["version"] == 1
        and type(lock.get("revision")) is int
        and lock["revision"] == 3,
        "dependency source lock format is unsupported",
    )
    packages = lock.get("package")
    _require(
        type(packages) is list and 0 < len(packages) <= 4096,
        "dependency source lock package bound differs",
    )
    assert isinstance(packages, list)
    allowed: set[tuple[str, str, str, str, int]] = set()
    names: set[tuple[str, str, str]] = set()
    for package in packages:
        _require(type(package) is dict, "dependency source lock package differs")
        if package.get("source") != {"registry": "https://pypi.org/simple"}:
            continue
        name, version = package.get("name"), package.get("version")
        _require(
            type(name) is str and type(version) is str,
            "dependency source lock package identity differs",
        )
        distributions = package.get("wheels", [])
        _require(type(distributions) is list, "dependency source lock wheels differ")
        for wheel in distributions:
            _require(
                type(wheel) is dict and len(allowed) < 65536,
                "dependency source lock wheel bound differs",
            )
            url, hashed, size = wheel.get("url"), wheel.get("hash"), wheel.get("size")
            _require(
                type(url) is str
                and type(hashed) is str
                and re.fullmatch(r"sha256:[0-9a-f]{64}", hashed) is not None
                and type(size) is int
                and 0 < size <= 4 * 1024**3,
                "dependency source lock wheel reference differs",
            )
            parsed = urlsplit(url)
            _require(
                parsed.scheme == "https"
                and parsed.netloc == "files.pythonhosted.org"
                and not parsed.query
                and not parsed.fragment,
                "dependency source lock origin differs",
            )
            basename = parsed.path.rsplit("/", 1)[-1]
            parsed_name, parsed_version, _, _ = parse_wheel_filename(basename)
            _require(
                parsed_name == canonicalize_name(name) and str(parsed_version) == version,
                "dependency source lock wheel identity differs",
            )
            key = (str(parsed_name), version, basename)
            _require(key not in names, "dependency source lock wheel is ambiguous")
            names.add(key)
            allowed.add((*key, hashed.removeprefix("sha256:"), size))
    return digest, frozenset(allowed)


def _inspect_dependency_files(
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
    purposes: tuple[str, ...],
) -> _Binding:
    """Read actual retained wheel bytes, including the candidate's dependency edges."""
    source = candidate_file_metadata(candidate)
    current = retained_input_metadata(files)
    _require(
        current.qualification_input_sha256 == source.qualification_input_sha256,
        "dependency input differs from its candidate binding",
    )
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    references = {item["role"]: item for item in document["files"]}
    tools = {item["role"]: item for item in document["toolIdentities"]}
    lock_digest, locked_wheels = _source_locked_wheels(candidate)
    observed = []
    for purpose in purposes:
        role = core._WHEELHOUSE_ROLES[purpose]
        manifest = json.loads(
            sealed_file_bytes(
                selected.seals,
                references[role]["relativePath"],
                4 * 1024**2,
            )
        )
        platform = "linux_x86_64" if purpose == "realtime_linux_runtime" else "windows_amd64"
        _require(manifest["platform"] == platform, "dependency purpose platform differs")
        if platform == "windows_amd64":
            _require(
                manifest["pythonVersion"] == document["expected"]["pythonFullVersion"],
                "dependency purpose Python differs",
            )
        wheels = {
            item["basename"]: sealed_file_bytes(selected.seals, item["relativePath"], 4 * 1024**3)
            for item in manifest["wheels"]
        }
        requirements = sealed_file_bytes(
            selected.seals,
            manifest["requirements"]["relativePath"],
            4 * 1024**2,
        )
        constraints = sealed_file_bytes(
            selected.seals,
            manifest["constraints"]["relativePath"],
            4 * 1024**2,
        )
        wheel_role = (
            "sdist_built_wheel"
            if purpose == "realtime_windows_sdist_built_runtime"
            else "direct_wheel"
        )
        for basename, raw in wheels.items():
            name, version, _, _ = parse_wheel_filename(basename)
            if name == "hermes-realtime" and purpose != "build":
                reference = references[wheel_role]
                _require(
                    (hashlib.sha256(raw).hexdigest(), len(raw))
                    == (reference["sha256"], reference["bytes"]),
                    "dependency candidate wheel differs from its genuine source binding",
                )
                continue
            _require(
                (str(name), str(version), basename, hashlib.sha256(raw).hexdigest(), len(raw))
                in locked_wheels,
                "dependency wheel differs from the candidate source lock",
            )
        if purpose != "build":
            reference = references[wheel_role]
            requirements = _include_candidate_wheel(
                wheels,
                requirements,
                selected.seals,
                reference["relativePath"],
                reference["sha256"],
            )
        inventory = inspect_wheelhouse_files(
            requirements=requirements,
            constraints=constraints,
            wheels=wheels,
            python_version=manifest["pythonVersion"],
            platform=platform,
            site_processing=False,
            roots=("hatchling",) if purpose == "build" else ("hermes-realtime",),
            root_extras={"hermes-realtime": ("local",)}
            if purpose
            in {"realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime"}
            else None,
        )
        if purpose == "build":
            hatchling = [item for item in inventory if item.name == "hatchling"]
            _require(
                len(hatchling) == 1
                and hatchling[0].version == tools["hatchling"]["version"] == "1.27.0"
                and hatchling[0].sha256 == tools["hatchling"]["artifact"]["sha256"],
                "dependency build tool differs from its admitted file role",
            )
            _require(
                tools["build_python"]["version"] == manifest["pythonVersion"],
                "dependency build Python differs from its tool role",
            )
        if purpose in {"realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime"}:
            for provider, package_name in (
                ("moonshine", "moonshine-voice"),
                ("kokoro", "kokoro-onnx"),
            ):
                distribution = [item for item in inventory if item.name == package_name]
                _require(
                    len(distribution) == 1
                    and distribution[0].version == document["expected"][provider + "Version"]
                    and distribution[0].sha256 == references[provider + "_distribution"]["sha256"],
                    "provider distribution differs from the admitted Windows dependency files",
                )
        observed.append(
            WheelhouseFileMetadataV1(purpose, manifest["pythonVersion"], platform, inventory)
        )
    candidate_file_metadata(candidate)
    retained_input_metadata(files)
    return _Binding(
        files,
        candidate,
        DependencyFileMetadataV1(
            current.qualification_input_sha256,
            lock_digest,
            tuple(observed),
        ),
    )


def bind_dependency_files(
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
) -> BoundDependencyFilesV1:
    """Bind all five purposes; a partial purpose cannot mint this capability."""
    binding = _inspect_dependency_files(files, candidate, tuple(sorted(core._WHEELHOUSE_ROLES)))
    receipt = object.__new__(BoundDependencyFilesV1)
    _BINDINGS[receipt] = binding
    return receipt


def bind_dependency_purpose(
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
    *,
    purpose: str,
) -> BoundDependencyPurposeV1:
    """Verify one installation's inputs without accepting the complete closure."""
    _require(
        type(purpose) is str and purpose in core._WHEELHOUSE_ROLES,
        "dependency purpose is unavailable",
    )
    binding = _inspect_dependency_files(files, candidate, (purpose,))
    receipt = object.__new__(BoundDependencyPurposeV1)
    _PURPOSE_BINDINGS[receipt] = binding
    return receipt


def dependency_file_metadata(receipt: BoundDependencyFilesV1) -> DependencyFileMetadataV1:
    if type(receipt) is not BoundDependencyFilesV1:
        raise TypeError("dependency file binding type differs")
    _require(receipt in _BINDINGS, "dependency file binding is unregistered")
    return _binding_metadata(_BINDINGS[receipt])


def _binding_metadata(binding: _Binding) -> DependencyFileMetadataV1:
    candidate_file_metadata(binding.candidate)
    _require(
        retained_input_metadata(binding.files).qualification_input_sha256
        == binding.metadata.qualification_input_sha256,
        "dependency file binding differs",
    )
    return binding.metadata


def dependency_purpose_metadata(receipt: BoundDependencyPurposeV1) -> DependencyFileMetadataV1:
    return _dependency_purpose_for_consumer(receipt).metadata


def _dependency_purpose_for_consumer(receipt: BoundDependencyPurposeV1) -> _Binding:
    if type(receipt) is not BoundDependencyPurposeV1:
        raise TypeError("dependency purpose capability type differs")
    _require(receipt in _PURPOSE_BINDINGS, "dependency purpose capability is unregistered")
    value = _PURPOSE_BINDINGS[receipt]
    _binding_metadata(value)
    return value


def _dependency_files_for_consumer(receipt: BoundDependencyFilesV1) -> _Binding:
    dependency_file_metadata(receipt)
    return _BINDINGS[receipt]


def _dependency_binding_for_consumer(
    receipt: BoundDependencyFilesV1 | BoundDependencyPurposeV1,
) -> _Binding:
    if isinstance(receipt, BoundDependencyPurposeV1):
        return _dependency_purpose_for_consumer(receipt)
    return _dependency_files_for_consumer(receipt)


def _dependency_wheels_for_consumer(
    receipt: BoundDependencyFilesV1 | BoundDependencyPurposeV1,
    purpose: str,
) -> tuple[dict[str, bytes], bytes, bytes]:
    """Reopen the same resolved recipe, including its separately bound candidate."""
    _require(
        type(purpose) is str and purpose in core._WHEELHOUSE_ROLES,
        "dependency purpose is unavailable",
    )
    value = _dependency_binding_for_consumer(receipt)
    _require(
        purpose in {item.purpose for item in value.metadata.wheelhouses},
        "dependency purpose is outside its verified coverage",
    )
    selected = _retained_input_files_for_consumer(value.files)
    document = json.loads(selected.document)
    references = {item["role"]: item for item in document["files"]}
    reference = references[core._WHEELHOUSE_ROLES[purpose]]
    manifest = json.loads(sealed_file_bytes(selected.seals, reference["relativePath"], 4 * 1024**2))
    wheels = {
        item["basename"]: sealed_file_bytes(selected.seals, item["relativePath"], 4 * 1024**3)
        for item in manifest["wheels"]
    }
    requirements = sealed_file_bytes(
        selected.seals, manifest["requirements"]["relativePath"], 4 * 1024**2
    )
    constraints = sealed_file_bytes(
        selected.seals, manifest["constraints"]["relativePath"], 4 * 1024**2
    )
    if purpose != "build":
        candidate = references[
            "sdist_built_wheel"
            if purpose == "realtime_windows_sdist_built_runtime"
            else "direct_wheel"
        ]
        requirements = _include_candidate_wheel(
            wheels, requirements, selected.seals, candidate["relativePath"], candidate["sha256"]
        )
    return wheels, requirements, constraints

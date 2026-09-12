"""Bind immutable source-locked build bytes before any candidate outputs exist.

This capability authenticates build inputs, not installation or build execution.
Consumers must install offline under independent process ownership and retain the
complete admitted tool environment until their final observed exit.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from packaging.utils import parse_wheel_filename

from scripts import candidate_source_archive_oracle as archives
from scripts.candidate_wheel import _metadata_source_files
from scripts.qualification_dependency_files import _archive_locked_wheels
from scripts.qualification_file_seals import sealed_file_bytes
from scripts.qualification_tool_distributions import ToolDistributionMetadataV1
from scripts.qualification_tool_environment import (
    ImmutableToolEnvironmentV1,
    _tool_image_for_consumer,
    tool_environment_metadata,
)
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


@dataclass(frozen=True, slots=True)
class BuildInputMetadataV1:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    source_lock_sha256: str
    project_sha256: str
    python_version: str
    requirements_sha256: str
    constraints_sha256: str
    distributions: tuple[WheelDistributionV1, ...]


class BoundBuildInputsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("build input bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _ToolImage:
    role: str
    version: str
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class _BuildInputs:
    archive: archives.VerifiedCandidateSourceArchiveV1
    identity: CandidateIdentityV1
    tools: ImmutableToolEnvironmentV1
    requirements: bytes
    constraints: bytes
    wheels: tuple[tuple[str, bytes], ...]
    metadata: BuildInputMetadataV1
    tool_distributions: tuple[ToolDistributionMetadataV1, ...]
    tool_images: tuple[_ToolImage, ...]


_BOUND: WeakKeyDictionary[BoundBuildInputsV1, _BuildInputs] = WeakKeyDictionary()


def bind_build_inputs(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
    tools: ImmutableToolEnvironmentV1,
    *,
    wheels: dict[str, bytes],
    requirements: bytes,
    constraints: bytes,
) -> BoundBuildInputsV1:
    """Require genuine source and live tool authority before parsing package bytes."""
    payload = archives._archive_bytes_for_consumer(archive, identity)
    source = archives.verified_candidate_source_archive_metadata(archive)
    _require(
        archives._archive_tool_capture_for_consumer(archive, identity)
        == tool_environment_metadata(tools),
        "build source capture tool distributions differ",
    )
    _, _, python_version = _tool_image_for_consumer(tools, "build_python")
    project, _ = _metadata_source_files(payload, source)
    _require(
        tomllib.loads(project.decode("utf-8")).get("build-system")
        == {"requires": ["hatchling==1.27.0"], "build-backend": "hatchling.build"},
        "build backend differs from the governed source profile",
    )
    lock_digest, allowed = _archive_locked_wheels(archive, identity)
    _require(type(wheels) is dict and 0 < len(wheels) <= 2048, "build wheel set exceeds its bound")
    frozen: list[tuple[str, bytes]] = []
    for basename, raw in wheels.items():
        _require(
            type(basename) is str and type(raw) is bytes and 0 < len(raw) <= 4 * 1024**3,
            "build wheel file exceeds its bound",
        )
        name, version, _, _ = parse_wheel_filename(basename)
        _require(
            (str(name), str(version), basename, hashlib.sha256(raw).hexdigest(), len(raw))
            in allowed,
            "build wheel differs from the candidate source lock",
        )
        frozen.append((basename, raw))
    inventory = inspect_wheelhouse_files(
        requirements=requirements,
        constraints=constraints,
        wheels=dict(frozen),
        python_version=python_version,
        platform="windows_amd64",
        roots=("hatchling",),
        site_processing=False,
    )
    _require(
        any(item.name == "hatchling" and item.version == "1.27.0" for item in inventory),
        "build backend distribution differs",
    )
    images = []
    for role in ("git", "uv", "build_python"):
        image, digest, version = _tool_image_for_consumer(tools, role)
        images.append(_ToolImage(role, version, digest, image.stat().st_size))
    tool_distributions = tool_environment_metadata(tools)
    metadata = BuildInputMetadataV1(
        source.candidate_head_oid,
        source.candidate_tree_oid,
        source.archive_sha256,
        lock_digest,
        hashlib.sha256(project).hexdigest(),
        python_version,
        hashlib.sha256(requirements).hexdigest(),
        hashlib.sha256(constraints).hexdigest(),
        inventory,
    )
    receipt = object.__new__(BoundBuildInputsV1)
    _BOUND[receipt] = _BuildInputs(
        archive,
        identity,
        tools,
        requirements,
        constraints,
        tuple(sorted(frozen)),
        metadata,
        tool_distributions,
        tuple(images),
    )
    return receipt


def _build_input_facts(receipt: BoundBuildInputsV1) -> _BuildInputs:
    """Retained byte comparisons, without permission to execute removed tools."""
    if type(receipt) is not BoundBuildInputsV1:
        raise TypeError("build input capability type differs")
    _require(receipt in _BOUND, "build input capability is unregistered")
    bound = _BOUND[receipt]
    archives._archive_bytes_for_consumer(bound.archive, bound.identity)
    _require(
        archives._archive_tool_capture_for_consumer(bound.archive, bound.identity)
        == bound.tool_distributions,
        "build source capture tool distributions differ",
    )
    return bound


def _build_inputs_for_consumer(receipt: BoundBuildInputsV1) -> _BuildInputs:
    bound = _build_input_facts(receipt)
    _require(
        tool_environment_metadata(bound.tools) == bound.tool_distributions,
        "build tool environment differs from its admitted distributions",
    )
    return bound


def build_input_metadata(receipt: BoundBuildInputsV1) -> BuildInputMetadataV1:
    return _build_inputs_for_consumer(receipt).metadata


@dataclass(frozen=True, slots=True)
class BuildFileMetadataV1:
    qualification_input_sha256: str
    source_commit: str
    source_archive_sha256: str
    tool_sha256s: tuple[tuple[str, str], ...]


class BoundBuildFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("build file bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _BuildFiles:
    inputs: BoundBuildInputsV1
    files: RetainedQualificationInputFilesV1
    metadata: BuildFileMetadataV1


_FILES: WeakKeyDictionary[BoundBuildFilesV1, _BuildFiles] = WeakKeyDictionary()


def bind_build_input_files(
    inputs: BoundBuildInputsV1,
    files: RetainedQualificationInputFilesV1,
) -> BoundBuildFilesV1:
    """Bind final file roles to prior inputs, never replace invocation evidence."""
    bound = _build_inputs_for_consumer(inputs)
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    identity = bound.identity
    _require(
        document["candidate"]
        == {
            "candidateCommit": identity.candidate_head_oid,
            "tree": identity.candidate_tree_oid,
            "baselineCommit": identity.canonical_baseline_oid,
            "canonicalDiffSha256": identity.canonical_diff_sha256,
            "version": "0.0.3",
        },
        "final build source differs from the pre-build authority",
    )
    tools = {item["role"]: item for item in document["toolIdentities"]}
    digests = []
    for image in bound.tool_images:
        declared = tools[image.role]
        reference = declared["artifact"]
        _require(
            (declared["version"], reference["sha256"], reference["bytes"])
            == (image.version, image.sha256, image.size),
            "final build tool file differs from its admitted environment",
        )
        digests.append((image.role, image.sha256))
    _require(
        document["expected"]["pythonFullVersion"] == bound.metadata.python_version,
        "final build Python differs from its admitted environment",
    )
    hatchling = next(item for item in bound.metadata.distributions if item.name == "hatchling")
    declared = tools["hatchling"]
    artifact = declared["artifact"]
    _require(
        declared["version"] == hatchling.version
        and artifact["sha256"] == hatchling.sha256
        and sealed_file_bytes(selected.seals, artifact["relativePath"], 4 * 1024**3)
        == next(raw for name, raw in bound.wheels if parse_wheel_filename(name)[0] == "hatchling"),
        "final build backend file differs from its source-locked input",
    )
    digests.append(("hatchling", hatchling.sha256))
    reference = next(
        item for item in document["files"] if item["role"] == "build_wheelhouse_manifest"
    )
    manifest = json.loads(sealed_file_bytes(selected.seals, reference["relativePath"], 4 * 1024**2))
    _require(
        (manifest["pythonVersion"], manifest["platform"])
        == (bound.metadata.python_version, "windows_amd64"),
        "final build target differs from its pre-build input",
    )
    for key, expected in (("requirements", bound.requirements), ("constraints", bound.constraints)):
        _require(
            sealed_file_bytes(selected.seals, manifest[key]["relativePath"], 4 * 1024**2)
            == expected,
            "final build requirements differ from their pre-build input",
        )
    _require(
        {
            item["basename"]: sealed_file_bytes(selected.seals, item["relativePath"], 4 * 1024**3)
            for item in manifest["wheels"]
        }
        == dict(bound.wheels),
        "final build wheel files differ from their pre-build inputs",
    )
    _build_inputs_for_consumer(inputs)
    metadata = BuildFileMetadataV1(
        retained_input_metadata(files).qualification_input_sha256,
        bound.metadata.source_commit,
        bound.metadata.source_archive_sha256,
        tuple(sorted(digests)),
    )
    receipt = object.__new__(BoundBuildFilesV1)
    _FILES[receipt] = _BuildFiles(inputs, files, metadata)
    return receipt


def build_file_metadata(receipt: BoundBuildFilesV1) -> BuildFileMetadataV1:
    if type(receipt) is not BoundBuildFilesV1:
        raise TypeError("build file capability type differs")
    _require(receipt in _FILES, "build file capability is unregistered")
    bound = _FILES[receipt]
    _build_input_facts(bound.inputs)
    _require(
        retained_input_metadata(bound.files).qualification_input_sha256
        == bound.metadata.qualification_input_sha256,
        "build file binding differs",
    )
    return bound.metadata

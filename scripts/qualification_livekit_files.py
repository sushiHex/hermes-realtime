"""Bind source-selected LiveKit archive bytes before final qualification inputs exist.

This admits a governed archive and its complete materialized namespace.  It does
not execute LiveKit or claim readiness, version output, process ownership, or
controller behavior.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts.qualification_build_inputs import (
    BoundBuildInputsV1,
    _build_input_facts,
    _build_inputs_for_consumer,
)
from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
    execution_file_metadata,
    owned_execution_files,
)
from scripts.qualification_file_seals import sealed_file_bytes
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_tool_distributions import _inspect_archive_members
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)

_WORKFLOW = ".github/workflows/release-gates.yml"
_BROWSER = "tests/integration/test_browser_self_acceptance.py"
_GATE = "scripts/real_natural_work_gate.py"
_ARCHIVE_MEMBER = "livekit-server.exe"
_MAX_ARCHIVE = 512 * 1024**2
_MAX_MEMBER = 128 * 1024**2
_ARCHIVE_FILE = "source/archive.zip"
_DISTRIBUTION_PREFIX = "distribution/"
_WORKFLOW_PIN = re.compile(
    rb"\$archive = Join-Path \$env:RUNNER_TEMP '"
    rb"(livekit_1\.13\.4_windows_amd64\.zip)'\r?\n"
    rb"\s*Invoke-WebRequest 'https://github\.com/livekit/livekit/releases/download/"
    rb"v1\.13\.4/livekit_1\.13\.4_windows_amd64\.zip' -OutFile \$archive\r?\n"
    rb"\s*\$actual = \(Get-FileHash \$archive -Algorithm SHA256\)\.Hash\.ToLowerInvariant\(\)\r?\n"
    rb"\s*\$expected = '([0-9a-f]{64})'"
)
_BROWSER_PIN = re.compile(rb'_PINNED_LIVEKIT_SHA256 = "([0-9a-f]{64})"')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _Policy:
    archive_sha256: str
    executable_sha256: str
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    gate_sha256: str


@dataclass(frozen=True, slots=True)
class LiveKitArchiveMetadataV1:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    archive_sha256: str
    archive_bytes: int
    executable_sha256: str
    gate_sha256: str
    members: tuple[tuple[str, str, int], ...]


class AdmittedLiveKitArchiveV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("LiveKit archive admissions are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _ArchiveBinding:
    inputs: BoundBuildInputsV1
    work: OwnedQualificationWorkV1
    files: ImmutableExecutionFilesV1
    metadata: LiveKitArchiveMetadataV1


_ARCHIVES: WeakKeyDictionary[AdmittedLiveKitArchiveV1, _ArchiveBinding] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class LiveKitFileMetadataV1:
    qualification_input_sha256: str
    candidate_commit: str
    archive_sha256: str
    executable_sha256: str
    member_count: int


class BoundLiveKitFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("LiveKit final bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _FinalBinding:
    inputs: BoundBuildInputsV1
    files: RetainedQualificationInputFilesV1
    metadata: LiveKitFileMetadataV1


_FINAL: WeakKeyDictionary[BoundLiveKitFilesV1, _FinalBinding] = WeakKeyDictionary()


def _source_member(payload: bytes, metadata: Any, path: str) -> bytes:
    member = next((item for item in metadata.manifest if item.path == path), None)
    _require(
        member is not None and member.kind == "file" and member.size <= 1024 * 1024,
        "LiveKit policy source differs",
    )
    assert member is not None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        stream = archive.extractfile(metadata.prefix + "/" + path)
        _require(stream is not None, "LiveKit policy source differs")
        assert stream is not None
        raw = stream.read(member.size + 1)
    _require(
        len(raw) == member.size and hashlib.sha256(raw).hexdigest() == member.sha256,
        "LiveKit policy source differs",
    )
    return raw


def _policy_from_bound(bound: Any) -> _Policy:
    payload = archives._archive_bytes_for_consumer(bound.archive, bound.identity)
    source = archives.verified_candidate_source_archive_metadata(bound.archive)
    workflow = _source_member(payload, source, _WORKFLOW)
    browser = _source_member(payload, source, _BROWSER)
    gate = _source_member(payload, source, _GATE)
    match = _WORKFLOW_PIN.findall(workflow)
    executable = _BROWSER_PIN.findall(browser)
    _require(len(match) == 1 and len(executable) == 1, "LiveKit source pin profile differs")
    asset, archive_sha = match[0]
    _require(asset == b"livekit_1.13.4_windows_amd64.zip", "LiveKit source asset differs")
    return _Policy(
        archive_sha.decode("ascii"),
        executable[0].decode("ascii"),
        source.candidate_head_oid,
        source.candidate_tree_oid,
        source.archive_sha256,
        hashlib.sha256(gate).hexdigest(),
    )


def _policy_from_inputs(inputs: BoundBuildInputsV1) -> _Policy:
    return _policy_from_bound(_build_inputs_for_consumer(inputs))


def _completed_policy_from_inputs(inputs: BoundBuildInputsV1) -> _Policy:
    """Retained source facts remain after the prefinal execution tree closes."""
    return _policy_from_bound(_build_input_facts(inputs))


def _inspect_archive(raw: bytes) -> tuple[dict[str, bytes], tuple[tuple[str, str, int], ...]]:
    _require(type(raw) is bytes and 0 < len(raw) <= _MAX_ARCHIVE, "LiveKit archive is bounded")
    try:
        files = dict(_inspect_archive_members(raw, "zip"))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError("LiveKit archive is invalid") from error
    _require(
        _ARCHIVE_MEMBER in files and all(files.values()),
        "LiveKit archive member differs",
    )
    inventory = tuple(
        sorted(
            (name, hashlib.sha256(value).hexdigest(), len(value)) for name, value in files.items()
        )
    )
    return files, inventory


def admit_livekit_archive(
    inputs: BoundBuildInputsV1, work: OwnedQualificationWorkV1, archive: bytes
) -> AdmittedLiveKitArchiveV1:
    _require(type(work) is OwnedQualificationWorkV1, "LiveKit work owner differs")
    work._accepting()
    policy = _policy_from_inputs(inputs)
    _require(
        hashlib.sha256(archive).hexdigest() == policy.archive_sha256,
        "LiveKit archive differs from source pin",
    )
    files, inventory = _inspect_archive(archive)
    _require(
        hashlib.sha256(files[_ARCHIVE_MEMBER]).hexdigest() == policy.executable_sha256,
        "LiveKit executable differs from source pin",
    )
    retained = work.enter(
        owned_execution_files(
            {
                _ARCHIVE_FILE: archive,
                **{_DISTRIBUTION_PREFIX + name: raw for name, raw in files.items()},
            }
        )
    )
    receipt = object.__new__(AdmittedLiveKitArchiveV1)
    _ARCHIVES[receipt] = _ArchiveBinding(
        inputs,
        work,
        retained,
        LiveKitArchiveMetadataV1(
            policy.source_commit,
            policy.source_tree,
            policy.source_archive_sha256,
            policy.archive_sha256,
            len(archive),
            policy.executable_sha256,
            policy.gate_sha256,
            inventory,
        ),
    )
    return receipt


def _archive_binding(receipt: AdmittedLiveKitArchiveV1) -> _ArchiveBinding:
    if type(receipt) is not AdmittedLiveKitArchiveV1:
        raise TypeError("LiveKit archive capability type differs")
    _require(receipt in _ARCHIVES, "LiveKit archive capability is unregistered")
    value = _ARCHIVES[receipt]
    value.work._accepting()
    _require(
        _policy_from_inputs(value.inputs).source_archive_sha256
        == value.metadata.source_archive_sha256,
        "LiveKit source authority differs",
    )
    expected = tuple(
        sorted(
            (
                (_ARCHIVE_FILE, value.metadata.archive_sha256, value.metadata.archive_bytes),
                *(
                    (_DISTRIBUTION_PREFIX + name, digest, size)
                    for name, digest, size in value.metadata.members
                ),
            )
        )
    )
    _require(execution_file_metadata(value.files) == expected, "LiveKit retained archive differs")
    return value


def livekit_archive_metadata(receipt: AdmittedLiveKitArchiveV1) -> LiveKitArchiveMetadataV1:
    return _archive_binding(receipt).metadata


def bind_livekit_files(
    archive: AdmittedLiveKitArchiveV1, files: RetainedQualificationInputFilesV1
) -> BoundLiveKitFilesV1:
    admitted = _archive_binding(archive)
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    candidate = document["candidate"]
    metadata = admitted.metadata
    _require(
        candidate["candidateCommit"] == metadata.source_commit
        and candidate["tree"] == metadata.source_tree,
        "LiveKit final candidate differs",
    )
    refs = {item["role"]: item for item in document["files"]}
    archive_ref, executable_ref = refs["livekit_archive"], refs["livekit_executable"]
    _require(
        sealed_file_bytes(selected.seals, archive_ref["relativePath"], _MAX_ARCHIVE)
        == _execution_files_for_consumer(admitted.files).joinpath(_ARCHIVE_FILE).read_bytes()
        and sealed_file_bytes(selected.seals, executable_ref["relativePath"], _MAX_MEMBER)
        == _execution_files_for_consumer(admitted.files)
        .joinpath(_DISTRIBUTION_PREFIX + _ARCHIVE_MEMBER)
        .read_bytes(),
        "LiveKit final files differ",
    )
    receipt = object.__new__(BoundLiveKitFilesV1)
    _FINAL[receipt] = _FinalBinding(
        admitted.inputs,
        files,
        LiveKitFileMetadataV1(
            retained_input_metadata(files).qualification_input_sha256,
            metadata.source_commit,
            metadata.archive_sha256,
            metadata.executable_sha256,
            len(metadata.members),
        ),
    )
    return receipt


def livekit_file_metadata(receipt: BoundLiveKitFilesV1) -> LiveKitFileMetadataV1:
    if type(receipt) is not BoundLiveKitFilesV1:
        raise TypeError("LiveKit final capability type differs")
    _require(receipt in _FINAL, "LiveKit final capability is unregistered")
    value = _FINAL[receipt]
    policy = _completed_policy_from_inputs(value.inputs)
    _require(
        (policy.source_commit, policy.archive_sha256, policy.executable_sha256)
        == (
            value.metadata.candidate_commit,
            value.metadata.archive_sha256,
            value.metadata.executable_sha256,
        ),
        "LiveKit retained source facts differ",
    )
    selected = _retained_input_files_for_consumer(value.files)
    document = json.loads(selected.document)
    refs = {item["role"]: item for item in document["files"]}
    archive_ref, executable_ref = refs["livekit_archive"], refs["livekit_executable"]
    archive = sealed_file_bytes(selected.seals, archive_ref["relativePath"], _MAX_ARCHIVE)
    executable = sealed_file_bytes(selected.seals, executable_ref["relativePath"], _MAX_MEMBER)
    _require(
        hashlib.sha256(archive).hexdigest() == value.metadata.archive_sha256
        and hashlib.sha256(executable).hexdigest() == value.metadata.executable_sha256
        and retained_input_metadata(value.files).qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "LiveKit final seals differ",
    )
    return value.metadata

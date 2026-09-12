"""Bind retained candidate file bytes; build and installation proof stays separate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts import candidate_wheel as wheels
from scripts import qualify_evidence_slice_zero as core
from scripts.qualification_file_seals import sealed_file_bytes
from scripts.qualification_sdist import inspect_candidate_sdist
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_SOURCE_ROLES = {
    "governing_plan": "docs/evidence-capture.md",
    "qualification_runner": "scripts/qualify_evidence_slice_zero.py",
    "hermes_pluginmanager_runner": "scripts/qualify_hermes_v020_pluginmanager.py",
    "benchmark_machine_schema": "scripts/schemas/benchmark-machine-v1.schema.json",
    "benchmark_report_schema": "scripts/schemas/benchmark-report-v1.schema.json",
    "wheelhouse_manifest_schema": "scripts/schemas/wheelhouse-manifest-v1.schema.json",
    "qualification_input_schema": "scripts/schemas/qualification-input-v1.schema.json",
    "qualification_report_schema": "scripts/schemas/qualification-report-v1.schema.json",
    "release_manifest_schema": "scripts/schemas/release-manifest-v1.schema.json",
}
_WHEEL_ROLES = (
    "direct_wheel",
    "direct_wheel_repeat",
    "sdist_built_wheel",
    "sdist_built_wheel_repeat",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class CandidateFileMetadataV1:
    qualification_input_sha256: str
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    wheel_sha256s: tuple[str, ...]
    sdist_sha256s: tuple[str, ...]
    execution_plan_sha256: str


class BoundCandidateFilesV1:
    """A live input-file binding, never an independent-build or installed receipt."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("candidate file bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    files: RetainedQualificationInputFilesV1
    archive: archives.VerifiedCandidateSourceArchiveV1
    identity: CandidateIdentityV1
    wheels: tuple[wheels.VerifiedCandidateWheelV1, ...]
    metadata: CandidateFileMetadataV1


_BINDINGS: WeakKeyDictionary[BoundCandidateFilesV1, _Binding] = WeakKeyDictionary()


def bind_candidate_files(
    files: RetainedQualificationInputFilesV1,
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
) -> BoundCandidateFilesV1:
    """Require the source authority before interpreting the selected file graph."""
    metadata = archives.verified_candidate_source_archive_metadata(archive)
    archives._archive_bytes_for_consumer(archive, identity)
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    expected = {
        "baselineCommit": identity.canonical_baseline_oid,
        "candidateCommit": identity.candidate_head_oid,
        "tree": identity.candidate_tree_oid,
        "canonicalDiffSha256": identity.canonical_diff_sha256,
        "version": "0.0.3",
    }
    _require(
        core.canonical_json_bytes(document["candidate"]) == core.canonical_json_bytes(expected),
        "input candidate identity differs from its source authority",
    )
    references = {item["role"]: item for item in document["files"]}
    source = {
        item.path: (item.sha256, item.size) for item in metadata.manifest if item.kind == "file"
    }
    for role, path in _SOURCE_ROLES.items():
        reference = references[role]
        _require(
            source.get(path) == (reference["sha256"], reference["bytes"]),
            "input source artifact differs from its candidate blob",
        )
    reference = references["candidate_source_archive"]
    _require(
        (reference["sha256"], reference["bytes"])
        == (metadata.archive_sha256, metadata.archive_bytes),
        "input archive differs from its genuine source authority",
    )
    accepted = tuple(
        wheels._verify_candidate_wheel_bytes_v1(
            archive,
            identity,
            sealed_file_bytes(selected.seals, references[role]["relativePath"], wheels._MAX_WHEEL),
            references[role]["sha256"],
        )
        for role in _WHEEL_ROLES
    )
    sdists = tuple(
        inspect_candidate_sdist(
            archive,
            identity,
            accepted[0],
            sealed_file_bytes(selected.seals, references[role]["relativePath"], 16 * 1024**2),
        )
        for role in ("sdist", "sdist_repeat")
    )
    policy = source.get("docs/qualification-execution.md")
    _require(policy is not None, "candidate execution protocol is absent")
    assert policy is not None and policy[0] is not None
    current = retained_input_metadata(files)
    observation = CandidateFileMetadataV1(
        current.qualification_input_sha256,
        metadata.candidate_head_oid,
        metadata.candidate_tree_oid,
        metadata.archive_sha256,
        tuple(
            wheels._wheel_for_consumer(item, archive, identity).wheel_sha256 for item in accepted
        ),
        tuple(item.sha256 for item in sdists),
        policy[0],
    )
    receipt = object.__new__(BoundCandidateFilesV1)
    _BINDINGS[receipt] = _Binding(files, archive, identity, accepted, observation)
    return receipt


def candidate_file_metadata(receipt: BoundCandidateFilesV1) -> CandidateFileMetadataV1:
    if type(receipt) is not BoundCandidateFilesV1:
        raise TypeError("candidate file binding type differs")
    _require(receipt in _BINDINGS, "candidate file binding is unregistered")
    binding = _BINDINGS[receipt]
    current = retained_input_metadata(binding.files)
    _require(
        current.qualification_input_sha256 == binding.metadata.qualification_input_sha256,
        "candidate file input digest differs",
    )
    archives._archive_bytes_for_consumer(binding.archive, binding.identity)
    for item in binding.wheels:
        wheels._wheel_for_consumer(item, binding.archive, binding.identity)
    return binding.metadata


def _candidate_files_for_consumer(receipt: BoundCandidateFilesV1) -> _Binding:
    candidate_file_metadata(receipt)
    return _BINDINGS[receipt]

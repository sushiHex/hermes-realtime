"""Produce source-bound artifacts through six independently owned offline builds.

Completed invocation and cleanup facts outlive disposable build environments.
Intermediate output seals remain live through copying into the final input root;
the final binding then retains that root's seals. This is build authority, not
complete installed, platform or physical qualification.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts import candidate_wheel as wheels
from scripts.qualification_build_environment import (
    _ARTIFACT_NAMES,
    InstalledBuildMetadataV1,
    _installed_build_for_consumer,
    _worker_observation,
    _workspace,
    install_build_environment,
    installed_build_metadata,
)
from scripts.qualification_build_inputs import (
    BoundBuildInputsV1,
    BuildInputMetadataV1,
    _build_inputs_for_consumer,
    build_input_metadata,
)
from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
    execution_file_metadata,
    owned_execution_files,
)
from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_sdist import _candidate_sdist_files
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_BUILD_ORDER = (
    ("direct_wheel", "wheel", None),
    ("direct_wheel_repeat", "wheel", None),
    ("sdist", "sdist", None),
    ("sdist_repeat", "sdist", None),
    ("sdist_built_wheel", "wheel", "sdist"),
    ("sdist_built_wheel_repeat", "wheel", "sdist_repeat"),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _archive_source_files(inputs: BoundBuildInputsV1) -> dict[str, bytes]:
    bound = _build_inputs_for_consumer(inputs)
    payload = archives._archive_bytes_for_consumer(bound.archive, bound.identity)
    metadata = archives.verified_candidate_source_archive_metadata(bound.archive)
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as source:
        for member in metadata.manifest:
            if member.kind == "dir":
                continue
            _require(member.kind == "file", "build source contains an indirect member")
            stream = source.extractfile(f"{metadata.prefix}/{member.path}")
            _require(stream is not None, "build source member is unreadable")
            assert stream is not None
            with stream:
                raw = stream.read(member.size + 1)
            _require(
                (len(raw), hashlib.sha256(raw).hexdigest()) == (member.size, member.sha256),
                "build source member differs from its archive",
            )
            contents[member.path] = raw
    return contents


@dataclass(frozen=True, slots=True)
class BuildArtifactMetadataV1:
    role: str
    sha256: str
    size: int
    input_sha256: str
    installed: InstalledBuildMetadataV1
    import_count: int
    file_origin_count: int


@dataclass(frozen=True, slots=True)
class CandidateBuildMetadataV1:
    inputs: BuildInputMetadataV1
    artifacts: tuple[BuildArtifactMetadataV1, ...]
    invocation_count: int


@dataclass(frozen=True, slots=True)
class _Produced:
    metadata: BuildArtifactMetadataV1
    contents: bytes
    files: ImmutableExecutionFilesV1
    invocations: tuple[CompletedToolInvocationV1, ...]
    work: OwnedQualificationWorkV1


class CandidateBuildsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("candidate builds are recipe-minted only")


@dataclass(frozen=True, slots=True)
class _Builds:
    inputs: BoundBuildInputsV1
    archive: archives.VerifiedCandidateSourceArchiveV1
    identity: CandidateIdentityV1
    produced: tuple[_Produced, ...]
    output_owner: OwnedQualificationWorkV1
    metadata: CandidateBuildMetadataV1


_BUILDS: WeakKeyDictionary[CandidateBuildsV1, _Builds] = WeakKeyDictionary()


def _produce_one(
    parent: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
    role: str,
    kind: str,
    source_files: dict[str, bytes],
    input_sha256: str,
    direct: wheels.VerifiedCandidateWheelV1 | None,
) -> tuple[_Produced, wheels.VerifiedCandidateWheelV1 | None]:
    bound = _build_inputs_for_consumer(inputs)
    with OwnedQualificationWorkV1() as work:
        environment = install_build_environment(work, inputs)
        installed = installed_build_metadata(environment)
        value = _installed_build_for_consumer(environment)
        source = work.enter(owned_execution_files(source_files))
        source_path = _execution_files_for_consumer(source)
        resources = _execution_files_for_consumer(value.resources)
        packages = _execution_files_for_consumer(value.files)
        workspace = work.enter(_workspace())
        output = workspace / "output"
        output.mkdir()
        invocation = work.run_tool(
            bound.tools,
            "build_python",
            (
                str(resources / "worker.py"),
                kind,
                str(packages),
                str(source_path),
                str(output),
                str(workspace / "build.json"),
            ),
            workspace,
        )
        installed_build_metadata(environment)
        _execution_files_for_consumer(source)
        observation = work.enter(retain_file_seals(workspace, ("build.json",)))
        counts = _worker_observation(
            sealed_file_bytes(observation, "build.json", 4096),
            _tool_invocation_for_consumer(invocation).process.pid,
            kind,
        )
        name = _ARTIFACT_NAMES[kind]
        assert name is not None
        _require(
            sorted(path.name for path in output.iterdir()) == [name],
            "build output namespace differs",
        )
        seals = work.enter(retain_file_seals(output, (name,)))
        raw = sealed_file_bytes(seals, name, 16 * 1024**2)
        digest = hashlib.sha256(raw).hexdigest()
        verified = None
        if kind == "wheel":
            verified = wheels._verify_candidate_wheel_bytes_v1(
                bound.archive, bound.identity, raw, digest
            )
        else:
            _require(direct is not None, "sdist lacks a prior source-bound wheel")
            assert direct is not None
            _candidate_sdist_files(bound.archive, bound.identity, direct, raw)
        # Preserve the intermediate before releasing the original output's seals.
        retained = parent.enter(owned_execution_files({role: raw}))
        metadata = BuildArtifactMetadataV1(role, digest, len(raw), input_sha256, installed, *counts)
        produced = _Produced(
            metadata, raw, retained, (value.installer, value.importer, invocation), work
        )
    _require(work._closed, "build environment cleanup is incomplete")
    return produced, verified


def _validate_produced(produced: tuple[_Produced, ...]) -> None:
    _require(
        tuple(item.metadata.role for item in produced) == tuple(row[0] for row in _BUILD_ORDER),
        "build roles or order differ",
    )
    invocations: list[tuple[int, int]] = []
    directories = []
    for item in produced:
        _require(
            item.metadata.installed.inputs == produced[0].metadata.installed.inputs
            and item.metadata.installed.worker_sha256
            == produced[0].metadata.installed.worker_sha256,
            "build candidate, dependency closure or worker differs",
        )
        _require(
            item.work._closed and item.work._unrecoverable is None,
            "build environment cleanup is incomplete",
        )
        _require(len(item.invocations) == 3, "build invocation set differs")
        values = tuple(_tool_invocation_for_consumer(value) for value in item.invocations)
        _require(
            tuple(tool_invocation_metadata(value).role for value in item.invocations)
            == ("uv", "build_python", "build_python"),
            "build invocation roles differ",
        )
        invocations.extend((value.process.pid, value.process.creation_filetime) for value in values)
        directories.append(values[-1].working_directory)
        _require(
            (hashlib.sha256(item.contents).hexdigest(), len(item.contents))
            == (item.metadata.sha256, item.metadata.size),
            "build retained output differs",
        )
    _require(len(set(invocations)) == 18, "build invocations are not independent")
    _require(len(set(directories)) == 6, "build output workspaces are not independent")
    by_role = {item.metadata.role: item for item in produced}
    direct = by_role["direct_wheel"]
    _require(
        all(
            by_role[role].contents == direct.contents
            for role in ("direct_wheel_repeat", "sdist_built_wheel", "sdist_built_wheel_repeat")
        )
        and by_role["sdist"].contents == by_role["sdist_repeat"].contents,
        "independent build outputs are not reproducible",
    )
    for role, _, upstream in _BUILD_ORDER:
        expected = (
            by_role[upstream].metadata.sha256
            if upstream
            else direct.metadata.installed.inputs.source_archive_sha256
        )
        _require(by_role[role].metadata.input_sha256 == expected, "build source lineage differs")


def build_candidate_artifacts(
    work: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
) -> CandidateBuildsV1:
    _require(type(work) is OwnedQualificationWorkV1, "build work owner type differs")
    work._accepting()
    try:
        bound = _build_inputs_for_consumer(inputs)
        source_files = _archive_source_files(inputs)
        metadata = build_input_metadata(inputs)
        produced: dict[str, _Produced] = {}
        direct = None
        for role, kind, upstream in _BUILD_ORDER:
            if upstream is None:
                selection, digest = source_files, metadata.source_archive_sha256
            else:
                prior = produced[upstream]
                execution_file_metadata(prior.files)
                assert direct is not None
                _, selection = _candidate_sdist_files(
                    bound.archive, bound.identity, direct, prior.contents
                )
                digest = prior.metadata.sha256
            result, verified = _produce_one(work, inputs, role, kind, selection, digest, direct)
            produced[role] = result
            if role == "direct_wheel":
                direct = verified
        outputs = tuple(produced.values())
        _validate_produced(outputs)
        _require(build_input_metadata(inputs) == metadata, "build input closure changed")
        receipt = object.__new__(CandidateBuildsV1)
        _BUILDS[receipt] = _Builds(
            inputs,
            bound.archive,
            bound.identity,
            outputs,
            work,
            CandidateBuildMetadataV1(metadata, tuple(item.metadata for item in outputs), 18),
        )
        return receipt
    except BaseException as error:
        work._retain_failure(error)
        raise


def _completed_builds(receipt: CandidateBuildsV1) -> _Builds:
    if type(receipt) is not CandidateBuildsV1:
        raise TypeError("candidate build capability type differs")
    _require(receipt in _BUILDS, "candidate build capability is unregistered")
    value = _BUILDS[receipt]
    archives._archive_bytes_for_consumer(value.archive, value.identity)
    _validate_produced(value.produced)
    return value


def candidate_build_metadata(receipt: CandidateBuildsV1) -> CandidateBuildMetadataV1:
    """Completed facts remain readable after the disposable build trees are removed."""
    return _completed_builds(receipt).metadata


def _candidate_builds_for_consumer(receipt: CandidateBuildsV1) -> _Builds:
    return _completed_builds(receipt)


def _candidate_build_bytes(receipt: CandidateBuildsV1) -> dict[str, bytes]:
    value = _completed_builds(receipt)
    for item in value.produced:
        _require(
            execution_file_metadata(item.files)
            == ((item.metadata.role, item.metadata.sha256, item.metadata.size),),
            "build intermediate seals differ",
        )
    return {item.metadata.role: item.contents for item in value.produced}


class BoundBuildOutputsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("build output bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class BuildOutputMetadataV1:
    qualification_input_sha256: str
    builds: CandidateBuildMetadataV1


@dataclass(frozen=True, slots=True)
class _BoundOutputs:
    builds: CandidateBuildsV1
    files: RetainedQualificationInputFilesV1
    metadata: BuildOutputMetadataV1


_OUTPUTS: WeakKeyDictionary[BoundBuildOutputsV1, _BoundOutputs] = WeakKeyDictionary()


def bind_build_output_files(
    builds: CandidateBuildsV1,
    files: RetainedQualificationInputFilesV1,
) -> BoundBuildOutputsV1:
    contents = _candidate_build_bytes(builds)
    completed = _completed_builds(builds)
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    identity = completed.identity
    _require(
        document["candidate"]
        == {
            "candidateCommit": identity.candidate_head_oid,
            "tree": identity.candidate_tree_oid,
            "baselineCommit": identity.canonical_baseline_oid,
            "canonicalDiffSha256": identity.canonical_diff_sha256,
            "version": "0.0.3",
        },
        "final build output candidate differs",
    )
    references = {item["role"]: item for item in document["files"]}
    for role, raw in contents.items():
        _require(
            sealed_file_bytes(selected.seals, references[role]["relativePath"], 16 * 1024**2)
            == raw,
            "final build role differs from its produced artifact",
        )
    metadata = BuildOutputMetadataV1(
        retained_input_metadata(files).qualification_input_sha256, completed.metadata
    )
    receipt = object.__new__(BoundBuildOutputsV1)
    _OUTPUTS[receipt] = _BoundOutputs(builds, files, metadata)
    return receipt


def build_output_metadata(receipt: BoundBuildOutputsV1) -> BuildOutputMetadataV1:
    if type(receipt) is not BoundBuildOutputsV1:
        raise TypeError("build output capability type differs")
    _require(receipt in _OUTPUTS, "build output capability is unregistered")
    value = _OUTPUTS[receipt]
    _require(
        candidate_build_metadata(value.builds) == value.metadata.builds,
        "completed build facts differ",
    )
    _require(
        retained_input_metadata(value.files).qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "final build input seals differ",
    )
    return value.metadata


def _build_outputs_for_consumer(receipt: BoundBuildOutputsV1) -> _BoundOutputs:
    build_output_metadata(receipt)
    return _OUTPUTS[receipt]

"""Install each bound Windows runtime purpose without activating a host or capture.

Live execution authority and completed installation facts have separate lifetimes.
The work owner must retain all runtime consumers until their final exit; completed
facts require its verified cleanup and the separate final input root's live seals.
Linux installation, provider construction and Hermes host execution are separate
authorities and cannot be inferred from these installed import observations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts.qualification_build_environment import _candidate_worker, _worker_observation
from scripts.qualification_builds import BoundBuildOutputsV1, build_output_metadata
from scripts.qualification_candidate_files import _candidate_files_for_consumer
from scripts.qualification_dependency_files import (
    BoundDependencyFilesV1,
    BoundDependencyPurposeV1,
    _dependency_binding_for_consumer,
    _dependency_wheels_for_consumer,
)
from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
)
from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes
from scripts.qualification_installation import _install_packages
from scripts.qualification_installed_files import InstalledFileMetadataV1
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_tool_environment import (
    ImmutableToolEnvironmentV1,
    _tool_image_for_consumer,
    tool_environment_metadata,
)
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)
from scripts.retained_qualification_inputs import _retained_input_files_for_consumer

_PURPOSES = frozenset(
    {
        "realtime_windows_direct_runtime",
        "realtime_windows_sdist_built_runtime",
        "hermes_v020_pluginmanager_runtime",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class RuntimeEnvironmentMetadataV1:
    qualification_input_sha256: str
    source_commit: str
    source_tree: str
    purpose: str
    worker_sha256: str
    installed: InstalledFileMetadataV1
    import_count: int
    file_origin_count: int


class InstalledRuntimeEnvironmentV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("installed runtime environments are recipe-minted only")


class CompletedRuntimeEnvironmentV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("completed runtime environments are recipe-minted only")


@dataclass(frozen=True, slots=True)
class _Runtime:
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1
    outputs: BoundBuildOutputsV1
    tools: ImmutableToolEnvironmentV1
    work: OwnedQualificationWorkV1
    files: ImmutableExecutionFilesV1
    resources: ImmutableExecutionFilesV1
    invocations: tuple[CompletedToolInvocationV1, CompletedToolInvocationV1]
    metadata: RuntimeEnvironmentMetadataV1


_LIVE: WeakKeyDictionary[InstalledRuntimeEnvironmentV1, _Runtime] = WeakKeyDictionary()
_COMPLETED: WeakKeyDictionary[CompletedRuntimeEnvironmentV1, _Runtime] = WeakKeyDictionary()


def _runtime_inputs(
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1,
    outputs: BoundBuildOutputsV1,
    tools: ImmutableToolEnvironmentV1,
    purpose: str,
) -> tuple[dict[str, bytes], bytes, bytes, bytes]:
    _require(type(purpose) is str and purpose in _PURPOSES, "runtime purpose is unavailable")
    value = _dependency_binding_for_consumer(dependencies)
    source = _candidate_files_for_consumer(value.candidate)
    produced = build_output_metadata(outputs)
    _require(
        produced.qualification_input_sha256 == value.metadata.qualification_input_sha256,
        "runtime dependencies and build outputs have different final inputs",
    )
    _require(
        (produced.builds.inputs.source_commit, produced.builds.inputs.source_tree)
        == (source.identity.candidate_head_oid, source.identity.candidate_tree_oid),
        "runtime build source differs",
    )
    selected = _retained_input_files_for_consumer(value.files)
    document = json.loads(selected.document)
    declared = {item["role"]: item for item in document["toolIdentities"]}
    for role in ("git", "uv", "build_python"):
        image, digest, version = _tool_image_for_consumer(tools, role)
        reference = declared[role]["artifact"]
        _require(
            (declared[role]["version"], reference["sha256"], reference["bytes"])
            == (version, digest, image.stat().st_size),
            "runtime tool identity differs",
        )
        if role == "build_python":
            _require(
                version == document["expected"]["pythonFullVersion"],
                "runtime Python version differs",
            )
    wheels, requirements, constraints = _dependency_wheels_for_consumer(dependencies, purpose)
    worker = _candidate_worker(source.archive, source.identity)
    return wheels, requirements, constraints, worker


def install_runtime_environment(
    work: OwnedQualificationWorkV1,
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1,
    outputs: BoundBuildOutputsV1,
    tools: ImmutableToolEnvironmentV1,
    *,
    purpose: str,
) -> InstalledRuntimeEnvironmentV1:
    _require(type(work) is OwnedQualificationWorkV1, "runtime work owner type differs")
    work._accepting()
    try:
        wheels, requirements, constraints, worker = _runtime_inputs(
            dependencies, outputs, tools, purpose
        )
        installed = _install_packages(
            work,
            tools,
            wheels=wheels,
            requirements=requirements,
            constraints=constraints,
            worker=worker,
        )
        packages = _execution_files_for_consumer(installed.files)
        resources = _execution_files_for_consumer(installed.resources)
        workspace = installed.workspace
        importer = work.run_tool(
            tools,
            "build_python",
            (
                str(resources / "worker.py"),
                purpose,
                str(packages),
                str(workspace),
                str(workspace),
                str(workspace / "imports.json"),
            ),
            workspace,
        )
        _execution_files_for_consumer(installed.files)
        observation = work.enter(retain_file_seals(workspace, ("imports.json",)))
        counts = _worker_observation(
            sealed_file_bytes(observation, "imports.json", 4096),
            _tool_invocation_for_consumer(importer).process.pid,
            purpose,
        )
        _require(
            _tool_invocation_for_consumer(installed.installer).process
            != _tool_invocation_for_consumer(importer).process,
            "runtime installation and import invocations are not distinct",
        )
        source = build_output_metadata(outputs).builds.inputs
        metadata = RuntimeEnvironmentMetadataV1(
            _dependency_binding_for_consumer(dependencies).metadata.qualification_input_sha256,
            source.source_commit,
            source.source_tree,
            purpose,
            hashlib.sha256(worker).hexdigest(),
            installed.inventory,
            *counts,
        )
        receipt = object.__new__(InstalledRuntimeEnvironmentV1)
        _LIVE[receipt] = _Runtime(
            dependencies,
            outputs,
            tools,
            work,
            installed.files,
            installed.resources,
            (installed.installer, importer),
            metadata,
        )
        return receipt
    except BaseException as error:
        work._retain_failure(error)
        raise


def _retained_facts(value: _Runtime) -> RuntimeEnvironmentMetadataV1:
    _require(
        _dependency_binding_for_consumer(value.dependencies).metadata.qualification_input_sha256
        == build_output_metadata(value.outputs).qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "runtime final input binding differs",
    )
    _require(
        all(tool_invocation_metadata(item).exit_code == 0 for item in value.invocations),
        "runtime invocation did not complete normally",
    )
    return value.metadata


def _installed_runtime_for_consumer(receipt: InstalledRuntimeEnvironmentV1) -> _Runtime:
    if type(receipt) is not InstalledRuntimeEnvironmentV1:
        raise TypeError("runtime environment capability type differs")
    _require(receipt in _LIVE, "runtime environment capability is unregistered")
    value = _LIVE[receipt]
    value.work._accepting()
    tool_environment_metadata(value.tools)
    _execution_files_for_consumer(value.files)
    _execution_files_for_consumer(value.resources)
    _retained_facts(value)
    return value


def installed_runtime_metadata(
    receipt: InstalledRuntimeEnvironmentV1,
) -> RuntimeEnvironmentMetadataV1:
    return _installed_runtime_for_consumer(receipt).metadata


def complete_runtime_environment(
    receipt: InstalledRuntimeEnvironmentV1,
) -> CompletedRuntimeEnvironmentV1:
    if type(receipt) is not InstalledRuntimeEnvironmentV1:
        raise TypeError("runtime environment capability type differs")
    _require(receipt in _LIVE, "runtime environment capability is unregistered")
    value = _LIVE[receipt]
    _require(
        value.work._closed and value.work._unrecoverable is None,
        "runtime environment cleanup is incomplete",
    )
    _retained_facts(value)
    completed = object.__new__(CompletedRuntimeEnvironmentV1)
    _COMPLETED[completed] = value
    return completed


def completed_runtime_metadata(
    receipt: CompletedRuntimeEnvironmentV1,
) -> RuntimeEnvironmentMetadataV1:
    if type(receipt) is not CompletedRuntimeEnvironmentV1:
        raise TypeError("completed runtime capability type differs")
    _require(receipt in _COMPLETED, "completed runtime capability is unregistered")
    return _retained_facts(_COMPLETED[receipt])

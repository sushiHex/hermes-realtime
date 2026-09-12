"""Install source-locked build dependencies through admitted owned tools.

The live capability binds actual installation, independently inspected bytes and
isolated imports. It does not prove any build, independent repeat, complete final
input closure or durable controller-death recovery. Its work owner retains every
file dependency until all owned consumers have finished.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from typing import Any
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts.qualification_build_inputs import (
    BoundBuildInputsV1,
    BuildInputMetadataV1,
    _build_inputs_for_consumer,
    build_input_metadata,
)
from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
)
from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes
from scripts.qualification_installation import _install_packages
from scripts.qualification_installation import _workspace as _workspace
from scripts.qualification_installed_files import InstalledFileMetadataV1
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_WORKER = "scripts/qualification_build_worker.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _build_worker_for_inputs(inputs: BoundBuildInputsV1) -> bytes:
    bound = _build_inputs_for_consumer(inputs)
    return _candidate_worker(bound.archive, bound.identity)


def _candidate_worker(
    archive: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
) -> bytes:
    payload = archives._archive_bytes_for_consumer(archive, identity)
    source = archives.verified_candidate_source_archive_metadata(archive)
    member = next((item for item in source.manifest if item.path == _WORKER), None)
    _require(
        member is not None and member.kind == "file" and 0 < member.size <= 512 * 1024,
        "candidate build worker is unavailable or unbounded",
    )
    assert member is not None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as contents:
        stream = contents.extractfile(f"{source.prefix}/{_WORKER}")
        _require(stream is not None, "candidate build worker is unreadable")
        assert stream is not None
        raw = stream.read(member.size + 1)
    _require(
        (len(raw), hashlib.sha256(raw).hexdigest()) == (member.size, member.sha256),
        "candidate build worker differs from its source authority",
    )
    return raw


_ARTIFACT_NAMES = {
    "imports": None,
    "wheel": "hermes_realtime-0.0.3-py3-none-any.whl",
    "sdist": "hermes_realtime-0.0.3.tar.gz",
    "realtime_windows_direct_runtime": None,
    "realtime_windows_sdist_built_runtime": None,
    "hermes_v020_pluginmanager_runtime": None,
}


def _worker_observation(raw: bytes, pid: int, kind: str) -> tuple[int, int]:
    _require(kind in _ARTIFACT_NAMES, "build worker operation is unavailable")
    _require(type(raw) is bytes and len(raw) <= 4096, "build import observation is unbounded")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        _require(len(dict(rows)) == len(rows), "build import observation is ambiguous")
        return dict(rows)

    value = json.loads(raw, object_pairs_hook=pairs)
    _require(
        type(value) is dict
        and set(value)
        == {
            "version",
            "kind",
            "pid",
            "imports",
            "file_origins",
            "source_fallback",
            "artifact",
        },
        "build import observation fields differ",
    )
    _require(
        type(value["version"]) is int
        and value["version"] == 1
        and value["kind"] == kind
        and value["artifact"] == _ARTIFACT_NAMES[kind],
        "build import observation protocol differs",
    )
    _require(
        type(value["pid"]) is int and value["pid"] == pid,
        "build import observation belongs to another process",
    )
    _require(
        type(value["imports"]) is int
        and value["imports"] == (5 if kind in {"imports", "wheel", "sdist"} else 4)
        and type(value["file_origins"]) is int
        and 5 <= value["file_origins"] <= 16384
        and value["source_fallback"] is False,
        "build import observation does not prove the closed import profile",
    )
    return value["imports"], value["file_origins"]


def _import_observation(raw: bytes, pid: int) -> tuple[int, int]:
    return _worker_observation(raw, pid, "imports")


@dataclass(frozen=True, slots=True)
class InstalledBuildMetadataV1:
    inputs: BuildInputMetadataV1
    files: InstalledFileMetadataV1
    import_count: int
    file_origin_count: int
    worker_sha256: str


class InstalledBuildEnvironmentV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("installed build environments are recipe-minted only")


@dataclass(frozen=True, slots=True)
class _Environment:
    inputs: BoundBuildInputsV1
    work: OwnedQualificationWorkV1
    files: ImmutableExecutionFilesV1
    resources: ImmutableExecutionFilesV1
    installer: CompletedToolInvocationV1
    importer: CompletedToolInvocationV1
    metadata: InstalledBuildMetadataV1


_LIVE: WeakKeyDictionary[InstalledBuildEnvironmentV1, _Environment] = WeakKeyDictionary()


def install_build_environment(
    work: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
) -> InstalledBuildEnvironmentV1:
    _require(type(work) is OwnedQualificationWorkV1, "build work owner type differs")
    work._accepting()
    try:
        return _install_build_environment(work, inputs)
    except BaseException as error:
        work._retain_failure(error)
        raise


def _install_build_environment(
    work: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
) -> InstalledBuildEnvironmentV1:
    bound = _build_inputs_for_consumer(inputs)
    _require(type(work) is OwnedQualificationWorkV1, "build work owner type differs")
    work._accepting()
    worker = _build_worker_for_inputs(inputs)
    installation = _install_packages(
        work,
        bound.tools,
        wheels=dict(bound.wheels),
        requirements=bound.requirements,
        constraints=bound.constraints,
        worker=worker,
    )
    files, resources = installation.files, installation.resources
    installer, inventory = installation.installer, installation.inventory
    workspace = installation.workspace
    source = _execution_files_for_consumer(resources)
    packages = _execution_files_for_consumer(files)
    report = workspace / "imports.json"
    importer = work.run_tool(
        bound.tools,
        "build_python",
        (
            str(source / "worker.py"),
            "imports",
            str(packages),
            str(workspace),
            str(workspace),
            str(report),
        ),
        workspace,
    )
    observation = work.enter(retain_file_seals(workspace, ("imports.json",)))
    counts = _import_observation(
        sealed_file_bytes(observation, "imports.json", 4096),
        _tool_invocation_for_consumer(importer).process.pid,
    )
    _require(
        _tool_invocation_for_consumer(installer).process
        != _tool_invocation_for_consumer(importer).process,
        "build installation and import invocations are not distinct",
    )
    _require(
        all(tool_invocation_metadata(value).exit_code == 0 for value in (installer, importer)),
        "build invocation did not complete normally",
    )
    metadata = InstalledBuildMetadataV1(
        build_input_metadata(inputs),
        inventory,
        *counts,
        hashlib.sha256(worker).hexdigest(),
    )
    receipt = object.__new__(InstalledBuildEnvironmentV1)
    _LIVE[receipt] = _Environment(inputs, work, files, resources, installer, importer, metadata)
    return receipt


def _installed_build_for_consumer(receipt: InstalledBuildEnvironmentV1) -> _Environment:
    if type(receipt) is not InstalledBuildEnvironmentV1:
        raise TypeError("installed build capability type differs")
    _require(receipt in _LIVE, "installed build capability is unregistered")
    value = _LIVE[receipt]
    value.work._accepting()
    _require(build_input_metadata(value.inputs) == value.metadata.inputs, "build inputs changed")
    _execution_files_for_consumer(value.files)
    _execution_files_for_consumer(value.resources)
    tool_invocation_metadata(value.installer)
    tool_invocation_metadata(value.importer)
    return value


def installed_build_metadata(receipt: InstalledBuildEnvironmentV1) -> InstalledBuildMetadataV1:
    return _installed_build_for_consumer(receipt).metadata

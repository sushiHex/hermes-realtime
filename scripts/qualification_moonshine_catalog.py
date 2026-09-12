"""Observe Moonshine 0.1.0's source-locked native STT resource selection."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tarfile
from dataclasses import dataclass
from typing import Any, cast
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts.qualification_builds import BoundBuildOutputsV1, build_output_metadata
from scripts.qualification_candidate_files import (
    BoundCandidateFilesV1,
    _candidate_files_for_consumer,
    candidate_file_metadata,
)
from scripts.qualification_dependency_files import (
    BoundDependencyFilesV1,
    BoundDependencyPurposeV1,
    _dependency_binding_for_consumer,
)
from scripts.qualification_execution_files import (
    _execution_files_for_consumer,
    execution_file_metadata,
    owned_execution_files,
)
from scripts.qualification_file_seals import retain_file_seals, sealed_file_bytes
from scripts.qualification_installation import _workspace
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_runtime_environment import (
    InstalledRuntimeEnvironmentV1,
    _installed_runtime_for_consumer,
    _Runtime,
)
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)
from scripts.qualify_evidence_slice_zero import canonical_json_bytes
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_WORKER = "scripts/qualification_moonshine_worker.py"
_PURPOSES = frozenset(
    {"realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime"}
)
_MODEL_BASE = "https://download.moonshine.ai/model/medium-streaming-en/quantized"
_SPELLING_BASE = "https://download.moonshine.ai/model/spelling-en"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_MAX_RESOURCE_BYTES = 512 * 1024**2
_MAX_TOTAL_BYTES = 1024**3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _Resource:
    group: str
    cache_path: str
    url: str
    size: int
    crc32c: str


@dataclass(frozen=True, slots=True)
class MoonshineCatalogMetadataV1:
    qualification_input_sha256: str
    source_commit: str
    source_tree: str
    purpose: str
    worker_sha256: str
    distribution_sha256: str
    python_api_sha256: str
    native_library_sha256: str
    catalog_identity_sha256: str
    primary_resource_count: int
    spelling_resource_count: int
    resource_bytes: int
    file_origin_count: int


class BoundMoonshineCatalogV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Moonshine catalog bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    work: OwnedQualificationWorkV1
    runtime: _Runtime
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1
    outputs: BoundBuildOutputsV1
    candidate: BoundCandidateFilesV1
    invocation: CompletedToolInvocationV1
    primary: tuple[_Resource, ...]
    spelling: tuple[_Resource, ...]
    metadata: MoonshineCatalogMetadataV1


_BINDINGS: WeakKeyDictionary[BoundMoonshineCatalogV1, _Binding] = WeakKeyDictionary()


def _archive_worker(
    archive: archives.VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1
) -> bytes:
    payload = archives._archive_bytes_for_consumer(archive, identity)
    source = archives.verified_candidate_source_archive_metadata(archive)
    member = next((item for item in source.manifest if item.path == _WORKER), None)
    _require(
        member is not None and member.kind == "file" and 0 < member.size <= 512 * 1024,
        "candidate Moonshine catalog worker is unavailable or unbounded",
    )
    assert member is not None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as contents:
        stream = contents.extractfile(f"{source.prefix}/{_WORKER}")
        _require(stream is not None, "candidate Moonshine catalog worker is unreadable")
        assert stream is not None
        raw = stream.read(member.size + 1)
    _require(
        (len(raw), hashlib.sha256(raw).hexdigest()) == (member.size, member.sha256),
        "candidate Moonshine catalog worker differs from its source authority",
    )
    return raw


def _candidate_worker(candidate: BoundCandidateFilesV1) -> bytes:
    bound = _candidate_files_for_consumer(candidate)
    return _archive_worker(bound.archive, bound.identity)


def _strict_object(raw: bytes) -> dict[str, Any]:
    _require(type(raw) is bytes and 0 < len(raw) <= 128 * 1024, "catalog observation is unbounded")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        _require(len(dict(items)) == len(items), "catalog observation is ambiguous")
        return dict(items)

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError("catalog observation is invalid") from None
    _require(type(value) is dict, "catalog observation is not an object")
    return cast(dict[str, Any], value)


def _manifest_group(
    value: object, *, base_url: str, count: int, group: str
) -> tuple[_Resource, ...]:
    _require(
        type(value) is dict and set(value) == {"base_url", "files"},
        "catalog group fields differ",
    )
    assert isinstance(value, dict)
    files = value["files"]
    _require(
        value["base_url"] == base_url and type(files) is list and len(files) == count,
        "catalog group profile differs",
    )
    resources: list[_Resource] = []
    for item in files:
        _require(
            type(item) is dict
            and set(item) == {"checksum", "checksum_type", "name", "size", "url"},
            "catalog resource fields differ",
        )
        assert isinstance(item, dict)
        name, size, checksum, url = (
            item["name"],
            item["size"],
            item["checksum"],
            item["url"],
        )
        _require(
            type(name) is str
            and _SAFE_NAME.fullmatch(name) is not None
            and type(size) is int
            and 0 < size <= _MAX_RESOURCE_BYTES
            and type(checksum) is str
            and item["checksum_type"] == "crc32c"
            and type(url) is str
            and url == f"{base_url}/{name}",
            "catalog resource identity differs",
        )
        try:
            decoded = base64.b64decode(checksum, validate=True)
        except ValueError:
            raise ValueError("catalog CRC32C is invalid") from None
        _require(
            len(decoded) == 4 and base64.b64encode(decoded).decode("ascii") == checksum,
            "catalog CRC32C is noncanonical",
        )
        resources.append(
            _Resource(group, url.removeprefix("https://"), url, size, checksum)
        )
    _require(
        tuple(row.cache_path for row in resources)
        == tuple(sorted(row.cache_path for row in resources))
        and len({row.cache_path.casefold() for row in resources}) == len(resources),
        "catalog resource namespace is ambiguous",
    )
    return tuple(resources)


def _catalogs(observation: dict[str, Any]) -> tuple[tuple[_Resource, ...], tuple[_Resource, ...]]:
    without = observation["without_spelling"]
    with_spelling = observation["with_spelling"]
    _require(
        type(without) is dict
        and set(without) == {"groups"}
        and type(without["groups"]) is list
        and len(without["groups"]) == 1
        and type(with_spelling) is dict
        and set(with_spelling) == {"groups"}
        and type(with_spelling["groups"]) is list
        and len(with_spelling["groups"]) == 2,
        "catalog spelling profile differs",
    )
    _require(
        with_spelling["groups"][0] == without["groups"][0],
        "catalog primary resources change with spelling selection",
    )
    primary = _manifest_group(
        without["groups"][0], base_url=_MODEL_BASE, count=7, group="primary"
    )
    spelling = _manifest_group(
        with_spelling["groups"][1], base_url=_SPELLING_BASE, count=2, group="spelling"
    )
    all_resources = primary + spelling
    _require(
        len({row.cache_path.casefold() for row in all_resources}) == len(all_resources)
        and sum(row.size for row in all_resources) <= _MAX_TOTAL_BYTES,
        "catalog resource closure is ambiguous or unbounded",
    )
    return primary, spelling


def _observation(
    raw: bytes,
    *,
    pid: int,
    native: tuple[str, int],
    python_api: tuple[str, int],
) -> tuple[tuple[_Resource, ...], tuple[_Resource, ...], int, str]:
    value = _strict_object(raw)
    _require(
        set(value)
        == {
            "version",
            "pid",
            "language",
            "model_arch",
            "native_library_sha256",
            "native_library_bytes",
            "python_api_sha256",
            "python_api_bytes",
            "file_origins",
            "source_fallback",
            "without_spelling",
            "with_spelling",
        },
        "catalog observation fields differ",
    )
    _require(
        type(value["version"]) is int
        and value["version"] == 1
        and type(value["pid"]) is int
        and value["pid"] == pid
        and value["language"] == "en"
        and type(value["model_arch"]) is int
        and value["model_arch"] == 5
        and (value["native_library_sha256"], value["native_library_bytes"]) == native
        and (value["python_api_sha256"], value["python_api_bytes"]) == python_api
        and type(value["file_origins"]) is int
        and 2 <= value["file_origins"] <= 16384
        and value["source_fallback"] is False,
        "catalog observation identity differs",
    )
    primary, spelling = _catalogs(value)
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "withoutSpelling": value["without_spelling"],
                "withSpelling": value["with_spelling"],
            },
            terminal_lf=False,
        )
    ).hexdigest()
    return primary, spelling, value["file_origins"], identity


def observe_moonshine_catalog(
    runtime: InstalledRuntimeEnvironmentV1,
) -> BoundMoonshineCatalogV1:
    """Run the archived fixed-profile catalog worker in a live installed runtime."""
    value = _installed_runtime_for_consumer(runtime)
    work = value.work
    try:
        purpose = value.metadata.purpose
        _require(purpose in _PURPOSES, "Moonshine catalog purpose is unavailable")
        closure = _dependency_binding_for_consumer(value.dependencies)
        covered = [row for row in closure.metadata.wheelhouses if row.purpose == purpose]
        _require(len(covered) == 1, "Moonshine catalog purpose is outside dependency coverage")
        provider = [row for row in covered[0].distributions if row.name == "moonshine-voice"]
        _require(
            len(provider) == 1 and provider[0].version == "0.1.0",
            "Moonshine catalog distribution differs",
        )
        expected = {path: (digest, size) for path, digest, size in provider[0].members}
        installed = {
            path: (digest, size) for path, digest, size in execution_file_metadata(value.files)
        }
        native = expected.get("moonshine_voice/moonshine.dll")
        python_api = expected.get("moonshine_voice/moonshine_api.py")
        _require(
            native is not None
            and python_api is not None
            and installed.get("moonshine_voice/moonshine.dll") == native
            and installed.get("moonshine_voice/moonshine_api.py") == python_api,
            "installed Moonshine catalog implementation differs",
        )
        assert native is not None and python_api is not None
        worker = _candidate_worker(closure.candidate)
        worker_files = work.enter(owned_execution_files({"worker.py": worker}))
        worker_root = _execution_files_for_consumer(worker_files)
        workspace = work.enter(_workspace())
        packages = _execution_files_for_consumer(value.files)
        report = workspace / "moonshine-catalog.json"
        invocation = work.run_tool(
            value.tools,
            "build_python",
            (str(worker_root / "worker.py"), str(packages), str(workspace), str(report)),
            workspace,
        )
        _execution_files_for_consumer(value.files)
        report_files = work.enter(retain_file_seals(workspace, (report.name,)))
        process = _tool_invocation_for_consumer(invocation).process
        primary, spelling, origins, identity = _observation(
            sealed_file_bytes(report_files, report.name, 128 * 1024),
            pid=process.pid,
            native=native,
            python_api=python_api,
        )
        _require(
            _installed_runtime_for_consumer(runtime) is value,
            "installed Moonshine runtime changed during catalog observation",
        )
        candidate = candidate_file_metadata(closure.candidate)
        outputs = build_output_metadata(value.outputs)
        _require(
            candidate.qualification_input_sha256
            == outputs.qualification_input_sha256
            == closure.metadata.qualification_input_sha256,
            "Moonshine catalog final input bindings differ",
        )
        metadata = MoonshineCatalogMetadataV1(
            candidate.qualification_input_sha256,
            candidate.source_commit,
            candidate.source_tree,
            purpose,
            hashlib.sha256(worker).hexdigest(),
            provider[0].sha256,
            python_api[0],
            native[0],
            identity,
            len(primary),
            len(spelling),
            sum(row.size for row in primary + spelling),
            origins,
        )
        receipt = object.__new__(BoundMoonshineCatalogV1)
        _BINDINGS[receipt] = _Binding(
            work,
            value,
            value.dependencies,
            value.outputs,
            closure.candidate,
            invocation,
            primary,
            spelling,
            metadata,
        )
        return receipt
    except BaseException as error:
        work._retain_failure(error)
        raise


def _binding(receipt: BoundMoonshineCatalogV1) -> _Binding:
    if type(receipt) is not BoundMoonshineCatalogV1:
        raise TypeError("Moonshine catalog binding type differs")
    _require(receipt in _BINDINGS, "Moonshine catalog binding is unregistered")
    value = _BINDINGS[receipt]
    _require(
        (not value.work._closing) or (value.work._closed and value.work._unrecoverable is None),
        "Moonshine catalog cleanup is incomplete",
    )
    closure = _dependency_binding_for_consumer(value.dependencies)
    candidate = candidate_file_metadata(value.candidate)
    outputs = build_output_metadata(value.outputs)
    _require(
        closure.candidate is value.candidate
        and candidate.qualification_input_sha256
        == outputs.qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "Moonshine catalog retained input bindings differ",
    )
    _require(
        tool_invocation_metadata(value.invocation).exit_code == 0,
        "Moonshine catalog invocation did not complete normally",
    )
    return value


def moonshine_catalog_metadata(
    receipt: BoundMoonshineCatalogV1,
) -> MoonshineCatalogMetadataV1:
    return _binding(receipt).metadata


def _moonshine_catalog_for_consumer(
    receipt: BoundMoonshineCatalogV1, *, include_spelling: bool
) -> tuple[_Resource, ...]:
    _require(type(include_spelling) is bool, "Moonshine spelling selection differs")
    value = _binding(receipt)
    return value.primary + value.spelling if include_spelling else value.primary

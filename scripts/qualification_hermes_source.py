"""Authenticate and bind the reviewed Hermes publisher source without executing it."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
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
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)
from scripts.source_archive_authority import SourceArchivePolicyV1, archive_member_validator

_REPOSITORY = "NousResearch/hermes-agent"
_RELEASE_TAG = "v2026.8.3"
_TAG_OBJECT = "7de39e700d2c329e15d32eb0b96e2f7cdd9fbdb2"
_COMMIT = "3c27eb6234bf91b8ceee9e9071591b31e9b148cb"
_TREE = "b217767ccb994605dad522e693fa1b4cdbc2f352"
# Official release and immutable source selection:
# https://github.com/NousResearch/hermes-agent/releases/tag/v2026.8.3
# https://codeload.github.com/NousResearch/hermes-agent/tar.gz/3c27eb6234bf91b8ceee9e9071591b31e9b148cb
_CODELOAD_SHA256 = "a68e96f385768ec6c466122bf21fcb697680a5f349c9a673badfbe27752b6928"
_CODELOAD_BYTES = 63_384_803
_CODELOAD_PREFIX = f"hermes-agent-{_COMMIT}"
_NORMALIZED_PREFIX = "hermes-agent-v0.20.0"
_SOURCE_TAR_SHA256 = "254b82ea69f80da5a473dc4d60ca709663ca01e6f37de1990f53713e1f81fcd4"
_SOURCE_TAR_BYTES = 146_739_200
_FILE_COUNT = 8_436
_TREE_BYTES = 139_651_582
_MAX_MEMBERS = 100_000
_MAX_FILE_BYTES = 16 * 1024**2
_MAX_TREE_BYTES = 256 * 1024**2
_CODELOAD_FILE = "publisher/hermes-agent.tar.gz"
_NORMALIZED_FILE = "source/hermes-agent-v0.20.0.tar"
_TREE_PREFIX = "publisher-tree/"
_HARNESS = "scripts/qualify_hermes_v020_pluginmanager.py"
_WORKER = "scripts/qualification_hermes_pluginmanager_worker.py"


class HermesPublisherSourceV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Hermes publisher sources are verifier-minted only")


class BoundHermesPublisherSourceV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Hermes publisher source bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class HermesPublisherSourceMetadataV1:
    repository: str
    release_tag: str
    tag_object: str
    commit: str
    tree: str
    codeload_sha256: str
    codeload_bytes: int
    source_tar_sha256: str
    source_tar_bytes: int
    normalized_tar_sha256: str
    normalized_tar_bytes: int
    member_count: int
    tree_bytes: int
    candidate_commit: str
    candidate_tree: str
    candidate_archive_sha256: str


@dataclass(frozen=True, slots=True)
class HermesPublisherFileMetadataV1:
    qualification_input_sha256: str
    source: HermesPublisherSourceMetadataV1


@dataclass(frozen=True, slots=True)
class _Source:
    inputs: BoundBuildInputsV1
    work: OwnedQualificationWorkV1
    files: ImmutableExecutionFilesV1
    metadata: HermesPublisherSourceMetadataV1
    members: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True, slots=True)
class _Final:
    inputs: BoundBuildInputsV1
    files: RetainedQualificationInputFilesV1
    preparation_owner: OwnedQualificationWorkV1
    metadata: HermesPublisherFileMetadataV1


_SOURCES: WeakKeyDictionary[HermesPublisherSourceV1, _Source] = WeakKeyDictionary()
_FINAL: WeakKeyDictionary[BoundHermesPublisherSourceV1, _Final] = WeakKeyDictionary()
_TRANSFERRED: WeakKeyDictionary[HermesPublisherSourceV1, str] = WeakKeyDictionary()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _policy(prefix: str, label: str) -> SourceArchivePolicyV1:
    return SourceArchivePolicyV1(
        prefix=prefix,
        workspace_child=".qualification-hermes-source",
        owner_marker=".qualification-hermes-source-owner",
        max_members=_MAX_MEMBERS,
        max_file_bytes=_MAX_FILE_BYTES,
        max_tree_bytes=_MAX_TREE_BYTES,
        error_label=label,
    )


def _read_members(raw: bytes, *, prefix: str, label: str) -> tuple[tuple[str, bytes, int], ...]:
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            validated = archive_member_validator(_policy(prefix, label))(archive)
            result: list[tuple[str, bytes, int]] = []
            for member in validated:
                source = archive.extractfile(member)
                _require(source is not None, f"{label} member is unreadable")
                assert source is not None
                with source:
                    payload = source.read(member.size + 1)
                _require(len(payload) == member.size, f"{label} member size differs")
                result.append((member.name.removeprefix(prefix + "/"), payload, member.mode))
    except (OSError, tarfile.TarError) as error:
        raise ValueError(f"{label} is invalid") from error
    _require(bool(result), f"{label} has no regular members")
    return tuple(result)


def _source_tar(codeload: bytes) -> bytes:
    try:
        return gzip.decompress(codeload)
    except OSError as error:
        raise ValueError("Hermes codeload compression is invalid") from error


def _normalize_source_tar(source_tar: bytes) -> tuple[bytes, tuple[tuple[str, bytes, int], ...]]:
    _require(type(source_tar) is bytes, "Hermes source tar bytes differ")
    output = io.BytesIO()
    try:
        with tarfile.open(fileobj=io.BytesIO(source_tar), mode="r:") as source:
            validated = archive_member_validator(
                _policy(_CODELOAD_PREFIX, "Hermes codeload source")
            )(source)
            payloads: dict[str, bytes] = {}
            rows: list[tuple[str, bytes, int]] = []
            for member in validated:
                stream = source.extractfile(member)
                _require(stream is not None, "Hermes codeload member is unreadable")
                assert stream is not None
                with stream:
                    payload = stream.read(member.size + 1)
                _require(len(payload) == member.size, "Hermes codeload member size differs")
                payloads[member.name] = payload
                rows.append(
                    (member.name.removeprefix(_CODELOAD_PREFIX + "/"), payload, member.mode)
                )
            with tarfile.open(fileobj=output, mode="w:", format=tarfile.USTAR_FORMAT) as archive:
                for member in source.getmembers():
                    _require(
                        member.isdir() or member.isreg(), "Hermes codeload member type differs"
                    )
                    relative = member.name.removeprefix(_CODELOAD_PREFIX).lstrip("/")
                    name = _NORMALIZED_PREFIX + ("/" + relative if relative else "")
                    info = tarfile.TarInfo(name + ("/" if member.isdir() else ""))
                    info.type = tarfile.DIRTYPE if member.isdir() else tarfile.REGTYPE
                    info.mode = member.mode
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    if member.isdir():
                        archive.addfile(info)
                        continue
                    payload = payloads[member.name]
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
    except (OSError, tarfile.TarError) as error:
        raise ValueError("Hermes codeload source is invalid") from error
    normalized = output.getvalue()
    normalized_rows = _read_members(
        normalized, prefix=_NORMALIZED_PREFIX, label="Hermes normalized source"
    )
    _require(
        tuple((name, payload, mode) for name, payload, mode in normalized_rows) == tuple(rows),
        "Hermes source normalization differs",
    )
    return normalized, tuple(rows)


def _source_member(inputs: BoundBuildInputsV1, path: str, *, live_tools: bool) -> bytes:
    bound = _build_inputs_for_consumer(inputs) if live_tools else _build_input_facts(inputs)
    payload = archives._archive_bytes_for_consumer(bound.archive, bound.identity)
    metadata = archives.verified_candidate_source_archive_metadata(bound.archive)
    member = next((item for item in metadata.manifest if item.path == path), None)
    _require(
        member is not None and member.kind == "file",
        "Hermes candidate source member is unavailable",
    )
    assert member is not None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        stream = archive.extractfile(metadata.prefix + "/" + path)
        _require(stream is not None, "Hermes candidate source member is unreadable")
        assert stream is not None
        raw = stream.read(member.size + 1)
    _require(
        len(raw) == member.size and hashlib.sha256(raw).hexdigest() == member.sha256,
        "Hermes candidate source member differs",
    )
    return raw


def admit_hermes_publisher_source(
    inputs: BoundBuildInputsV1, work: OwnedQualificationWorkV1, codeload: bytes
) -> HermesPublisherSourceV1:
    """Admit only the reviewed publisher archive before final input files exist."""
    _require(type(work) is OwnedQualificationWorkV1, "Hermes work owner differs")
    work._accepting()
    bound = _build_inputs_for_consumer(inputs)
    _require(
        type(codeload) is bytes
        and len(codeload) == _CODELOAD_BYTES
        and hashlib.sha256(codeload).hexdigest() == _CODELOAD_SHA256,
        "Hermes codeload differs from reviewed publisher source",
    )
    source_tar = _source_tar(codeload)
    _require(
        (
            hashlib.sha256(source_tar).hexdigest(),
            len(source_tar),
        )
        == (_SOURCE_TAR_SHA256, _SOURCE_TAR_BYTES),
        "Hermes publisher source tar differs",
    )
    normalized, rows = _normalize_source_tar(source_tar)
    _require(
        (len(rows), sum(len(row[1]) for row in rows)) == (_FILE_COUNT, _TREE_BYTES),
        "Hermes publisher source inventory differs",
    )
    contents = {
        _CODELOAD_FILE: codeload,
        _NORMALIZED_FILE: normalized,
        **{_TREE_PREFIX + name: payload for name, payload, _ in rows},
    }
    retained = work.enter(owned_execution_files(contents))
    source = archives.verified_candidate_source_archive_metadata(bound.archive)
    receipt = object.__new__(HermesPublisherSourceV1)
    _SOURCES[receipt] = _Source(
        inputs,
        work,
        retained,
        HermesPublisherSourceMetadataV1(
            _REPOSITORY,
            _RELEASE_TAG,
            _TAG_OBJECT,
            _COMMIT,
            _TREE,
            _CODELOAD_SHA256,
            _CODELOAD_BYTES,
            _SOURCE_TAR_SHA256,
            _SOURCE_TAR_BYTES,
            hashlib.sha256(normalized).hexdigest(),
            len(normalized),
            _FILE_COUNT,
            _TREE_BYTES,
            source.candidate_head_oid,
            source.candidate_tree_oid,
            source.archive_sha256,
        ),
        tuple(
            (name, hashlib.sha256(payload).hexdigest(), len(payload)) for name, payload, _ in rows
        ),
    )
    return receipt


def _source(receipt: HermesPublisherSourceV1) -> _Source:
    if type(receipt) is not HermesPublisherSourceV1:
        raise TypeError("Hermes publisher source capability type differs")
    _require(receipt in _SOURCES, "Hermes publisher source capability is unregistered")
    value = _SOURCES[receipt]
    value.work._accepting()
    source = archives.verified_candidate_source_archive_metadata(
        _build_inputs_for_consumer(value.inputs).archive
    )
    _require(
        (source.candidate_head_oid, source.candidate_tree_oid, source.archive_sha256)
        == (
            value.metadata.candidate_commit,
            value.metadata.candidate_tree,
            value.metadata.candidate_archive_sha256,
        ),
        "Hermes candidate source authority differs",
    )
    expected = tuple(
        sorted(
            (
                (_CODELOAD_FILE, value.metadata.codeload_sha256, value.metadata.codeload_bytes),
                (
                    _NORMALIZED_FILE,
                    value.metadata.normalized_tar_sha256,
                    value.metadata.normalized_tar_bytes,
                ),
                *((_TREE_PREFIX + name, digest, size) for name, digest, size in value.members),
            )
        )
    )
    _require(execution_file_metadata(value.files) == expected, "Hermes retained source differs")
    return value


def hermes_publisher_source_metadata(
    receipt: HermesPublisherSourceV1,
) -> HermesPublisherSourceMetadataV1:
    return _source(receipt).metadata


def bind_hermes_publisher_source(
    source: HermesPublisherSourceV1, files: RetainedQualificationInputFilesV1
) -> BoundHermesPublisherSourceV1:
    admitted = _source(source)
    _require(source not in _TRANSFERRED, "Hermes publisher source was already transferred")
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    candidate = document["candidate"]
    metadata = admitted.metadata
    _require(
        (candidate["candidateCommit"], candidate["tree"])
        == (metadata.candidate_commit, metadata.candidate_tree),
        "Hermes final candidate differs",
    )
    refs = {item["role"]: item for item in document["files"]}
    archive_ref, harness_ref = refs["hermes_source_archive"], refs["hermes_pluginmanager_runner"]
    normalized = (
        _execution_files_for_consumer(admitted.files).joinpath(_NORMALIZED_FILE).read_bytes()
    )
    harness = _source_member(admitted.inputs, _HARNESS, live_tools=True)
    _require(
        sealed_file_bytes(selected.seals, archive_ref["relativePath"], 256 * 1024**2) == normalized
        and sealed_file_bytes(selected.seals, harness_ref["relativePath"], 16 * 1024**2) == harness,
        "Hermes final source or harness differs",
    )
    receipt = object.__new__(BoundHermesPublisherSourceV1)
    _FINAL[receipt] = _Final(
        admitted.inputs,
        files,
        admitted.work,
        HermesPublisherFileMetadataV1(
            retained_input_metadata(files).qualification_input_sha256, metadata
        ),
    )
    _TRANSFERRED[source] = retained_input_metadata(files).qualification_input_sha256
    return receipt


def hermes_publisher_file_metadata(
    receipt: BoundHermesPublisherSourceV1,
) -> HermesPublisherFileMetadataV1:
    if type(receipt) is not BoundHermesPublisherSourceV1:
        raise TypeError("Hermes publisher final capability type differs")
    _require(receipt in _FINAL, "Hermes publisher final capability is unregistered")
    value = _FINAL[receipt]
    metadata = value.metadata.source
    bound = _build_input_facts(value.inputs)
    source = archives.verified_candidate_source_archive_metadata(bound.archive)
    _require(
        (source.candidate_head_oid, source.candidate_tree_oid, source.archive_sha256)
        == (metadata.candidate_commit, metadata.candidate_tree, metadata.candidate_archive_sha256),
        "Hermes retained candidate source differs",
    )
    selected = _retained_input_files_for_consumer(value.files)
    document = json.loads(selected.document)
    refs = {item["role"]: item for item in document["files"]}
    archive = sealed_file_bytes(
        selected.seals, refs["hermes_source_archive"]["relativePath"], 256 * 1024**2
    )
    harness = sealed_file_bytes(
        selected.seals, refs["hermes_pluginmanager_runner"]["relativePath"], 16 * 1024**2
    )
    expected_harness = _source_member(value.inputs, _HARNESS, live_tools=False)
    _require(
        hashlib.sha256(archive).hexdigest() == metadata.normalized_tar_sha256
        and len(archive) == metadata.normalized_tar_bytes
        and harness == expected_harness
        and retained_input_metadata(value.files).qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "Hermes final source seals differ",
    )
    return value.metadata


def _bound_hermes_source_for_consumer(receipt: BoundHermesPublisherSourceV1) -> _Final:
    """Return a live final binding only after its source and harness seals revalidate."""
    hermes_publisher_file_metadata(receipt)
    return _FINAL[receipt]


def _bound_hermes_execution_inputs(
    receipt: BoundHermesPublisherSourceV1,
) -> tuple[dict[str, bytes], bytes, bytes]:
    """Return exact final source and candidate-archived harness/worker bytes."""
    value = _bound_hermes_source_for_consumer(receipt)
    selected = _retained_input_files_for_consumer(value.files)
    document = json.loads(selected.document)
    refs = {item["role"]: item for item in document["files"]}
    normalized = sealed_file_bytes(
        selected.seals,
        refs["hermes_source_archive"]["relativePath"],
        _MAX_TREE_BYTES,
    )
    rows = _read_members(
        normalized,
        prefix=_NORMALIZED_PREFIX,
        label="Hermes final normalized source",
    )
    _require(
        len(rows) == value.metadata.source.member_count
        and sum(len(payload) for _, payload, _ in rows) == value.metadata.source.tree_bytes,
        "Hermes final source inventory differs",
    )
    harness = sealed_file_bytes(
        selected.seals,
        refs["hermes_pluginmanager_runner"]["relativePath"],
        16 * 1024**2,
    )
    _require(
        harness == _source_member(value.inputs, _HARNESS, live_tools=False),
        "Hermes final harness differs from candidate source",
    )
    worker = _source_member(value.inputs, _WORKER, live_tools=False)
    _require(0 < len(worker) <= 512 * 1024, "Hermes PluginManager worker is unbounded")
    return {name: payload for name, payload, _ in rows}, harness, worker

"""Acquire source-locked Moonshine resources before the final input root exists."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from packaging.utils import parse_wheel_filename

from scripts.qualification_build_inputs import (
    BoundBuildInputsV1,
    _build_input_facts,
    _build_inputs_for_consumer,
)
from scripts.qualification_dependency_files import _archive_locked_wheels
from scripts.qualification_execution_files import (
    _execution_files_for_consumer,
    owned_execution_files,
)
from scripts.qualification_file_seals import (
    RetainedFileSealsV1,
    retain_file_seals,
    sealed_file_bytes,
)
from scripts.qualification_installation import _workspace
from scripts.qualification_moonshine_catalog import (
    _archive_worker,
    _observation,
    _strict_object,
)
from scripts.qualification_moonshine_worker import _crc32c
from scripts.qualification_owned_work import (
    OwnedQualificationCleanupError,
    OwnedQualificationWorkV1,
)
from scripts.qualification_tool_process import (
    CompletedToolInvocationV1,
    _tool_invocation_for_consumer,
    tool_invocation_metadata,
)
from scripts.qualification_wheelhouse import WheelDistributionV1, _inspect, _target
from scripts.qualify_evidence_slice_zero import canonical_json_bytes

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPORT = "moonshine-preparation.json"
_MAX_REPORT_BYTES = 256 * 1024
_ERROR_REPORT = "moonshine-preparation-error.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class PreparedMoonshineResourceV1:
    group: str
    cache_path: str
    url: str
    size: int
    crc32c: str
    sha256: str


@dataclass(frozen=True, slots=True)
class MoonshinePreparationMetadataV1:
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    worker_sha256: str
    distribution_basename: str
    distribution_sha256: str
    python_api_sha256: str
    native_library_sha256: str
    catalog_identity_sha256: str
    resource_count: int
    resource_bytes: int


class PreparedMoonshineResourcesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("prepared Moonshine resources are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Preparation:
    resource_work: OwnedQualificationWorkV1
    execution_work: OwnedQualificationWorkV1
    inputs: BoundBuildInputsV1
    invocation: CompletedToolInvocationV1
    resource_root: Path
    report_seals: RetainedFileSealsV1
    resource_seals: RetainedFileSealsV1
    report_sha256: str
    resources: tuple[PreparedMoonshineResourceV1, ...]
    metadata: MoonshinePreparationMetadataV1


_PREPARED: WeakKeyDictionary[PreparedMoonshineResourcesV1, _Preparation] = WeakKeyDictionary()


def _wheel(
    inputs: BoundBuildInputsV1, basename: str, raw: bytes
) -> tuple[WheelDistributionV1, dict[str, bytes]]:
    bound = _build_inputs_for_consumer(inputs)
    _require(
        type(basename) is str
        and "/" not in basename
        and "\\" not in basename
        and type(raw) is bytes,
        "Moonshine preparation distribution identity differs",
    )
    try:
        name, version, _, tags = parse_wheel_filename(basename)
    except ValueError:
        raise ValueError("Moonshine preparation distribution filename is invalid") from None
    _require(
        str(name) == "moonshine-voice"
        and str(version) == "0.1.0"
        and str(next(iter(tags))) == "py3-none-win_amd64"
        and len(tags) == 1,
        "Moonshine preparation distribution profile differs",
    )
    digest = hashlib.sha256(raw).hexdigest()
    _, locked = _archive_locked_wheels(bound.archive, bound.identity)
    _require(
        (str(name), str(version), basename, digest, len(raw)) in locked,
        "Moonshine preparation distribution is outside the source lock",
    )
    environment, target_tags = _target(bound.metadata.python_version, "windows_amd64")
    distribution = _inspect(basename, raw, environment, target_tags, site_processing=False)
    _require(
        distribution.name == "moonshine-voice"
        and distribution.version == "0.1.0"
        and distribution.sha256 == digest
        and not any(".data/" in path for path, _, _ in distribution.members),
        "Moonshine preparation wheel contents differ",
    )
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for path, member_digest, size in distribution.members:
            payload = archive.read(path)
            _require(
                (hashlib.sha256(payload).hexdigest(), len(payload)) == (member_digest, size),
                "Moonshine preparation wheel member differs",
            )
            contents[path] = payload
    return distribution, contents


def _prepared_report(
    raw: bytes,
    *,
    pid: int,
    native: tuple[str, int],
    python_api: tuple[str, int],
) -> tuple[tuple[PreparedMoonshineResourceV1, ...], str]:
    value = _strict_object(raw)
    _require(set(value) == {"catalog", "resources"}, "Moonshine preparation fields differ")
    catalog = value["catalog"]
    _require(type(catalog) is dict, "Moonshine preparation catalog differs")
    primary, spelling, _, identity = _observation(
        canonical_json_bytes(catalog, terminal_lf=False),
        pid=pid,
        native=native,
        python_api=python_api,
    )
    selected = primary + spelling
    observed = value["resources"]
    _require(
        type(observed) is list and len(observed) == len(selected) == 9,
        "Moonshine preparation resource closure differs",
    )
    rows = []
    for expected, item in zip(selected, observed, strict=True):
        _require(
            type(item) is dict
            and set(item) == {"cache_path", "crc32c", "sha256", "size", "url"},
            "Moonshine prepared resource fields differ",
        )
        assert isinstance(item, dict)
        _require(
            (
                item["cache_path"],
                item["url"],
                item["size"],
                item["crc32c"],
            )
            == (expected.cache_path, expected.url, expected.size, expected.crc32c)
            and type(item["sha256"]) is str
            and _SHA256.fullmatch(item["sha256"]) is not None,
            "Moonshine prepared resource identity differs",
        )
        rows.append(
            PreparedMoonshineResourceV1(
                expected.group,
                expected.cache_path,
                expected.url,
                expected.size,
                expected.crc32c,
                item["sha256"],
            )
        )
    return tuple(rows), identity


def _verify_resource(seals: RetainedFileSealsV1, resource: PreparedMoonshineResourceV1) -> bytes:
    payload = cast(bytes, sealed_file_bytes(seals, resource.cache_path, 512 * 1024**2))
    crc32c = base64.b64encode(_crc32c(payload).to_bytes(4, "big")).decode("ascii")
    _require(
        (len(payload), crc32c, hashlib.sha256(payload).hexdigest())
        == (resource.size, resource.crc32c, resource.sha256),
        "prepared Moonshine resource bytes differ",
    )
    return payload


def _worker_failure(root: Path) -> str | None:
    path = root / _ERROR_REPORT
    if not path.is_file() or path.stat().st_size > 4096:
        return None
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeError, json.JSONDecodeError, OSError):
        return None
    if (
        type(value) is dict
        and set(value) == {"error", "stage"}
        and type(value["error"]) is str
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value["error"])
        and type(value["stage"]) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,255}", value["stage"])
    ):
        return f"{value['stage']} ({value['error']})"
    return None


def _binding(receipt: PreparedMoonshineResourcesV1) -> _Preparation:
    if type(receipt) is not PreparedMoonshineResourcesV1:
        raise TypeError("prepared Moonshine resource capability type differs")
    _require(receipt in _PREPARED, "prepared Moonshine resource capability is unregistered")
    value = _PREPARED[receipt]
    _require(
        not value.resource_work._closing
        and value.execution_work._closed
        and value.execution_work._unrecoverable is None,
        "Moonshine preparation resource owner is unavailable or execution cleanup is incomplete",
    )
    source = _build_input_facts(value.inputs).metadata
    _require(
        (source.source_commit, source.source_tree, source.source_archive_sha256)
        == (
            value.metadata.source_commit,
            value.metadata.source_tree,
            value.metadata.source_archive_sha256,
        ),
        "Moonshine preparation source binding differs",
    )
    _require(
        tool_invocation_metadata(value.invocation).exit_code == 0
        and hashlib.sha256(
            sealed_file_bytes(value.report_seals, _REPORT, _MAX_REPORT_BYTES)
        ).hexdigest()
        == value.report_sha256,
        "Moonshine preparation invocation or report differs",
    )
    return value


def moonshine_preparation_metadata(
    receipt: PreparedMoonshineResourcesV1,
) -> MoonshinePreparationMetadataV1:
    return _binding(receipt).metadata


def _moonshine_preparation_for_consumer(
    receipt: PreparedMoonshineResourcesV1,
) -> _Preparation:
    return _binding(receipt)


def _prepared_resource_bytes(
    receipt: PreparedMoonshineResourcesV1, cache_path: str
) -> bytes:
    value = _binding(receipt)
    selected = [item for item in value.resources if item.cache_path == cache_path]
    _require(len(selected) == 1, "prepared Moonshine resource is unavailable")
    return _verify_resource(value.resource_seals, selected[0])


@contextmanager
def prepare_moonshine_resources(
    work: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
    *,
    distribution_basename: str,
    distribution: bytes,
) -> Iterator[PreparedMoonshineResourcesV1]:
    """Acquire the fixed medium-English cache closure under pre-final authority."""
    _require(type(work) is OwnedQualificationWorkV1, "Moonshine preparation owner type differs")
    work._accepting()
    resource_work = OwnedQualificationWorkV1()
    resource_root = resource_work.enter(_workspace())
    execution_work = OwnedQualificationWorkV1()
    try:
        with execution_work:
            bound = _build_inputs_for_consumer(inputs)
            package, contents = _wheel(inputs, distribution_basename, distribution)
            expected = {path: (digest, size) for path, digest, size in package.members}
            native = expected.get("moonshine_voice/moonshine.dll")
            python_api = expected.get("moonshine_voice/moonshine_api.py")
            _require(
                native is not None and python_api is not None,
                "Moonshine preparation implementation is unavailable",
            )
            assert native is not None and python_api is not None
            worker = _archive_worker(bound.archive, bound.identity)
            namespace = execution_work.enter(owned_execution_files(contents))
            worker_files = execution_work.enter(owned_execution_files({"worker.py": worker}))
            packages = _execution_files_for_consumer(namespace)
            worker_root = _execution_files_for_consumer(worker_files)
            execution = execution_work.enter(_workspace())
            invocation = execution_work.run_tool(
                bound.tools,
                "build_python",
                (
                    str(worker_root / "worker.py"),
                    str(packages),
                    str(execution),
                    str(resource_root),
                    "acquire",
                ),
                execution,
                timeout_milliseconds=600_000,
            )
            _execution_files_for_consumer(namespace)
    except OwnedQualificationCleanupError as error:
        resource_work._retain_failure(error)
        retained = OwnedQualificationCleanupError(resource_work)
        work._retain_failure(retained)
        raise retained from error
    except BaseException as error:
        stage = _worker_failure(resource_root)
        failure = (
            ValueError(f"Moonshine acquisition worker failed at {stage}")
            if stage is not None
            else error
        )
        work._retain_failure(failure)
        try:
            resource_work.close()
        except BaseException as cleanup:
            work._retain_failure(cleanup)
            raise OwnedQualificationCleanupError(work) from BaseExceptionGroup(
                "Moonshine preparation and resource cleanup failed", [failure, cleanup]
            )
        raise failure from error
    try:
        report_seals = resource_work.enter(retain_file_seals(resource_root, (_REPORT,)))
        report = sealed_file_bytes(report_seals, _REPORT, _MAX_REPORT_BYTES)
        resources, catalog_identity = _prepared_report(
            report,
            pid=_tool_invocation_for_consumer(invocation).process.pid,
            native=native,
            python_api=python_api,
        )
        expected_files = {_REPORT, *(item.cache_path for item in resources)}
        actual_files = {
            path.relative_to(resource_root).as_posix()
            for path in resource_root.rglob("*")
            if path.is_file()
        }
        _require(actual_files == expected_files, "Moonshine prepared namespace differs")
        resource_seals = resource_work.enter(
            retain_file_seals(resource_root, tuple(item.cache_path for item in resources))
        )
        for resource in resources:
            _verify_resource(resource_seals, resource)
        source = bound.metadata
        metadata = MoonshinePreparationMetadataV1(
            source.source_commit,
            source.source_tree,
            source.source_archive_sha256,
            hashlib.sha256(worker).hexdigest(),
            distribution_basename,
            package.sha256,
            python_api[0],
            native[0],
            catalog_identity,
            len(resources),
            sum(item.size for item in resources),
        )
        receipt = object.__new__(PreparedMoonshineResourcesV1)
        _PREPARED[receipt] = _Preparation(
            resource_work,
            execution_work,
            inputs,
            invocation,
            resource_root,
            report_seals,
            resource_seals,
            hashlib.sha256(report).hexdigest(),
            resources,
            metadata,
        )
    except BaseException as error:
        work._retain_failure(error)
        try:
            resource_work.close()
        except BaseException as cleanup:
            work._retain_failure(cleanup)
            raise OwnedQualificationCleanupError(work) from BaseExceptionGroup(
                "Moonshine preparation validation and cleanup failed", [error, cleanup]
            )
        raise
    try:
        yield receipt
    finally:
        try:
            resource_work.close()
        except BaseException as cleanup:
            work._retain_failure(cleanup)
            raise
        else:
            _PREPARED.pop(receipt, None)

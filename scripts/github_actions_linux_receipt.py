"""Authenticate a Linux qualification receipt with direct GitHub HTTPS observations.

GitHub metadata may be visible without a credential, while artifact ZIP download
can require an existing GitHub Actions-read credential.  This module neither
provisions nor discovers that credential.  It is sent only to api.github.com;
the narrowly accepted signed-storage redirect receives no bearer header.

This is a final-binding consumer only: its `BoundDependencyPurposeV1` input
requires the final retained qualification inputs.  It cannot establish the
stage-2 Linux prerequisite before that input exists.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts import qualification_linux_receipt as payloads
from scripts.qualification_candidate_files import (
    BoundCandidateFilesV1,
    _candidate_files_for_consumer,
)
from scripts.qualification_candidate_files import (
    _Binding as _CandidateBinding,
)
from scripts.qualification_dependency_files import (
    BoundDependencyPurposeV1,
    _dependency_binding_for_consumer,
    _dependency_wheels_for_consumer,
)
from scripts.qualification_file_seals import sealed_file_bytes
from scripts.qualification_linux_image import AdmittedLinuxRuntimeImageV1, linux_image_metadata
from scripts.retained_qualification_inputs import _retained_input_files_for_consumer
from scripts.task13_artifact_orchestrator import CandidateIdentityV1

_API = "https://api.github.com"
_OWNER = "sushiHex"
_REPOSITORY = "hermes-realtime"
_WORKFLOW = ".github/workflows/release-gates.yml"
_REPOSITORY_ID = payloads._REPOSITORY_ID
_WORKFLOW_ID = payloads._WORKFLOW_ID
_ATTEMPT = payloads._ATTEMPT
_JOB = payloads._JOB
_ARTIFACT = payloads._ARTIFACT
_MEMBER = payloads._MEMBER


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class _LinuxReceiptFacts:
    candidate_commit: str
    candidate_tree: str
    source_archive_sha256: str
    workflow_bytes: bytes
    direct_wheel_sha256: str
    direct_wheel_bytes: int
    direct_wheel_basename: str
    wheelhouse_manifest_sha256: str
    source_lock_sha256: str
    requirements_sha256: str
    constraints_sha256: str
    wheels: tuple[tuple[str, str, int], ...]
    image_reference: str
    image_config_sha256: str
    image_layers: tuple[str, ...]
    image_diff_ids: tuple[str, ...]
    image_python_version: str


@dataclass(frozen=True, slots=True)
class LinuxReceiptMetadataV1:
    run_id: int
    attempt: int
    artifact_sha256: str
    candidate_commit: str
    direct_wheel_sha256: str
    cleanup_removed: bool


@dataclass(frozen=True, slots=True)
class _RetainedReceipt:
    linux_runtime: BoundDependencyPurposeV1
    image: AdmittedLinuxRuntimeImageV1
    facts: _LinuxReceiptFacts
    metadata: LinuxReceiptMetadataV1
    payload: payloads.LinuxReceiptPayloadV1


@dataclass(frozen=True, slots=True)
class _AuthenticatedObservation:
    facts: _LinuxReceiptFacts
    metadata: LinuxReceiptMetadataV1
    payload: payloads.LinuxReceiptPayloadV1


class AuthenticatedLinuxNullCaptureReceiptV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Linux qualification receipts are verifier-minted only")


_RECEIPTS: WeakKeyDictionary[AuthenticatedLinuxNullCaptureReceiptV1, _RetainedReceipt] = (
    WeakKeyDictionary()
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, request: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        raise urllib.error.HTTPError(newurl, code, "redirect refused", headers, fp)


class _HttpsTransport:
    """Private, fixed-origin transport.  It carries no caller-controlled authority."""

    def __init__(self, api_bearer: str | None) -> None:
        if api_bearer is not None:
            _require(
                type(api_bearer) is str
                and 0 < len(api_bearer) <= 4096
                and "\r" not in api_bearer
                and "\n" not in api_bearer,
                "GitHub receipt credential differs",
            )
        self._api_bearer = api_bearer

    def get(self, url: str, headers: dict[str, str]) -> object:
        parsed = urlsplit(url)
        _require(
            parsed.scheme == "https"
            and parsed.netloc == "api.github.com"
            and not parsed.username
            and not parsed.password,
            "GitHub receipt URL differs",
        )
        request_headers = dict(headers)
        if self._api_bearer is not None:
            request_headers["Authorization"] = "Bearer " + self._api_bearer
        request = urllib.request.Request(url, headers=request_headers, method="GET")
        opener = urllib.request.build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=20) as response:
                _require(
                    response.url == url and response.status == 200,
                    "GitHub receipt response differs",
                )
                raw = response.read(16 * 1024**2 + 1)
        except urllib.error.HTTPError as error:
            if error.code != 302 or not url.endswith("/zip"):
                raise ValueError("GitHub receipt observation failed") from None
            try:
                location = error.headers.get("Location")
                _require(type(location) is str, "GitHub receipt artifact redirect differs")
                assert isinstance(location, str)
                redirect = urlsplit(location)
                _require(
                    redirect.scheme == "https"
                    and redirect.port is None
                    and not redirect.username
                    and not redirect.password
                    and (
                        redirect.hostname == "artifactcache.actions.githubusercontent.com"
                        or (
                            redirect.hostname is not None
                            and redirect.hostname.endswith(".blob.core.windows.net")
                        )
                    ),
                    "GitHub receipt artifact redirect differs",
                )
                # The signed storage URL came from the authenticated GitHub API.
                # It receives no API bearer header and cannot redirect again.
                storage = urllib.request.Request(
                    location,
                    headers={"Accept": "application/octet-stream"},
                    method="GET",
                )
                with opener.open(storage, timeout=20) as response:
                    _require(
                        response.url == location and response.status == 200,
                        "GitHub receipt artifact response differs",
                    )
                    raw = response.read(16 * 1024**2 + 1)
            except (OSError, urllib.error.HTTPError, ValueError):
                raise ValueError("GitHub receipt artifact observation failed") from None
        except OSError:
            raise ValueError("GitHub receipt observation failed") from None
        _require(len(raw) <= 16 * 1024**2, "GitHub receipt response exceeds its bound")
        if url.endswith("/zip"):
            return raw
        try:
            return json.loads(raw, object_pairs_hook=_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("GitHub receipt JSON is invalid") from error


def _object(rows: list[tuple[str, object]]) -> dict[str, object]:
    value = dict(rows)
    _require(len(value) == len(rows), "GitHub receipt JSON has duplicate fields")
    return value


def _https_transport(api_bearer: str | None) -> _HttpsTransport:
    return _HttpsTransport(api_bearer)


def _workflow_from_source(
    archive_receipt: archives.VerifiedCandidateSourceArchiveV1,
    identity: CandidateIdentityV1,
) -> bytes:
    archive = archives._archive_bytes_for_consumer(archive_receipt, identity)
    # The source authority already authenticates the exact candidate tree.  Read
    # the workflow from that opaque archive, never a caller-supplied pathname.
    archive_metadata = archives.verified_candidate_source_archive_metadata(archive_receipt)
    member = next((item for item in archive_metadata.manifest if item.path == _WORKFLOW), None)
    _require(
        member is not None and member.kind == "file" and 0 < member.size <= 1024 * 1024,
        "candidate workflow is unavailable",
    )
    assert member is not None
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as contents:
        stream = contents.extractfile(archive_metadata.prefix + "/" + _WORKFLOW)
        _require(stream is not None, "candidate workflow is unreadable")
        assert stream is not None
        raw = stream.read(member.size + 1)
    _require((len(raw), _sha(raw)) == (member.size, member.sha256), "candidate workflow differs")
    return raw


def _workflow_from_archive(candidate: BoundCandidateFilesV1) -> tuple[bytes, _CandidateBinding]:
    bound = _candidate_files_for_consumer(candidate)
    return _workflow_from_source(bound.archive, bound.identity), bound


def _facts_from_capabilities(
    linux_runtime: BoundDependencyPurposeV1,
    image: AdmittedLinuxRuntimeImageV1,
) -> _LinuxReceiptFacts:
    dependency = _dependency_binding_for_consumer(linux_runtime)
    _require(
        {item.purpose for item in dependency.metadata.wheelhouses} == {"realtime_linux_runtime"},
        "Linux receipt needs exactly the Linux dependency purpose",
    )
    workflow, bound = _workflow_from_archive(dependency.candidate)
    selected = _retained_input_files_for_consumer(dependency.files)
    document: dict[str, Any] = json.loads(selected.document)
    refs = {item["role"]: item for item in document["files"]}
    direct = refs["direct_wheel"]
    _require(
        type(direct.get("basename")) is str,
        "Linux receipt direct wheel basename differs",
    )
    direct_basename = cast(str, direct["basename"])
    direct_raw = sealed_file_bytes(selected.seals, direct["relativePath"], 16 * 1024**2)
    _require(
        (_sha(direct_raw), len(direct_raw)) == (direct["sha256"], direct["bytes"]),
        "Linux receipt direct wheel differs",
    )
    wheelhouse_ref = refs["linux_runtime_wheelhouse_manifest"]
    wheelhouse_raw = sealed_file_bytes(selected.seals, wheelhouse_ref["relativePath"], 4 * 1024**2)
    actual_wheels, requirements, constraints = _dependency_wheels_for_consumer(
        linux_runtime, "realtime_linux_runtime"
    )
    image_metadata = linux_image_metadata(image)
    return _LinuxReceiptFacts(
        bound.metadata.source_commit,
        bound.metadata.source_tree,
        bound.metadata.source_archive_sha256,
        workflow,
        _sha(direct_raw),
        len(direct_raw),
        direct_basename,
        _sha(wheelhouse_raw),
        dependency.metadata.source_lock_sha256,
        _sha(requirements),
        _sha(constraints),
        tuple(sorted((name, _sha(raw), len(raw)) for name, raw in actual_wheels.items())),
        image_metadata.image_reference,
        image_metadata.config_sha256,
        image_metadata.layer_sha256s,
        image_metadata.layer_diff_sha256s,
        image_metadata.python_version,
    )


def _timestamp(value: object, label: str) -> datetime:
    _require(type(value) is str and value.endswith("Z"), label + " differs")
    assert isinstance(value, str)
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00").astimezone(UTC)
    except ValueError as error:
        raise ValueError(label + " differs") from error


def _record(transport: _HttpsTransport, path: str) -> dict[str, object]:
    value = transport.get(
        _API + "/repos/" + _OWNER + "/" + _REPOSITORY + path,
        {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    _require(type(value) is dict, "GitHub receipt service record differs")
    return cast(dict[str, object], value)


def _service_workflow(transport: _HttpsTransport, commit: str) -> bytes:
    service = _record(
        transport,
        "/contents/.github/workflows/release-gates.yml?ref=" + commit,
    )
    size = service.get("size")
    _require(
        service.get("type") == "file"
        and service.get("path") == _WORKFLOW
        and service.get("encoding") == "base64"
        and type(service.get("content")) is str
        and type(size) is int
        and 0 < size <= 1024 * 1024,
        "GitHub receipt workflow source differs",
    )
    content = cast(str, service["content"])
    assert isinstance(size, int)
    try:
        encoded = content.encode("ascii")
        _require(
            re.fullmatch(rb"[A-Za-z0-9+/=\n]*", encoded) is not None,
            "GitHub receipt workflow source differs",
        )
        raw = base64.b64decode(encoded, validate=False)
    except (ValueError, UnicodeEncodeError) as error:
        raise ValueError("GitHub receipt workflow source differs") from error
    _require(len(raw) == size, "GitHub receipt workflow source differs")
    return raw


def _match_payload(
    value: payloads.LinuxReceiptPayloadV1, facts: _LinuxReceiptFacts, run_id: int
) -> None:
    _require(
        (value.repository_id, value.workflow_id, value.run_id, value.attempt)
        == (_REPOSITORY_ID, _WORKFLOW_ID, run_id, _ATTEMPT),
        "Linux receipt service identity differs",
    )
    _require(
        (
            value.candidate_commit,
            value.candidate_tree,
            value.source_archive_sha256,
            value.workflow_sha256,
        )
        == (
            facts.candidate_commit,
            facts.candidate_tree,
            facts.source_archive_sha256,
            _sha(facts.workflow_bytes),
        ),
        "Linux receipt candidate or workflow differs",
    )
    _require(
        (value.direct_wheel_sha256, value.direct_wheel_bytes, value.direct_wheel_basename)
        == (facts.direct_wheel_sha256, facts.direct_wheel_bytes, facts.direct_wheel_basename),
        "Linux receipt direct wheel differs",
    )
    _require(
        (
            value.wheelhouse_manifest_sha256,
            value.source_lock_sha256,
            value.requirements_sha256,
            value.constraints_sha256,
            value.wheels,
        )
        == (
            facts.wheelhouse_manifest_sha256,
            facts.source_lock_sha256,
            facts.requirements_sha256,
            facts.constraints_sha256,
            facts.wheels,
        ),
        "Linux receipt wheelhouse differs",
    )
    _require(
        (
            value.image_reference,
            value.image_config_sha256,
            value.image_layers,
            value.image_diff_ids,
            value.python_version,
        )
        == (
            facts.image_reference,
            facts.image_config_sha256,
            facts.image_layers,
            facts.image_diff_ids,
            facts.image_python_version,
        ),
        "Linux receipt image differs",
    )


def _authenticate_observation(
    run_id: int,
    facts: _LinuxReceiptFacts,
    *,
    github_api_bearer: str | None = None,
) -> _AuthenticatedObservation:
    """Observe the fixed GitHub service once against phase-owned expected facts."""
    _require(
        type(run_id) is int and not isinstance(run_id, bool) and run_id > 0,
        "Linux receipt run ID differs",
    )
    if type(facts) is not _LinuxReceiptFacts:
        raise TypeError("Linux receipt expected facts type differs")
    transport = _https_transport(github_api_bearer)
    attempt = _record(transport, f"/actions/runs/{run_id}/attempts/{_ATTEMPT}")
    repository = attempt.get("repository")
    _require(
        attempt.get("id") == run_id
        and attempt.get("run_attempt") == _ATTEMPT
        and attempt.get("workflow_id") == _WORKFLOW_ID
        and type(repository) is dict
        and cast(dict[str, object], repository).get("id") == _REPOSITORY_ID
        and attempt.get("head_sha") == facts.candidate_commit
        and attempt.get("path") == _WORKFLOW
        and attempt.get("status") == "completed"
        and attempt.get("conclusion") == "success",
        "GitHub receipt attempt differs",
    )
    started, finished = (
        _timestamp(attempt.get("created_at"), "GitHub receipt attempt start"),
        _timestamp(attempt.get("updated_at"), "GitHub receipt attempt finish"),
    )
    _require(started < finished, "GitHub receipt attempt timestamps differ")
    _require(
        _service_workflow(transport, facts.candidate_commit) == facts.workflow_bytes,
        "GitHub receipt workflow source differs",
    )
    job_record = _record(transport, f"/actions/runs/{run_id}/attempts/{_ATTEMPT}/jobs?per_page=100")
    jobs = job_record.get("jobs")
    _require(type(jobs) is list, "GitHub receipt jobs differ")
    jobs = cast(list[object], jobs)
    _require(job_record.get("total_count") == len(jobs), "GitHub receipt job list is incomplete")
    matched = [job for job in jobs if type(job) is dict and job.get("name") == _JOB]
    _require(
        len(matched) == 1
        and matched[0].get("head_sha") == facts.candidate_commit
        and matched[0].get("status") == "completed"
        and matched[0].get("conclusion") == "success"
        and matched[0].get("labels") == ["ubuntu-24.04"],
        "GitHub receipt job differs",
    )
    job_started = _timestamp(matched[0].get("started_at"), "GitHub receipt job start")
    job_finished = _timestamp(matched[0].get("completed_at"), "GitHub receipt job finish")
    _require(
        started <= job_started < job_finished <= finished, "GitHub receipt job timestamps differ"
    )
    artifact_record = _record(transport, f"/actions/runs/{run_id}/artifacts?per_page=100")
    artifacts = artifact_record.get("artifacts")
    _require(type(artifacts) is list, "GitHub receipt artifacts differ")
    artifacts = cast(list[object], artifacts)
    _require(
        artifact_record.get("total_count") == len(artifacts),
        "GitHub receipt artifact list is incomplete",
    )
    selected = [item for item in artifacts if type(item) is dict and item.get("name") == _ARTIFACT]
    _require(len(selected) == 1, "GitHub receipt artifact differs")
    artifact = cast(dict[str, object], selected[0])
    link = artifact.get("workflow_run")
    artifact_id = artifact.get("id")
    artifact_digest = artifact.get("digest")
    _require(
        type(link) is dict
        and link.get("id") == run_id
        and link.get("repository_id") == _REPOSITORY_ID
        and link.get("head_repository_id") == _REPOSITORY_ID
        and link.get("head_sha") == facts.candidate_commit
        and artifact.get("expired") is False
        and type(artifact_id) is int
        and artifact_id > 0
        and type(artifact_digest) is str
        and artifact_digest.startswith("sha256:"),
        "GitHub receipt artifact binding differs",
    )
    assert isinstance(artifact_id, int) and isinstance(artifact_digest, str)
    _require(
        job_started
        <= _timestamp(artifact.get("created_at"), "GitHub receipt artifact start")
        <= _timestamp(artifact.get("updated_at"), "GitHub receipt artifact update")
        <= job_finished
        <= finished,
        "GitHub receipt artifact is outside attempt 1",
    )
    archive = transport.get(
        _API + "/repos/" + _OWNER + "/" + _REPOSITORY + f"/actions/artifacts/{artifact_id}/zip",
        {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"},
    )
    _require(
        type(archive) is bytes and _sha(archive) == artifact_digest.removeprefix("sha256:"),
        "GitHub receipt artifact digest differs",
    )
    archive_bytes = cast(bytes, archive)
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as bundle:
            _require(
                bundle.namelist() == [_MEMBER]
                and not bundle.getinfo(_MEMBER).is_dir()
                and bundle.getinfo(_MEMBER).file_size <= 128 * 1024,
                "GitHub receipt artifact members differ",
            )
            raw = bundle.read(_MEMBER)
    except zipfile.BadZipFile as error:
        raise ValueError("GitHub receipt artifact is not a ZIP") from error
    payload = payloads.parse_linux_receipt_payload(raw)
    _match_payload(payload, facts, run_id)
    receipt_metadata = LinuxReceiptMetadataV1(
        run_id,
        _ATTEMPT,
        _sha(archive_bytes),
        payload.candidate_commit,
        payload.direct_wheel_sha256,
        payload.cleanup_removed,
    )
    return _AuthenticatedObservation(facts, receipt_metadata, payload)


def authenticate_linux_receipt(
    run_id: int,
    linux_runtime: BoundDependencyPurposeV1,
    image: AdmittedLinuxRuntimeImageV1,
    *,
    github_api_bearer: str | None = None,
) -> AuthenticatedLinuxNullCaptureReceiptV1:
    """Bind a service receipt to final live inputs; this is not a stage-2 verifier."""
    facts = _facts_from_capabilities(linux_runtime, image)
    observed = _authenticate_observation(
        run_id, facts, github_api_bearer=github_api_bearer
    )
    receipt = object.__new__(AuthenticatedLinuxNullCaptureReceiptV1)
    _RECEIPTS[receipt] = _RetainedReceipt(
        linux_runtime, image, facts, observed.metadata, observed.payload
    )
    return receipt


def linux_receipt_metadata(
    receipt: AuthenticatedLinuxNullCaptureReceiptV1,
) -> LinuxReceiptMetadataV1:
    if type(receipt) is not AuthenticatedLinuxNullCaptureReceiptV1:
        raise TypeError("Linux receipt capability type differs")
    _require(receipt in _RECEIPTS, "Linux receipt capability is unregistered")
    retained = _RECEIPTS[receipt]
    # Keep the source/dependency receipt live: final input seals are re-read
    # through this binding, so closing that authority expires this capability.
    _require(
        _facts_from_capabilities(retained.linux_runtime, retained.image) == retained.facts,
        "Linux receipt source authority differs",
    )
    return retained.metadata


def _linux_receipt_payload_for_consumer(
    receipt: AuthenticatedLinuxNullCaptureReceiptV1,
) -> payloads.LinuxReceiptPayloadV1:
    """Return bounded authenticated execution facts while final seals remain live."""
    linux_receipt_metadata(receipt)
    return _RECEIPTS[receipt].payload

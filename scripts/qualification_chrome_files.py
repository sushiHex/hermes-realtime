"""Admit a complete reviewed Chrome image and bind its final version directory.

This establishes publisher bytes before any browser execution. It neither adopts
an operator's installation nor creates a browser profile or an execution owner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts.qualification_candidate_files import BoundCandidateFilesV1, candidate_file_metadata
from scripts.qualification_file_seals import sealed_file_bytes, sealed_file_metadata
from scripts.qualification_tool_distributions import _inspect_archive_members
from scripts.qualify_evidence_slice_zero import canonical_json_bytes
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _nested_chrome,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)


@dataclass(frozen=True, slots=True)
class _ChromePolicy:
    version: str
    archive_sha256: str
    archive_bytes: int


# Official Chrome for Testing win64 archive, acquired through default Windows
# HTTPS validation without redirects. Selection changes require source review;
# neither a moving channel nor a caller-supplied manifest authorizes new bytes.
# https://googlechromelabs.github.io/chrome-for-testing/153.0.8010.36.json
# https://storage.googleapis.com/chrome-for-testing-public/153.0.8010.36/win64/chrome-win64.zip
_POLICY = _ChromePolicy(
    "153.0.8010.36",
    "8edfaa0923c11a30a9315a5e7e5794c5efb60146edea7e3f749f7fdc2aa026cb",
    205134000,
)
_PREFIX = "chrome-win64/"
_EXECUTABLE = "chrome.exe"
_MAX_MEMBER = 384 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ChromeDistributionMetadataV1:
    version: str
    archive_sha256: str
    executable_sha256: str
    version_manifest_sha256: str
    file_count: int
    expanded_bytes: int


class AdmittedChromeDistributionV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Chrome distributions are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Distribution:
    files: tuple[tuple[str, bytes], ...]
    manifest: bytes
    metadata: ChromeDistributionMetadataV1


_DISTRIBUTIONS: WeakKeyDictionary[AdmittedChromeDistributionV1, _Distribution] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class ChromeFileMetadataV1:
    qualification_input_sha256: str
    candidate_commit: str
    distribution: ChromeDistributionMetadataV1


class BoundChromeFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Chrome final bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    files: RetainedQualificationInputFilesV1
    candidate: BoundCandidateFilesV1
    metadata: ChromeFileMetadataV1


_BOUND: WeakKeyDictionary[BoundChromeFilesV1, _Binding] = WeakKeyDictionary()


def admit_chrome_distribution(archive: bytes) -> AdmittedChromeDistributionV1:
    """Authenticate compressed bytes before parsing the complete publisher image."""
    _require(
        type(archive) is bytes
        and len(archive) == _POLICY.archive_bytes
        and hashlib.sha256(archive).hexdigest() == _POLICY.archive_sha256,
        "Chrome archive differs from the admitted publisher distribution",
    )
    members = _inspect_archive_members(archive, "zip", max_member_bytes=_MAX_MEMBER)
    _require(
        bool(members) and all(name.startswith(_PREFIX) and raw for name, raw in members),
        "Chrome publisher namespace differs",
    )
    files = tuple((name.removeprefix(_PREFIX), raw) for name, raw in members)
    executable = dict(files).get(_EXECUTABLE)
    _require(executable is not None, "Chrome publisher image is absent")
    assert executable is not None
    inventory = [
        {"name": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        for name, raw in files
    ]
    manifest = canonical_json_bytes(
        {"schemaVersion": 1, "chromeVersion": _POLICY.version, "files": inventory}
    )
    receipt = object.__new__(AdmittedChromeDistributionV1)
    _DISTRIBUTIONS[receipt] = _Distribution(
        files,
        manifest,
        ChromeDistributionMetadataV1(
            _POLICY.version,
            _POLICY.archive_sha256,
            hashlib.sha256(executable).hexdigest(),
            hashlib.sha256(manifest).hexdigest(),
            len(files),
            sum(len(raw) for _, raw in files),
        ),
    )
    return receipt


def _distribution(receipt: AdmittedChromeDistributionV1) -> _Distribution:
    if type(receipt) is not AdmittedChromeDistributionV1:
        raise TypeError("Chrome distribution capability type differs")
    _require(receipt in _DISTRIBUTIONS, "Chrome distribution capability is unregistered")
    return _DISTRIBUTIONS[receipt]


def chrome_distribution_metadata(
    receipt: AdmittedChromeDistributionV1,
) -> ChromeDistributionMetadataV1:
    return _distribution(receipt).metadata


def chrome_version_manifest(receipt: AdmittedChromeDistributionV1) -> bytes:
    """Return the canonical inventory derived from every admitted image member."""
    return _distribution(receipt).manifest


def _chrome_distribution_files(
    receipt: AdmittedChromeDistributionV1,
) -> tuple[tuple[str, bytes], ...]:
    return _distribution(receipt).files


def bind_chrome_files(
    distribution: AdmittedChromeDistributionV1,
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
) -> BoundChromeFilesV1:
    admitted = _distribution(distribution)
    source = candidate_file_metadata(candidate)
    selected = _retained_input_files_for_consumer(files)
    current = retained_input_metadata(files)
    _require(
        current.qualification_input_sha256 == source.qualification_input_sha256,
        "Chrome source and files require the same final input",
    )
    document = json.loads(selected.document)
    refs = {row["role"]: row for row in document["files"]}
    manifest = refs["chrome_version_directory_manifest"]["relativePath"]
    _require(
        document["expected"]["chromeVersion"] == admitted.metadata.version
        and sealed_file_bytes(selected.seals, manifest, 4 * 1024**2) == admitted.manifest
        and refs["chrome_executable"]["relativePath"] == _nested_chrome(manifest, _EXECUTABLE),
        "Chrome final version directory differs from the publisher image",
    )
    observed = {name: (digest, size) for name, digest, size in sealed_file_metadata(selected.seals)}
    _require(
        all(
            observed.get(_nested_chrome(manifest, name))
            == (hashlib.sha256(raw).hexdigest(), len(raw))
            for name, raw in admitted.files
        ),
        "Chrome final files differ from the publisher image",
    )
    receipt = object.__new__(BoundChromeFilesV1)
    _BOUND[receipt] = _Binding(
        files,
        candidate,
        ChromeFileMetadataV1(
            current.qualification_input_sha256, source.source_commit, admitted.metadata
        ),
    )
    return receipt


def chrome_file_metadata(receipt: BoundChromeFilesV1) -> ChromeFileMetadataV1:
    if type(receipt) is not BoundChromeFilesV1:
        raise TypeError("Chrome final capability type differs")
    _require(receipt in _BOUND, "Chrome final capability is unregistered")
    bound = _BOUND[receipt]
    _require(
        retained_input_metadata(bound.files).qualification_input_sha256
        == candidate_file_metadata(bound.candidate).qualification_input_sha256
        == bound.metadata.qualification_input_sha256,
        "Chrome final input seals differ",
    )
    return bound.metadata


def _chrome_files_for_consumer(receipt: BoundChromeFilesV1) -> _Binding:
    chrome_file_metadata(receipt)
    return _BOUND[receipt]

"""Retain the strict v1 input graph without claiming builds or installed execution.

This lease establishes selected immutable file bytes. The complete input authority
must additionally own source/build provenance, installed namespaces and platform
receipts. A byte lease alone cannot authorize candidate execution or qualification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast
from weakref import WeakKeyDictionary

from scripts import qualify_evidence_slice_zero as core
from scripts.qualification_file_seals import (
    RetainedFileSealsV1,
    retain_file_seals,
    sealed_file_bytes,
    sealed_file_metadata,
)

_MAX_MANIFEST_BYTES = 4 * 1024**2


class RetainedQualificationInputFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("input file leases are created by their owner only")


@dataclass(frozen=True, slots=True)
class _Inputs:
    seals: RetainedFileSealsV1
    metadata: core.QualificationInputClosure
    document: bytes


_LIVE: WeakKeyDictionary[RetainedQualificationInputFilesV1, _Inputs] = WeakKeyDictionary()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(seals: RetainedFileSealsV1, member: str) -> dict[str, Any]:
    value = core.load_strict_canonical_json(
        sealed_file_bytes(seals, member, _MAX_MANIFEST_BYTES), source="sealed input manifest"
    )
    _require(type(value) is dict, "sealed input manifest is not an object")
    return cast(dict[str, Any], value)


def _relative(value: object) -> str:
    core._safe_posix_path(value, label="sealed input member")
    assert isinstance(value, str)
    return value


def _nested(parent: str, name: object) -> str:
    return str(PurePosixPath(parent).parent / _relative(name))


def _nested_chrome(parent: str, name: object) -> str:
    return str(PurePosixPath(parent).parent / core._chrome_resource_name(name))


def _members(
    document: dict[str, Any],
    direct: RetainedFileSealsV1,
    manifest: str,
) -> tuple[str, ...]:
    files = {item["role"]: item["relativePath"] for item in document["files"]}
    members = {manifest}

    def add(name: object, *, chrome: bool = False) -> None:
        relative = core._chrome_resource_name(name) if chrome else _relative(name)
        _require(
            relative not in members or (chrome and relative == files["chrome_executable"]),
            "sealed input roles alias a path",
        )
        members.add(relative)
        _require(len(members) <= 16384, "sealed input graph exceeds its bound")

    for item in document["files"]:
        add(item["relativePath"])
    for tool in document["toolIdentities"]:
        add(tool["artifact"]["relativePath"])
    for role in core._WHEELHOUSE_ROLES.values():
        wheelhouse = _read(direct, files[role])
        for reference in (
            wheelhouse["requirements"],
            wheelhouse["constraints"],
            *wheelhouse["wheels"],
        ):
            add(reference["relativePath"])
    for provider in ("moonshine", "kokoro"):
        path = files[f"{provider}_model_manifest"]
        for reference in _read(direct, path)["resources"]:
            add(_nested(path, reference["name"]))
    path = files["chrome_version_directory_manifest"]
    for reference in _read(direct, path)["files"]:
        add(_nested_chrome(path, reference["name"]), chrome=True)
    return tuple(sorted(members))


@contextmanager
def retain_qualification_input_files(
    root: Path,
    manifest: str,
    expected_sha256: str,
) -> Iterator[RetainedQualificationInputFilesV1]:
    """Seal the manifest before discovering and retaining its transitive graph."""
    _relative(manifest)
    _require(
        type(expected_sha256) is str and core._SHA256.fullmatch(expected_sha256) is not None,
        "sealed input digest is invalid",
    )
    with retain_file_seals(root, (manifest,)) as manifest_seal:
        raw = sealed_file_bytes(manifest_seal, manifest, _MAX_MANIFEST_BYTES)
        _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "sealed input digest differs")
        document = _read(manifest_seal, manifest)
        core._validate_qualification_input_schema(document)
        direct_members = tuple(
            sorted(
                {_relative(item["relativePath"]) for item in document["files"]}
                | {
                    _relative(tool["artifact"]["relativePath"])
                    for tool in document["toolIdentities"]
                }
            )
        )
        with retain_file_seals(root, direct_members) as direct:
            members = _members(document, direct, manifest)
            with retain_file_seals(root, members) as seals:
                roles = {item["role"]: root / item["relativePath"] for item in document["files"]}
                metadata = core.verify_qualification_input_closure(
                    qualification_input_root=root,
                    qualification_input_manifest=root / manifest,
                    expected_qualification_input_sha256=expected_sha256,
                    plan=roles["governing_plan"],
                    candidate_source_archive=roles["candidate_source_archive"],
                    runner_path=roles["qualification_runner"],
                )
                receipt = object.__new__(RetainedQualificationInputFilesV1)
                _LIVE[receipt] = _Inputs(seals, metadata, raw)
                try:
                    yield receipt
                finally:
                    del _LIVE[receipt]


def retained_input_metadata(
    receipt: RetainedQualificationInputFilesV1,
) -> core.QualificationInputClosure:
    if type(receipt) is not RetainedQualificationInputFilesV1:
        raise TypeError("input file lease type differs")
    _require(receipt in _LIVE, "input file lease is closed or unregistered")
    value = _LIVE[receipt]
    sealed_file_metadata(value.seals)
    return value.metadata


def _retained_input_files_for_consumer(receipt: RetainedQualificationInputFilesV1) -> _Inputs:
    retained_input_metadata(receipt)
    return _LIVE[receipt]

"""Admit the reviewed Codex publisher package and bind its final executable.

This authenticates a complete immutable publisher package before any parsing.  It
neither extracts or runs Codex nor claims an installed environment, sandbox, or
runtime behavior.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import dataclass
from typing import Any, cast
from weakref import WeakKeyDictionary

from scripts import candidate_source_archive_oracle as archives
from scripts.qualification_candidate_files import (
    BoundCandidateFilesV1,
    _candidate_files_for_consumer,
    candidate_file_metadata,
)
from scripts.qualification_file_seals import sealed_file_bytes, sealed_file_metadata
from scripts.qualification_tool_distributions import _inspect_archive_members
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _retained_input_files_for_consumer,
    retained_input_metadata,
)

# Official Codex CLI release and publisher package.  Selection changes require
# source review; the release digest is checked before the archive is parsed.
# https://github.com/openai/codex/releases/tag/rust-v0.145.0
# https://github.com/openai/codex/releases/download/rust-v0.145.0/codex-package-x86_64-pc-windows-msvc.tar.gz
_RELEASE = "rust-v0.145.0"
_ARCHIVE_SHA256 = "8d0d281346aedf63c4cc3922997df822fbb8881f7ffb2b57416f48e8c52a734e"
_ARCHIVE_BYTES = 145216684
_VERSION = "0.145.0"
_TARGET = "x86_64-pc-windows-msvc"
_VARIANT = "codex"
_ENTRYPOINT = "bin/codex.exe"
_RESOURCES = "codex-resources"
_PATH = "codex-path"
_FIXTURE = "tests/fixtures/codex_app_server_dynamic_tools.json"
_REPORT_SCHEMA = "scripts/schemas/qualification-report-v1.schema.json"
_MAX_MEMBER = 384 * 1024**2
_MAX_FIXTURE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _Member:
    name: str
    size: int
    sha256: str


_MEMBERS = (
    _Member(
        "bin/codex-code-mode-host.exe",
        53605168,
        "de58d3bd9fb88c44555de1104d06fba78e207bce7115d92691b42f6b0f87f3b7",
    ),
    _Member(
        "bin/codex.exe",
        359245096,
        "83751f15cb6a0a7b97df67752c001e3fe1c20e18ffbfec3ff63567296205eb6c",
    ),
    _Member(
        "codex-package.json",
        215,
        "d15dd152401ec63697fb4888d3dec75a849ac85c11aa69256bccc0355e0b7ddd",
    ),
    _Member(
        "codex-path/rg.exe",
        4218880,
        "14231169855ec5205cf5a1b6f1db358ff4aed4247c86b69ce8aae647c77f6680",
    ),
    _Member(
        "codex-resources/codex-command-runner.exe",
        1271088,
        "09531442d178aefb4c849745e95a000f52d5910a13944638269d9991cb08319b",
    ),
    _Member(
        "codex-resources/codex-windows-sandbox-setup.exe",
        8807728,
        "c981b438d0959e33f90f6b8b1a9656c4f803a1b82ebdd97e2150d2b8543a0c31",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{label} repeats a key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ValueError(f"{label} is invalid") from None
    _require(type(value) is dict, f"{label} is not an object")
    return cast(dict[str, Any], value)


@dataclass(frozen=True, slots=True)
class CodexDistributionMetadataV1:
    release: str
    version: str
    target: str
    archive_sha256: str
    executable_sha256: str
    file_count: int
    expanded_bytes: int


class AdmittedCodexDistributionV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Codex distributions are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Distribution:
    files: tuple[tuple[str, bytes], ...]
    metadata: CodexDistributionMetadataV1


_DISTRIBUTIONS: WeakKeyDictionary[AdmittedCodexDistributionV1, _Distribution] = WeakKeyDictionary()


@dataclass(frozen=True, slots=True)
class CodexFileMetadataV1:
    qualification_input_sha256: str
    candidate_commit: str
    version: str
    model: str
    effort: str
    distribution: CodexDistributionMetadataV1


class BoundCodexFilesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Codex final bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    files: RetainedQualificationInputFilesV1
    candidate: BoundCandidateFilesV1
    distribution: AdmittedCodexDistributionV1
    metadata: CodexFileMetadataV1


_BOUND: WeakKeyDictionary[BoundCodexFilesV1, _Binding] = WeakKeyDictionary()


def _package_layout(raw: bytes) -> None:
    value = _strict_object(raw, label="Codex package layout")
    _require(
        set(value)
        == {
            "layoutVersion",
            "version",
            "target",
            "variant",
            "entrypoint",
            "resourcesDir",
            "pathDir",
        }
        and type(value["layoutVersion"]) is int
        and value["layoutVersion"] == 1
        and value["version"] == _VERSION
        and value["target"] == _TARGET
        and value["variant"] == _VARIANT
        and value["entrypoint"] == _ENTRYPOINT
        and value["resourcesDir"] == _RESOURCES
        and value["pathDir"] == _PATH,
        "Codex package layout differs",
    )


def admit_codex_distribution(archive: bytes) -> AdmittedCodexDistributionV1:
    """Authenticate the full reviewed publisher package before parsing it."""
    _require(
        type(archive) is bytes
        and len(archive) == _ARCHIVE_BYTES
        and hashlib.sha256(archive).hexdigest() == _ARCHIVE_SHA256,
        "Codex archive differs from the admitted publisher distribution",
    )
    try:
        files = _inspect_archive_members(archive, "tar.gz", max_member_bytes=_MAX_MEMBER)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Codex publisher archive is invalid") from None
    observed = tuple(
        _Member(name, len(raw), hashlib.sha256(raw).hexdigest()) for name, raw in files
    )
    _require(observed == _MEMBERS, "Codex publisher namespace differs")
    file_map = dict(files)
    _package_layout(file_map["codex-package.json"])
    receipt = object.__new__(AdmittedCodexDistributionV1)
    _DISTRIBUTIONS[receipt] = _Distribution(
        files,
        CodexDistributionMetadataV1(
            _RELEASE,
            _VERSION,
            _TARGET,
            _ARCHIVE_SHA256,
            _MEMBERS[1].sha256,
            len(files),
            sum(len(raw) for _, raw in files),
        ),
    )
    return receipt


def _distribution(receipt: AdmittedCodexDistributionV1) -> _Distribution:
    if type(receipt) is not AdmittedCodexDistributionV1:
        raise TypeError("Codex distribution capability type differs")
    _require(receipt in _DISTRIBUTIONS, "Codex distribution capability is unregistered")
    return _DISTRIBUTIONS[receipt]


def codex_distribution_metadata(
    receipt: AdmittedCodexDistributionV1,
) -> CodexDistributionMetadataV1:
    return _distribution(receipt).metadata


def _codex_distribution_files(
    receipt: AdmittedCodexDistributionV1,
) -> tuple[tuple[str, bytes], ...]:
    """Private complete package inventory retained for later materialization."""
    return _distribution(receipt).files


def _source_member(candidate: BoundCandidateFilesV1, path: str) -> bytes:
    binding = _candidate_files_for_consumer(candidate)
    metadata = archives.verified_candidate_source_archive_metadata(binding.archive)
    member = next((item for item in metadata.manifest if item.path == path), None)
    _require(
        member is not None
        and member.kind == "file"
        and member.size <= _MAX_FIXTURE
        and member.sha256 is not None,
        "Codex candidate fixture is unavailable",
    )
    assert member is not None and member.sha256 is not None
    payload = archives._archive_bytes_for_consumer(binding.archive, binding.identity)
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            stream = archive.extractfile(metadata.prefix + "/" + path)
            _require(stream is not None, "Codex candidate fixture is unavailable")
            assert stream is not None
            with stream:
                raw = stream.read(member.size + 1)
    except (OSError, tarfile.TarError):
        raise ValueError("Codex candidate fixture is unavailable") from None
    _require(
        len(raw) == member.size and hashlib.sha256(raw).hexdigest() == member.sha256,
        "Codex candidate fixture differs",
    )
    return raw


def _source_fixture(candidate: BoundCandidateFilesV1) -> tuple[str, str]:
    fixture = _strict_object(_source_member(candidate, _FIXTURE), label="Codex candidate fixture")
    _require(
        type(fixture.get("codexVersion")) is str and type(fixture.get("codexBinarySha256")) is str,
        "Codex candidate fixture differs",
    )
    return cast(str, fixture["codexVersion"]), cast(str, fixture["codexBinarySha256"])


def _source_provider_constants(candidate: BoundCandidateFilesV1) -> tuple[str, str]:
    schema = _strict_object(_source_member(candidate, _REPORT_SCHEMA), label="Codex report schema")
    definitions = schema.get("$defs")
    _require(type(definitions) is dict, "Codex report schema differs")
    provider = cast(dict[str, Any], definitions).get("ProviderV1")
    _require(type(provider) is dict, "Codex report schema differs")
    properties = cast(dict[str, Any], provider).get("properties")
    _require(type(properties) is dict, "Codex report schema differs")
    properties = cast(dict[str, Any], properties)
    model = properties.get("codexModel")
    effort = properties.get("codexEffort")
    _require(type(model) is dict and type(effort) is dict, "Codex report schema differs")
    model = cast(dict[str, Any], model)
    effort = cast(dict[str, Any], effort)
    _require(
        model == {"const": "gpt-5.6-terra", "type": "string"}
        and effort == {"const": "low", "type": "string"},
        "Codex report schema differs",
    )
    return cast(str, model["const"]), cast(str, effort["const"])


def _expected(document: bytes, candidate: BoundCandidateFilesV1) -> tuple[str, str, str]:
    value = _strict_object(document, label="Codex final input")
    expected = value.get("expected")
    _require(type(expected) is dict, "Codex final expected values differ")
    expected = cast(dict[str, Any], expected)
    version = expected.get("codexVersion")
    model = expected.get("codexModel")
    effort = expected.get("codexEffort")
    _require(
        type(version) is str
        and type(model) is str
        and type(effort) is str
        and version == _VERSION
        and (model, effort) == _source_provider_constants(candidate),
        "Codex final expected values differ",
    )
    return cast(str, version), cast(str, model), cast(str, effort)


def _executable_reference(document: bytes) -> str:
    value = _strict_object(document, label="Codex final input")
    listed = value.get("files")
    _require(type(listed) is list, "Codex final executable is unavailable")
    listed = cast(list[Any], listed)
    references = [
        row for row in listed if type(row) is dict and row.get("role") == "codex_executable"
    ]
    _require(
        len(references) == 1 and type(references[0].get("relativePath")) is str,
        "Codex final executable is unavailable",
    )
    return cast(str, references[0]["relativePath"])


def bind_codex_files(
    distribution: AdmittedCodexDistributionV1,
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
) -> BoundCodexFilesV1:
    """Bind the reviewed executable and source-selected fixture to final inputs."""
    admitted = _distribution(distribution)
    candidate_binding = _candidate_files_for_consumer(candidate)
    source = candidate_file_metadata(candidate)
    selected = _retained_input_files_for_consumer(files)
    current = retained_input_metadata(files)
    _require(
        candidate_binding.files is files
        and current.qualification_input_sha256 == source.qualification_input_sha256,
        "Codex source and files require the same final input",
    )
    version, model, effort = _expected(selected.document, candidate)
    fixture_version, fixture_sha256 = _source_fixture(candidate)
    executable = next(raw for name, raw in admitted.files if name == _ENTRYPOINT)
    reference = _executable_reference(selected.document)
    _require(
        fixture_version == "codex-cli " + version
        and fixture_sha256 == admitted.metadata.executable_sha256
        and sealed_file_bytes(selected.seals, reference, _MAX_MEMBER) == executable,
        "Codex final executable differs from the publisher package",
    )
    receipt = object.__new__(BoundCodexFilesV1)
    _BOUND[receipt] = _Binding(
        files,
        candidate,
        distribution,
        CodexFileMetadataV1(
            current.qualification_input_sha256,
            source.source_commit,
            version,
            model,
            effort,
            admitted.metadata,
        ),
    )
    return receipt


def codex_file_metadata(receipt: BoundCodexFilesV1) -> CodexFileMetadataV1:
    if type(receipt) is not BoundCodexFilesV1:
        raise TypeError("Codex final capability type differs")
    _require(receipt in _BOUND, "Codex final capability is unregistered")
    value = _BOUND[receipt]
    admitted = _distribution(value.distribution)
    candidate_binding = _candidate_files_for_consumer(value.candidate)
    source = candidate_file_metadata(value.candidate)
    selected = _retained_input_files_for_consumer(value.files)
    current = retained_input_metadata(value.files)
    version, model, effort = _expected(selected.document, value.candidate)
    fixture_version, fixture_sha256 = _source_fixture(value.candidate)
    observed = dict(
        (name, (digest, size)) for name, digest, size in sealed_file_metadata(selected.seals)
    )
    _require(
        candidate_binding.files is value.files
        and current.qualification_input_sha256
        == source.qualification_input_sha256
        == value.metadata.qualification_input_sha256
        and (version, model, effort)
        == (value.metadata.version, value.metadata.model, value.metadata.effort)
        and fixture_version == "codex-cli " + version
        and fixture_sha256 == admitted.metadata.executable_sha256
        and observed.get(_executable_reference(selected.document))
        == (admitted.metadata.executable_sha256, _MEMBERS[1].size),
        "Codex final seals differ",
    )
    return value.metadata


def _codex_files_for_consumer(receipt: BoundCodexFilesV1) -> _Binding:
    codex_file_metadata(receipt)
    return _BOUND[receipt]

"""Bind Kokoro model bytes to the candidate's existing resource selections.

This pre-load authority does not import a provider, construct a native model,
prove installed execution, or admit the remaining full-host prerequisites.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from scripts import candidate_wheel as wheels
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
from scripts.qualification_file_seals import sealed_file_bytes
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _nested,
    _read,
    _retained_input_files_for_consumer,
)

_SOURCE = "hermes_realtime/providers/kokoro.py"
_PURPOSES = ("realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime")
_MAX_RESOURCE = 384 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _Asset:
    name: str
    size: int
    sha256: str


def _source_assets(raw: bytes) -> tuple[_Asset, ...]:
    """Read the reviewed literal profile without importing candidate code."""
    _require(type(raw) is bytes and 0 < len(raw) <= 512 * 1024, "asset source is unavailable")
    try:
        module = ast.parse(raw)
    except (SyntaxError, ValueError) as error:
        raise ValueError("asset source cannot be parsed") from error
    assets: list[_Asset] = []
    for name in ("_MODEL_ASSET", "_VOICES_ASSET"):
        writes = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ]
        definitions = [
            node
            for node in module.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
        ]
        _require(len(writes) == len(definitions) == 1, "asset selection is absent or ambiguous")
        call = definitions[0].value
        _require(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_PinnedAsset"
            and not call.args
            and len(call.keywords) == 3,
            "asset selection is not the supported literal profile",
        )
        assert isinstance(call, ast.Call)
        _require(
            {item.arg for item in call.keywords} == {"filename", "size", "sha256"},
            "asset selection fields differ",
        )
        try:
            values = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
        except (ValueError, TypeError, SyntaxError) as error:
            raise ValueError("asset selection is not literal") from error
        filename, size, digest = values["filename"], values["size"], values["sha256"]
        _require(
            type(filename) is str
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", filename) is not None
            and type(size) is int
            and 0 < size <= _MAX_RESOURCE
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
            "asset selection identity or bound differs",
        )
        assets.append(_Asset(filename, size, digest))
    _require(len({row.name.casefold() for row in assets}) == 2, "asset names are ambiguous")
    return tuple(sorted(assets, key=lambda row: row.name))


@dataclass(frozen=True, slots=True)
class KokoroResourceMetadataV1:
    qualification_input_sha256: str
    source_commit: str
    purpose: str
    provider_source_sha256: str
    distribution_sha256: str
    model_identity_sha256: str
    resource_count: int
    resource_bytes: int


class BoundKokoroResourcesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Kokoro resource bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    files: RetainedQualificationInputFilesV1
    candidate: BoundCandidateFilesV1
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1
    manifest: str
    assets: tuple[_Asset, ...]
    metadata: KokoroResourceMetadataV1


_BINDINGS: WeakKeyDictionary[BoundKokoroResourcesV1, _Binding] = WeakKeyDictionary()


def bind_kokoro_resources(
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1,
    *,
    purpose: str,
) -> BoundKokoroResourcesV1:
    """Authenticate source and installed-purpose inputs before reading model data."""
    _require(type(purpose) is str and purpose in _PURPOSES, "Kokoro resource purpose differs")
    bound = _candidate_files_for_consumer(candidate)
    closure = _dependency_binding_for_consumer(dependencies)
    _require(
        bound.files is files and closure.files is files and closure.candidate is candidate,
        "Kokoro resources require the same candidate and input authorities",
    )
    covered = [row for row in closure.metadata.wheelhouses if row.purpose == purpose]
    _require(len(covered) == 1, "Kokoro resource purpose is outside dependency coverage")
    provider = [row for row in covered[0].distributions if row.name == "kokoro-onnx"]
    _require(len(provider) == 1 and provider[0].version == "0.6.1", "Kokoro distribution differs")
    # The wheel owner already compared every runtime blob with the genuine archive.
    source = wheels._wheel_for_consumer(bound.wheels[0], bound.archive, bound.identity).members
    raw = source.get(_SOURCE, b"")
    assets = _source_assets(raw)
    selected = _retained_input_files_for_consumer(files)
    document = json.loads(selected.document)
    reference = next(row for row in document["files"] if row["role"] == "kokoro_model_manifest")
    manifest = _read(selected.seals, reference["relativePath"])
    expected = [{"name": row.name, "bytes": row.size, "sha256": row.sha256} for row in assets]
    _require(manifest["resources"] == expected, "Kokoro resources differ from candidate assets")
    for asset in assets:
        payload = sealed_file_bytes(
            selected.seals, _nested(reference["relativePath"], asset.name), _MAX_RESOURCE
        )
        _require(
            (len(payload), hashlib.sha256(payload).hexdigest()) == (asset.size, asset.sha256),
            "Kokoro resource bytes differ from candidate assets",
        )
    candidate_metadata = candidate_file_metadata(candidate)
    _dependency_binding_for_consumer(dependencies)
    metadata = KokoroResourceMetadataV1(
        candidate_metadata.qualification_input_sha256,
        candidate_metadata.source_commit,
        purpose,
        hashlib.sha256(raw).hexdigest(),
        provider[0].sha256,
        manifest["modelIdentitySha256"],
        len(assets),
        sum(row.size for row in assets),
    )
    receipt = object.__new__(BoundKokoroResourcesV1)
    _BINDINGS[receipt] = _Binding(
        files, candidate, dependencies, reference["relativePath"], assets, metadata
    )
    return receipt


def _binding(receipt: BoundKokoroResourcesV1) -> _Binding:
    if type(receipt) is not BoundKokoroResourcesV1:
        raise TypeError("Kokoro resource binding type differs")
    _require(receipt in _BINDINGS, "Kokoro resource binding is unregistered")
    value = _BINDINGS[receipt]
    candidate_file_metadata(value.candidate)
    _dependency_binding_for_consumer(value.dependencies)
    return value


def kokoro_resource_metadata(receipt: BoundKokoroResourcesV1) -> KokoroResourceMetadataV1:
    return _binding(receipt).metadata


def _kokoro_resources_for_consumer(
    receipt: BoundKokoroResourcesV1,
) -> tuple[tuple[str, bytes], ...]:
    value = _binding(receipt)
    selected = _retained_input_files_for_consumer(value.files)
    return tuple(
        (
            row.name,
            sealed_file_bytes(selected.seals, _nested(value.manifest, row.name), _MAX_RESOURCE),
        )
        for row in value.assets
    )

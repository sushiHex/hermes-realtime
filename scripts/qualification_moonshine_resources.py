"""Bind acquired Moonshine resources to the final sealed input closure.

This authority transfers authenticated publisher bytes into the final input
namespace.  It does not import Moonshine, construct a model, or prove runtime
execution.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from weakref import WeakKeyDictionary

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
from scripts.qualification_moonshine_catalog import (
    BoundMoonshineCatalogV1,
    _moonshine_catalog_for_consumer,
    moonshine_catalog_metadata,
)
from scripts.qualification_moonshine_catalog import (
    _binding as _catalog_binding,
)
from scripts.qualification_moonshine_preparation import (
    PreparedMoonshineResourcesV1,
    PreparedMoonshineResourceV1,
    _moonshine_preparation_for_consumer,
)
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualify_evidence_slice_zero import canonical_json_bytes
from scripts.retained_qualification_inputs import (
    RetainedQualificationInputFilesV1,
    _nested,
    _read,
    _retained_input_files_for_consumer,
)

_PURPOSES = frozenset({"realtime_windows_direct_runtime", "realtime_windows_sdist_built_runtime"})
_MAX_RESOURCE_BYTES = 512 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class MoonshineResourceMetadataV1:
    qualification_input_sha256: str
    source_commit: str
    source_tree: str
    source_archive_sha256: str
    purpose: str
    worker_sha256: str
    distribution_sha256: str
    python_api_sha256: str
    native_library_sha256: str
    catalog_identity_sha256: str
    model_identity_sha256: str
    resource_count: int
    resource_bytes: int


class BoundMoonshineResourcesV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Moonshine resource bindings are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _Binding:
    files: RetainedQualificationInputFilesV1
    candidate: BoundCandidateFilesV1
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1
    catalog: BoundMoonshineCatalogV1
    preparation_owner: OwnedQualificationWorkV1
    manifest: str
    resources: tuple[PreparedMoonshineResourceV1, ...]
    metadata: MoonshineResourceMetadataV1


_BINDINGS: WeakKeyDictionary[BoundMoonshineResourcesV1, _Binding] = WeakKeyDictionary()


def _preparation_cleanup_complete(owner: OwnedQualificationWorkV1) -> bool:
    return owner._closing and owner._closed and owner._unrecoverable is None


def bind_moonshine_resources(
    files: RetainedQualificationInputFilesV1,
    candidate: BoundCandidateFilesV1,
    dependencies: BoundDependencyFilesV1 | BoundDependencyPurposeV1,
    prepared: PreparedMoonshineResourcesV1,
    catalog: BoundMoonshineCatalogV1,
    *,
    purpose: str,
) -> BoundMoonshineResourcesV1:
    """Transfer the fixed medium-English publisher closure into final inputs."""
    _require(type(purpose) is str and purpose in _PURPOSES, "Moonshine resource purpose differs")
    candidate_binding = _candidate_files_for_consumer(candidate)
    dependency_binding = _dependency_binding_for_consumer(dependencies)
    catalog_binding = _catalog_binding(catalog)
    preparation = _moonshine_preparation_for_consumer(prepared)
    candidate_metadata = candidate_file_metadata(candidate)
    catalog_metadata = moonshine_catalog_metadata(catalog)
    preparation_metadata = preparation.metadata
    final = _retained_input_files_for_consumer(files)

    _require(
        candidate_binding.files is files
        and dependency_binding.files is files
        and dependency_binding.candidate is candidate
        and catalog_binding.candidate is candidate
        and catalog_binding.dependencies is dependencies,
        "Moonshine resources require the same candidate and input authorities",
    )
    _require(
        candidate_metadata.qualification_input_sha256
        == dependency_binding.metadata.qualification_input_sha256
        == catalog_metadata.qualification_input_sha256
        == final.metadata.qualification_input_sha256,
        "Moonshine resource final input bindings differ",
    )
    _require(
        catalog_metadata.purpose == purpose,
        "Moonshine resource catalog purpose differs",
    )
    covered = [row for row in dependency_binding.metadata.wheelhouses if row.purpose == purpose]
    _require(len(covered) == 1, "Moonshine resource purpose is outside dependency coverage")
    providers = [row for row in covered[0].distributions if row.name == "moonshine-voice"]
    _require(
        len(providers) == 1 and providers[0].version == "0.1.0",
        "Moonshine resource distribution differs",
    )
    provider = providers[0]
    _require(
        (
            preparation_metadata.source_commit,
            preparation_metadata.source_tree,
            preparation_metadata.source_archive_sha256,
        )
        == (
            candidate_metadata.source_commit,
            candidate_metadata.source_tree,
            candidate_metadata.source_archive_sha256,
        )
        and (
            catalog_metadata.source_commit,
            catalog_metadata.source_tree,
        )
        == (candidate_metadata.source_commit, candidate_metadata.source_tree),
        "Moonshine resource source authority differs",
    )
    _require(
        provider.sha256
        == preparation_metadata.distribution_sha256
        == catalog_metadata.distribution_sha256,
        "Moonshine resource distribution authority differs",
    )
    _require(
        (
            preparation_metadata.worker_sha256,
            preparation_metadata.python_api_sha256,
            preparation_metadata.native_library_sha256,
            preparation_metadata.catalog_identity_sha256,
        )
        == (
            catalog_metadata.worker_sha256,
            catalog_metadata.python_api_sha256,
            catalog_metadata.native_library_sha256,
            catalog_metadata.catalog_identity_sha256,
        ),
        "Moonshine resource catalog authority differs",
    )

    catalog_resources = _moonshine_catalog_for_consumer(catalog, include_spelling=True)
    resources = preparation.resources
    _require(
        len(catalog_resources) == len(resources) == 9
        and catalog_metadata.primary_resource_count == 7
        and catalog_metadata.spelling_resource_count == 2
        and preparation_metadata.resource_count == 9
        and preparation_metadata.resource_bytes
        == catalog_metadata.resource_bytes
        == sum(row.size for row in resources),
        "Moonshine resource closure count or size differs",
    )
    _require(
        tuple((row.group, row.cache_path, row.url, row.size, row.crc32c) for row in resources)
        == tuple(
            (row.group, row.cache_path, row.url, row.size, row.crc32c) for row in catalog_resources
        ),
        "Moonshine prepared resources differ from the native catalog",
    )

    document = json.loads(final.document)
    references = [row for row in document["files"] if row["role"] == "moonshine_model_manifest"]
    _require(len(references) == 1, "Moonshine model manifest is unavailable")
    reference = references[0]
    manifest = _read(final.seals, reference["relativePath"])
    expected_resources = [
        {"name": row.cache_path, "bytes": row.size, "sha256": row.sha256}
        for row in sorted(resources, key=lambda item: item.cache_path)
    ]
    identity = hashlib.sha256(
        canonical_json_bytes(
            {row["name"]: row["sha256"] for row in expected_resources},
            terminal_lf=False,
        )
    ).hexdigest()
    _require(
        set(manifest) == {"schemaVersion", "provider", "modelIdentitySha256", "resources"}
        and manifest["schemaVersion"] == 1
        and manifest["provider"] == "moonshine"
        and manifest["modelIdentitySha256"] == identity
        and manifest["resources"] == expected_resources,
        "Moonshine final resource manifest differs from prepared publisher bytes",
    )
    for resource in resources:
        original = sealed_file_bytes(
            preparation.resource_seals, resource.cache_path, _MAX_RESOURCE_BYTES
        )
        final_bytes = sealed_file_bytes(
            final.seals,
            _nested(reference["relativePath"], resource.cache_path),
            _MAX_RESOURCE_BYTES,
        )
        _require(
            len(original) == resource.size
            and hashlib.sha256(original).hexdigest() == resource.sha256
            and final_bytes == original,
            "Moonshine final resource bytes differ from prepared publisher bytes",
        )
    _require(
        _moonshine_preparation_for_consumer(prepared) is preparation,
        "Moonshine preparation changed during resource transfer",
    )
    moonshine_catalog_metadata(catalog)
    candidate_file_metadata(candidate)
    _dependency_binding_for_consumer(dependencies)

    metadata = MoonshineResourceMetadataV1(
        candidate_metadata.qualification_input_sha256,
        candidate_metadata.source_commit,
        candidate_metadata.source_tree,
        candidate_metadata.source_archive_sha256,
        purpose,
        preparation_metadata.worker_sha256,
        provider.sha256,
        preparation_metadata.python_api_sha256,
        preparation_metadata.native_library_sha256,
        preparation_metadata.catalog_identity_sha256,
        identity,
        len(resources),
        sum(row.size for row in resources),
    )
    receipt = object.__new__(BoundMoonshineResourcesV1)
    _BINDINGS[receipt] = _Binding(
        files,
        candidate,
        dependencies,
        catalog,
        preparation.resource_work,
        reference["relativePath"],
        resources,
        metadata,
    )
    return receipt


def _binding(receipt: BoundMoonshineResourcesV1) -> _Binding:
    if type(receipt) is not BoundMoonshineResourcesV1:
        raise TypeError("Moonshine resource binding type differs")
    _require(receipt in _BINDINGS, "Moonshine resource binding is unregistered")
    value = _BINDINGS[receipt]
    _require(
        _preparation_cleanup_complete(value.preparation_owner),
        "Moonshine preparation resource cleanup is incomplete",
    )
    candidate = candidate_file_metadata(value.candidate)
    dependency = _dependency_binding_for_consumer(value.dependencies)
    catalog = moonshine_catalog_metadata(value.catalog)
    _retained_input_files_for_consumer(value.files)
    _require(
        dependency.files is value.files
        and dependency.candidate is value.candidate
        and candidate.qualification_input_sha256
        == dependency.metadata.qualification_input_sha256
        == catalog.qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "Moonshine resource retained input bindings differ",
    )
    return value


def moonshine_resource_metadata(
    receipt: BoundMoonshineResourcesV1,
) -> MoonshineResourceMetadataV1:
    return _binding(receipt).metadata


def _moonshine_resources_for_consumer(
    receipt: BoundMoonshineResourcesV1,
) -> tuple[tuple[str, bytes], ...]:
    value = _binding(receipt)
    final = _retained_input_files_for_consumer(value.files)
    resources = tuple(
        (
            row.cache_path,
            sealed_file_bytes(
                final.seals,
                _nested(value.manifest, row.cache_path),
                _MAX_RESOURCE_BYTES,
            ),
        )
        for row in sorted(value.resources, key=lambda item: item.cache_path)
    )
    _binding(receipt)
    return resources

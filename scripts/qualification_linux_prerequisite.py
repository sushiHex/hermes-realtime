"""Bind a Linux receipt before the final qualification document exists.

The preparation boundary accepts only source-locked generic ``linux_x86_64`` or
pure wheels.  Manylinux compatibility still needs an actual Linux ABI observer;
this module does not widen the preliminary wheel inspector's platform authority.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, cast
from weakref import WeakKeyDictionary

from packaging.markers import Environment
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import parse_wheel_filename

from scripts import github_actions_linux_receipt as service
from scripts import qualify_evidence_slice_zero as core
from scripts.qualification_build_environment import _ARTIFACT_NAMES
from scripts.qualification_build_inputs import (
    BoundBuildInputsV1,
    _build_inputs_for_consumer,
    build_input_metadata,
)
from scripts.qualification_builds import (
    CandidateBuildsV1,
    _candidate_build_bytes,
    candidate_build_metadata,
)
from scripts.qualification_dependency_files import (
    BoundDependencyPurposeV1,
    _archive_locked_wheels,
    _dependency_binding_for_consumer,
    _include_candidate_wheel_bytes,
)
from scripts.qualification_execution_files import (
    ImmutableExecutionFilesV1,
    _execution_files_for_consumer,
    execution_file_metadata,
    owned_execution_files,
)
from scripts.qualification_linux_image import (
    AdmittedLinuxRuntimeImageV1,
    LinuxRuntimeImageMetadataV1,
    linux_image_metadata,
)
from scripts.qualification_linux_receipt import LinuxReceiptPayloadV1
from scripts.qualification_owned_work import OwnedQualificationWorkV1
from scripts.qualification_wheelhouse import (
    AuthenticatedLinuxWheelTargetV1,
    _authenticated_linux_binding_for_consumer,
    _inspect_wheelhouse_for_target,
    _LinuxWheelRecipeBinding,
    _mint_authenticated_linux_target,
    _target,
)
from scripts.retained_qualification_inputs import RetainedQualificationInputFilesV1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class LinuxPrerequisiteMetadataV1:
    source_commit: str
    source_tree: str
    direct_wheel_sha256: str
    wheelhouse_manifest_sha256: str
    source_lock_sha256: str
    python_version: str
    wheel_count: int


class PreparedLinuxReceiptInputsV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("Linux receipt inputs are preparation-minted only")


@dataclass(frozen=True, slots=True)
class _Prepared:
    inputs: BoundBuildInputsV1
    builds: CandidateBuildsV1
    image: AdmittedLinuxRuntimeImageV1
    image_metadata: LinuxRuntimeImageMetadataV1
    files: ImmutableExecutionFilesV1
    contents: tuple[tuple[str, bytes], ...]
    direct_wheel: bytes
    wheels: tuple[tuple[str, bytes], ...]
    requirements: bytes
    constraints: bytes
    expected: service._LinuxReceiptFacts
    metadata: LinuxPrerequisiteMetadataV1


_PREPARED: WeakKeyDictionary[PreparedLinuxReceiptInputsV1, _Prepared] = WeakKeyDictionary()


def _manifest_document(raw: bytes) -> dict[str, Any]:
    value = core.load_strict_canonical_json(raw, source="prefinal Linux wheelhouse")
    _require(type(value) is dict, "prefinal Linux wheelhouse differs")
    return cast(dict[str, Any], value)


def _owned_recipe(
    work: OwnedQualificationWorkV1,
    manifest_path: str,
    manifest: bytes,
    closure: dict[str, bytes],
) -> tuple[ImmutableExecutionFilesV1, dict[str, bytes], dict[str, Any]]:
    _require(
        type(manifest_path) is str
        and type(manifest) is bytes
        and type(closure) is dict
        and manifest_path not in closure,
        "prefinal Linux wheelhouse input types differ",
    )
    contents = {manifest_path: manifest, **closure}
    snapshot = work.enter(owned_execution_files(contents))
    root = _execution_files_for_consumer(snapshot)
    artifacts: dict[str, core.VerifiedArtifact] = {}
    core._verify_wheelhouse(
        root,
        root / manifest_path,
        purpose="realtime_linux_runtime",
        artifacts=artifacts,
        seen_transitive_paths=set(),
    )
    document = _manifest_document(manifest)
    references = [document["requirements"], document["constraints"], *document["wheels"]]
    expected_paths = {manifest_path, *(item["relativePath"] for item in references)}
    _require(set(contents) == expected_paths, "prefinal Linux wheelhouse closure differs")
    return snapshot, contents, document


def _expected_facts(
    inputs: BoundBuildInputsV1,
    builds: CandidateBuildsV1,
    image: AdmittedLinuxRuntimeImageV1,
    manifest: bytes,
    contents: dict[str, bytes],
    document: dict[str, Any],
) -> tuple[service._LinuxReceiptFacts, bytes, str, dict[str, bytes], bytes, bytes]:
    bound = _build_inputs_for_consumer(inputs)
    input_metadata = build_input_metadata(inputs)
    build_metadata = candidate_build_metadata(builds)
    _require(build_metadata.inputs == input_metadata, "Linux prerequisite build inputs differ")
    outputs = _candidate_build_bytes(builds)
    direct = outputs["direct_wheel"]
    direct_name = _ARTIFACT_NAMES["wheel"]
    assert direct_name is not None
    direct_digest = _sha(direct)
    lock_digest, locked = _archive_locked_wheels(bound.archive, bound.identity)
    image_metadata = linux_image_metadata(image)
    _require(
        document["platform"] == "linux_x86_64"
        and document["pythonVersion"] == image_metadata.python_version,
        "prefinal Linux wheelhouse target differs",
    )
    wheels: dict[str, bytes] = {}
    for reference in document["wheels"]:
        basename = reference["basename"]
        raw = contents[reference["relativePath"]]
        name, version, _, _ = parse_wheel_filename(basename)
        if name == "hermes-realtime":
            _require(
                basename == direct_name and raw == direct,
                "prefinal Linux candidate wheel differs from its build",
            )
        else:
            _require(
                (str(name), str(version), basename, _sha(raw), len(raw)) in locked,
                "prefinal Linux wheel differs from the candidate source lock",
            )
        wheels[basename] = raw
    requirements = contents[document["requirements"]["relativePath"]]
    constraints = contents[document["constraints"]["relativePath"]]
    requirements = _include_candidate_wheel_bytes(
        wheels, requirements, direct_name, direct, direct_digest
    )
    source = input_metadata
    expected = service._LinuxReceiptFacts(
        source.source_commit,
        source.source_tree,
        source.source_archive_sha256,
        service._workflow_from_source(bound.archive, bound.identity),
        direct_digest,
        len(direct),
        direct_name,
        _sha(manifest),
        lock_digest,
        _sha(requirements),
        _sha(constraints),
        tuple(sorted((name, _sha(raw), len(raw)) for name, raw in wheels.items())),
        image_metadata.image_reference,
        image_metadata.config_sha256,
        image_metadata.layer_sha256s,
        image_metadata.layer_diff_sha256s,
        image_metadata.python_version,
    )
    return expected, direct, lock_digest, wheels, requirements, constraints


def prepare_linux_receipt_inputs(
    work: OwnedQualificationWorkV1,
    inputs: BoundBuildInputsV1,
    builds: CandidateBuildsV1,
    image: AdmittedLinuxRuntimeImageV1,
    *,
    manifest_path: str,
    manifest: bytes,
    closure: dict[str, bytes],
) -> PreparedLinuxReceiptInputsV1:
    """Own a prefinal Linux recipe bound to genuine source and build outputs."""
    if type(work) is not OwnedQualificationWorkV1:
        raise TypeError("Linux prerequisite work owner type differs")
    work._accepting()
    try:
        snapshot, contents, document = _owned_recipe(work, manifest_path, manifest, closure)
        expected, direct, lock_digest, wheels, requirements, constraints = _expected_facts(
            inputs, builds, image, manifest, contents, document
        )
        image_metadata = linux_image_metadata(image)
        receipt = object.__new__(PreparedLinuxReceiptInputsV1)
        _PREPARED[receipt] = _Prepared(
            inputs,
            builds,
            image,
            image_metadata,
            snapshot,
            tuple(sorted(contents.items())),
            direct,
            tuple(sorted(wheels.items())),
            requirements,
            constraints,
            expected,
            LinuxPrerequisiteMetadataV1(
                expected.candidate_commit,
                expected.candidate_tree,
                expected.direct_wheel_sha256,
                expected.wheelhouse_manifest_sha256,
                lock_digest,
                image_metadata.python_version,
                len(expected.wheels),
            ),
        )
        return receipt
    except BaseException as error:
        work._retain_failure(error)
        raise


def _prepared(receipt: PreparedLinuxReceiptInputsV1) -> _Prepared:
    if type(receipt) is not PreparedLinuxReceiptInputsV1:
        raise TypeError("prepared Linux receipt capability type differs")
    _require(receipt in _PREPARED, "prepared Linux receipt capability is unregistered")
    value = _PREPARED[receipt]
    _require(
        candidate_build_metadata(value.builds).inputs == build_input_metadata(value.inputs),
        "prepared Linux build authority differs",
    )
    _require(
        _candidate_build_bytes(value.builds)["direct_wheel"] == value.direct_wheel,
        "prepared Linux direct wheel seal differs",
    )
    _require(linux_image_metadata(value.image) == value.image_metadata, "Linux image differs")
    _require(
        execution_file_metadata(value.files)
        == tuple((name, _sha(raw), len(raw)) for name, raw in value.contents),
        "prepared Linux wheelhouse seals differ",
    )
    return value


def linux_prerequisite_metadata(
    receipt: PreparedLinuxReceiptInputsV1,
) -> LinuxPrerequisiteMetadataV1:
    return _prepared(receipt).metadata


class AuthenticatedPrefinalLinuxReceiptV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("prefinal Linux receipts are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _PrefinalReceipt:
    prepared: PreparedLinuxReceiptInputsV1
    observation: service._AuthenticatedObservation
    target: AuthenticatedLinuxWheelTargetV1


_PREFINAL: WeakKeyDictionary[AuthenticatedPrefinalLinuxReceiptV1, _PrefinalReceipt] = (
    WeakKeyDictionary()
)
_TRANSFERRED: WeakKeyDictionary[AuthenticatedPrefinalLinuxReceiptV1, str] = WeakKeyDictionary()


def _authenticated_linux_target(
    payload: LinuxReceiptPayloadV1,
) -> tuple[Environment, set[Tag]]:
    """Derive compatible tags only from service-authenticated native facts."""
    _require(
        payload.soabi == "cpython-311-x86_64-linux-gnu"
        and payload.ext_suffix == ".cpython-311-x86_64-linux-gnu.so"
        and payload.multiarch == "x86_64-linux-gnu",
        "authenticated Linux ABI differs",
    )
    match = re.fullmatch(r"2\.([0-9]{1,2})", payload.glibc)
    _require(match is not None and 5 <= int(match.group(1)) <= 99, "authenticated glibc differs")
    assert match is not None
    minor = int(match.group(1))
    platforms = [f"manylinux_2_{value}_x86_64" for value in range(minor, 4, -1)]
    if minor >= 17:
        platforms.append("manylinux2014_x86_64")
    if minor >= 12:
        platforms.append("manylinux2010_x86_64")
    if minor >= 5:
        platforms.append("manylinux1_x86_64")
    platforms.append("linux_x86_64")
    tags = set(cpython_tags((3, 11), abis=["cp311"], platforms=platforms))
    tags.update(compatible_tags((3, 11), interpreter="cp311", platforms=platforms))
    environment, _ = _target(payload.python_version, "linux_x86_64")
    return environment, tags


def _verify_authenticated_recipe(
    prepared: _Prepared, payload: LinuxReceiptPayloadV1
) -> AuthenticatedLinuxWheelTargetV1:
    environment, tags = _authenticated_linux_target(payload)
    _inspect_wheelhouse_for_target(
        requirements=prepared.requirements,
        constraints=prepared.constraints,
        wheels=dict(prepared.wheels),
        environment=environment,
        tags=tags,
        roots=("hermes-realtime",),
        site_processing=False,
    )
    expected = prepared.expected
    return _mint_authenticated_linux_target(
        environment,
        tags,
        _LinuxWheelRecipeBinding(
            expected.candidate_commit,
            expected.candidate_tree,
            expected.source_archive_sha256,
            (
                expected.direct_wheel_basename,
                expected.direct_wheel_sha256,
                expected.direct_wheel_bytes,
            ),
            expected.wheelhouse_manifest_sha256,
            expected.source_lock_sha256,
            expected.requirements_sha256,
            expected.constraints_sha256,
            expected.wheels,
        ),
        (
            expected.image_reference,
            expected.image_config_sha256,
            expected.image_layers,
            expected.image_diff_ids,
        ),
    )


def authenticate_prefinal_linux_receipt(
    run_id: int,
    prepared: PreparedLinuxReceiptInputsV1,
    *,
    github_api_bearer: str | None = None,
) -> AuthenticatedPrefinalLinuxReceiptV1:
    value = _prepared(prepared)
    observation = service._authenticate_observation(
        run_id, value.expected, github_api_bearer=github_api_bearer
    )
    target = _verify_authenticated_recipe(value, observation.payload)
    _require(_prepared(prepared).expected == observation.facts, "prefinal Linux inputs changed")
    receipt = object.__new__(AuthenticatedPrefinalLinuxReceiptV1)
    _PREFINAL[receipt] = _PrefinalReceipt(prepared, observation, target)
    return receipt


def prefinal_linux_receipt_metadata(
    receipt: AuthenticatedPrefinalLinuxReceiptV1,
) -> service.LinuxReceiptMetadataV1:
    if type(receipt) is not AuthenticatedPrefinalLinuxReceiptV1:
        raise TypeError("prefinal Linux receipt capability type differs")
    _require(receipt in _PREFINAL, "prefinal Linux receipt capability is unregistered")
    value = _PREFINAL[receipt]
    _require(
        _prepared(value.prepared).expected == value.observation.facts,
        "prefinal Linux receipt inputs differ",
    )
    return value.observation.metadata


def authenticated_linux_wheel_target(
    receipt: AuthenticatedPrefinalLinuxReceiptV1,
) -> AuthenticatedLinuxWheelTargetV1:
    prefinal_linux_receipt_metadata(receipt)
    return _PREFINAL[receipt].target


@dataclass(frozen=True, slots=True)
class BoundPrefinalLinuxReceiptMetadataV1:
    qualification_input_sha256: str
    service: service.LinuxReceiptMetadataV1


class BoundPrefinalLinuxReceiptV1:
    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("bound prefinal Linux receipts are verifier-minted only")


@dataclass(frozen=True, slots=True)
class _BoundReceipt:
    linux_runtime: BoundDependencyPurposeV1
    image: AdmittedLinuxRuntimeImageV1
    expected: service._LinuxReceiptFacts
    observation: service._AuthenticatedObservation
    metadata: BoundPrefinalLinuxReceiptMetadataV1


_BOUND: WeakKeyDictionary[BoundPrefinalLinuxReceiptV1, _BoundReceipt] = WeakKeyDictionary()


def bind_prefinal_linux_receipt(
    receipt: AuthenticatedPrefinalLinuxReceiptV1,
    files: RetainedQualificationInputFilesV1,
    linux_runtime: BoundDependencyPurposeV1,
    image: AdmittedLinuxRuntimeImageV1,
) -> BoundPrefinalLinuxReceiptV1:
    if type(receipt) is not AuthenticatedPrefinalLinuxReceiptV1:
        raise TypeError("prefinal Linux receipt capability type differs")
    _require(receipt not in _TRANSFERRED, "prefinal Linux receipt was already transferred")
    _require(receipt in _PREFINAL, "prefinal Linux receipt capability is unregistered")
    prefinal = _PREFINAL[receipt]
    prepared = _prepared(prefinal.prepared)
    _require(
        prefinal.observation.facts == prepared.expected,
        "authenticated prefinal Linux facts differ",
    )
    final_facts = service._facts_from_capabilities(linux_runtime, image)
    _, target_image = _authenticated_linux_binding_for_consumer(prefinal.target)
    _require(final_facts == prepared.expected, "final Linux inputs differ from prefinal receipt")
    _require(
        target_image
        == (
            final_facts.image_reference,
            final_facts.image_config_sha256,
            final_facts.image_layers,
            final_facts.image_diff_ids,
        ),
        "final Linux image differs from authenticated target",
    )
    _require(_prepared(prefinal.prepared).expected == final_facts, "prefinal Linux seals changed")
    binding = _dependency_binding_for_consumer(linux_runtime)
    _require(
        binding.files is files and binding.linux_target is prefinal.target,
        "final Linux receipt lease or target differs",
    )
    result = object.__new__(BoundPrefinalLinuxReceiptV1)
    metadata = BoundPrefinalLinuxReceiptMetadataV1(
        binding.metadata.qualification_input_sha256, prefinal.observation.metadata
    )
    _BOUND[result] = _BoundReceipt(
        linux_runtime, image, final_facts, prefinal.observation, metadata
    )
    _TRANSFERRED[receipt] = metadata.qualification_input_sha256
    return result


def bound_prefinal_linux_receipt_metadata(
    receipt: BoundPrefinalLinuxReceiptV1,
) -> BoundPrefinalLinuxReceiptMetadataV1:
    if type(receipt) is not BoundPrefinalLinuxReceiptV1:
        raise TypeError("bound prefinal Linux receipt capability type differs")
    _require(receipt in _BOUND, "bound prefinal Linux receipt capability is unregistered")
    value = _BOUND[receipt]
    current = service._facts_from_capabilities(value.linux_runtime, value.image)
    _require(current == value.expected, "bound Linux final input seals differ")
    _require(value.observation.facts == current, "bound Linux authenticated facts differ")
    service._match_payload(value.observation.payload, current, value.observation.metadata.run_id)
    _require(
        _dependency_binding_for_consumer(value.linux_runtime).metadata.qualification_input_sha256
        == value.metadata.qualification_input_sha256,
        "bound Linux final input identity differs",
    )
    return value.metadata


def _bound_linux_receipt_payload_for_consumer(
    receipt: BoundPrefinalLinuxReceiptV1,
) -> LinuxReceiptPayloadV1:
    bound_prefinal_linux_receipt_metadata(receipt)
    return _BOUND[receipt].observation.payload

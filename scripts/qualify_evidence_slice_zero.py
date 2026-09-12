"""Strict qualification input/report validation and closed scenario registration.

Input validation reopens direct and transitive artifacts without launching a
process. The registered source-equivalence and packaged producers own their
execution boundaries; other scenarios refuse execution until their producers exist.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

if TYPE_CHECKING:
    from scripts.candidate_source_archive_oracle import VerifiedCandidateSourceArchiveV1
    from scripts.candidate_wheel import VerifiedCandidateWheelV1
    from scripts.capacity_rollover import ObservedCapacityRolloverV1
    from scripts.deterministic_equivalence import ObservedEquivalenceV1
    from scripts.full_purge_cleanup import ObservedFullPurgeV1
    from scripts.over_budget_turn import ObservedOverBudgetTurnV1
    from scripts.owned_close_faults import ObservedOwnedCloseFaultsV1
    from scripts.revoke_race import ObservedRevokeRaceV1
    from scripts.spool_crash_matrix import ObservedSpoolCrashMatrixV1
    from scripts.synthetic_fault_matrix import ObservedSyntheticFaultV1
    from scripts.task13_artifact_orchestrator import CandidateIdentityV1


class QualificationInputError(ValueError):
    """The immutable qualification input is malformed, noncanonical, or stale."""


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """One byte-verified member of the closed input set, without a host path."""

    logical_id: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class QualificationInputClosure:
    """The canonical input digest plus its sorted direct/transitive artifact closure."""

    qualification_input_sha256: str
    verified_artifacts: tuple[VerifiedArtifact, ...]


class ScenarioIdV1(StrEnum):
    """The exact governed Task-12 scenario order; aliases are intentionally absent."""

    DETERMINISTIC_EQUIVALENCE = "deterministic_equivalence"
    PHYSICAL_CAPTURE_DISABLED = "physical_capture_disabled"
    PHYSICAL_AVAILABLE_UNCONSENTED = "physical_available_unconsented"
    PHYSICAL_MICROPHONE_RESPONSE = "physical_microphone_response"
    PHYSICAL_TYPED_RESPONSE = "physical_typed_response"
    PHYSICAL_UNMUTED_TRANSPORT = "physical_unmuted_transport"
    PHYSICAL_MUTED_TRANSPORT = "physical_muted_transport"
    PHYSICAL_INTERRUPTION_MATRIX = "physical_interruption_matrix"
    PHYSICAL_RECONNECT = "physical_reconnect"
    PHYSICAL_MEDIA_REPLACEMENT = "physical_media_replacement"
    REVOKE_RACE = "revoke_race"
    CAPACITY_ROLLOVER = "capacity_rollover"
    OVER_BUDGET_TURN = "over_budget_turn"
    SPOOL_CRASH_MATRIX = "spool_crash_matrix"
    INSTALLED_HOST_CRASH_MATRIX = "installed_host_crash_matrix"
    SYNTHETIC_FAULT_MATRIX = "synthetic_fault_matrix"
    WINDOWS_FILESYSTEM_MATRIX = "windows_filesystem_matrix"
    WINDOWS_VOLUME_FULL = "windows_volume_full"
    FULL_PURGE_CLEANUP = "full_purge_cleanup"
    OWNED_CLOSE_FAULTS = "owned_close_faults"


_SCENARIO_IDS_V1 = tuple(ScenarioIdV1)
_LOWER_HEX = frozenset("0123456789abcdef")


def _require_lower_hex(value: object, *, length: int, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{label} must be exact lowercase hexadecimal")
    return value


@dataclass(frozen=True, slots=True)
class CandidateBindingV1:
    """Candidate bytes and source identity shared by every unavailable attempt."""

    qualification_input_sha256: str
    source_commit: str
    source_tree: str

    def __post_init__(self) -> None:
        if type(self) is not CandidateBindingV1:
            raise TypeError("candidate binding subclasses are forbidden")
        _require_lower_hex(
            self.qualification_input_sha256,
            length=64,
            label="qualification input SHA-256",
        )
        _require_lower_hex(self.source_commit, length=40, label="source commit")
        _require_lower_hex(self.source_tree, length=40, label="source tree")


@dataclass(frozen=True, slots=True)
class ScenarioAttemptIdentityV1:
    """One exact scenario invocation identity, without observations or outcomes."""

    scenario_id: ScenarioIdV1
    ordinal: int
    candidate: CandidateBindingV1
    runner_pid: int
    runner_creation_filetime: int

    def __post_init__(self) -> None:
        if type(self) is not ScenarioAttemptIdentityV1:
            raise TypeError("scenario attempt subclasses are forbidden")
        if type(self.scenario_id) is not ScenarioIdV1:
            raise TypeError("scenario ID must be the exact governed enum")
        if type(self.ordinal) is not int or not 0 <= self.ordinal < len(_SCENARIO_IDS_V1):
            raise ValueError("scenario ordinal is invalid")
        if _SCENARIO_IDS_V1[self.ordinal] is not self.scenario_id:
            raise ValueError("scenario ID and ordinal are confused")
        if type(self.candidate) is not CandidateBindingV1:
            raise TypeError("candidate binding must be exact")
        if type(self.runner_pid) is not int or self.runner_pid <= 0:
            raise ValueError("runner PID is invalid")
        if type(self.runner_creation_filetime) is not int or self.runner_creation_filetime <= 0:
            raise ValueError("runner creation FILETIME is invalid")


class ProducerUnavailableV1(RuntimeError):
    """A registered scenario has no governed producer capability yet."""

    def __init__(self, scenario_id: ScenarioIdV1) -> None:
        if type(scenario_id) is not ScenarioIdV1:
            raise TypeError("unavailable producer scenario ID is invalid")
        self.scenario_id = scenario_id
        super().__init__(f"producer unavailable: {scenario_id.value}")


@dataclass(frozen=True, slots=True)
class _UnavailableScenarioProducerV1:
    scenario_id: ScenarioIdV1

    def __post_init__(self) -> None:
        if (
            type(self) is not _UnavailableScenarioProducerV1
            or type(self.scenario_id) is not ScenarioIdV1
        ):
            raise TypeError("unavailable producer identity is invalid")

    def __call__(self, attempt: ScenarioAttemptIdentityV1) -> NoReturn:
        if type(attempt) is not ScenarioAttemptIdentityV1:
            raise TypeError("scenario attempt identity must be exact")
        if attempt.scenario_id is not self.scenario_id:
            raise ValueError("scenario attempt does not match unavailable producer")
        raise ProducerUnavailableV1(self.scenario_id)


@dataclass(frozen=True, slots=True)
class UnavailableScenarioRegistrationV1:
    """Closed registration only; it cannot return a raw record or report row."""

    scenario_id: ScenarioIdV1
    produce: _UnavailableScenarioProducerV1

    def __post_init__(self) -> None:
        if type(self) is not UnavailableScenarioRegistrationV1:
            raise TypeError("unavailable registration subclasses are forbidden")
        if type(self.scenario_id) is not ScenarioIdV1:
            raise TypeError("registration scenario ID must be exact")
        if (
            type(self.produce) is not _UnavailableScenarioProducerV1
            or self.produce.scenario_id is not self.scenario_id
        ):
            raise ValueError("unavailable producer registration is confused")


@dataclass(frozen=True, slots=True)
class DeterministicEquivalenceRegistrationV1:
    """The one available producer, with an exact archive rather than supplied facts."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.DETERMINISTIC_EQUIVALENCE

    def __post_init__(self) -> None:
        if (
            type(self) is not DeterministicEquivalenceRegistrationV1
            or self.scenario_id is not ScenarioIdV1.DETERMINISTIC_EQUIVALENCE
        ):
            raise TypeError("deterministic producer registration is not exact")

    def produce(
        self,
        archive: VerifiedCandidateSourceArchiveV1,
        identity: CandidateIdentityV1,
        *,
        livekit_executable: Path,
        livekit_sha256: str,
    ) -> ObservedEquivalenceV1:
        if self is not SCENARIO_REGISTRY_V1[0]:
            raise ValueError("deterministic producer registration is not canonical")
        from scripts.deterministic_equivalence import produce_deterministic_equivalence_v1

        return produce_deterministic_equivalence_v1(
            archive,
            identity,
            livekit_executable=livekit_executable,
            livekit_sha256=livekit_sha256,
        )


@dataclass(frozen=True, slots=True)
class RevokeRaceRegistrationV1:
    """The packaged revocation producer at its unchanged canonical ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.REVOKE_RACE

    def __post_init__(self) -> None:
        if (
            type(self) is not RevokeRaceRegistrationV1
            or self.scenario_id is not ScenarioIdV1.REVOKE_RACE
        ):
            raise TypeError("revocation registration must be exact")

    def produce(
        self, archive: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1,
        wheel: VerifiedCandidateWheelV1,
        *, livekit_executable: Path, livekit_sha256: str,
    ) -> ObservedRevokeRaceV1:
        if self is not SCENARIO_REGISTRY_V1[10]:
            raise ValueError("revocation producer registration is not canonical")
        from scripts.revoke_race import produce_revoke_race_v1

        return produce_revoke_race_v1(
            archive, identity, wheel,
            livekit_executable=livekit_executable, livekit_sha256=livekit_sha256,
        )


@dataclass(frozen=True, slots=True)
class CapacityRolloverRegistrationV1:
    """The packaged capacity rollover producer at its unchanged canonical ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.CAPACITY_ROLLOVER

    def __post_init__(self) -> None:
        if (
            type(self) is not CapacityRolloverRegistrationV1
            or self.scenario_id is not ScenarioIdV1.CAPACITY_ROLLOVER
        ):
            raise TypeError("capacity rollover registration must be exact")

    def produce(
        self, archive: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1,
        wheel: VerifiedCandidateWheelV1,
        *, livekit_executable: Path, livekit_sha256: str,
    ) -> ObservedCapacityRolloverV1:
        if self is not SCENARIO_REGISTRY_V1[11]:
            raise ValueError("capacity rollover producer registration is not canonical")
        from scripts.capacity_rollover import produce_capacity_rollover_v1

        return produce_capacity_rollover_v1(
            archive, identity, wheel,
            livekit_executable=livekit_executable, livekit_sha256=livekit_sha256,
        )


@dataclass(frozen=True, slots=True)
class OverBudgetTurnRegistrationV1:
    """The packaged capture admission overflow producer at its unchanged canonical ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.OVER_BUDGET_TURN

    def __post_init__(self) -> None:
        if (
            type(self) is not OverBudgetTurnRegistrationV1
            or self.scenario_id is not ScenarioIdV1.OVER_BUDGET_TURN
        ):
            raise TypeError("capture admission overflow registration must be exact")

    def produce(
        self,
        archive: VerifiedCandidateSourceArchiveV1,
        identity: CandidateIdentityV1,
        wheel: VerifiedCandidateWheelV1,
        *,
        livekit_executable: Path,
        livekit_sha256: str,
    ) -> ObservedOverBudgetTurnV1:
        if self is not SCENARIO_REGISTRY_V1[12]:
            raise ValueError("capture admission overflow producer registration is not canonical")
        from scripts.over_budget_turn import produce_over_budget_turn_v1

        return produce_over_budget_turn_v1(
            archive,
            identity,
            wheel,
            livekit_executable=livekit_executable,
            livekit_sha256=livekit_sha256,
        )



@dataclass(frozen=True, slots=True)
class SpoolCrashMatrixRegistrationV1:
    """The packaged spool crash producer at its governed ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.SPOOL_CRASH_MATRIX

    def __post_init__(self) -> None:
        if (
            type(self) is not SpoolCrashMatrixRegistrationV1
            or self.scenario_id is not ScenarioIdV1.SPOOL_CRASH_MATRIX
        ):
            raise TypeError("spool crash registration must be exact")

    def produce(
        self, archive: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1,
        wheel: VerifiedCandidateWheelV1,
    ) -> ObservedSpoolCrashMatrixV1:
        if self is not SCENARIO_REGISTRY_V1[13]:
            raise ValueError("spool crash producer registration is not canonical")
        from scripts.spool_crash_matrix import produce_spool_crash_matrix_v1

        return produce_spool_crash_matrix_v1(archive, identity, wheel)


@dataclass(frozen=True, slots=True)
class SyntheticFaultRegistrationV1:
    """The five-case synthetic producer at its governed ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.SYNTHETIC_FAULT_MATRIX

    def __post_init__(self) -> None:
        if (type(self) is not SyntheticFaultRegistrationV1
                or self.scenario_id is not ScenarioIdV1.SYNTHETIC_FAULT_MATRIX):
            raise TypeError("synthetic fault registration must be exact")

    def produce(self, archive: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1,
                wheel: VerifiedCandidateWheelV1) -> ObservedSyntheticFaultV1:
        if self is not SCENARIO_REGISTRY_V1[15]:
            raise ValueError("synthetic fault producer registration is not canonical")
        from scripts.synthetic_fault_matrix import produce_synthetic_fault_v1

        return produce_synthetic_fault_v1(archive, identity, wheel)


@dataclass(frozen=True, slots=True)
class FullPurgeRegistrationV1:
    """The real-filesystem full-purge producer at its governed ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.FULL_PURGE_CLEANUP

    def __post_init__(self) -> None:
        if (
            type(self) is not FullPurgeRegistrationV1
            or self.scenario_id is not ScenarioIdV1.FULL_PURGE_CLEANUP
        ):
            raise TypeError("full-purge registration must be exact")

    def produce(self, archive: VerifiedCandidateSourceArchiveV1, identity: CandidateIdentityV1,
                wheel: VerifiedCandidateWheelV1) -> ObservedFullPurgeV1:
        if self is not SCENARIO_REGISTRY_V1[18]:
            raise ValueError("full-purge producer registration is not canonical")
        from scripts.full_purge_cleanup import produce_full_purge_v1

        return produce_full_purge_v1(archive, identity, wheel)


@dataclass(frozen=True, slots=True)
class OwnedCloseFaultsRegistrationV1:
    """The packaged owned-close faults producer at its unchanged canonical ordinal."""

    scenario_id: ScenarioIdV1 = ScenarioIdV1.OWNED_CLOSE_FAULTS

    def __post_init__(self) -> None:
        if (
            type(self) is not OwnedCloseFaultsRegistrationV1
            or self.scenario_id is not ScenarioIdV1.OWNED_CLOSE_FAULTS
        ):
            raise TypeError("owned-close faults registration must be exact")

    def produce(
        self,
        archive: VerifiedCandidateSourceArchiveV1,
        identity: CandidateIdentityV1,
        wheel: VerifiedCandidateWheelV1,
        *,
        livekit_executable: Path,
        livekit_sha256: str,
    ) -> ObservedOwnedCloseFaultsV1:
        if self is not SCENARIO_REGISTRY_V1[19]:
            raise ValueError("owned-close faults producer registration is not canonical")
        from scripts.owned_close_faults import produce_owned_close_faults_v1

        return produce_owned_close_faults_v1(
            archive,
            identity,
            wheel,
            livekit_executable=livekit_executable,
            livekit_sha256=livekit_sha256,
        )



_UNAVAILABLE_SCENARIO_IDS_V1 = tuple(
    scenario for scenario in _SCENARIO_IDS_V1
    if scenario not in {
        ScenarioIdV1.DETERMINISTIC_EQUIVALENCE,
        ScenarioIdV1.REVOKE_RACE,
        ScenarioIdV1.CAPACITY_ROLLOVER,
        ScenarioIdV1.OVER_BUDGET_TURN,
        ScenarioIdV1.SPOOL_CRASH_MATRIX,
        ScenarioIdV1.SYNTHETIC_FAULT_MATRIX,
        ScenarioIdV1.FULL_PURGE_CLEANUP,
        ScenarioIdV1.OWNED_CLOSE_FAULTS,
    }
)
UNAVAILABLE_SCENARIO_REGISTRY_V1 = tuple(
    UnavailableScenarioRegistrationV1(
        scenario_id=scenario_id,
        produce=_UnavailableScenarioProducerV1(scenario_id),
    )
    for scenario_id in _UNAVAILABLE_SCENARIO_IDS_V1
)

DETERMINISTIC_EQUIVALENCE_REGISTRATION_V1 = DeterministicEquivalenceRegistrationV1()
REVOKE_RACE_REGISTRATION_V1 = RevokeRaceRegistrationV1()
CAPACITY_ROLLOVER_REGISTRATION_V1 = CapacityRolloverRegistrationV1()
OVER_BUDGET_TURN_REGISTRATION_V1 = OverBudgetTurnRegistrationV1()
SPOOL_CRASH_MATRIX_REGISTRATION_V1 = SpoolCrashMatrixRegistrationV1()
SYNTHETIC_FAULT_REGISTRATION_V1 = SyntheticFaultRegistrationV1()
FULL_PURGE_REGISTRATION_V1 = FullPurgeRegistrationV1()
OWNED_CLOSE_FAULTS_REGISTRATION_V1 = OwnedCloseFaultsRegistrationV1()

_ScenarioRegistrationV1 = (
    UnavailableScenarioRegistrationV1 | DeterministicEquivalenceRegistrationV1
    | RevokeRaceRegistrationV1 | CapacityRolloverRegistrationV1
    | OverBudgetTurnRegistrationV1 | SpoolCrashMatrixRegistrationV1 | FullPurgeRegistrationV1
    | OwnedCloseFaultsRegistrationV1 | SyntheticFaultRegistrationV1
)
_REGISTRATIONS_V1: tuple[_ScenarioRegistrationV1, ...] = (
    *UNAVAILABLE_SCENARIO_REGISTRY_V1,
    DETERMINISTIC_EQUIVALENCE_REGISTRATION_V1,
    REVOKE_RACE_REGISTRATION_V1,
    CAPACITY_ROLLOVER_REGISTRATION_V1,
    OVER_BUDGET_TURN_REGISTRATION_V1,
    SPOOL_CRASH_MATRIX_REGISTRATION_V1,
    SYNTHETIC_FAULT_REGISTRATION_V1,
    FULL_PURGE_REGISTRATION_V1,
    OWNED_CLOSE_FAULTS_REGISTRATION_V1,
)
_REGISTRATIONS_BY_ID_V1 = {
    registration.scenario_id: registration for registration in _REGISTRATIONS_V1
}
SCENARIO_REGISTRY_V1 = tuple(
    _REGISTRATIONS_BY_ID_V1[scenario_id] for scenario_id in _SCENARIO_IDS_V1
)


def validate_unavailable_scenario_registry_v1(
    registry: object,
) -> tuple[UnavailableScenarioRegistrationV1, ...]:
    if type(registry) is not tuple:
        raise TypeError("unavailable scenario registry must be an exact tuple")
    if registry is not UNAVAILABLE_SCENARIO_REGISTRY_V1:
        raise ValueError("unavailable scenario registry is not canonical")
    if len(registry) != len(_UNAVAILABLE_SCENARIO_IDS_V1):
        raise ValueError("unavailable registry does not match its scenario IDs")
    for expected, registration in zip(_UNAVAILABLE_SCENARIO_IDS_V1, registry, strict=True):
        if type(registration) is not UnavailableScenarioRegistrationV1:
            raise TypeError("unavailable scenario registration must be exact")
        if registration.scenario_id is not expected:
            raise ValueError("unavailable scenario registry order is invalid")
        if (
            type(registration.produce) is not _UnavailableScenarioProducerV1
            or registration.produce.scenario_id is not registration.scenario_id
        ):
            raise ValueError("unavailable scenario producer binding is invalid")
    return registry


def invoke_unavailable_scenario_v1(
    registration: UnavailableScenarioRegistrationV1,
    attempt: ScenarioAttemptIdentityV1,
) -> NoReturn:
    if type(registration) is not UnavailableScenarioRegistrationV1:
        raise TypeError("unavailable scenario registration must be exact")
    if type(attempt) is not ScenarioAttemptIdentityV1:
        raise TypeError("scenario attempt identity type is invalid")
    if registration is not SCENARIO_REGISTRY_V1[attempt.ordinal]:
        raise ValueError("scenario registration is not canonical for attempt ordinal")
    if registration.scenario_id is not attempt.scenario_id:
        raise ValueError("registration and scenario attempt are confused")
    registration.produce(attempt)


_FILE_ROLES = frozenset(
    {
        "governing_plan",
        "candidate_source_archive",
        "qualification_runner",
        "direct_wheel",
        "direct_wheel_repeat",
        "sdist",
        "sdist_repeat",
        "sdist_built_wheel",
        "sdist_built_wheel_repeat",
        "build_wheelhouse_manifest",
        "windows_direct_runtime_wheelhouse_manifest",
        "windows_sdist_built_runtime_wheelhouse_manifest",
        "linux_runtime_wheelhouse_manifest",
        "hermes_runtime_wheelhouse_manifest",
        "benchmark_machine_schema",
        "benchmark_report_schema",
        "wheelhouse_manifest_schema",
        "qualification_input_schema",
        "qualification_report_schema",
        "release_manifest_schema",
        "benchmark_report",
        "benchmark_machine_manifest",
        "hermes_source_archive",
        "hermes_pluginmanager_runner",
        "chrome_executable",
        "chrome_version_directory_manifest",
        "livekit_archive",
        "livekit_executable",
        "codex_executable",
        "moonshine_distribution",
        "moonshine_model_manifest",
        "kokoro_distribution",
        "kokoro_model_manifest",
    }
)
_TOOL_ROLES = frozenset({"git", "uv", "build_python", "hatchling"})
_TOOL_ARTIFACT_ROLES = {
    "git": "git_executable",
    "uv": "uv_executable",
    "build_python": "build_python_executable",
    "hatchling": "hatchling_wheel",
}
_WHEELHOUSE_ROLES = {
    "build": "build_wheelhouse_manifest",
    "realtime_windows_direct_runtime": "windows_direct_runtime_wheelhouse_manifest",
    "realtime_windows_sdist_built_runtime": "windows_sdist_built_runtime_wheelhouse_manifest",
    "realtime_linux_runtime": "linux_runtime_wheelhouse_manifest",
    "hermes_v020_pluginmanager_runtime": "hermes_runtime_wheelhouse_manifest",
}
_REQUESTED_SCENARIO_IDS = (
    "deterministic_equivalence",
    "physical_capture_disabled",
    "physical_available_unconsented",
    "physical_microphone_response",
    "physical_typed_response",
    "physical_unmuted_transport",
    "physical_muted_transport",
    "physical_interruption_matrix",
    "physical_reconnect",
    "physical_media_replacement",
    "revoke_race",
    "capacity_rollover",
    "over_budget_turn",
    "spool_crash_matrix",
    "installed_host_crash_matrix",
    "synthetic_fault_matrix",
    "windows_filesystem_matrix",
    "windows_volume_full",
    "full_purge_cleanup",
    "owned_close_faults",
)
_EXPECTED_SCHEMA_SHA256_ROLES = {
    "benchmarkMachineSchemaSha256": "benchmark_machine_schema",
    "benchmarkReportSchemaSha256": "benchmark_report_schema",
    "wheelhouseManifestSchemaSha256": "wheelhouse_manifest_schema",
    "qualificationInputSchemaSha256": "qualification_input_schema",
    "qualificationReportSchemaSha256": "qualification_report_schema",
    "releaseManifestSchemaSha256": "release_manifest_schema",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._-]+\Z")
_CHROME_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\Z")
_REQUIREMENT_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})(?=\s|$)")
_QUALIFICATION_INPUT_SCHEMA_PATH = (
    Path(__file__).parent / "schemas" / "qualification-input-v1.schema.json"
)


def _fail(message: str) -> NoReturn:
    raise QualificationInputError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate object key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> NoReturn:
    _fail(f"non-finite JSON number {value!r} is forbidden")


def _reject_floats_and_nulls(value: object) -> None:
    if value is None:
        _fail("JSON null is forbidden")
    if type(value) is float:
        _fail("JSON floats are forbidden")
    if type(value) is dict:
        for item in value.values():
            _reject_floats_and_nulls(item)
    elif type(value) is list:
        for item in value:
            _reject_floats_and_nulls(item)


def canonical_json_bytes(document: object, *, terminal_lf: bool = True) -> bytes:
    """Serialize the sole accepted compact, key-sorted UTF-8 JSON representation."""

    _reject_floats_and_nulls(document)
    try:
        result = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QualificationInputError("JSON value cannot be canonically encoded") from error
    return result + (b"\n" if terminal_lf else b"")


def load_strict_canonical_json(raw: bytes, *, source: str) -> object:
    """Decode only exact canonical JSON with one terminal LF and no duplicate keys."""

    if type(raw) is not bytes:
        _fail(f"{source}: JSON bytes must be exact bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{source}: UTF-8 BOM is forbidden")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        _fail(f"{source}: canonical JSON requires exactly one terminal LF and no CR")
    try:
        document = json.loads(
            raw[:-1].decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, QualificationInputError) as error:
        raise QualificationInputError(f"{source}: malformed JSON") from error
    _reject_floats_and_nulls(document)
    if canonical_json_bytes(document) != raw:
        _fail(f"{source}: JSON is not canonical compact sorted-key UTF-8")
    return document


def _validate_qualification_input_schema(document: object) -> None:
    """Validate against the exact canonical schema shipped beside this runner."""

    try:
        jsonschema = importlib.import_module("jsonschema")
        jsonschema_exceptions = importlib.import_module("jsonschema.exceptions")
    except ImportError as error:
        raise QualificationInputError(
            "qualification input schema validator is unavailable"
        ) from error
    try:
        schema_bytes = _QUALIFICATION_INPUT_SCHEMA_PATH.read_bytes()
    except OSError as error:
        raise QualificationInputError(
            "qualification input schema cannot be read beside runner"
        ) from error
    schema = load_strict_canonical_json(schema_bytes, source="qualification input schema")
    if type(schema) is not dict:
        _fail("qualification input schema root must be an object")
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
    except jsonschema_exceptions.SchemaError as error:
        raise QualificationInputError(
            "qualification input schema is not valid Draft 2020-12"
        ) from error
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        location = ".".join(str(part) for part in errors[0].absolute_path) or "<root>"
        _fail(f"qualification input manifest: schema violation at {location}")


def _expect_exact_object(value: object, keys: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        _fail(f"{label}: object keys are not exact")
    return value


def _expect_list(value: object, *, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label}: expected array")
    return value


def _expect_string(value: object, *, label: str) -> str:
    if type(value) is not str:
        _fail(f"{label}: expected string")
    return value


def _safe_posix_path(value: object, *, label: str) -> str:
    path = _expect_string(value, label=label)
    if not path or path.startswith("/") or "\\" in path or ":" in path:
        _fail(f"{label}: path is not repository-free POSIX")
    components = path.split("/")
    if any(
        component in {"", ".", ".."} or not _SAFE_COMPONENT.fullmatch(component)
        for component in components
    ):
        _fail(f"{label}: unsafe path component")
    return path


def _chrome_resource_name(value: object) -> str:
    """Keep ordinary publisher names; direct artifact references stay strict."""
    from scripts.qualification_file_seals import _relative_windows_member

    name = _expect_string(value, label="Chrome resource name")
    if len(name) > 512:
        _fail("Chrome resource name exceeds its bound")
    try:
        _relative_windows_member(name)
    except ValueError as error:
        raise QualificationInputError("Chrome resource name is unsafe") from error
    return name


def _is_reparse_or_link(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise QualificationInputError(f"{path}: could not stat input path") from error
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _validated_root(path: Path) -> Path:
    try:
        candidate = path.absolute()
        if not candidate.is_dir() or _is_reparse_or_link(candidate):
            _fail("qualification input root must be an ordinary directory")
        return candidate.resolve(strict=True)
    except OSError as error:
        raise QualificationInputError("qualification input root cannot be resolved") from error


def _path_beneath(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = root
    for component in relative_path.split("/"):
        candidate = candidate / component
        if _is_reparse_or_link(candidate):
            _fail(f"{label}: reparse point or symlink is forbidden")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise QualificationInputError(f"{label}: path escapes or is absent") from error
    if not resolved.is_file():
        _fail(f"{label}: artifact must be an ordinary file")
    return resolved


def _relative_path(root: Path, path: Path, *, label: str) -> str:
    try:
        absolute = path.absolute()
        if _is_reparse_or_link(absolute):
            _fail(f"{label}: reparse point or symlink is forbidden")
        resolved = absolute.resolve(strict=True)
        return resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as error:
        raise QualificationInputError(
            f"{label}: must resolve beneath qualification input root"
        ) from error


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise QualificationInputError(f"{path}: unable to read artifact") from error
    return digest.hexdigest(), size


def _verify_ref(
    root: Path,
    value: object,
    *,
    expected_role: str | None,
    label: str,
) -> tuple[dict[str, Any], Path]:
    reference = _expect_exact_object(
        value,
        frozenset({"role", "relativePath", "basename", "sha256", "bytes"}),
        label=label,
    )
    role = _expect_string(reference["role"], label=f"{label}.role")
    if expected_role is not None and role != expected_role:
        _fail(f"{label}: unexpected artifact role")
    relative = _safe_posix_path(reference["relativePath"], label=f"{label}.relativePath")
    basename = _expect_string(reference["basename"], label=f"{label}.basename")
    if basename != relative.rsplit("/", maxsplit=1)[-1] or not _SAFE_COMPONENT.fullmatch(basename):
        _fail(f"{label}: basename does not match safe relative path")
    claimed_sha = _expect_string(reference["sha256"], label=f"{label}.sha256")
    if not _SHA256.fullmatch(claimed_sha):
        _fail(f"{label}: digest must be lowercase SHA-256")
    claimed_bytes = reference["bytes"]
    if type(claimed_bytes) is not int or not 1 <= claimed_bytes <= 2**63 - 1:
        _fail(f"{label}: bytes must be a bounded exact integer")
    path = _path_beneath(root, relative, label=label)
    actual_sha, actual_bytes = _sha256_and_size(path)
    if (actual_sha, actual_bytes) != (claimed_sha, claimed_bytes):
        _fail(f"{label}: artifact bytes do not match reference")
    return reference, path


def _verify_external_match(path: Path, reference: dict[str, Any], *, label: str) -> Path:
    """Bind an invocation path to a root-resolved ArtifactRef by bytes, not text."""

    try:
        absolute = path.absolute()
        if _is_reparse_or_link(absolute) or not absolute.is_file():
            _fail(f"{label}: actual path must be an ordinary file")
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise QualificationInputError(f"{label}: actual path cannot be resolved") from error
    actual_sha, actual_bytes = _sha256_and_size(resolved)
    if (actual_sha, actual_bytes) != (reference["sha256"], reference["bytes"]):
        _fail(f"{label}: actual bytes do not match input-bound artifact")
    return resolved


def _append_artifact(
    artifacts: dict[str, VerifiedArtifact], *, logical_id: str, reference: dict[str, Any]
) -> None:
    if logical_id in artifacts:
        _fail(f"duplicate verified artifact logical ID {logical_id}")
    artifacts[logical_id] = VerifiedArtifact(
        logical_id=logical_id,
        sha256=reference["sha256"],
        bytes=reference["bytes"],
    )


def _load_canonical_file(path: Path, *, label: str) -> object:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise QualificationInputError(f"{label}: unable to read JSON") from error
    return load_strict_canonical_json(raw, source=label)


def _verify_wheelhouse(
    root: Path,
    path: Path,
    *,
    purpose: str,
    artifacts: dict[str, VerifiedArtifact],
    seen_transitive_paths: set[str],
) -> None:
    document = _expect_exact_object(
        _load_canonical_file(path, label=f"wheelhouse {purpose}"),
        frozenset(
            {
                "schemaVersion",
                "purpose",
                "pythonVersion",
                "platform",
                "requirements",
                "constraints",
                "wheels",
            }
        ),
        label=f"wheelhouse {purpose}",
    )
    if document["schemaVersion"] != 1 or document["purpose"] != purpose:
        _fail(f"wheelhouse {purpose}: wrong identity")
    _expect_string(document["pythonVersion"], label=f"wheelhouse {purpose}.pythonVersion")
    if document["platform"] not in {"windows_amd64", "linux_x86_64", "any"}:
        _fail(f"wheelhouse {purpose}: invalid platform")

    requirements, requirements_path = _verify_ref(
        root,
        document["requirements"],
        expected_role="requirements",
        label=f"wheelhouse {purpose}.requirements",
    )
    constraints, _ = _verify_ref(
        root,
        document["constraints"],
        expected_role="constraints",
        label=f"wheelhouse {purpose}.constraints",
    )
    for kind, reference in (("requirements", requirements), ("constraints", constraints)):
        relative = reference["relativePath"]
        if relative in seen_transitive_paths:
            _fail(f"wheelhouse {purpose}: duplicate transitive artifact path")
        seen_transitive_paths.add(relative)
        _append_artifact(artifacts, logical_id=f"wheelhouse:{purpose}:{kind}", reference=reference)

    wheels = _expect_list(document["wheels"], label=f"wheelhouse {purpose}.wheels")
    if not wheels:
        _fail(f"wheelhouse {purpose}: wheels cannot be empty")
    wheel_hashes: set[str] = set()
    wheel_paths: set[str] = set()
    wheel_basenames: set[str] = set()
    wheel_directories: set[Path] = set()
    for index, item in enumerate(wheels):
        reference, wheel_path = _verify_ref(
            root, item, expected_role="wheel", label=f"wheelhouse {purpose}.wheels[{index}]"
        )
        relative = reference["relativePath"]
        if (
            relative in seen_transitive_paths
            or relative in wheel_paths
            or reference["basename"].casefold() in wheel_basenames
            or reference["sha256"] in wheel_hashes
        ):
            _fail(f"wheelhouse {purpose}: duplicate transitive wheel")
        seen_transitive_paths.add(relative)
        wheel_paths.add(relative)
        wheel_basenames.add(reference["basename"].casefold())
        wheel_hashes.add(reference["sha256"])
        wheel_directories.add(wheel_path.parent)
        _append_artifact(
            artifacts,
            logical_id=f"wheelhouse:{purpose}:wheel:{reference['basename']}",
            reference=reference,
        )

    try:
        requirement_text = requirements_path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as error:
        raise QualificationInputError(
            f"wheelhouse {purpose}: requirements are not UTF-8"
        ) from error
    hashes = _REQUIREMENT_HASH.findall(requirement_text)
    if (
        not hashes
        or set(hashes) != wheel_hashes
        or any(hashes.count(value) != 1 for value in hashes)
    ):
        _fail(f"wheelhouse {purpose}: requirements hashes do not close the wheel set")
    for directory in wheel_directories:
        try:
            listed: set[str] = set()
            for child in directory.rglob("*"):
                if _is_reparse_or_link(child):
                    _fail(f"wheelhouse {purpose}: reparse point or symlink is forbidden")
                if child.is_file() and child.suffix.casefold() == ".whl":
                    listed.add(child.resolve(strict=True).relative_to(root).as_posix())
                elif not child.is_file() and not child.is_dir():
                    _fail(f"wheelhouse {purpose}: unsupported wheel-directory entry")
        except QualificationInputError:
            raise
        except (OSError, ValueError) as error:
            raise QualificationInputError(
                f"wheelhouse {purpose}: cannot enumerate wheel directory"
            ) from error
        if listed != wheel_paths:
            _fail(f"wheelhouse {purpose}: unlisted wheel is present")


def _verify_provider_manifest(
    root: Path,
    path: Path,
    *,
    provider: str,
    artifacts: dict[str, VerifiedArtifact],
    seen_transitive_paths: set[str],
) -> None:
    document = _expect_exact_object(
        _load_canonical_file(path, label=f"provider {provider}"),
        frozenset({"schemaVersion", "provider", "modelIdentitySha256", "resources"}),
        label=f"provider {provider}",
    )
    if document["schemaVersion"] != 1 or document["provider"] != provider:
        _fail(f"provider {provider}: wrong identity")
    model_identity = _expect_string(
        document["modelIdentitySha256"], label=f"provider {provider}.modelIdentitySha256"
    )
    if not _SHA256.fullmatch(model_identity):
        _fail(f"provider {provider}: model identity is invalid")
    resources = _expect_list(document["resources"], label=f"provider {provider}.resources")
    if not resources:
        _fail(f"provider {provider}: resources cannot be empty")
    resource_map: dict[str, str] = {}
    previous_name = ""
    for index, item in enumerate(resources):
        resource = _expect_exact_object(
            item,
            frozenset({"name", "bytes", "sha256"}),
            label=f"provider {provider}.resources[{index}]",
        )
        name = _safe_posix_path(
            resource["name"], label=f"provider {provider}.resources[{index}].name"
        )
        if name <= previous_name or name in resource_map:
            _fail(f"provider {provider}: resources are not sorted unique")
        previous_name = name
        claimed_bytes = resource["bytes"]
        claimed_sha = _expect_string(
            resource["sha256"], label=f"provider {provider}.resources[{index}].sha256"
        )
        if (
            type(claimed_bytes) is not int
            or claimed_bytes < 1
            or not _SHA256.fullmatch(claimed_sha)
        ):
            _fail(f"provider {provider}: invalid resource reference")
        resource_path = _path_beneath(
            root,
            _relative_path(root, path.parent / name, label="provider resource"),
            label="provider resource",
        )
        actual_sha, actual_bytes = _sha256_and_size(resource_path)
        if (actual_sha, actual_bytes) != (claimed_sha, claimed_bytes):
            _fail(f"provider {provider}: resource bytes do not match")
        relative = _relative_path(root, resource_path, label="provider resource")
        if relative in seen_transitive_paths:
            _fail(f"provider {provider}: duplicate transitive artifact path")
        seen_transitive_paths.add(relative)
        resource_map[name] = claimed_sha
        _append_artifact(
            artifacts,
            logical_id=f"provider:{provider}:resource:{name}",
            reference={"sha256": claimed_sha, "bytes": claimed_bytes},
        )
    if (
        hashlib.sha256(canonical_json_bytes(resource_map, terminal_lf=False)).hexdigest()
        != model_identity
    ):
        _fail(f"provider {provider}: model identity does not bind resources")


def _verify_chrome_manifest(
    root: Path,
    path: Path,
    chrome_executable: Path,
    artifacts: dict[str, VerifiedArtifact],
    seen_transitive_paths: set[str],
) -> None:
    document = _expect_exact_object(
        _load_canonical_file(path, label="Chrome version-directory manifest"),
        frozenset({"schemaVersion", "chromeVersion", "files"}),
        label="Chrome version-directory manifest",
    )
    if document["schemaVersion"] != 1:
        _fail("Chrome version-directory manifest: unsupported schema version")
    chrome_version = _expect_string(
        document["chromeVersion"], label="Chrome version-directory manifest.chromeVersion"
    )
    if not _CHROME_VERSION.fullmatch(chrome_version) or path.parent.name != chrome_version:
        _fail("Chrome version-directory manifest: version does not bind directory leaf")
    if chrome_executable.parent != path.parent:
        _fail("Chrome version-directory manifest: executable is outside version directory")
    files = _expect_list(document["files"], label="Chrome version-directory manifest.files")
    if not files:
        _fail("Chrome version-directory manifest: files cannot be empty")
    manifest_paths: set[str] = set()
    previous_name = ""
    executable_name: str | None = None
    for index, item in enumerate(files):
        file = _expect_exact_object(
            item, frozenset({"name", "bytes", "sha256"}), label=f"Chrome manifest.files[{index}]"
        )
        name = _chrome_resource_name(file["name"])
        if name <= previous_name or name in manifest_paths:
            _fail("Chrome version-directory manifest: files are not sorted unique")
        previous_name = name
        claimed_bytes = file["bytes"]
        claimed_sha = _expect_string(file["sha256"], label=f"Chrome manifest.files[{index}].sha256")
        if (
            type(claimed_bytes) is not int
            or claimed_bytes < 1
            or not _SHA256.fullmatch(claimed_sha)
        ):
            _fail("Chrome version-directory manifest: invalid file reference")
        file_path = _path_beneath(
            root, _relative_path(root, path.parent / name, label="Chrome file"), label="Chrome file"
        )
        actual_sha, actual_bytes = _sha256_and_size(file_path)
        if (actual_sha, actual_bytes) != (claimed_sha, claimed_bytes):
            _fail("Chrome version-directory manifest: file bytes do not match")
        relative = _relative_path(root, file_path, label="Chrome file")
        if relative in seen_transitive_paths:
            _fail("Chrome version-directory manifest: duplicate transitive artifact path")
        seen_transitive_paths.add(relative)
        manifest_paths.add(name)
        if file_path == chrome_executable:
            executable_name = name
        _append_artifact(
            artifacts,
            # Publisher names can contain spaces and directory separators. A
            # digest identifies every name under the existing public ID grammar;
            # the sealed version manifest retains the exact relative spelling.
            logical_id="chrome:file:" + hashlib.sha256(name.encode("utf-8")).hexdigest(),
            reference={"sha256": claimed_sha, "bytes": claimed_bytes},
        )
    if executable_name is None:
        _fail("Chrome version-directory manifest: bound executable is absent")
    try:
        actual_files: set[str] = set()
        for candidate in path.parent.rglob("*"):
            if _is_reparse_or_link(candidate):
                _fail("Chrome version-directory manifest: reparse point or symlink is forbidden")
            if candidate == path:
                continue
            if candidate.is_file():
                actual_files.add(candidate.relative_to(path.parent).as_posix())
            elif not candidate.is_dir():
                _fail("Chrome version-directory manifest: unsupported directory entry")
    except OSError as error:
        raise QualificationInputError("Chrome version directory cannot be enumerated") from error
    if actual_files != manifest_paths:
        _fail("Chrome version-directory manifest: file set is incomplete")


def verify_qualification_input_closure(
    *,
    qualification_input_root: Path,
    qualification_input_manifest: Path,
    expected_qualification_input_sha256: str,
    plan: Path,
    candidate_source_archive: Path,
    runner_path: Path | None = None,
) -> QualificationInputClosure:
    """Reopen and byte-verify the entire immutable input closure before a launch.

    The caller supplies paths for the plan/archive (and, in tests, a runner
    override) so a digest string alone can never stand in for the real bytes.
    """

    if not _SHA256.fullmatch(expected_qualification_input_sha256):
        _fail("expected qualification-input SHA-256 is invalid")
    root = _validated_root(Path(qualification_input_root))
    manifest_relative = _relative_path(
        root, Path(qualification_input_manifest), label="qualification input manifest"
    )
    manifest_path = _path_beneath(root, manifest_relative, label="qualification input manifest")
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise QualificationInputError("qualification input manifest cannot be read") from error
    actual_input_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_input_hash != expected_qualification_input_sha256:
        _fail("qualification input SHA-256 does not match immutable bytes")
    raw_document = load_strict_canonical_json(manifest_bytes, source="qualification input manifest")
    _validate_qualification_input_schema(raw_document)
    document = _expect_exact_object(
        raw_document,
        frozenset(
            {
                "schemaVersion",
                "candidate",
                "files",
                "toolIdentities",
                "expected",
                "requestedScenarioIds",
            }
        ),
        label="qualification input manifest",
    )
    if document["schemaVersion"] != 1:
        _fail("qualification input manifest: unsupported schema version")
    requested_scenarios = _expect_list(
        document["requestedScenarioIds"],
        label="qualification input manifest.requestedScenarioIds",
    )
    if tuple(requested_scenarios) != _REQUESTED_SCENARIO_IDS:
        _fail("qualification input manifest: requested scenarios are not exact ordered closure")

    files = _expect_list(document["files"], label="qualification input manifest.files")
    if len(files) != len(_FILE_ROLES):
        _fail("qualification input manifest: files role cardinality is wrong")
    direct_refs: dict[str, tuple[dict[str, Any], Path]] = {}
    direct_paths: set[str] = set()
    artifacts: dict[str, VerifiedArtifact] = {}
    previous_file_role = ""
    for index, item in enumerate(files):
        reference, path = _verify_ref(root, item, expected_role=None, label=f"files[{index}]")
        role = reference["role"]
        if (
            role not in _FILE_ROLES
            or role in direct_refs
            or role in _TOOL_ARTIFACT_ROLES.values()
            or role <= previous_file_role
        ):
            _fail(
                "qualification input manifest: files roles are not exact "
                "sorted unique closure roles"
            )
        previous_file_role = role
        if reference["relativePath"] in direct_paths:
            _fail("qualification input manifest: duplicate direct artifact path")
        direct_paths.add(reference["relativePath"])
        direct_refs[role] = (reference, path)
        _append_artifact(artifacts, logical_id=f"file:{role}", reference=reference)
    if set(direct_refs) != _FILE_ROLES:
        _fail("qualification input manifest: missing required direct artifact role")

    _verify_external_match(
        _QUALIFICATION_INPUT_SCHEMA_PATH,
        direct_refs["qualification_input_schema"][0],
        label="qualification input schema",
    )
    expected = document["expected"]
    if type(expected) is not dict:
        _fail("qualification input manifest.expected: expected object")
    for expected_key, direct_role in _EXPECTED_SCHEMA_SHA256_ROLES.items():
        if expected.get(expected_key) != direct_refs[direct_role][0]["sha256"]:
            _fail(
                f"qualification input manifest.expected.{expected_key}: "
                "digest does not bind direct schema reference"
            )

    _verify_external_match(Path(plan), direct_refs["governing_plan"][0], label="governing plan")
    _verify_external_match(
        Path(candidate_source_archive),
        direct_refs["candidate_source_archive"][0],
        label="candidate source archive",
    )
    effective_runner = Path(__file__) if runner_path is None else Path(runner_path)
    _verify_external_match(
        effective_runner,
        direct_refs["qualification_runner"][0],
        label="qualification runner",
    )

    tool_identities = _expect_list(
        document["toolIdentities"], label="qualification input manifest.toolIdentities"
    )
    if len(tool_identities) != len(_TOOL_ROLES):
        _fail("qualification input manifest: tool identity cardinality is wrong")
    seen_tools: set[str] = set()
    previous_tool = ""
    for index, item in enumerate(tool_identities):
        identity = _expect_exact_object(
            item, frozenset({"role", "version", "artifact"}), label=f"toolIdentities[{index}]"
        )
        role = _expect_string(identity["role"], label=f"toolIdentities[{index}].role")
        if role not in _TOOL_ROLES or role in seen_tools or role <= previous_tool:
            _fail("qualification input manifest: tool identities are not sorted unique")
        previous_tool = role
        seen_tools.add(role)
        _expect_string(identity["version"], label=f"toolIdentities[{index}].version")
        reference, _ = _verify_ref(
            root,
            identity["artifact"],
            expected_role=_TOOL_ARTIFACT_ROLES[role],
            label=f"toolIdentities[{index}].artifact",
        )
        if reference["relativePath"] in direct_paths:
            _fail("qualification input manifest: tool artifact duplicates direct artifact path")
        direct_paths.add(reference["relativePath"])
        _append_artifact(artifacts, logical_id=f"tool:{role}", reference=reference)
    if seen_tools != _TOOL_ROLES:
        _fail("qualification input manifest: missing tool identity")

    seen_transitive_paths: set[str] = set()
    for purpose, role in _WHEELHOUSE_ROLES.items():
        _, manifest_path_for_purpose = direct_refs[role]
        _verify_wheelhouse(
            root,
            manifest_path_for_purpose,
            purpose=purpose,
            artifacts=artifacts,
            seen_transitive_paths=seen_transitive_paths,
        )
    for provider in ("moonshine", "kokoro"):
        _, provider_path = direct_refs[f"{provider}_model_manifest"]
        _verify_provider_manifest(
            root,
            provider_path,
            provider=provider,
            artifacts=artifacts,
            seen_transitive_paths=seen_transitive_paths,
        )
    _, chrome_manifest = direct_refs["chrome_version_directory_manifest"]
    _, chrome_executable = direct_refs["chrome_executable"]
    _verify_chrome_manifest(
        root,
        chrome_manifest,
        chrome_executable,
        artifacts,
        seen_transitive_paths,
    )

    verified = tuple(artifacts[key] for key in sorted(artifacts))
    return QualificationInputClosure(
        qualification_input_sha256=actual_input_hash,
        verified_artifacts=verified,
    )


_QUALIFICATION_REPORT_SCHEMA_PATH = (
    Path(__file__).parent / "schemas" / "qualification-report-v1.schema.json"
)
_RELEASE_MANIFEST_SCHEMA_PATH = (
    Path(__file__).parent / "schemas" / "release-manifest-v1.schema.json"
)
_RELEASE_WHEELHOUSE_ROLES = (
    "build_wheelhouse_manifest",
    "hermes_runtime_wheelhouse_manifest",
    "linux_runtime_wheelhouse_manifest",
    "windows_direct_runtime_wheelhouse_manifest",
    "windows_sdist_built_runtime_wheelhouse_manifest",
)
_RELEASE_GATES = frozenset(
    {
        "windowsTests",
        "ruff",
        "mypy",
        "compileall",
        "webTestsBuild",
        "generatedStaticClean",
        "releaseGate",
        "deterministicEquivalence",
        "benchmark",
        "reproducibleBuild",
        "artifactParity",
        "cleanInstallWindows",
        "cleanInstallLinux",
        "linuxNullCapture",
        "hermesPluginManager",
        "physicalQualification",
        "uninstallRetention",
    }
)


def _validate_schema_document(document: object, *, schema_path: Path, label: str) -> None:
    """Fail closed when Draft 2020-12 validation is unavailable or rejects data."""

    try:
        jsonschema = importlib.import_module("jsonschema")
        jsonschema_exceptions = importlib.import_module("jsonschema.exceptions")
    except ImportError as error:
        raise QualificationInputError(f"{label}: schema validator is unavailable") from error
    try:
        schema_raw = schema_path.read_bytes()
    except OSError as error:
        raise QualificationInputError(f"{label}: local schema cannot be read") from error
    schema = load_strict_canonical_json(schema_raw, source=f"{label} schema")
    if type(schema) is not dict:
        _fail(f"{label}: local schema must be an object")
    try:
        validator_type = jsonschema.Draft202012Validator
        validator_type.check_schema(schema)
        errors = sorted(
            validator_type(schema).iter_errors(document), key=lambda error: str(error.path)
        )
    except jsonschema_exceptions.SchemaError as error:
        raise QualificationInputError(f"{label}: local schema is invalid") from error
    if errors:
        raise QualificationInputError(f"{label}: schema validation failed: {errors[0].message}")


def _semantic_input_context(
    *,
    qualification_input_root: Path,
    qualification_input_manifest: Path,
    expected_qualification_input_sha256: str,
    plan: Path,
    candidate_source_archive: Path,
    runner_path: Path | None,
) -> tuple[QualificationInputClosure, dict[str, Any], dict[str, dict[str, Any]], Path]:
    """Revalidate immutable inputs and expose only their already-verified metadata."""

    closure = verify_qualification_input_closure(
        qualification_input_root=qualification_input_root,
        qualification_input_manifest=qualification_input_manifest,
        expected_qualification_input_sha256=expected_qualification_input_sha256,
        plan=plan,
        candidate_source_archive=candidate_source_archive,
        runner_path=runner_path,
    )
    root = _validated_root(Path(qualification_input_root))
    manifest_relative = _relative_path(
        root, Path(qualification_input_manifest), label="qualification input manifest"
    )
    manifest_bytes = _path_beneath(
        root, manifest_relative, label="qualification input manifest"
    ).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != closure.qualification_input_sha256:
        _fail("qualification input manifest changed during semantic validation")
    document = load_strict_canonical_json(
        manifest_bytes,
        source="qualification input manifest",
    )
    if type(document) is not dict:
        _fail("qualification input manifest: expected object")
    files = document.get("files")
    if type(files) is not list:
        _fail("qualification input manifest: files must be a list")
    references: dict[str, dict[str, Any]] = {}
    for reference in files:
        if type(reference) is not dict or type(reference.get("role")) is not str:
            _fail("qualification input manifest: invalid direct reference")
        references[reference["role"]] = reference
    if set(references) != _FILE_ROLES:
        _fail("qualification input manifest: direct reference closure changed")
    return closure, document, references, root


def _expected_verified_artifacts(closure: QualificationInputClosure) -> list[dict[str, Any]]:
    return [
        {"logicalId": artifact.logical_id, "sha256": artifact.sha256, "bytes": artifact.bytes}
        for artifact in closure.verified_artifacts
    ]


def _exact(value: object, expected: object, *, label: str) -> None:
    if value != expected:
        _fail(f"{label}: does not equal its immutable input binding")


def _strict_utc(value: object, *, label: str) -> datetime:
    if type(value) is not str:
        _fail(f"{label}: expected UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise QualificationInputError(f"{label}: invalid UTC timestamp") from error
    return parsed


def _model_identity(root: Path, reference: dict[str, Any], *, provider: str) -> str:
    relative = _expect_string(reference.get("relativePath"), label=f"{provider} manifest path")
    document = load_strict_canonical_json(
        _path_beneath(root, relative, label=f"{provider} model manifest").read_bytes(),
        source=f"{provider} model manifest",
    )
    if type(document) is not dict or not _SHA256.fullmatch(
        str(document.get("modelIdentitySha256", ""))
    ):
        _fail(f"{provider} model manifest: missing canonical model identity")
    return str(document["modelIdentitySha256"])


def _validate_coupled_capacity_semantics_v1(scenarios: list[Any]) -> None:
    synthetic = [
        scenario for scenario in scenarios if scenario["scenarioId"] == "synthetic_fault_matrix"
    ]
    if len(synthetic) != 1:
        _fail("qualification report: synthetic fault scenario cardinality is wrong")
    measurements = synthetic[0]["measurements"]
    governed = {
        "maxQueueRecords": 64,
        "maxQueueCanonicalBytes": 2_097_152,
        "maxQueuePhysicalItems": 64,
        "maxCanonicalRecordBytes": 32_768,
        "queueRecordCount": 64,
    }
    for field, expected in governed.items():
        if measurements[field] != expected:
            _fail(f"qualification report: coupled-capacity {field} is not governed")
    if measurements["maxQueueCanonicalBytes"] != (
        measurements["maxQueueRecords"] * measurements["maxCanonicalRecordBytes"]
    ):
        _fail("qualification report: coupled-capacity byte limit is not the lawful product")
    if not 0 <= measurements["queueCanonicalBytes"] < measurements["maxQueueCanonicalBytes"]:
        _fail("qualification report: canonical bytes did not remain below byte capacity")
    if not 0 <= measurements["queuePhysicalCount"] <= measurements["maxQueuePhysicalItems"]:
        _fail("qualification report: physical item count exceeds its governed maximum")


def _validate_topology_semantics_v1(
    topology: dict[str, Any],
    scenarios: list[Any],
    references: dict[str, dict[str, Any]],
    input_document: dict[str, Any],
) -> None:
    jobs = _expect_list(topology["jobs"], label="qualification report topology jobs")
    expected_job_ids = [scenario["scenarioId"] for scenario in scenarios[1:]]
    if [job["scenarioId"] for job in jobs] != expected_job_ids:
        _fail("qualification report: Job records are not exact governed scenario order")
    sockets = _expect_list(
        topology["rtcSocketInventory"], label="qualification report RTC socket inventory"
    )
    socket_keys = [
        (socket["protocol"], socket["port"], socket["addressClass"]) for socket in sockets
    ]
    if (
        not sockets
        or socket_keys != sorted(socket_keys)
        or len(set(socket_keys)) != len(socket_keys)
    ):
        _fail("qualification report: RTC socket inventory is not nonempty sorted unique")
    if any(socket["addressClass"] != "loopback" for socket in sockets):
        _fail("qualification report: RTC socket inventory contains non-loopback authority")

    runner_identity = (topology["runnerPid"], topology["runnerCreationFiletime"])
    seen_identities: set[tuple[int, int]] = set()
    creation_by_pid: dict[int, int] = {}
    artifact_roles = {
        "chrome_owned_descendant": "chrome_executable",
        "livekit_owned_descendant": "livekit_executable",
        "codex_owned_descendant": "codex_executable",
    }
    build_python = next(
        (
            identity["artifact"]
            for identity in _expect_list(
                input_document["toolIdentities"], label="qualification input tool identities"
            )
            if identity.get("artifact", {}).get("role") == "build_python_executable"
        ),
        None,
    )
    if not isinstance(build_python, dict):
        _fail("qualification report: build Python executable identity is unavailable")
    role_policy = {
        "host_root": (True, build_python),
        "host_runtime": (False, build_python),
        "hermes_pluginmanager_owned_descendant": (False, build_python),
        **{role: (False, references[artifact]) for role, artifact in artifact_roles.items()},
    }
    for job in jobs:
        processes = _expect_list(
            job["processes"], label=f"qualification report topology {job['scenarioId']} processes"
        )
        process_keys = [(process["pid"], process["creationFiletime"]) for process in processes]
        if not processes or process_keys != sorted(process_keys):
            _fail("qualification report: Job process records are not nonempty sorted identities")
        job_identities = set(process_keys)
        if len(job_identities) != len(processes):
            _fail("qualification report: duplicate process identity")
        process_by_identity = {
            (process["pid"], process["creationFiletime"]): process for process in processes
        }
        root_pids = {
            process["pid"]
            for process in processes
            if process["parentPid"] == runner_identity[0]
            and process["parentCreationFiletime"] == runner_identity[1]
            and process["processGroupId"] == process["pid"]
            and process["role"] == "host_root"
        }
        if not root_pids:
            _fail("qualification report: scenario Job has no retained root process")
        for process in processes:
            if process["scenarioId"] != job["scenarioId"]:
                _fail("qualification report: process scenario does not match owning Job")
            identity = (process["pid"], process["creationFiletime"])
            if identity in seen_identities:
                _fail("qualification report: process identity appears in multiple records")
            seen_identities.add(identity)
            previous_creation = creation_by_pid.setdefault(
                process["pid"], process["creationFiletime"]
            )
            if previous_creation != process["creationFiletime"]:
                _fail("qualification report: PID was reused with a different creation identity")
            parent_identity = (
                process["parentPid"],
                process["parentCreationFiletime"],
            )
            policy = role_policy.get(process["role"])
            if policy is None:
                _fail("qualification report: process role is not closed")
            root_allowed, reference = policy
            if root_allowed:
                if (
                    parent_identity != runner_identity
                    or process["processGroupId"] != process["pid"]
                ):
                    _fail("qualification report: scenario root authority is not runner-bound")
            else:
                if parent_identity not in job_identities or parent_identity == identity:
                    _fail("qualification report: descendant parent is not a distinct Job identity")
                if process["processGroupId"] not in root_pids:
                    _fail("qualification report: descendant process group has no scenario root")
                parent = process_by_identity[parent_identity]
                if parent["creationFiletime"] > process["creationFiletime"]:
                    _fail("qualification report: child predates its retained parent identity")
            if (
                process["imageBasename"] != reference["basename"]
                or process["imageSha256"] != reference["sha256"]
            ):
                _fail("qualification report: process executable identity is not input-bound")


def validate_qualification_report(
    *,
    qualification_input_root: Path,
    qualification_input_manifest: Path,
    expected_qualification_input_sha256: str,
    plan: Path,
    candidate_source_archive: Path,
    report_bytes: bytes,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Validate report bytes without launching, emitting, or publishing anything."""

    closure, input_document, references, root = _semantic_input_context(
        qualification_input_root=qualification_input_root,
        qualification_input_manifest=qualification_input_manifest,
        expected_qualification_input_sha256=expected_qualification_input_sha256,
        plan=plan,
        candidate_source_archive=candidate_source_archive,
        runner_path=runner_path,
    )
    _verify_external_match(
        _QUALIFICATION_REPORT_SCHEMA_PATH,
        references["qualification_report_schema"],
        label="qualification report schema",
    )
    raw_report = load_strict_canonical_json(report_bytes, source="qualification report")
    _validate_schema_document(
        raw_report, schema_path=_QUALIFICATION_REPORT_SCHEMA_PATH, label="qualification report"
    )
    if type(raw_report) is not dict:
        _fail("qualification report: expected object")
    report = raw_report
    _exact(
        report.get("qualificationInputSha256"),
        closure.qualification_input_sha256,
        label="qualification report input SHA-256",
    )
    _exact(
        report.get("governingPlanSha256"),
        references["governing_plan"]["sha256"],
        label="qualification report governing plan SHA-256",
    )
    _exact(
        report.get("benchmarkReportSha256"),
        references["benchmark_report"]["sha256"],
        label="qualification report benchmark SHA-256",
    )
    _exact(
        report.get("candidate"), input_document["candidate"], label="qualification report candidate"
    )
    _exact(
        report.get("verifiedArtifacts"),
        _expected_verified_artifacts(closure),
        label="qualification report verified artifact closure",
    )

    expected = input_document["expected"]
    environment = report["environment"]
    topology = report["topology"]
    providers = report["providers"]
    for key in ("pythonFullVersion", "pythonArchitecture", "chromeVersion"):
        _exact(environment[key], expected[key], label=f"qualification report environment.{key}")
    _exact(
        environment["chromeVersionDirectoryManifestSha256"],
        references["chrome_version_directory_manifest"]["sha256"],
        label="qualification report chrome manifest",
    )
    _exact(
        topology["livekitVersion"],
        expected["livekitVersion"],
        label="qualification report topology livekit version",
    )
    provider_bindings = {
        "codexExecutableVersion": expected["codexVersion"],
        "codexExecutableSha256": references["codex_executable"]["sha256"],
        "codexModel": expected["codexModel"],
        "codexEffort": expected["codexEffort"],
        "moonshineDistributionVersion": expected["moonshineVersion"],
        "moonshineDistributionSha256": references["moonshine_distribution"]["sha256"],
        "moonshineModelIdentitySha256": _model_identity(
            root, references["moonshine_model_manifest"], provider="moonshine"
        ),
        "kokoroDistributionVersion": expected["kokoroVersion"],
        "kokoroDistributionSha256": references["kokoro_distribution"]["sha256"],
        "kokoroModelIdentitySha256": _model_identity(
            root, references["kokoro_model_manifest"], provider="kokoro"
        ),
    }
    for key, value in provider_bindings.items():
        _exact(providers[key], value, label=f"qualification report providers.{key}")

    scenarios = _expect_list(report["scenarios"], label="qualification report scenarios")
    attestations = _expect_list(
        report["operatorAttestations"], label="qualification report operator attestations"
    )
    attestation_ids = [item["attestationId"] for item in attestations]
    if attestation_ids != sorted(set(attestation_ids)):
        _fail("qualification report operator attestations are not sorted unique")
    referenced_ids: list[str] = []
    scenarios_pass = True
    for scenario in scenarios:
        cases = _expect_list(scenario["caseResults"], label="qualification report scenario cases")
        ids = _expect_list(
            scenario["attestationIds"], label="qualification report scenario attestations"
        )
        if ids != sorted(set(ids)):
            _fail("qualification report scenario attestation IDs are not sorted unique")
        referenced_ids.extend(ids)
        scenarios_pass = (
            scenarios_pass and scenario["outcome"] == "pass" and scenario["exitCode"] == 0
        )
        scenarios_pass = scenarios_pass and all(case["outcome"] == "pass" for case in cases)
    if sorted(referenced_ids) != attestation_ids:
        _fail("qualification report attestations are not reference-closed")
    _validate_coupled_capacity_semantics_v1(scenarios)
    _validate_topology_semantics_v1(topology, scenarios, references, input_document)
    attestations_pass = all(item["confirmed"] is True for item in attestations)
    jobs = _expect_list(topology["jobs"], label="qualification report topology jobs")
    topology_pass = (
        topology["signalingLoopbackOnly"] is True
        and topology["rtcSocketInventoryRecorded"] is True
        and all(job["activeProcessCountAfterCleanup"] == 0 for job in jobs)
    )
    cleanup = _expect_exact_object(
        report["cleanup"], frozenset(report["cleanup"].keys()), label="qualification report cleanup"
    )
    cleanup_pass = all(value is True for value in cleanup.values())
    derived_passed = scenarios_pass and attestations_pass and topology_pass and cleanup_pass
    _exact(report["passed"], derived_passed, label="qualification report passed derivation")
    if _strict_utc(
        report["startedAtUtc"], label="qualification report startedAtUtc"
    ) >= _strict_utc(report["finishedAtUtc"], label="qualification report finishedAtUtc"):
        _fail("qualification report timestamps are not strictly ordered")
    _semantic_input_context(
        qualification_input_root=qualification_input_root,
        qualification_input_manifest=qualification_input_manifest,
        expected_qualification_input_sha256=expected_qualification_input_sha256,
        plan=plan,
        candidate_source_archive=candidate_source_archive,
        runner_path=runner_path,
    )
    _verify_external_match(
        _QUALIFICATION_REPORT_SCHEMA_PATH,
        references["qualification_report_schema"],
        label="qualification report schema",
    )
    return report


def validate_release_manifest(
    *,
    qualification_input_root: Path,
    qualification_input_manifest: Path,
    expected_qualification_input_sha256: str,
    plan: Path,
    candidate_source_archive: Path,
    qualification_report_bytes: bytes,
    release_manifest_bytes: bytes,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    """Validate a release manifest as a pure, fail-closed semantic operation."""

    report = validate_qualification_report(
        qualification_input_root=qualification_input_root,
        qualification_input_manifest=qualification_input_manifest,
        expected_qualification_input_sha256=expected_qualification_input_sha256,
        plan=plan,
        candidate_source_archive=candidate_source_archive,
        report_bytes=qualification_report_bytes,
        runner_path=runner_path,
    )
    closure, input_document, references, _ = _semantic_input_context(
        qualification_input_root=qualification_input_root,
        qualification_input_manifest=qualification_input_manifest,
        expected_qualification_input_sha256=expected_qualification_input_sha256,
        plan=plan,
        candidate_source_archive=candidate_source_archive,
        runner_path=runner_path,
    )
    _verify_external_match(
        _RELEASE_MANIFEST_SCHEMA_PATH,
        references["release_manifest_schema"],
        label="release manifest schema",
    )
    raw_manifest = load_strict_canonical_json(release_manifest_bytes, source="release manifest")
    _validate_schema_document(
        raw_manifest, schema_path=_RELEASE_MANIFEST_SCHEMA_PATH, label="release manifest"
    )
    if type(raw_manifest) is not dict:
        _fail("release manifest: expected object")
    manifest = raw_manifest
    _exact(manifest["candidate"], input_document["candidate"], label="release manifest candidate")
    _exact(
        manifest["sourceArchive"],
        references["candidate_source_archive"],
        label="release manifest source archive",
    )
    _exact(
        manifest["artifacts"],
        _expected_verified_artifacts(closure),
        label="release manifest artifact closure",
    )
    _exact(
        manifest["wheelhouseManifests"],
        [references[role] for role in _RELEASE_WHEELHOUSE_ROLES],
        label="release manifest wheelhouse closure",
    )
    _exact(
        manifest["qualificationInputSha256"],
        closure.qualification_input_sha256,
        label="release manifest input SHA-256",
    )
    _exact(
        manifest["qualificationReportSha256"],
        hashlib.sha256(qualification_report_bytes).hexdigest(),
        label="release manifest report SHA-256",
    )
    _exact(
        manifest["benchmarkReportSha256"],
        references["benchmark_report"]["sha256"],
        label="release manifest benchmark SHA-256",
    )
    environment = manifest["environment"]
    expected = input_document["expected"]
    for key in (
        "windowsEdition",
        "windowsBuild",
        "windowsArchitecture",
        "pythonFullVersion",
        "pythonArchitecture",
        "sqliteVersion",
    ):
        _exact(
            environment[key],
            report["environment"][key],
            label=f"release manifest environment.{key}",
        )
    _exact(
        environment["pythonFullVersion"],
        expected["pythonFullVersion"],
        label="release manifest environment.pythonFullVersion input binding",
    )
    _exact(
        environment["pythonArchitecture"],
        expected["pythonArchitecture"],
        label="release manifest environment.pythonArchitecture input binding",
    )
    _exact(
        environment["toolIdentities"],
        input_document["toolIdentities"],
        label="release manifest tool identities",
    )
    gates = (
        environment  # keep all release observations schema-bound; no callback result is trusted.
    )
    del gates
    gate_results = manifest["gateResults"]
    if set(gate_results) != _RELEASE_GATES:
        _fail("release manifest gate rows are not the exact 17-gate closure")
    gates_pass = all(result["outcome"] == "pass" for result in gate_results.values())
    _exact(
        manifest["passed"],
        report["passed"] and gates_pass,
        label="release manifest passed derivation",
    )
    _semantic_input_context(
        qualification_input_root=qualification_input_root,
        qualification_input_manifest=qualification_input_manifest,
        expected_qualification_input_sha256=expected_qualification_input_sha256,
        plan=plan,
        candidate_source_archive=candidate_source_archive,
        runner_path=runner_path,
    )
    _verify_external_match(
        _RELEASE_MANIFEST_SCHEMA_PATH,
        references["release_manifest_schema"],
        label="release manifest schema",
    )
    return manifest


# Private t12-3 Windows Job/process-authority machinery.  This intentionally has
# no producer, report, publication, CLI, or physical-execution entry point.
_CREATE_SUSPENDED = 0x00000004
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x00001000
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200


class _JOBOBJECT_BASIC_LIMIT_INFORMATION_V1(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IO_COUNTERS_V1(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION_V1),
        ("IoInfo", _IO_COUNTERS_V1),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION_V1(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", ctypes.c_uint32),
        ("TotalProcesses", ctypes.c_uint32),
        ("ActiveProcesses", ctypes.c_uint32),
        ("TotalTerminatedProcesses", ctypes.c_uint32),
    ]


class _JOBOBJECT_BASIC_PROCESS_ID_LIST_HEADER_V1(ctypes.Structure):
    _fields_ = [
        ("NumberOfAssignedProcesses", ctypes.c_uint32),
        ("NumberOfProcessIdsInList", ctypes.c_uint32),
    ]


def _job_pid_list_buffer_bytes_v1(max_active_processes: int) -> int:
    if type(max_active_processes) is not int or not 1 <= max_active_processes <= 4_096:
        _windows_fail("Job PID-list limit is outside the closed bounded range")
    return ctypes.sizeof(_JOBOBJECT_BASIC_PROCESS_ID_LIST_HEADER_V1) + (
        max_active_processes * ctypes.sizeof(ctypes.c_size_t)
    )


class _WindowsScenarioJobError(RuntimeError):
    """A private scenario process authority cannot prove its closed lifecycle."""


class _WindowsPlatformError(_WindowsScenarioJobError):
    """The concrete Win32 adapter was used outside Windows."""


@dataclass(frozen=True, slots=True)
class _WindowsJobLimitsV1:
    max_active_processes: int
    max_process_memory_bytes: int
    max_job_memory_bytes: int


@dataclass(frozen=True, slots=True)
class _WindowsScenarioSpecV1:
    scenario_id: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    working_directory: str
    timeout_milliseconds: int
    inherited_handles: tuple[int, ...]
    limits: _WindowsJobLimitsV1
    no_window: bool = False


@dataclass(frozen=True, slots=True)
class _WindowsRunnerIdentityV1:
    pid: int
    creation_filetime: int


@dataclass(frozen=True, slots=True)
class _WindowsKernelLaunchV1:
    pid: int
    process_handle: int
    thread_handle: int


@dataclass(frozen=True, slots=True)
class _WindowsKernelProcessV1:
    pid: int
    parent_pid: int
    parent_creation_filetime: int
    creation_filetime: int
    image_basename: str
    image_sha256: str
    scenario_id: str | None = None


@dataclass(frozen=True, slots=True)
class _WindowsRoleRuleV1:
    role: str
    image_basename: str
    image_sha256: str
    parent_roles: frozenset[str]
    root: bool


@dataclass(frozen=True, slots=True)
class _WindowsProcessIdentityV1:
    scenario_id: str
    pid: int
    parent_pid: int
    parent_creation_filetime: int
    creation_filetime: int
    image_basename: str
    image_sha256: str


@dataclass(frozen=True, slots=True)
class _WindowsBoundProcessV1:
    identity: _WindowsProcessIdentityV1
    role: str
    process_handle: int
    thread_handle: int | None
    root: bool


@dataclass(frozen=True, slots=True)
class _WindowsMembershipSnapshotV1:
    checkpoint: str
    members: tuple[_WindowsBoundProcessV1, ...]


@dataclass(frozen=True, slots=True)
class _WindowsOperationFailureV1:
    operation: str
    handle: int | None
    message: str


@dataclass(frozen=True, slots=True)
class _WindowsFinalizationResultV1:
    pre_cleanup_snapshot: _WindowsMembershipSnapshotV1 | None
    terminated_job: bool
    waited_handles: tuple[int, ...]
    zero_active_observed: bool
    closed_handles: tuple[int, ...]
    failed_handles: tuple[int, ...]
    failures: tuple[_WindowsOperationFailureV1, ...]
    closed: bool


class _WindowsFinalizationError(_WindowsScenarioJobError):
    def __init__(
        self,
        result: _WindowsFinalizationResultV1,
        owner: _WindowsScenarioJobV1 | None = None,
    ) -> None:
        self.result = result
        self.owner = owner
        super().__init__("Windows scenario finalization left retryable retained authority")


def _windows_fail(message: str) -> NoReturn:
    raise _WindowsScenarioJobError(message)


def _valid_windows_handle_v1(value: object) -> TypeGuard[int]:
    return type(value) is int and value > 0 and value != ctypes.c_void_p(-1).value


def _validate_windows_spec_v1(spec: _WindowsScenarioSpecV1) -> None:
    if not isinstance(spec, _WindowsScenarioSpecV1):
        _windows_fail("scenario spec has the wrong closed DTO type")
    if type(spec.no_window) is not bool:
        _windows_fail("scenario window policy must be an exact boolean")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", spec.scenario_id):
        _windows_fail("scenario ID is not a closed safe identifier")
    if (
        not isinstance(spec.command, tuple)
        or not spec.command
        or any(not isinstance(value, str) or not value or "\x00" in value for value in spec.command)
    ):
        _windows_fail("command is not a nonempty NUL-free exact tuple")
    if not isinstance(spec.environment, tuple):
        _windows_fail("environment is not an exact tuple")
    names: set[str] = set()
    for item in spec.environment:
        if not isinstance(item, tuple) or len(item) != 2:
            _windows_fail("environment item is not a name/value pair")
        name, value = item
        canonical_name = name.casefold() if isinstance(name, str) else ""
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or not name
            or "=" in name
            or "\x00" in name
            or "\x00" in value
            or canonical_name in names
        ):
            _windows_fail("environment is duplicate or not Win32-safe")
        names.add(canonical_name)
    if (
        tuple(sorted(spec.environment, key=lambda item: (item[0].casefold(), item[0])))
        != spec.environment
    ):
        _windows_fail("environment pairs must be deterministic Windows name order")
    if not isinstance(spec.working_directory, str) or not re.fullmatch(
        r"[A-Za-z]:\\[^\x00]*", spec.working_directory
    ):
        _windows_fail("working directory is not an absolute Windows path")
    working_directory = PureWindowsPath(spec.working_directory)
    if any(
        part in {".", ".."}
        or part.endswith((" ", "."))
        or any(ord(character) < 32 or ord(character) == 127 for character in part)
        for part in working_directory.parts[1:]
    ):
        _windows_fail("working directory contains an unsafe Windows path component")
    if type(spec.timeout_milliseconds) is not int or not 1 <= spec.timeout_milliseconds <= 600_000:
        _windows_fail("timeout is outside the closed bounded range")
    if (
        type(spec.inherited_handles) is not tuple
        or not spec.inherited_handles
        or any(not _valid_windows_handle_v1(handle) for handle in spec.inherited_handles)
        or len(set(spec.inherited_handles)) != len(spec.inherited_handles)
    ):
        _windows_fail("inherited handles are not an exact unique positive allowlist")
    limits = spec.limits
    if (
        not isinstance(limits, _WindowsJobLimitsV1)
        or any(
            type(value) is not int or value <= 0
            for value in (
                limits.max_active_processes,
                limits.max_process_memory_bytes,
                limits.max_job_memory_bytes,
            )
        )
        or not 1 <= limits.max_active_processes <= 4_096
        or limits.max_process_memory_bytes > limits.max_job_memory_bytes
        or limits.max_job_memory_bytes > (1 << 63) - 1
    ):
        _windows_fail("Job limits are invalid or internally inconsistent")


def _validate_kernel_identity_v1(value: _WindowsKernelProcessV1) -> None:
    if (
        not isinstance(value, _WindowsKernelProcessV1)
        or type(value.pid) is not int
        or value.pid <= 0
        or type(value.parent_pid) is not int
        or value.parent_pid <= 0
        or type(value.parent_creation_filetime) is not int
        or value.parent_creation_filetime <= 0
        or type(value.creation_filetime) is not int
        or value.creation_filetime <= 0
        or not re.fullmatch(r"[^\\/:*?\"<>|\x00]+", value.image_basename)
        or not _SHA256.fullmatch(value.image_sha256)
        or (
            value.scenario_id is not None
            and not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value.scenario_id)
        )
    ):
        _windows_fail("kernel returned an invalid process identity")


class _WindowsScenarioJobV1:
    """Single private owner of one exact Job, its roots, and observed descendants."""

    def __init__(
        self,
        kernel: Any,
        spec: _WindowsScenarioSpecV1,
        runner_identity: _WindowsRunnerIdentityV1,
        role_rules: tuple[_WindowsRoleRuleV1, ...],
    ) -> None:
        _validate_windows_spec_v1(spec)
        if (
            not isinstance(runner_identity, _WindowsRunnerIdentityV1)
            or type(runner_identity.pid) is not int
            or type(runner_identity.creation_filetime) is not int
            or runner_identity.pid <= 0
            or runner_identity.creation_filetime <= 0
        ):
            _windows_fail("runner PID/creation FILETIME identity is invalid")
        if not isinstance(role_rules, tuple) or not role_rules:
            _windows_fail("role rules are required before native action")
        roles: set[str] = set()
        root_rules = 0
        for rule in role_rules:
            if (
                not isinstance(rule, _WindowsRoleRuleV1)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", rule.role)
                or rule.role in roles
                or type(rule.root) is not bool
                or not re.fullmatch(r"[^\\/:*?\"<>|\x00]+", rule.image_basename)
                or not _SHA256.fullmatch(rule.image_sha256)
                or not isinstance(rule.parent_roles, frozenset)
            ):
                _windows_fail("role rules are not closed and valid")
            roles.add(rule.role)
            root_rules += int(rule.root)
        if root_rules != 1 or any(
            not rule.root and not rule.parent_roles <= roles for rule in role_rules
        ):
            _windows_fail("role rules do not define one closed root and descendants")
        self._kernel = kernel
        self._spec = spec
        self._runner_identity = runner_identity
        self._rules = role_rules
        self._job_handle: int | None = None
        self._members: dict[int, _WindowsBoundProcessV1] = {}
        self._identities: dict[int, _WindowsProcessIdentityV1] = {}
        self._pending_process_handles: dict[int, int] = {}
        self._owned_handles: dict[int, str] = {}
        self._directly_terminated_handles: set[int] = set()
        self._unassigned_process_handles: set[int] = set()
        self._job_assigned = False
        self._zero_active_observed = False
        self._last_finalization: _WindowsFinalizationResultV1 | None = None

    @property
    def last_finalization(self) -> _WindowsFinalizationResultV1 | None:
        return self._last_finalization

    def _ensure_job(self) -> int:
        if self._job_handle is None:
            handle = self._kernel.create_job(self._spec.limits)
            if not _valid_windows_handle_v1(handle):
                _windows_fail("kernel returned an invalid Job handle")
            self._job_handle = handle
        return self._job_handle

    def _bind(
        self,
        raw: _WindowsKernelProcessV1,
        process_handle: int,
        thread_handle: int | None,
        root: bool,
    ) -> _WindowsBoundProcessV1:
        _validate_kernel_identity_v1(raw)
        if raw.scenario_id is not None and raw.scenario_id != self._spec.scenario_id:
            _windows_fail("member is bound to another scenario")
        identity = _WindowsProcessIdentityV1(
            self._spec.scenario_id,
            raw.pid,
            raw.parent_pid,
            raw.parent_creation_filetime,
            raw.creation_filetime,
            raw.image_basename,
            raw.image_sha256,
        )
        prior = self._identities.get(raw.pid)
        if prior is not None and prior != identity:
            _windows_fail("PID creation-time or immutable identity conflict")
        applicable = [
            rule
            for rule in self._rules
            if rule.image_basename == raw.image_basename
            and rule.image_sha256 == raw.image_sha256
            and rule.root == root
        ]
        if len(applicable) != 1:
            _windows_fail(
                f"member image cannot be classified into a closed role: {raw.image_basename}"
            )
        rule = applicable[0]
        if root:
            if (raw.parent_pid, raw.parent_creation_filetime) != (
                self._runner_identity.pid,
                self._runner_identity.creation_filetime,
            ):
                _windows_fail("root parent is not the retained runner identity")
        else:
            parent = self._members.get(raw.parent_pid)
            if (
                parent is None
                or (parent.identity.pid, parent.identity.creation_filetime)
                != (raw.parent_pid, raw.parent_creation_filetime)
                or parent.role not in rule.parent_roles
            ):
                _windows_fail("descendant parent identity is unresolved or disallowed")
        existing = self._members.get(raw.pid)
        if existing is not None and (existing.identity != identity or existing.role != rule.role):
            _windows_fail("duplicate member has a conflicting identity or role")
        bound = existing or _WindowsBoundProcessV1(
            identity, rule.role, process_handle, thread_handle, root
        )
        self._identities[raw.pid] = identity
        self._members[raw.pid] = bound
        self._owned_handles.setdefault(process_handle, "process")
        if thread_handle is not None:
            self._owned_handles.setdefault(thread_handle, "thread")
        return bound

    def launch_root(self) -> _WindowsBoundProcessV1:
        flags = (
            _CREATE_SUSPENDED
            | _CREATE_NEW_PROCESS_GROUP
            | _CREATE_UNICODE_ENVIRONMENT
            | _EXTENDED_STARTUPINFO_PRESENT
            | (_CREATE_NO_WINDOW if self._spec.no_window else 0)
        )
        try:
            job = self._ensure_job()
            launch = self._kernel.create_process_suspended(
                self._spec.command,
                self._spec.environment,
                self._spec.working_directory,
                self._spec.inherited_handles,
                flags,
            )
            process_handle = getattr(launch, "process_handle", None)
            thread_handle = getattr(launch, "thread_handle", None)
            pid = getattr(launch, "pid", None)
            if (
                _valid_windows_handle_v1(process_handle)
                and process_handle not in self._owned_handles
            ):
                self._owned_handles[process_handle] = "process"
                self._unassigned_process_handles.add(process_handle)
            if _valid_windows_handle_v1(thread_handle) and thread_handle not in self._owned_handles:
                self._owned_handles[thread_handle] = "thread"
            if (
                not isinstance(launch, _WindowsKernelLaunchV1)
                or type(pid) is not int
                or pid <= 0
                or not _valid_windows_handle_v1(process_handle)
                or not _valid_windows_handle_v1(thread_handle)
                or process_handle == thread_handle
                or self._owned_handles.get(process_handle) != "process"
                or self._owned_handles.get(thread_handle) != "thread"
            ):
                _windows_fail("kernel returned invalid CreateProcessW handles")
            self._pending_process_handles[pid] = process_handle
            raw = self._kernel.query_process_identity(launch.process_handle)
            if raw.pid != launch.pid:
                _windows_fail("CreateProcess PID does not match retained process handle")
            root = self._bind(raw, launch.process_handle, launch.thread_handle, True)
            self._kernel.assign_process_to_job(job, launch.process_handle)
            self._job_assigned = True
            self._pending_process_handles.pop(launch.pid, None)
            self._unassigned_process_handles.discard(launch.process_handle)
        except BaseException as primary:
            try:
                self.finalize()
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "root launch and retained-authority cleanup both failed",
                    [primary, cleanup],
                ) from None
            raise
        try:
            self._kernel.resume_thread(launch.thread_handle)
        except BaseException as primary:
            try:
                self.finalize()
            except _WindowsFinalizationError as cleanup:
                raise BaseExceptionGroup(
                    "root resume and finalization both failed", [primary, cleanup]
                ) from None
            raise
        return root

    def checkpoint(self, label: str) -> _WindowsMembershipSnapshotV1:
        if not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", label):
            _windows_fail("checkpoint label is invalid")
        if self._job_handle is None:
            _windows_fail("cannot snapshot an uncreated Job")
        pids = self._kernel.query_job_processes(self._job_handle)
        if (
            not isinstance(pids, tuple)
            or any(type(pid) is not int or pid <= 0 for pid in pids)
            or len(set(pids)) != len(pids)
        ):
            _windows_fail("Job membership query is not a complete unique PID set")
        current = set(pids)
        pending: dict[int, tuple[_WindowsKernelProcessV1, int]] = {}
        # Capture new lifetimes before potentially expensive rechecks of known
        # images. Their existing retained handles already preserve identity.
        for pid in sorted(pids, key=lambda value: value in self._members):
            existing = self._members.get(pid)
            if existing is None:
                handle = self._pending_process_handles.get(pid)
                if handle is None:
                    handle = self._kernel.open_process(
                        pid, _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION
                    )
                    if not _valid_windows_handle_v1(handle) or handle in self._owned_handles:
                        _windows_fail(
                            "new Job member did not yield a unique retained process handle"
                        )
                    self._owned_handles[handle] = "process"
                    self._pending_process_handles[pid] = handle
                raw = self._kernel.query_process_identity(handle)
                _validate_kernel_identity_v1(raw)
                if raw.pid != pid:
                    _windows_fail("Job member PID does not match its retained process handle")
                pending[pid] = (raw, handle)
            else:
                raw = self._kernel.query_process_identity(existing.process_handle)
                self._bind(raw, existing.process_handle, existing.thread_handle, existing.root)
        while pending:
            progressed = False
            for pid, (raw, handle) in tuple(pending.items()):
                if raw.parent_pid not in self._members:
                    continue
                self._bind(raw, handle, None, False)
                self._pending_process_handles.pop(pid, None)
                pending.pop(pid)
                progressed = True
            if not progressed:
                _windows_fail("Job membership contains an unresolved parent identity graph")
        for member in self._members.values():
            if (
                member.identity.pid in current
                and not member.root
                and member.identity.parent_pid not in current
            ):
                parent = self._members.get(member.identity.parent_pid)
                if label != "pre-cleanup" or parent is None:
                    _windows_fail("observed descendant parent is outside the Job")
                # A retained exited root can leave a console helper alive for
                # final Job termination. Require the exact parent handle to be
                # signaled; missing membership alone never proves parent exit.
                self._kernel.wait(parent.process_handle, 0)
        return _WindowsMembershipSnapshotV1(
            label, tuple(self._members[pid] for pid in sorted(current))
        )

    def retain_auxiliary_handle(self, handle: int, kind: str) -> None:
        if (
            not _valid_windows_handle_v1(handle)
            or kind not in {"stdout", "console"}
            or handle in self._owned_handles
        ):
            _windows_fail("auxiliary handle ownership is invalid")
        self._owned_handles[handle] = kind

    def finalize(self) -> _WindowsFinalizationResultV1:
        failures: list[_WindowsOperationFailureV1] = []
        deferred_handles: set[int] = set()
        defer_job_close = False
        closed_handles: list[int] = []
        waited: list[int] = []
        snapshot: _WindowsMembershipSnapshotV1 | None = None
        terminated = False
        if self._job_handle is not None and not self._zero_active_observed:
            try:
                snapshot = self.checkpoint("pre-cleanup")
            except BaseException as error:
                defer_job_close = True
                failures.append(
                    _WindowsOperationFailureV1("pre_cleanup_snapshot", self._job_handle, str(error))
                )
        if self._job_handle is not None and self._unassigned_process_handles:
            for handle in tuple(sorted(self._unassigned_process_handles)):
                if handle in self._directly_terminated_handles:
                    continue
                try:
                    self._kernel.terminate_process(handle)
                    self._directly_terminated_handles.add(handle)
                except BaseException as error:
                    defer_job_close = True
                    deferred_handles.add(handle)
                    failures.append(
                        _WindowsOperationFailureV1("terminate_process", handle, str(error))
                    )
        if self._job_handle is not None:
            try:
                self._kernel.terminate_job(self._job_handle)
                terminated = True
            except BaseException as error:
                defer_job_close = True
                failures.append(
                    _WindowsOperationFailureV1("terminate_job", self._job_handle, str(error))
                )
        for handle, kind in tuple(self._owned_handles.items()):
            if kind not in {"process", "thread"}:
                continue
            try:
                self._kernel.wait(handle, self._spec.timeout_milliseconds)
                waited.append(handle)
            except BaseException as error:
                deferred_handles.add(handle)
                failures.append(_WindowsOperationFailureV1("wait", handle, str(error)))
        for member in tuple(self._members.values()):
            if member.process_handle not in self._owned_handles:
                continue
            try:
                raw = self._kernel.query_process_identity(member.process_handle)
                _validate_kernel_identity_v1(raw)
                if (
                    _WindowsProcessIdentityV1(
                        self._spec.scenario_id,
                        raw.pid,
                        raw.parent_pid,
                        raw.parent_creation_filetime,
                        raw.creation_filetime,
                        raw.image_basename,
                        raw.image_sha256,
                    )
                    != member.identity
                ):
                    _windows_fail("retained process identity changed during finalization")
            except BaseException as error:
                deferred_handles.add(member.process_handle)
                failures.append(
                    _WindowsOperationFailureV1("identity", member.process_handle, str(error))
                )
        if self._job_handle is not None and not self._zero_active_observed:
            try:
                active_count = self._kernel.query_job_active_process_count(self._job_handle)
                if type(active_count) is not int or active_count != 0:
                    _windows_fail("Job active-process count is not proven zero")
                self._zero_active_observed = True
            except BaseException as error:
                defer_job_close = True
                failures.append(
                    _WindowsOperationFailureV1("zero_active", self._job_handle, str(error))
                )
        close_order = tuple(
            handle
            for handle, kind in self._owned_handles.items()
            if kind in {"thread", "process", "stdout", "console"}
        )
        for handle in close_order:
            if handle in deferred_handles:
                continue
            try:
                self._kernel.close_handle(handle)
                closed_handles.append(handle)
                self._owned_handles.pop(handle, None)
                self._directly_terminated_handles.discard(handle)
                self._unassigned_process_handles.discard(handle)
                for pid, pending_handle in tuple(self._pending_process_handles.items()):
                    if pending_handle == handle:
                        self._pending_process_handles.pop(pid, None)
            except BaseException as error:
                failures.append(_WindowsOperationFailureV1("close", handle, str(error)))
        if self._job_handle is not None and not defer_job_close:
            job = self._job_handle
            try:
                self._kernel.close_handle(job)
                closed_handles.append(job)
                self._job_handle = None
            except BaseException as error:
                failures.append(_WindowsOperationFailureV1("close", job, str(error)))
        finalize_transients = getattr(self._kernel, "finalize_transient_handles", None)
        if callable(finalize_transients):
            try:
                finalize_transients()
            except BaseException as error:
                failures.append(_WindowsOperationFailureV1("transient_close", None, str(error)))
        failed_handles = tuple(
            sorted(
                {
                    failure.handle
                    for failure in failures
                    if failure.handle is not None
                    and (
                        failure.handle in self._owned_handles or failure.handle == self._job_handle
                    )
                }
            )
        )
        result = _WindowsFinalizationResultV1(
            snapshot,
            terminated,
            tuple(waited),
            self._zero_active_observed,
            tuple(closed_handles),
            failed_handles,
            tuple(failures),
            self._job_handle is None and not self._owned_handles,
        )
        self._last_finalization = result
        if failures:
            raise _WindowsFinalizationError(result, self)
        return result


def _run_with_windows_scenario_job_finalization_v1(
    job: _WindowsScenarioJobV1, body: Callable[[], Any]
) -> Any:
    """Private outer owner helper that preserves a primary and finalization failure."""
    primary: BaseException | None = None
    value: Any = None
    try:
        value = body()
    except BaseException as error:
        primary = error
    try:
        job.finalize()
    except _WindowsFinalizationError as finalization:
        if primary is not None:
            raise BaseExceptionGroup(
                "scenario body and finalization both failed", [primary, finalization]
            ) from None
        raise
    if primary is not None:
        raise primary
    return value


def _configure_windows_api_v1(api: Any) -> None:
    handle = ctypes.c_void_p
    pointer = ctypes.c_void_p
    dword = ctypes.c_uint32
    boolean = ctypes.c_int32
    size_t_pointer = ctypes.POINTER(ctypes.c_size_t)
    dword_pointer = ctypes.POINTER(dword)
    prototypes: dict[str, tuple[list[Any], Any]] = {
        "CreateJobObjectW": ([pointer, ctypes.c_wchar_p], handle),
        "SetInformationJobObject": ([handle, ctypes.c_int, pointer, dword], boolean),
        "CloseHandle": ([handle], boolean),
        "InitializeProcThreadAttributeList": (
            [pointer, dword, dword, size_t_pointer],
            boolean,
        ),
        "UpdateProcThreadAttribute": (
            [
                pointer,
                ctypes.c_size_t,
                ctypes.c_size_t,
                pointer,
                ctypes.c_size_t,
                pointer,
                size_t_pointer,
            ],
            boolean,
        ),
        "DeleteProcThreadAttributeList": ([pointer], None),
        "CreateProcessW": (
            [
                ctypes.c_wchar_p,
                ctypes.c_wchar_p,
                pointer,
                pointer,
                boolean,
                dword,
                pointer,
                ctypes.c_wchar_p,
                pointer,
                pointer,
            ],
            boolean,
        ),
        "AssignProcessToJobObject": ([handle, handle], boolean),
        "ResumeThread": ([handle], dword),
        "CreateToolhelp32Snapshot": ([dword, dword], handle),
        "Process32FirstW": ([handle, pointer], boolean),
        "Process32NextW": ([handle, pointer], boolean),
        "GetProcessTimes": ([handle, pointer, pointer, pointer, pointer], boolean),
        "QueryFullProcessImageNameW": ([handle, dword, ctypes.c_wchar_p, dword_pointer], boolean),
        "GetProcessId": ([handle], dword),
        "QueryInformationJobObject": (
            [handle, ctypes.c_int, pointer, dword, dword_pointer],
            boolean,
        ),
        "OpenProcess": ([dword, boolean, dword], handle),
        "TerminateProcess": ([handle, dword], boolean),
        "TerminateJobObject": ([handle, dword], boolean),
        "WaitForSingleObject": ([handle, dword], dword),
    }
    for name, (argtypes, restype) in prototypes.items():
        function = getattr(api, name)
        function.argtypes = argtypes
        function.restype = restype


class _PROCESS_INFORMATION_V1(ctypes.Structure):
    _fields_ = [
        ("hProcess", ctypes.c_void_p),
        ("hThread", ctypes.c_void_p),
        ("dwProcessId", ctypes.c_uint32),
        ("dwThreadId", ctypes.c_uint32),
    ]


class _CtypesWindowsKernelV1:
    """Real lazy ctypes boundary; import-safe everywhere and operational only on Windows."""

    def __init__(self, *, platform: str | None = None) -> None:
        self._platform = os.name if platform is None else platform
        self._kernel32: Any | None = None
        self._job_max_active_processes: dict[int, int] = {}
        self._retryable_handles: set[int] = set()
        self._retained_process_identities: dict[int, _WindowsKernelProcessV1] = {}

    @property
    def retryable_handles(self) -> tuple[int, ...]:
        return tuple(sorted(self._retryable_handles))

    def _close_or_retain(self, handle: int) -> None:
        try:
            if not self._api().CloseHandle(ctypes.c_void_p(handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self._retryable_handles.add(handle)
            raise
        self._retryable_handles.discard(handle)
        self._job_max_active_processes.pop(handle, None)
        self._retained_process_identities.pop(handle, None)

    def finalize_transient_handles(self) -> None:
        failures: list[BaseException] = []
        for handle in tuple(sorted(self._retryable_handles)):
            try:
                self._close_or_retain(handle)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("Win32 transient handles remain retryable", failures)

    def _api(self) -> Any:
        if self._platform != "nt":
            raise _WindowsPlatformError("Win32 process authority is unavailable off Windows")
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise _WindowsPlatformError("Task 12 Win32 process authority requires 64-bit Python")
        if self._kernel32 is None:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            _configure_windows_api_v1(self._kernel32)
        return self._kernel32

    @staticmethod
    def _handle(value: Any) -> int:
        if not _valid_windows_handle_v1(value):
            raise ctypes.WinError(ctypes.get_last_error())
        return value

    def create_job(self, limits: _WindowsJobLimitsV1) -> int:
        api = self._api()
        api.CreateJobObjectW.restype = ctypes.c_void_p
        job = self._handle(api.CreateJobObjectW(None, None))
        info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_V1()
        info.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            | _JOB_OBJECT_LIMIT_ACTIVE_PROCESS
            | _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            | _JOB_OBJECT_LIMIT_JOB_MEMORY
        )
        info.BasicLimitInformation.ActiveProcessLimit = limits.max_active_processes
        info.ProcessMemoryLimit = limits.max_process_memory_bytes
        info.JobMemoryLimit = limits.max_job_memory_bytes
        if not api.SetInformationJobObject(
            ctypes.c_void_p(job), 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            primary = ctypes.WinError(ctypes.get_last_error())
            try:
                self._close_or_retain(job)
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "Job configuration and close both failed", [primary, cleanup]
                ) from None
            raise primary
        self._job_max_active_processes[job] = limits.max_active_processes
        return job

    def create_process_suspended(
        self,
        command: tuple[str, ...],
        environment: tuple[tuple[str, str], ...],
        cwd: str,
        handles: tuple[int, ...],
        flags: int,
    ) -> _WindowsKernelLaunchV1:
        if (
            type(handles) is not tuple
            or not handles
            or any(not _valid_windows_handle_v1(handle) for handle in handles)
            or len(set(handles)) != len(handles)
        ):
            _windows_fail("concrete inherited handles are not an exact valid allowlist")
        api = self._api()

        class _STARTUPINFO(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("lpReserved", ctypes.c_void_p),
                ("lpDesktop", ctypes.c_wchar_p),
                ("lpTitle", ctypes.c_wchar_p),
                ("dwX", ctypes.c_uint32),
                ("dwY", ctypes.c_uint32),
                ("dwXSize", ctypes.c_uint32),
                ("dwYSize", ctypes.c_uint32),
                ("dwXCountChars", ctypes.c_uint32),
                ("dwYCountChars", ctypes.c_uint32),
                ("dwFillAttribute", ctypes.c_uint32),
                ("dwFlags", ctypes.c_uint32),
                ("wShowWindow", ctypes.c_uint16),
                ("cbReserved2", ctypes.c_uint16),
                ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", ctypes.c_void_p),
                ("hStdOutput", ctypes.c_void_p),
                ("hStdError", ctypes.c_void_p),
            ]

        class _STARTUPINFOEX(ctypes.Structure):
            _fields_ = [("StartupInfo", _STARTUPINFO), ("lpAttributeList", ctypes.c_void_p)]

        size = ctypes.c_size_t()
        api.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
        attributes = (ctypes.c_byte * size.value)()
        if not api.InitializeProcThreadAttributeList(
            ctypes.byref(attributes), 1, 0, ctypes.byref(size)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        inherited = (ctypes.c_void_p * len(handles))(*handles)
        try:
            if not api.UpdateProcThreadAttribute(
                ctypes.byref(attributes),
                0,
                0x00020002,
                ctypes.byref(inherited),
                ctypes.sizeof(inherited),
                None,
                None,
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            startup = _STARTUPINFOEX()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.lpAttributeList = ctypes.cast(ctypes.byref(attributes), ctypes.c_void_p)
            process = _PROCESS_INFORMATION_V1()
            env_block = "".join(f"{name}={value}\0" for name, value in environment) + "\0"
            command_line = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(command)))
            if not api.CreateProcessW(
                None,
                command_line,
                None,
                None,
                True,
                flags,
                ctypes.c_wchar_p(env_block),
                cwd,
                ctypes.byref(startup),
                ctypes.byref(process),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                return _WindowsKernelLaunchV1(
                    int(process.dwProcessId),
                    self._handle(process.hProcess),
                    self._handle(process.hThread),
                )
            except BaseException as primary:
                cleanup_errors: list[BaseException] = []
                for raw_handle in (process.hProcess, process.hThread):
                    handle = int(raw_handle or 0)
                    if handle in {0, ctypes.c_void_p(-1).value}:
                        continue
                    try:
                        self._close_or_retain(handle)
                    except BaseException as cleanup:
                        cleanup_errors.append(cleanup)
                if cleanup_errors:
                    raise BaseExceptionGroup(
                        "CreateProcessW handle conversion and cleanup both failed",
                        [primary, *cleanup_errors],
                    ) from None
                raise
        finally:
            api.DeleteProcThreadAttributeList(ctypes.byref(attributes))

    def assign_process_to_job(self, job: int, process: int) -> None:
        if not self._api().AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(process)):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume_thread(self, thread: int) -> None:
        if self._api().ResumeThread(ctypes.c_void_p(thread)) == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())

    def _parent_pid(self, pid: int) -> int:
        api = self._api()

        class _PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_uint32),
                ("cntUsage", ctypes.c_uint32),
                ("th32ProcessID", ctypes.c_uint32),
                ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_uint32),
                ("cntThreads", ctypes.c_uint32),
                ("th32ParentProcessID", ctypes.c_uint32),
                ("pcPriClassBase", ctypes.c_int32),
                ("dwFlags", ctypes.c_uint32),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        snapshot = self._handle(api.CreateToolhelp32Snapshot(0x00000002, 0))
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        try:
            if not api.Process32FirstW(ctypes.c_void_p(snapshot), ctypes.byref(entry)):
                raise ctypes.WinError(ctypes.get_last_error())
            while True:
                if int(entry.th32ProcessID) == pid:
                    parent = int(entry.th32ParentProcessID)
                    if parent <= 0:
                        _windows_fail("Toolhelp returned no parent PID for retained process")
                    return parent
                entry.dwSize = ctypes.sizeof(entry)
                if not api.Process32NextW(ctypes.c_void_p(snapshot), ctypes.byref(entry)):
                    if ctypes.get_last_error() == 18:
                        break
                    raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self._close_or_retain(snapshot)
        _windows_fail("retained process disappeared before parent identity observation")

    def _parent_identity(self, pid: int) -> tuple[int, int]:
        parent_pid = self._parent_pid(pid)
        parent_handle = self._handle(
            self._api().OpenProcess(
                _SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, parent_pid
            )
        )
        try:
            observed_pid = int(self._api().GetProcessId(ctypes.c_void_p(parent_handle)))
            if observed_pid != parent_pid:
                _windows_fail("opened parent handle did not preserve the Toolhelp parent PID")
            creation = ctypes.c_uint64()
            if not self._api().GetProcessTimes(
                ctypes.c_void_p(parent_handle),
                ctypes.byref(creation),
                ctypes.byref(ctypes.c_uint64()),
                ctypes.byref(ctypes.c_uint64()),
                ctypes.byref(ctypes.c_uint64()),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            return parent_pid, int(creation.value)
        finally:
            self._close_or_retain(parent_handle)

    def query_process_identity(self, handle: int) -> _WindowsKernelProcessV1:
        api = self._api()
        creation = ctypes.c_uint64()
        if not api.GetProcessTimes(
            ctypes.c_void_p(handle),
            ctypes.byref(creation),
            ctypes.byref(ctypes.c_uint64()),
            ctypes.byref(ctypes.c_uint64()),
            ctypes.byref(ctypes.c_uint64()),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        pid = int(api.GetProcessId(ctypes.c_void_p(handle)))
        retained = self._retained_process_identities.get(handle)
        if retained is not None:
            if (pid, int(creation.value)) != (retained.pid, retained.creation_filetime):
                _windows_fail("retained process handle identity changed")
            if api.WaitForSingleObject(ctypes.c_void_p(handle), 0) == 0:
                # Windows may no longer expose the image name or Toolhelp row
                # after exit. The still-retained process handle proves which
                # previously observed process terminated; a bare PID cannot.
                return retained
        size = ctypes.c_uint32(32768)
        image = ctypes.create_unicode_buffer(size.value)
        if not api.QueryFullProcessImageNameW(
            ctypes.c_void_p(handle), 0, image, ctypes.byref(size)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            # Exit may begin between the zero-time wait and image query. Only
            # a bounded successful wait on this previously identified handle
            # permits its cached immutable facts; a live inaccessible process
            # still fails. PID and creation time were rechecked above.
            if retained is not None and api.WaitForSingleObject(ctypes.c_void_p(handle), 1000) == 0:
                return retained
            raise error
        path = Path(image.value)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        parent_pid, parent_creation = (
            self._parent_identity(pid)
            if retained is None
            else (retained.parent_pid, retained.parent_creation_filetime)
        )
        identity = _WindowsKernelProcessV1(
            pid, parent_pid, parent_creation, int(creation.value), path.name, digest
        )
        self._retained_process_identities[handle] = identity
        return identity

    def query_job_processes(self, job: int) -> tuple[int, ...]:
        api = self._api()
        max_active = self._job_max_active_processes.get(job)
        if max_active is None:
            _windows_fail("Job PID-list query lacks retained limit authority")
        size = _job_pid_list_buffer_bytes_v1(max_active)
        buffer = (ctypes.c_byte * size)()
        returned = ctypes.c_uint32()
        if not api.QueryInformationJobObject(
            ctypes.c_void_p(job), 3, ctypes.byref(buffer), size, ctypes.byref(returned)
        ):
            if ctypes.get_last_error() == 234:
                _windows_fail("Job PID-list exceeded its retained active-process limit")
            raise ctypes.WinError(ctypes.get_last_error())
        header = ctypes.cast(
            buffer, ctypes.POINTER(_JOBOBJECT_BASIC_PROCESS_ID_LIST_HEADER_V1)
        ).contents
        assigned = int(header.NumberOfAssignedProcesses)
        count = int(header.NumberOfProcessIdsInList)
        if count != assigned or assigned > max_active or count > max_active:
            _windows_fail("Job PID-list returned impossible bounded counts")
        process_ids = ctypes.cast(
            ctypes.byref(buffer, ctypes.sizeof(header)), ctypes.POINTER(ctypes.c_size_t)
        )
        pids = tuple(int(process_ids[index]) for index in range(count))
        if any(pid <= 0 for pid in pids) or len(set(pids)) != len(pids):
            _windows_fail("Job PID-list returned invalid or duplicate identities")
        return pids

    def open_process(self, pid: int, access: int) -> int:
        return self._handle(self._api().OpenProcess(access, False, pid))

    def terminate_process(self, handle: int) -> None:
        if not self._api().TerminateProcess(ctypes.c_void_p(handle), 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate_job(self, job: int) -> None:
        if not self._api().TerminateJobObject(ctypes.c_void_p(job), 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait(self, handle: int, timeout_ms: int) -> None:
        result = self._api().WaitForSingleObject(ctypes.c_void_p(handle), timeout_ms)
        if result != 0:
            raise _WindowsScenarioJobError("retained handle did not signal within its bounded wait")

    def query_job_active_process_count(self, job: int) -> int:
        info = _JOBOBJECT_BASIC_ACCOUNTING_INFORMATION_V1()
        if not self._api().QueryInformationJobObject(
            ctypes.c_void_p(job), 1, ctypes.byref(info), ctypes.sizeof(info), None
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(info.ActiveProcesses)

    def close_handle(self, handle: int) -> None:
        self._close_or_retain(handle)

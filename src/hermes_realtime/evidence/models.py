"""Closed, immutable V1 models for realtime evidence capture.

This module is deliberately import-inert and standard-library-only.  It owns
validation and pure model composition; queues, storage, lifecycle mutation,
HTTP routing, and authority consumption belong to later implementation tasks.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from enum import StrEnum
from functools import cache
from typing import Any, NoReturn, SupportsIndex, TypeVar, cast

CONSENT_VERSION = "realtime-evidence-consent-v1"

_MAX_UNSIGNED_63 = 2**63 - 1
_MAX_BROWSER_INTEGER = 2**53 - 1
_MAX_EVENT_SEQUENCE = 9_216
_MAX_TEXT_CODEPOINTS = 4_096
_MAX_SEGMENTS = 256
_MAX_CHUNKS = 4_096
_MAX_QUEUE_RECORDS = 64
_MAX_QUEUE_CANONICAL_BYTES = 2_097_152
_MAX_RETENTION_HOURS = 168
_DEFAULT_JSON_LIMIT = 2_097_152
_EVENT_JSON_LIMIT = 65_536
_MAX_JSON_DEPTH = 32

_UUID4_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\Z"
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_UTC_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z\Z")


class EvidenceModelError(Exception):
    """Base class for rejected evidence-model input."""


class _EvidenceTypeError(TypeError, EvidenceModelError):
    """Wrong runtime type at a closed model boundary."""


class _EvidenceValueError(ValueError, EvidenceModelError):
    """Correct runtime type with an invalid value or combination."""


class InputSource(StrEnum):
    MICROPHONE = "microphone"
    TYPED = "typed"


class TurnKind(StrEnum):
    USER_RESPONSE = "user_response"
    PROACTIVE_UPDATE = "proactive_update"
    REPLAY = "replay"


class ConversationOperationKind(StrEnum):
    RESPONSE = "response"
    PROACTIVE = "proactive"
    REPLAY = "replay"


class CommandRoutingOutcome(StrEnum):
    NOT_COMMAND = "not_command"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    INVALID = "invalid"


class EventKind(StrEnum):
    SESSION_OPENED = "session_opened"
    BINDING_OPENED = "binding_opened"
    TURN_OPENED = "turn_opened"
    USER_FINAL_ACCEPTED = "user_final_accepted"
    COMMAND_ROUTED = "command_routed"
    ASSISTANT_SEGMENT_GENERATED = "assistant_segment_generated"
    ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL = (
        "assistant_chunk_transport_confirmed_full"
    )
    TURN_SNAPSHOT = "turn_snapshot"
    TURN_SETTLED = "turn_settled"
    BINDING_CLOSED = "binding_closed"
    SESSION_SEAL_REQUESTED = "session_seal_requested"
    SESSION_TAINTED = "session_tainted"


EvidenceEventKind = EventKind


class PersistentSessionState(StrEnum):
    OPEN = "open"
    SEALED = "sealed"
    TAINTED = "tainted"


class RuntimeSessionPhase(StrEnum):
    OPEN = "open"
    EXPIRING = "expiring"
    CLOSING = "closing"
    SEAL_QUEUED = "seal_queued"
    REVOKING = "revoking"
    STOPPED = "stopped"


class ConsentEpochState(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"
    REVOKED = "revoked"


class TerminalDisposition(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    REVOKED = "revoked"
    FAILED = "failed"


class TerminalReason(StrEnum):
    AUTHORITATIVE_CLOSE_COMPLETED = "authoritative_close_completed"
    BARGE_IN = "barge_in"
    STOP_SPEAKING = "stop_speaking"
    RESPONSE_REPLACED = "response_replaced"
    BINDING_CLOSED = "binding_closed"
    RETENTION_EXPIRED = "retention_expired"
    HOST_SHUTDOWN = "host_shutdown"
    CALLER_CANCELLED = "caller_cancelled"
    CONSENT_REVOKED = "consent_revoked"
    TASK_SPAWN_FAILED = "task_spawn_failed"
    PROVIDER_FAILED = "provider_failed"
    TRANSPORT_FAILED = "transport_failed"
    LEDGER_CLOSE_FAILED = "ledger_close_failed"
    LIFECYCLE_FAILED = "lifecycle_failed"


class BindingCloseReason(StrEnum):
    CLIENT_CLOSED = "client_closed"
    BINDING_REPLACED = "binding_replaced"
    MEDIA_INCARNATION_REPLACED = "media_incarnation_replaced"
    PROJECTION_RESYNC = "projection_resync"
    CAPACITY_ROLLOVER = "capacity_rollover"
    RETENTION_ROLLOVER = "retention_rollover"
    CONSENT_REVOKED = "consent_revoked"
    INACTIVITY_EXPIRED = "inactivity_expired"
    RECEIVER_TRANSPORT_FAILED = "receiver_transport_failed"
    HOST_SHUTDOWN = "host_shutdown"


_ROLLOVER_CLOSE_REASONS = (
    BindingCloseReason.CAPACITY_ROLLOVER,
    BindingCloseReason.RETENTION_ROLLOVER,
)


class TaintCode(StrEnum):
    ADMISSION_GAP = "admission_gap"
    OVERSIZE = "oversize"
    DENY_FILTER = "deny_filter"
    WRITER_FAULT = "writer_fault"
    QUOTA_EXCEEDED = "quota_exceeded"
    EVENT_ID_CONFLICT = "event_id_conflict"
    SEQUENCE_CONFLICT = "sequence_conflict"
    LINEAGE_INVALID = "lineage_invalid"
    TERMINAL_MISSING = "terminal_missing"
    TERMINAL_CONFLICT = "terminal_conflict"
    SPAWN_FAILED = "spawn_failed"
    LEASE_CAPACITY_EXHAUSTED = "lease_capacity_exhausted"
    CLOCK_ROLLBACK = "clock_rollback"
    TTL_EXPIRED = "ttl_expired"
    STALE_OPEN_RECOVERY = "stale_open_recovery"
    SEAL_VALIDATION_FAILED = "seal_validation_failed"
    PURGE_FAILED = "purge_failed"
    CORRUPT_STORE = "corrupt_store"
    SHUTDOWN_INCOMPLETE = "shutdown_incomplete"


class ConflictReason(StrEnum):
    EVENT_ID_ENVELOPE_MISMATCH = "event_id_envelope_mismatch"
    CROSS_SESSION_EVENT_ID = "cross_session_event_id"
    SESSION_SEQUENCE_CLAIMED = "session_sequence_claimed"
    SEALED_SESSION_REUSE = "sealed_session_reuse"


class ErasureReason(StrEnum):
    REVOKED = "revoked"
    TTL = "ttl"
    CLOCK_ROLLBACK = "clock_rollback"
    UNCLEAN_EPOCH = "unclean_epoch"


class ErasureScope(StrEnum):
    SESSION = "session"
    CONSENT_EPOCH = "consent_epoch"
    STORE = "store"


class CaptureState(StrEnum):
    UNAVAILABLE = "unavailable"
    IDLE = "idle"
    ACTIVE = "active"
    REVOKED_PURGING = "revoked_purging"
    PURGE_FAILED = "purge_failed"
    FAULTED = "faulted"


class OwnerState(StrEnum):
    ABSENT = "absent"
    RECOVERY_ONLY = "recovery_only"
    RUNNING = "running"
    DRAINING = "draining"
    FAULTED = "faulted"
    STOPPED = "stopped"


class QueueReservationClass(StrEnum):
    ORDINARY = "ordinary"
    TERMINAL = "terminal"


class WriterFault(StrEnum):
    OWNERSHIP_UNAVAILABLE = "ownership_unavailable"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    PATH_INVALID = "path_invalid"
    ASSET_MISMATCH = "asset_mismatch"
    STORE_CORRUPT = "store_corrupt"
    PURGE_REQUIRED = "purge_required"
    SQLITE_FAULT = "sqlite_fault"
    QUOTA_UNAVAILABLE = "quota_unavailable"
    DRAIN_TIMEOUT = "drain_timeout"


class SentinelState(StrEnum):
    CLEAR = "clear"
    FULL_PURGE_PENDING = "full_purge_pending"
    CLOCK_ROLLBACK_PURGE_PENDING = "clock_rollback_purge_pending"
    FIRST_CREATE_PENDING = "first_create_pending"


class ControlResult(StrEnum):
    CONSENT_ACTIVATED = "consent_activated"
    REVOKE_DURABLY_SCHEDULED = "revoke_durably_scheduled"
    PURGE_COMPLETED = "purge_completed"


class ControlError(StrEnum):
    INVALID_CONTROL = "invalid_control"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    CONTROL_SEQUENCE_CONFLICT = "control_sequence_conflict"
    CAPTURE_UNAVAILABLE = "capture_unavailable"
    STATUS_CAPACITY_UNAVAILABLE = "status_capacity_unavailable"
    CONTROL_PENDING = "control_pending"
    CONTROL_TIMEOUT = "control_timeout"
    WRITER_UNAVAILABLE = "writer_unavailable"
    REVOKE_NOT_DURABLE = "revoke_not_durable"
    PURGE_FAILED = "purge_failed"


class AppendDisposition(StrEnum):
    ADMITTED = "admitted"
    DISABLED = "disabled"
    CONSENT_MISSING = "consent_missing"
    INVALID_AUTHORITY = "invalid_authority"
    SESSION_CLOSING = "session_closing"
    SESSION_TAINTED = "session_tainted"
    DROPPED_CAPACITY = "dropped_capacity"
    REJECTED_OVERSIZE = "rejected_oversize"
    REJECTED_DENIED = "rejected_denied"
    QUOTA_EXCEEDED = "quota_exceeded"
    WRITER_FAULT = "writer_fault"


class CommandDisposition(StrEnum):
    ADMITTED = "admitted"
    DISABLED = "disabled"
    CONSENT_MISSING = "consent_missing"
    INVALID_AUTHORITY = "invalid_authority"
    SESSION_CLOSING = "session_closing"
    SESSION_TAINTED = "session_tainted"
    DROPPED_CAPACITY = "dropped_capacity"
    WRITER_FAULT = "writer_fault"


class CauseDisposition(StrEnum):
    RECORDED = "recorded"
    ALREADY_RECORDED = "already_recorded"
    INVALID_AUTHORITY = "invalid_authority"
    CAUSE_SET_FROZEN = "cause_set_frozen"


class RolloverDisposition(StrEnum):
    ROLLOVER_QUEUED = "rollover_queued"
    ACTIVE_LEASES = "active_leases"
    REPLAYABLE_TURNS = "replayable_turns"
    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    INVALID_AUTHORITY = "invalid_authority"
    SESSION_TAINTED = "session_tainted"
    WRITER_FAULT = "writer_fault"


class ExpiryDisposition(StrEnum):
    ROLLOVER_QUEUED = "rollover_queued"
    ERASURE_DURABLY_SCHEDULED = "erasure_durably_scheduled"
    ALREADY_EXPIRING = "already_expiring"
    WRITER_FAULT = "writer_fault"


class ConsentDisposition(StrEnum):
    CONSENT_ACTIVATED = "consent_activated"
    ALREADY_ACTIVATED = "already_activated"
    CREATE_PENDING = "create_pending"
    CREATE_FAILED = "create_failed"
    CONTROL_TIMED_OUT = "control_timed_out"


class SealDisposition(StrEnum):
    SEAL_QUEUED = "seal_queued"
    ALREADY_QUEUED = "already_queued"
    ACTIVE_LEASES = "active_leases"
    REVOKED = "revoked"
    SESSION_TAINTED = "session_tainted"
    WRITER_FAULT = "writer_fault"


class RevokeDisposition(StrEnum):
    CLOSED_NOT_DURABLE = "closed_not_durable"
    REVOKE_DURABLY_SCHEDULED = "revoke_durably_scheduled"
    ALREADY_SCHEDULED = "already_scheduled"
    PURGE_COMPLETED = "purge_completed"
    PURGE_FAILED = "purge_failed"
    CONTROL_TIMED_OUT = "control_timed_out"
    WRITER_FAULT = "writer_fault"


class PurgeDisposition(StrEnum):
    PURGE_COMPLETED = "purge_completed"
    ALREADY_ABSENT = "already_absent"
    OWNERSHIP_UNAVAILABLE = "ownership_unavailable"
    PURGE_FAILED = "purge_failed"
    UNSUPPORTED_PLATFORM = "unsupported_platform"


class DrainDisposition(StrEnum):
    DRAIN_QUEUED = "drain_queued"
    ALREADY_QUEUED = "already_queued"
    STOPPED = "stopped"
    TIMED_OUT = "timed_out"
    WRITER_FAULT = "writer_fault"


class RecoveryDisposition(StrEnum):
    ABSENT = "absent"
    RECOVERED = "recovered"
    PURGE_COMPLETED = "purge_completed"
    OWNERSHIP_UNAVAILABLE = "ownership_unavailable"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    FAULTED = "faulted"


class StoreDisposition(StrEnum):
    COMMITTED = "committed"
    IDEMPOTENT = "idempotent"
    CONFLICT_TAINTED = "conflict_tainted"
    REJECTED_STATE = "rejected_state"
    FAULTED = "faulted"


class ExpiryMode(StrEnum):
    ROLLOVER = "rollover"
    ERASE_STUCK = "erase_stuck"


SessionExpiryMode = ExpiryMode


class EligibilityOutcome(StrEnum):
    ELIGIBLE = "eligible"
    NEVER_PERSISTED = "never_persisted"
    PERSISTED_BUT_EXCLUDED = "persisted_but_excluded"
    PURGED = "purged"


EvidenceQualificationOutcome = EligibilityOutcome


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _field_name(field_name: str) -> str:
    if type(field_name) is not str:
        raise _EvidenceTypeError("field_name must be an exact built-in string")
    if not field_name:
        raise _EvidenceValueError("field_name must not be empty")
    return field_name


def _require_exact_bool(value: object, field_name: str) -> bool:
    name = _field_name(field_name)
    if type(value) is not bool:
        raise _EvidenceTypeError(f"{name} must be an exact built-in bool")
    return value


def _bounded_integer(value: object, field_name: str, minimum: int, maximum: int) -> int:
    name = _field_name(field_name)
    if type(value) is not int:
        raise _EvidenceTypeError(f"{name} must be an exact built-in int")
    integer = value
    if integer < minimum or integer > maximum:
        raise _EvidenceValueError(f"{name} is outside its closed range")
    return integer


def _unsigned_63(value: object, field_name: str) -> int:
    return _bounded_integer(value, field_name, 1, _MAX_UNSIGNED_63)


def _browser_integer(value: object, field_name: str) -> int:
    return _bounded_integer(value, field_name, 1, _MAX_BROWSER_INTEGER)


def _count(value: object, field_name: str, maximum: int) -> int:
    return _bounded_integer(value, field_name, 0, maximum)


def _require_version(value: object, field_name: str = "protocol_version") -> int:
    version = _bounded_integer(value, field_name, 1, 1)
    return version


def _require_enum(value: object, enum_type: type[_EnumT], field_name: str) -> _EnumT:
    name = _field_name(field_name)
    if type(value) is not enum_type:
        raise _EvidenceTypeError(f"{name} must be an exact {enum_type.__name__}")
    member_name = getattr(value, "name", None)
    if type(member_name) is not str or enum_type.__members__.get(member_name) is not value:
        raise _EvidenceValueError(f"{name} is not a registered V1 enum member")
    return value


def _enum_from_primitive(value: object, enum_type: type[_EnumT], field_name: str) -> _EnumT:
    name = _field_name(field_name)
    if type(value) is not str:
        raise _EvidenceTypeError(f"{name} must be an exact built-in string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _EvidenceValueError(f"{name} is not a supported value") from exc


def _require_literal(value: object, literal: str, field_name: str) -> str:
    name = _field_name(field_name)
    if type(value) is not str:
        raise _EvidenceTypeError(f"{name} must be an exact built-in string")
    text = value
    if text != literal:
        raise _EvidenceValueError(f"{name} must equal its V1 literal")
    return text


def validate_canonical_uuid4(value: object, *, field_name: str = "uuid") -> str:
    """Validate one exact lowercase, hyphenated RFC-4122 UUIDv4 string."""

    name = _field_name(field_name)
    if type(value) is not str:
        raise _EvidenceTypeError(f"{name} must be an exact built-in string")
    text = value
    if _UUID4_PATTERN.fullmatch(text) is None:
        raise _EvidenceValueError(f"{name} must be a canonical UUIDv4")
    return text


def _optional_uuid4(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return validate_canonical_uuid4(value, field_name=field_name)


def validate_sha256_hex(value: object, *, field_name: str = "sha256") -> str:
    """Validate a lowercase 64-character SHA-256 hexadecimal rendering."""

    name = _field_name(field_name)
    if type(value) is not str:
        raise _EvidenceTypeError(f"{name} must be an exact built-in string")
    text = value
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise _EvidenceValueError(f"{name} must be lowercase SHA-256 hexadecimal text")
    return text


def validate_canonical_utc(value: object, *, field_name: str = "utc") -> str:
    """Validate exact ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` UTC text."""

    name = _field_name(field_name)
    if type(value) is not str:
        raise _EvidenceTypeError(f"{name} must be an exact built-in string")
    text = value
    if _UTC_PATTERN.fullmatch(text) is None:
        raise _EvidenceValueError(f"{name} must use the canonical UTC form")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise _EvidenceValueError(f"{name} must contain a possible UTC instant") from exc
    return text


def validate_evidence_text(value: object, *, field_name: str = "text") -> str:
    """Validate bounded exact source text without normalizing or rewriting it."""

    name = _field_name(field_name)
    if type(value) is not str:
        raise _EvidenceTypeError(f"{name} must be an exact built-in string")
    text = value
    if len(text) > _MAX_TEXT_CODEPOINTS:
        raise _EvidenceValueError(f"{name} exceeds 4096 Unicode code points")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        raise _EvidenceValueError(f"{name} contains a surrogate code point")
    return text


def _validate_json_string(value: str, field_name: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise _EvidenceValueError(f"{field_name} contains a surrogate code point")
    return value


def _validate_json_value(
    value: object,
    field_name: str = "json",
    *,
    depth: int = 0,
) -> object:
    if depth > _MAX_JSON_DEPTH:
        raise _EvidenceValueError("JSON nesting exceeds the V1 depth limit")
    if value is None:
        return None
    if type(value) is bool:
        return value
    if type(value) is int:
        return value
    if type(value) is str:
        return _validate_json_string(value, field_name)
    if type(value) is list:
        return [
            _validate_json_value(item, f"{field_name}[]", depth=depth + 1)
            for item in cast(list[object], value)
        ]
    if type(value) is dict:
        source = cast(dict[object, object], value)
        for key in source:
            if type(key) is not str:
                raise _EvidenceTypeError("JSON object keys must be exact built-in strings")
        result: dict[str, object] = {}
        for raw_key, item in source.items():
            key = _validate_json_string(cast(str, raw_key), f"{field_name} key")
            result[key] = _validate_json_value(
                item,
                f"{field_name}.{key}",
                depth=depth + 1,
            )
        return result
    raise _EvidenceTypeError("JSON values must use exact V1 primitive container types")


def _reject_json_number(value: str) -> NoReturn:
    raise _EvidenceValueError("JSON floating-point and nonfinite numbers are forbidden")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _EvidenceValueError("duplicate JSON object key")
        result[key] = value
    return result


def _decode_json_document(
    document: object,
    max_bytes: int,
) -> str:
    limit = _bounded_integer(max_bytes, "max_bytes", 1, _MAX_QUEUE_CANONICAL_BYTES)
    if type(document) is bytes:
        raw = document
        if len(raw) > limit:
            raise _EvidenceValueError("JSON document exceeds its byte limit")
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _EvidenceValueError("JSON document is not strict UTF-8") from exc
    raise _EvidenceTypeError("JSON document must be exact built-in bytes")


def _parse_json_object(
    document: object,
    *,
    max_bytes: int,
) -> dict[str, object]:
    text = _decode_json_document(document, max_bytes)
    try:
        parsed: object = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_float=_reject_json_number,
            parse_constant=_reject_json_number,
        )
    except EvidenceModelError:
        raise
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise _EvidenceValueError("invalid strict JSON document") from exc
    validated = _validate_json_value(parsed)
    if type(validated) is not dict:
        raise _EvidenceTypeError("strict JSON root must be an object")
    return cast(dict[str, object], validated)


def parse_strict_json_object(
    document: object,
    *,
    max_bytes: int = _DEFAULT_JSON_LIMIT,
) -> dict[str, object]:
    """Parse exact UTF-8 bytes as strict, finite, integer-only JSON object data."""

    return _parse_json_object(document, max_bytes=max_bytes)


class _FinalModel:
    __slots__ = ()

    def __init_subclass__(cls) -> None:
        if any(
            base is not _FinalModel and issubclass(base, _FinalModel)
            for base in cls.__bases__
        ):
            raise TypeError("closed V1 model classes cannot be subclassed")
        super().__init_subclass__()


class _PayloadMapping(Mapping[str, object]):
    __slots__ = ()

    def __init_subclass__(cls) -> None:
        if any(
            base is not _PayloadMapping and issubclass(base, _PayloadMapping)
            for base in cls.__bases__
        ):
            raise TypeError("closed V1 payload classes cannot be subclassed")
        super().__init_subclass__()

    def __getitem__(self, key: str) -> object:
        if type(key) is not str:
            raise TypeError("payload key must be an exact built-in string")
        return event_payload_to_primitive(cast(EvidencePayloadV1, self))[key]

    def __iter__(self) -> Iterator[str]:
        return iter(event_payload_to_primitive(cast(EvidencePayloadV1, self)))

    def __len__(self) -> int:
        return len(event_payload_to_primitive(cast(EvidencePayloadV1, self)))


@dataclass(frozen=True, slots=True)
class SessionOpenedPayloadV1(_PayloadMapping):
    consent_epoch_id: str
    binding_id: str
    consent_version: str
    disclosure_digest: str
    retention_hours: int
    microphone_accepted: bool
    typed_accepted: bool
    predecessor_session_id: str | None

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        _require_literal(self.consent_version, CONSENT_VERSION, "consent_version")
        validate_sha256_hex(self.disclosure_digest, field_name="disclosure_digest")
        _bounded_integer(self.retention_hours, "retention_hours", 1, _MAX_RETENTION_HOURS)
        microphone = _require_exact_bool(self.microphone_accepted, "microphone_accepted")
        typed = _require_exact_bool(self.typed_accepted, "typed_accepted")
        if not microphone and not typed:
            raise _EvidenceValueError("an opened session must have an accepted source")
        _optional_uuid4(self.predecessor_session_id, "predecessor_session_id")


@dataclass(frozen=True, slots=True)
class BindingOpenedPayloadV1(_PayloadMapping):
    binding_id: str
    binding_generation: int
    microphone_available: bool
    typed_available: bool

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        _unsigned_63(self.binding_generation, "binding_generation")
        _require_exact_bool(self.microphone_available, "microphone_available")
        _require_exact_bool(self.typed_available, "typed_available")


@dataclass(frozen=True, slots=True)
class TurnOpenedPayloadV1(_PayloadMapping):
    evidence_turn_id: str
    turn_kind: TurnKind
    utterance_id: str | None
    replay_of_evidence_turn_id: str | None

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        kind = _require_enum(self.turn_kind, TurnKind, "turn_kind")
        utterance = _optional_uuid4(self.utterance_id, "utterance_id")
        replay = _optional_uuid4(
            self.replay_of_evidence_turn_id,
            "replay_of_evidence_turn_id",
        )
        if kind is TurnKind.USER_RESPONSE:
            valid = utterance is not None and replay is None
        elif kind is TurnKind.REPLAY:
            valid = utterance is None and replay is not None
        elif kind is TurnKind.PROACTIVE_UPDATE:
            valid = utterance is None and replay is None
        else:
            raise _EvidenceValueError("turn kind has no V1 lineage rule")
        if not valid:
            raise _EvidenceValueError("turn_opened conditional lineage is invalid")


@dataclass(frozen=True, slots=True)
class UserFinalAcceptedPayloadV1(_PayloadMapping):
    utterance_id: str
    evidence_turn_id: str
    source: InputSource
    routing_disposition: str
    text: str

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.utterance_id, field_name="utterance_id")
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        _require_enum(self.source, InputSource, "source")
        _require_literal(self.routing_disposition, "response", "routing_disposition")
        validate_evidence_text(self.text)


@dataclass(frozen=True, slots=True)
class CommandRoutedPayloadV1(_PayloadMapping):
    utterance_id: str
    source: InputSource
    routing_disposition: str

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.utterance_id, field_name="utterance_id")
        _require_enum(self.source, InputSource, "source")
        _require_literal(self.routing_disposition, "command", "routing_disposition")


@dataclass(frozen=True, slots=True)
class AssistantSegmentGeneratedPayloadV1(_PayloadMapping):
    evidence_turn_id: str
    evidence_segment_id: str
    segment_ordinal: int
    text: str

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        validate_canonical_uuid4(self.evidence_segment_id, field_name="evidence_segment_id")
        _bounded_integer(self.segment_ordinal, "segment_ordinal", 1, _MAX_SEGMENTS)
        validate_evidence_text(self.text)


@dataclass(frozen=True, slots=True)
class AssistantChunkTransportConfirmedFullPayloadV1(_PayloadMapping):
    evidence_turn_id: str
    evidence_segment_id: str | None
    synthesis_attempt_id: str
    transport_attempt_id: str
    evidence_chunk_id: str
    chunk_ordinal: int
    text: str | None

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        segment = _optional_uuid4(self.evidence_segment_id, "evidence_segment_id")
        validate_canonical_uuid4(self.synthesis_attempt_id, field_name="synthesis_attempt_id")
        validate_canonical_uuid4(self.transport_attempt_id, field_name="transport_attempt_id")
        validate_canonical_uuid4(self.evidence_chunk_id, field_name="evidence_chunk_id")
        _bounded_integer(self.chunk_ordinal, "chunk_ordinal", 1, _MAX_CHUNKS)
        text = None if self.text is None else validate_evidence_text(self.text)
        if (segment is None) != (text is None):
            raise _EvidenceValueError("chunk segment identity and text must be paired")


@dataclass(frozen=True, slots=True)
class TurnSnapshotPayloadV1(_PayloadMapping):
    evidence_turn_id: str
    turn_kind: TurnKind
    generated_segment_count: int
    queued_chunk_count: int
    started_chunk_count: int
    transport_confirmed_full_count: int
    model_context_admitted: bool
    assistant_delivery_context_recorded: bool

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        kind = _require_enum(self.turn_kind, TurnKind, "turn_kind")
        generated = _count(
            self.generated_segment_count,
            "generated_segment_count",
            _MAX_SEGMENTS,
        )
        queued = _count(self.queued_chunk_count, "queued_chunk_count", _MAX_CHUNKS)
        started = _count(self.started_chunk_count, "started_chunk_count", _MAX_CHUNKS)
        confirmed = _count(
            self.transport_confirmed_full_count,
            "transport_confirmed_full_count",
            _MAX_CHUNKS,
        )
        _require_exact_bool(self.model_context_admitted, "model_context_admitted")
        _require_exact_bool(
            self.assistant_delivery_context_recorded,
            "assistant_delivery_context_recorded",
        )
        if confirmed > started or started > queued:
            raise _EvidenceValueError("turn snapshot chunk counts contradict transport order")
        if kind is not TurnKind.USER_RESPONSE and generated != 0:
            raise _EvidenceValueError("non-user turns cannot report generated segments")


_FAILED_REASONS = (
    TerminalReason.TASK_SPAWN_FAILED,
    TerminalReason.PROVIDER_FAILED,
    TerminalReason.TRANSPORT_FAILED,
    TerminalReason.LEDGER_CLOSE_FAILED,
    TerminalReason.LIFECYCLE_FAILED,
)
_INTERRUPTED_REASONS = (
    TerminalReason.BARGE_IN,
    TerminalReason.STOP_SPEAKING,
    TerminalReason.RESPONSE_REPLACED,
)
_CANCELLED_REASONS = (
    TerminalReason.BINDING_CLOSED,
    TerminalReason.RETENTION_EXPIRED,
    TerminalReason.HOST_SHUTDOWN,
    TerminalReason.CALLER_CANCELLED,
)


def _valid_terminal_pair(
    disposition: TerminalDisposition,
    reason: TerminalReason,
) -> bool:
    if disposition is TerminalDisposition.COMPLETED:
        return reason is TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED
    if disposition is TerminalDisposition.REVOKED:
        return reason is TerminalReason.CONSENT_REVOKED
    if disposition is TerminalDisposition.FAILED:
        return reason in _FAILED_REASONS
    if disposition is TerminalDisposition.INTERRUPTED:
        return reason in _INTERRUPTED_REASONS
    if disposition is TerminalDisposition.CANCELLED:
        return reason in _CANCELLED_REASONS
    return False


@dataclass(frozen=True, slots=True)
class TurnSettledPayloadV1(_PayloadMapping):
    evidence_turn_id: str
    terminal_disposition: TerminalDisposition
    terminal_reason: TerminalReason
    context_committed: bool
    generated_segment_count: int
    transport_confirmed_full_count: int

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        disposition = _require_enum(
            self.terminal_disposition,
            TerminalDisposition,
            "terminal_disposition",
        )
        reason = _require_enum(self.terminal_reason, TerminalReason, "terminal_reason")
        committed = _require_exact_bool(self.context_committed, "context_committed")
        generated = _count(
            self.generated_segment_count,
            "generated_segment_count",
            _MAX_SEGMENTS,
        )
        confirmed = _count(
            self.transport_confirmed_full_count,
            "transport_confirmed_full_count",
            _MAX_CHUNKS,
        )
        if not _valid_terminal_pair(disposition, reason):
            raise _EvidenceValueError("terminal disposition and reason do not form a V1 pair")
        if (
            disposition is TerminalDisposition.COMPLETED
            and not committed
            and (generated != 0 or confirmed != 0)
        ):
            raise _EvidenceValueError("completed output requires an authoritative context commit")


@dataclass(frozen=True, slots=True)
class BindingClosedPayloadV1(_PayloadMapping):
    binding_id: str
    close_reason: BindingCloseReason

    def __post_init__(self) -> None:
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        _require_enum(self.close_reason, BindingCloseReason, "close_reason")


@dataclass(frozen=True, slots=True)
class SessionSealRequestedPayloadV1(_PayloadMapping):
    final_event_sequence: int
    consent_epoch_id: str
    consent_version: str
    disclosure_digest: str

    def __post_init__(self) -> None:
        _bounded_integer(
            self.final_event_sequence,
            "final_event_sequence",
            1,
            _MAX_EVENT_SEQUENCE,
        )
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        _require_literal(self.consent_version, CONSENT_VERSION, "consent_version")
        validate_sha256_hex(self.disclosure_digest, field_name="disclosure_digest")


@dataclass(frozen=True, slots=True)
class SessionTaintedPayloadV1(_PayloadMapping):
    taint_code: TaintCode

    def __post_init__(self) -> None:
        _require_enum(self.taint_code, TaintCode, "taint_code")


EvidencePayloadV1 = (
    SessionOpenedPayloadV1
    | BindingOpenedPayloadV1
    | TurnOpenedPayloadV1
    | UserFinalAcceptedPayloadV1
    | CommandRoutedPayloadV1
    | AssistantSegmentGeneratedPayloadV1
    | AssistantChunkTransportConfirmedFullPayloadV1
    | TurnSnapshotPayloadV1
    | TurnSettledPayloadV1
    | BindingClosedPayloadV1
    | SessionSealRequestedPayloadV1
    | SessionTaintedPayloadV1
)

_PAYLOAD_CLASSES = (
    SessionOpenedPayloadV1,
    BindingOpenedPayloadV1,
    TurnOpenedPayloadV1,
    UserFinalAcceptedPayloadV1,
    CommandRoutedPayloadV1,
    AssistantSegmentGeneratedPayloadV1,
    AssistantChunkTransportConfirmedFullPayloadV1,
    TurnSnapshotPayloadV1,
    TurnSettledPayloadV1,
    BindingClosedPayloadV1,
    SessionSealRequestedPayloadV1,
    SessionTaintedPayloadV1,
)


def event_payload_to_primitive(payload: EvidencePayloadV1) -> dict[str, object]:
    """Project one exact payload to a fresh exact-key primitive dictionary."""

    payload = _reconstruct_payload(payload)
    if type(payload) is SessionOpenedPayloadV1:
        return {
            "consent_epoch_id": payload.consent_epoch_id,
            "binding_id": payload.binding_id,
            "consent_version": payload.consent_version,
            "disclosure_digest": payload.disclosure_digest,
            "retention_hours": payload.retention_hours,
            "microphone_accepted": payload.microphone_accepted,
            "typed_accepted": payload.typed_accepted,
            "predecessor_session_id": payload.predecessor_session_id,
        }
    if type(payload) is BindingOpenedPayloadV1:
        return {
            "binding_id": payload.binding_id,
            "binding_generation": payload.binding_generation,
            "microphone_available": payload.microphone_available,
            "typed_available": payload.typed_available,
        }
    if type(payload) is TurnOpenedPayloadV1:
        opened_result: dict[str, object] = {
            "evidence_turn_id": payload.evidence_turn_id,
            "turn_kind": payload.turn_kind.value,
        }
        if payload.utterance_id is not None:
            opened_result["utterance_id"] = payload.utterance_id
        if payload.replay_of_evidence_turn_id is not None:
            opened_result["replay_of_evidence_turn_id"] = payload.replay_of_evidence_turn_id
        return opened_result
    if type(payload) is UserFinalAcceptedPayloadV1:
        return {
            "utterance_id": payload.utterance_id,
            "evidence_turn_id": payload.evidence_turn_id,
            "source": payload.source.value,
            "routing_disposition": payload.routing_disposition,
            "text": payload.text,
        }
    if type(payload) is CommandRoutedPayloadV1:
        return {
            "utterance_id": payload.utterance_id,
            "source": payload.source.value,
            "routing_disposition": payload.routing_disposition,
        }
    if type(payload) is AssistantSegmentGeneratedPayloadV1:
        return {
            "evidence_turn_id": payload.evidence_turn_id,
            "evidence_segment_id": payload.evidence_segment_id,
            "segment_ordinal": payload.segment_ordinal,
            "text": payload.text,
        }
    if type(payload) is AssistantChunkTransportConfirmedFullPayloadV1:
        confirmed_result: dict[str, object] = {
            "evidence_turn_id": payload.evidence_turn_id,
            "synthesis_attempt_id": payload.synthesis_attempt_id,
            "transport_attempt_id": payload.transport_attempt_id,
            "evidence_chunk_id": payload.evidence_chunk_id,
            "chunk_ordinal": payload.chunk_ordinal,
        }
        if payload.evidence_segment_id is not None:
            confirmed_result["evidence_segment_id"] = payload.evidence_segment_id
            confirmed_result["text"] = payload.text
        return confirmed_result
    if type(payload) is TurnSnapshotPayloadV1:
        return {
            "evidence_turn_id": payload.evidence_turn_id,
            "turn_kind": payload.turn_kind.value,
            "generated_segment_count": payload.generated_segment_count,
            "queued_chunk_count": payload.queued_chunk_count,
            "started_chunk_count": payload.started_chunk_count,
            "transport_confirmed_full_count": payload.transport_confirmed_full_count,
            "model_context_admitted": payload.model_context_admitted,
            "assistant_delivery_context_recorded": (
                payload.assistant_delivery_context_recorded
            ),
        }
    if type(payload) is TurnSettledPayloadV1:
        return {
            "evidence_turn_id": payload.evidence_turn_id,
            "terminal_disposition": payload.terminal_disposition.value,
            "terminal_reason": payload.terminal_reason.value,
            "context_committed": payload.context_committed,
            "generated_segment_count": payload.generated_segment_count,
            "transport_confirmed_full_count": payload.transport_confirmed_full_count,
        }
    if type(payload) is BindingClosedPayloadV1:
        return {
            "binding_id": payload.binding_id,
            "close_reason": payload.close_reason.value,
        }
    if type(payload) is SessionSealRequestedPayloadV1:
        return {
            "final_event_sequence": payload.final_event_sequence,
            "consent_epoch_id": payload.consent_epoch_id,
            "consent_version": payload.consent_version,
            "disclosure_digest": payload.disclosure_digest,
        }
    if type(payload) is SessionTaintedPayloadV1:
        return {"taint_code": payload.taint_code.value}
    raise _EvidenceTypeError("payload must be one exact V1 payload class")


event_payload_to_primitive_v1 = event_payload_to_primitive


def _payload_class(event_kind: EventKind) -> type[_PayloadMapping]:
    if event_kind is EventKind.SESSION_OPENED:
        return SessionOpenedPayloadV1
    if event_kind is EventKind.BINDING_OPENED:
        return BindingOpenedPayloadV1
    if event_kind is EventKind.TURN_OPENED:
        return TurnOpenedPayloadV1
    if event_kind is EventKind.USER_FINAL_ACCEPTED:
        return UserFinalAcceptedPayloadV1
    if event_kind is EventKind.COMMAND_ROUTED:
        return CommandRoutedPayloadV1
    if event_kind is EventKind.ASSISTANT_SEGMENT_GENERATED:
        return AssistantSegmentGeneratedPayloadV1
    if event_kind is EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL:
        return AssistantChunkTransportConfirmedFullPayloadV1
    if event_kind is EventKind.TURN_SNAPSHOT:
        return TurnSnapshotPayloadV1
    if event_kind is EventKind.TURN_SETTLED:
        return TurnSettledPayloadV1
    if event_kind is EventKind.BINDING_CLOSED:
        return BindingClosedPayloadV1
    if event_kind is EventKind.SESSION_SEAL_REQUESTED:
        return SessionSealRequestedPayloadV1
    if event_kind is EventKind.SESSION_TAINTED:
        return SessionTaintedPayloadV1
    raise _EvidenceValueError("unsupported V1 event kind")


def _expect_payload_keys(
    payload: object,
    expected: frozenset[str],
) -> dict[str, object]:
    if type(payload) is not dict:
        raise _EvidenceTypeError("event payload must be an exact built-in dict")
    source = cast(dict[object, object], payload)
    for key in source:
        if type(key) is not str:
            raise _EvidenceTypeError("event payload keys must be exact built-in strings")
    if frozenset(cast(dict[str, object], source)) != expected:
        raise _EvidenceValueError("event payload has a missing or unknown key")
    return cast(dict[str, object], source)


@cache
def _payload_field_names(model_type: type[_PayloadMapping]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(cast(Any, model_type)))


def _reconstruct_payload(payload: EvidencePayloadV1) -> EvidencePayloadV1:
    if type(payload) not in _PAYLOAD_CLASSES:
        raise _EvidenceTypeError("payload must be one exact V1 payload class")
    model_type = type(payload)
    values = {name: getattr(payload, name) for name in _payload_field_names(model_type)}
    reconstructed = model_type(**values)
    return reconstructed


def validate_event_payload(
    event_kind: EventKind,
    payload: object,
    *,
    turn_kind: TurnKind | None = None,
) -> EvidencePayloadV1:
    """Validate and defensively reconstruct the payload for one exact event kind."""

    kind = _require_enum(event_kind, EventKind, "event_kind")
    if turn_kind is not None:
        turn_context = _require_enum(turn_kind, TurnKind, "turn_kind")
    else:
        turn_context = None
    expected_class = _payload_class(kind)
    if type(payload) is expected_class:
        result = _reconstruct_payload(cast(EvidencePayloadV1, payload))
    elif type(payload) is dict:
        if kind is EventKind.SESSION_OPENED:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {
                        "consent_epoch_id",
                        "binding_id",
                        "consent_version",
                        "disclosure_digest",
                        "retention_hours",
                        "microphone_accepted",
                        "typed_accepted",
                        "predecessor_session_id",
                    }
                ),
            )
            result = SessionOpenedPayloadV1(
                consent_epoch_id=cast(str, data["consent_epoch_id"]),
                binding_id=cast(str, data["binding_id"]),
                consent_version=cast(str, data["consent_version"]),
                disclosure_digest=cast(str, data["disclosure_digest"]),
                retention_hours=cast(int, data["retention_hours"]),
                microphone_accepted=cast(bool, data["microphone_accepted"]),
                typed_accepted=cast(bool, data["typed_accepted"]),
                predecessor_session_id=cast(str | None, data["predecessor_session_id"]),
            )
        elif kind is EventKind.BINDING_OPENED:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {
                        "binding_id",
                        "binding_generation",
                        "microphone_available",
                        "typed_available",
                    }
                ),
            )
            result = BindingOpenedPayloadV1(
                binding_id=cast(str, data["binding_id"]),
                binding_generation=cast(int, data["binding_generation"]),
                microphone_available=cast(bool, data["microphone_available"]),
                typed_available=cast(bool, data["typed_available"]),
            )
        elif kind is EventKind.TURN_OPENED:
            if type(payload) is not dict:
                raise _EvidenceTypeError("turn_opened payload must be an exact dict")
            raw = cast(dict[str, object], payload)
            if "turn_kind" not in raw:
                raise _EvidenceValueError("turn_opened payload is missing turn_kind")
            parsed_kind = _enum_from_primitive(raw["turn_kind"], TurnKind, "turn_kind")
            base_keys = {"evidence_turn_id", "turn_kind"}
            if parsed_kind is TurnKind.USER_RESPONSE:
                expected_keys = frozenset(base_keys | {"utterance_id"})
            elif parsed_kind is TurnKind.REPLAY:
                expected_keys = frozenset(base_keys | {"replay_of_evidence_turn_id"})
            elif parsed_kind is TurnKind.PROACTIVE_UPDATE:
                expected_keys = frozenset(base_keys)
            else:
                raise _EvidenceValueError("turn kind has no V1 payload shape")
            data = _expect_payload_keys(payload, expected_keys)
            result = TurnOpenedPayloadV1(
                evidence_turn_id=cast(str, data["evidence_turn_id"]),
                turn_kind=parsed_kind,
                utterance_id=cast(str | None, data.get("utterance_id")),
                replay_of_evidence_turn_id=cast(
                    str | None,
                    data.get("replay_of_evidence_turn_id"),
                ),
            )
        elif kind is EventKind.USER_FINAL_ACCEPTED:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {
                        "utterance_id",
                        "evidence_turn_id",
                        "source",
                        "routing_disposition",
                        "text",
                    }
                ),
            )
            result = UserFinalAcceptedPayloadV1(
                utterance_id=cast(str, data["utterance_id"]),
                evidence_turn_id=cast(str, data["evidence_turn_id"]),
                source=_enum_from_primitive(data["source"], InputSource, "source"),
                routing_disposition=cast(str, data["routing_disposition"]),
                text=cast(str, data["text"]),
            )
        elif kind is EventKind.COMMAND_ROUTED:
            data = _expect_payload_keys(
                payload,
                frozenset({"utterance_id", "source", "routing_disposition"}),
            )
            result = CommandRoutedPayloadV1(
                utterance_id=cast(str, data["utterance_id"]),
                source=_enum_from_primitive(data["source"], InputSource, "source"),
                routing_disposition=cast(str, data["routing_disposition"]),
            )
        elif kind is EventKind.ASSISTANT_SEGMENT_GENERATED:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {"evidence_turn_id", "evidence_segment_id", "segment_ordinal", "text"}
                ),
            )
            result = AssistantSegmentGeneratedPayloadV1(
                evidence_turn_id=cast(str, data["evidence_turn_id"]),
                evidence_segment_id=cast(str, data["evidence_segment_id"]),
                segment_ordinal=cast(int, data["segment_ordinal"]),
                text=cast(str, data["text"]),
            )
        elif kind is EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL:
            base = {
                "evidence_turn_id",
                "synthesis_attempt_id",
                "transport_attempt_id",
                "evidence_chunk_id",
                "chunk_ordinal",
            }
            raw = cast(dict[str, object], payload)
            has_segment = "evidence_segment_id" in raw
            has_text = "text" in raw
            if has_segment != has_text:
                raise _EvidenceValueError("chunk segment identity and text must be paired")
            conditional_keys = {"evidence_segment_id", "text"} if has_text else set()
            expected_keys = frozenset(base | conditional_keys)
            data = _expect_payload_keys(payload, expected_keys)
            result = AssistantChunkTransportConfirmedFullPayloadV1(
                evidence_turn_id=cast(str, data["evidence_turn_id"]),
                evidence_segment_id=cast(str | None, data.get("evidence_segment_id")),
                synthesis_attempt_id=cast(str, data["synthesis_attempt_id"]),
                transport_attempt_id=cast(str, data["transport_attempt_id"]),
                evidence_chunk_id=cast(str, data["evidence_chunk_id"]),
                chunk_ordinal=cast(int, data["chunk_ordinal"]),
                text=cast(str | None, data.get("text")),
            )
        elif kind is EventKind.TURN_SNAPSHOT:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {
                        "evidence_turn_id",
                        "turn_kind",
                        "generated_segment_count",
                        "queued_chunk_count",
                        "started_chunk_count",
                        "transport_confirmed_full_count",
                        "model_context_admitted",
                        "assistant_delivery_context_recorded",
                    }
                ),
            )
            result = TurnSnapshotPayloadV1(
                evidence_turn_id=cast(str, data["evidence_turn_id"]),
                turn_kind=_enum_from_primitive(data["turn_kind"], TurnKind, "turn_kind"),
                generated_segment_count=cast(int, data["generated_segment_count"]),
                queued_chunk_count=cast(int, data["queued_chunk_count"]),
                started_chunk_count=cast(int, data["started_chunk_count"]),
                transport_confirmed_full_count=cast(
                    int,
                    data["transport_confirmed_full_count"],
                ),
                model_context_admitted=cast(bool, data["model_context_admitted"]),
                assistant_delivery_context_recorded=cast(
                    bool,
                    data["assistant_delivery_context_recorded"],
                ),
            )
        elif kind is EventKind.TURN_SETTLED:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {
                        "evidence_turn_id",
                        "terminal_disposition",
                        "terminal_reason",
                        "context_committed",
                        "generated_segment_count",
                        "transport_confirmed_full_count",
                    }
                ),
            )
            result = TurnSettledPayloadV1(
                evidence_turn_id=cast(str, data["evidence_turn_id"]),
                terminal_disposition=_enum_from_primitive(
                    data["terminal_disposition"],
                    TerminalDisposition,
                    "terminal_disposition",
                ),
                terminal_reason=_enum_from_primitive(
                    data["terminal_reason"],
                    TerminalReason,
                    "terminal_reason",
                ),
                context_committed=cast(bool, data["context_committed"]),
                generated_segment_count=cast(int, data["generated_segment_count"]),
                transport_confirmed_full_count=cast(
                    int,
                    data["transport_confirmed_full_count"],
                ),
            )
        elif kind is EventKind.BINDING_CLOSED:
            data = _expect_payload_keys(
                payload,
                frozenset({"binding_id", "close_reason"}),
            )
            result = BindingClosedPayloadV1(
                binding_id=cast(str, data["binding_id"]),
                close_reason=_enum_from_primitive(
                    data["close_reason"],
                    BindingCloseReason,
                    "close_reason",
                ),
            )
        elif kind is EventKind.SESSION_SEAL_REQUESTED:
            data = _expect_payload_keys(
                payload,
                frozenset(
                    {
                        "final_event_sequence",
                        "consent_epoch_id",
                        "consent_version",
                        "disclosure_digest",
                    }
                ),
            )
            result = SessionSealRequestedPayloadV1(
                final_event_sequence=cast(int, data["final_event_sequence"]),
                consent_epoch_id=cast(str, data["consent_epoch_id"]),
                consent_version=cast(str, data["consent_version"]),
                disclosure_digest=cast(str, data["disclosure_digest"]),
            )
        elif kind is EventKind.SESSION_TAINTED:
            data = _expect_payload_keys(payload, frozenset({"taint_code"}))
            result = SessionTaintedPayloadV1(
                taint_code=_enum_from_primitive(data["taint_code"], TaintCode, "taint_code")
            )
        else:
            raise _EvidenceValueError("unsupported V1 event kind")
    else:
        raise _EvidenceTypeError("event kind requires its exact payload class or primitive dict")

    if type(result) is AssistantSegmentGeneratedPayloadV1:
        if turn_context is not None and turn_context is not TurnKind.USER_RESPONSE:
            raise _EvidenceValueError("generated segments belong only to user-response turns")
    elif type(result) is AssistantChunkTransportConfirmedFullPayloadV1:
        carries_text = result.evidence_segment_id is not None
        if turn_context is TurnKind.USER_RESPONSE and not carries_text:
            raise _EvidenceValueError("user-response confirmations require segment text")
        if turn_context in (TurnKind.PROACTIVE_UPDATE, TurnKind.REPLAY) and carries_text:
            raise _EvidenceValueError("non-user confirmations must be content-free")
    elif (
        type(result) is TurnSnapshotPayloadV1
        and turn_context is not None
        and result.turn_kind is not turn_context
    ):
        raise _EvidenceValueError("turn snapshot kind contradicts owning turn")
    return result


def parse_event_payload_json(
    event_kind: EventKind,
    document: object,
    *,
    turn_kind: TurnKind | None = None,
) -> EvidencePayloadV1:
    data = _parse_json_object(
        document,
        max_bytes=_EVENT_JSON_LIMIT,
    )
    return validate_event_payload(event_kind, data, turn_kind=turn_kind)


@dataclass(frozen=True, slots=True)
class EvidenceSnapshotV1(_FinalModel):
    schema_version: int
    installation_id: str
    producer_instance_id: str
    logical_session_id: str
    event_id: str
    event_sequence: int
    event_kind: EventKind
    payload: EvidencePayloadV1

    def __post_init__(self) -> None:
        _require_version(self.schema_version, "schema_version")
        validate_canonical_uuid4(self.installation_id, field_name="installation_id")
        validate_canonical_uuid4(
            self.producer_instance_id,
            field_name="producer_instance_id",
        )
        validate_canonical_uuid4(self.logical_session_id, field_name="logical_session_id")
        validate_canonical_uuid4(self.event_id, field_name="event_id")
        sequence = _bounded_integer(
            self.event_sequence,
            "event_sequence",
            1,
            _MAX_EVENT_SEQUENCE,
        )
        kind = _require_enum(self.event_kind, EventKind, "event_kind")
        expected_payload_class = _payload_class(kind)
        if type(self.payload) is not expected_payload_class:
            raise _EvidenceTypeError(
                "snapshot event kind requires its exact V1 payload dataclass"
            )
        payload = validate_event_payload(kind, self.payload)
        if (
            type(payload) is SessionSealRequestedPayloadV1
            and payload.final_event_sequence != sequence
        ):
            raise _EvidenceValueError("seal payload sequence must equal its envelope sequence")
        object.__setattr__(self, "payload", payload)


def _copy_snapshot(snapshot: object) -> EvidenceSnapshotV1:
    if type(snapshot) is not EvidenceSnapshotV1:
        raise _EvidenceTypeError("snapshot must be an exact EvidenceSnapshotV1")
    return EvidenceSnapshotV1(
        schema_version=snapshot.schema_version,
        installation_id=snapshot.installation_id,
        producer_instance_id=snapshot.producer_instance_id,
        logical_session_id=snapshot.logical_session_id,
        event_id=snapshot.event_id,
        event_sequence=snapshot.event_sequence,
        event_kind=snapshot.event_kind,
        payload=snapshot.payload,
    )


def evidence_snapshot_to_primitive(snapshot: EvidenceSnapshotV1) -> dict[str, object]:
    copied = _copy_snapshot(snapshot)
    return {
        "schema_version": copied.schema_version,
        "installation_id": copied.installation_id,
        "producer_instance_id": copied.producer_instance_id,
        "logical_session_id": copied.logical_session_id,
        "event_id": copied.event_id,
        "event_sequence": copied.event_sequence,
        "event_kind": copied.event_kind.value,
        "payload": event_payload_to_primitive(copied.payload),
    }


evidence_snapshot_to_primitive_v1 = evidence_snapshot_to_primitive


def parse_evidence_snapshot_json(document: object) -> EvidenceSnapshotV1:
    data = parse_strict_json_object(document, max_bytes=_EVENT_JSON_LIMIT)
    expected = frozenset(
        {
            "schema_version",
            "installation_id",
            "producer_instance_id",
            "logical_session_id",
            "event_id",
            "event_sequence",
            "event_kind",
            "payload",
        }
    )
    if frozenset(data) != expected:
        raise _EvidenceValueError("snapshot envelope has a missing or unknown key")
    kind = _enum_from_primitive(data["event_kind"], EventKind, "event_kind")
    payload = validate_event_payload(kind, data["payload"])
    return EvidenceSnapshotV1(
        schema_version=cast(int, data["schema_version"]),
        installation_id=cast(str, data["installation_id"]),
        producer_instance_id=cast(str, data["producer_instance_id"]),
        logical_session_id=cast(str, data["logical_session_id"]),
        event_id=cast(str, data["event_id"]),
        event_sequence=cast(int, data["event_sequence"]),
        event_kind=kind,
        payload=payload,
    )


@dataclass(slots=True)
class _TurnHistory:
    kind: TurnKind
    utterance_id: str | None
    replay_of_evidence_turn_id: str | None
    user_final_seen: bool = False
    next_segment_ordinal: int = 1
    next_chunk_ordinal: int = 1
    segments: dict[str, str] | None = None
    chunk_ids: set[str] | None = None
    snapshot: TurnSnapshotPayloadV1 | None = None
    settlement: TurnSettledPayloadV1 | None = None
    active_replay_turn_id: str | None = None
    replay_retired: bool = False
    settled: bool = False

    def __post_init__(self) -> None:
        if self.segments is None:
            self.segments = {}
        if self.chunk_ids is None:
            self.chunk_ids = set()


def _turn_for_event(
    turns: dict[str, _TurnHistory],
    evidence_turn_id: str,
) -> _TurnHistory:
    turn = turns.get(evidence_turn_id)
    if turn is None:
        raise _EvidenceValueError("turn event does not reference an open V1 turn")
    if turn.settled:
        raise _EvidenceValueError("no turn event may follow settlement")
    return turn


def _require_session_source_acceptance(
    source: InputSource,
    session_opened: SessionOpenedPayloadV1,
    binding_opened: BindingOpenedPayloadV1,
) -> None:
    exact_source = _require_enum(source, InputSource, "source")
    if exact_source is InputSource.MICROPHONE:
        accepted = session_opened.microphone_accepted
        available = binding_opened.microphone_available
    elif exact_source is InputSource.TYPED:
        accepted = session_opened.typed_accepted
        available = binding_opened.typed_available
    else:
        raise _EvidenceValueError("input source has no V1 consent rule")
    if not accepted:
        raise _EvidenceValueError("event source was not accepted for this session")
    if not available:
        raise _EvidenceValueError("event source was not available for this binding")


def _is_replayable_source(turn: _TurnHistory) -> bool:
    snapshot = turn.snapshot
    settlement = turn.settlement
    return (
        turn.kind is TurnKind.USER_RESPONSE
        and turn.user_final_seen
        and turn.settled
        and not turn.replay_retired
        and turn.active_replay_turn_id is None
        and snapshot is not None
        and settlement is not None
        and snapshot.generated_segment_count > 0
        and snapshot.queued_chunk_count
        > snapshot.transport_confirmed_full_count
        and settlement.terminal_disposition is TerminalDisposition.FAILED
        and settlement.terminal_reason is TerminalReason.TRANSPORT_FAILED
    )


def classify_event_retry(
    existing: EvidenceSnapshotV1,
    incoming: EvidenceSnapshotV1,
    *,
    target_sealed: bool,
) -> StoreDisposition | ConflictReason:
    """Classify an exact retry, changed reuse, or sequence claim."""

    sealed = _require_exact_bool(target_sealed, "target_sealed")
    original = _copy_snapshot(existing)
    candidate = _copy_snapshot(incoming)
    if original.event_id == candidate.event_id:
        if original.logical_session_id != candidate.logical_session_id:
            return ConflictReason.CROSS_SESSION_EVENT_ID
        if original == candidate:
            return StoreDisposition.IDEMPOTENT
        return ConflictReason.EVENT_ID_ENVELOPE_MISMATCH
    if sealed:
        return ConflictReason.SEALED_SESSION_REUSE
    if (
        original.logical_session_id == candidate.logical_session_id
        and original.event_sequence == candidate.event_sequence
    ):
        return ConflictReason.SESSION_SEQUENCE_CLAIMED
    return StoreDisposition.COMMITTED


def _normalize_event_history(events: object) -> tuple[EvidenceSnapshotV1, ...]:
    if type(events) is not tuple:
        raise _EvidenceTypeError("events must be an exact built-in tuple")
    source = cast(tuple[object, ...], events)
    if not source:
        raise _EvidenceValueError("event history must not be empty")
    normalized: list[EvidenceSnapshotV1] = []
    by_event_id: dict[str, EvidenceSnapshotV1] = {}
    by_sequence: dict[int, EvidenceSnapshotV1] = {}
    for raw_event in source:
        event = _copy_snapshot(raw_event)
        previous_id = by_event_id.get(event.event_id)
        if previous_id is not None:
            classification = classify_event_retry(
                previous_id,
                event,
                target_sealed=False,
            )
            if classification is StoreDisposition.IDEMPOTENT:
                continue
            raise _EvidenceValueError("event history contains conflicting event-ID reuse")
        previous_sequence = by_sequence.get(event.event_sequence)
        if previous_sequence is not None:
            raise _EvidenceValueError("event history contains a claimed session sequence")
        by_event_id[event.event_id] = event
        by_sequence[event.event_sequence] = event
        normalized.append(event)
    first = normalized[0]
    for expected_sequence, event in enumerate(normalized, 1):
        if event.event_sequence != expected_sequence:
            raise _EvidenceValueError("event history sequence is not contiguous")
        if (
            event.installation_id != first.installation_id
            or event.producer_instance_id != first.producer_instance_id
            or event.logical_session_id != first.logical_session_id
        ):
            raise _EvidenceValueError("event history crosses an envelope lineage")
    return tuple(normalized)


def validate_event_sequence(events: tuple[EvidenceSnapshotV1, ...]) -> None:
    """Validate the closed static V1 event history and permitted exact retries."""

    history = _normalize_event_history(events)
    if len(history) < 2:
        raise _EvidenceValueError("event history is missing its two opening records")
    if history[0].event_kind is not EventKind.SESSION_OPENED:
        raise _EvidenceValueError("sequence 1 must be session_opened")
    if history[1].event_kind is not EventKind.BINDING_OPENED:
        raise _EvidenceValueError("sequence 2 must be binding_opened")
    session_opened = cast(SessionOpenedPayloadV1, history[0].payload)
    binding_opened = cast(BindingOpenedPayloadV1, history[1].payload)
    if session_opened.binding_id != binding_opened.binding_id:
        raise _EvidenceValueError("opening records disagree on binding identity")

    turns: dict[str, _TurnHistory] = {}
    segment_ids: set[str] = set()
    chunk_ids: set[str] = set()
    consumed_utterance_ids: set[str] = set()
    binding_closed = False
    session_terminal = False

    for event in history[2:]:
        if session_terminal:
            raise _EvidenceValueError("no session event may follow seal or taint")
        kind = event.event_kind
        payload = event.payload
        if kind in (EventKind.SESSION_OPENED, EventKind.BINDING_OPENED):
            raise _EvidenceValueError("session opening records cannot repeat")
        if kind is EventKind.COMMAND_ROUTED:
            if binding_closed:
                raise _EvidenceValueError("command event follows binding close")
            routed = cast(CommandRoutedPayloadV1, payload)
            _require_session_source_acceptance(routed.source, session_opened, binding_opened)
            if routed.utterance_id in consumed_utterance_ids:
                raise _EvidenceValueError("utterance identity is already consumed")
            consumed_utterance_ids.add(routed.utterance_id)
            continue
        if kind is EventKind.TURN_OPENED:
            if binding_closed:
                raise _EvidenceValueError("turn opens after binding close")
            opened = cast(TurnOpenedPayloadV1, payload)
            if opened.evidence_turn_id in turns:
                raise _EvidenceValueError("evidence turn identity is reused")
            if opened.turn_kind is TurnKind.USER_RESPONSE:
                utterance_id = cast(str, opened.utterance_id)
                if utterance_id in consumed_utterance_ids:
                    raise _EvidenceValueError("utterance identity is already consumed")
                consumed_utterance_ids.add(utterance_id)
            elif opened.turn_kind is TurnKind.REPLAY:
                source_id = cast(str, opened.replay_of_evidence_turn_id)
                if source_id == opened.evidence_turn_id:
                    raise _EvidenceValueError("replay turn cannot reference itself")
                source_turn = turns.get(source_id)
                if source_turn is None or not _is_replayable_source(source_turn):
                    raise _EvidenceValueError(
                        "replay source is not current and replayable in this session"
                    )
                source_turn.active_replay_turn_id = opened.evidence_turn_id
            elif opened.turn_kind is not TurnKind.PROACTIVE_UPDATE:
                raise _EvidenceValueError("turn kind has no V1 history path")
            turns[opened.evidence_turn_id] = _TurnHistory(
                kind=opened.turn_kind,
                utterance_id=opened.utterance_id,
                replay_of_evidence_turn_id=opened.replay_of_evidence_turn_id,
            )
            continue
        if kind is EventKind.USER_FINAL_ACCEPTED:
            accepted = cast(UserFinalAcceptedPayloadV1, payload)
            turn = _turn_for_event(turns, accepted.evidence_turn_id)
            if turn.kind is not TurnKind.USER_RESPONSE:
                raise _EvidenceValueError("user-final event belongs only to a user turn")
            if turn.user_final_seen or turn.snapshot is not None:
                raise _EvidenceValueError("user-final event is duplicated or late")
            if accepted.utterance_id != turn.utterance_id:
                raise _EvidenceValueError("user-final utterance lineage does not match")
            _require_session_source_acceptance(accepted.source, session_opened, binding_opened)
            turn.user_final_seen = True
            continue
        if kind is EventKind.ASSISTANT_SEGMENT_GENERATED:
            generated = cast(AssistantSegmentGeneratedPayloadV1, payload)
            turn = _turn_for_event(turns, generated.evidence_turn_id)
            if turn.kind is not TurnKind.USER_RESPONSE or not turn.user_final_seen:
                raise _EvidenceValueError("generated segment has no user-response authority")
            if turn.snapshot is not None:
                raise _EvidenceValueError("generated segment follows the turn snapshot")
            if generated.segment_ordinal != turn.next_segment_ordinal:
                raise _EvidenceValueError("segment ordinal is not the exact next ordinal")
            if generated.evidence_segment_id in segment_ids:
                raise _EvidenceValueError("segment identity is reused")
            segments = cast(dict[str, str], turn.segments)
            segments[generated.evidence_segment_id] = generated.text
            segment_ids.add(generated.evidence_segment_id)
            turn.next_segment_ordinal += 1
            continue
        if kind is EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL:
            confirmed = cast(AssistantChunkTransportConfirmedFullPayloadV1, payload)
            turn = _turn_for_event(turns, confirmed.evidence_turn_id)
            if turn.kind is TurnKind.USER_RESPONSE and not turn.user_final_seen:
                raise _EvidenceValueError("user confirmation precedes user-final acceptance")
            if turn.snapshot is not None:
                raise _EvidenceValueError("confirmation follows the turn snapshot")
            if confirmed.chunk_ordinal != turn.next_chunk_ordinal:
                raise _EvidenceValueError("chunk ordinal is not the exact next ordinal")
            if confirmed.evidence_chunk_id in chunk_ids:
                raise _EvidenceValueError("chunk identity is reused")
            if turn.kind is TurnKind.USER_RESPONSE:
                if confirmed.evidence_segment_id is None or confirmed.text is None:
                    raise _EvidenceValueError("user confirmation must carry exact segment text")
                segments = cast(dict[str, str], turn.segments)
                if segments.get(confirmed.evidence_segment_id) != confirmed.text:
                    raise _EvidenceValueError("confirmed text does not match its segment")
            elif confirmed.evidence_segment_id is not None or confirmed.text is not None:
                raise _EvidenceValueError("non-user confirmation must be content-free")
            cast(set[str], turn.chunk_ids).add(confirmed.evidence_chunk_id)
            chunk_ids.add(confirmed.evidence_chunk_id)
            turn.next_chunk_ordinal += 1
            continue
        if kind is EventKind.TURN_SNAPSHOT:
            snapshot_payload = cast(TurnSnapshotPayloadV1, payload)
            turn = _turn_for_event(turns, snapshot_payload.evidence_turn_id)
            if turn.snapshot is not None:
                raise _EvidenceValueError("turn snapshot is duplicated")
            if turn.kind is TurnKind.USER_RESPONSE and not turn.user_final_seen:
                raise _EvidenceValueError("user turn snapshot precedes user-final acceptance")
            if snapshot_payload.turn_kind is not turn.kind:
                raise _EvidenceValueError("turn snapshot kind does not match opening")
            generated_count = turn.next_segment_ordinal - 1
            confirmed_count = turn.next_chunk_ordinal - 1
            if (
                snapshot_payload.generated_segment_count != generated_count
                or snapshot_payload.transport_confirmed_full_count != confirmed_count
            ):
                raise _EvidenceValueError("turn snapshot counts do not match preceding records")
            turn.snapshot = snapshot_payload
            continue
        if kind is EventKind.TURN_SETTLED:
            settled = cast(TurnSettledPayloadV1, payload)
            turn = _turn_for_event(turns, settled.evidence_turn_id)
            settlement_snapshot = turn.snapshot
            if settlement_snapshot is None:
                raise _EvidenceValueError("turn settlement precedes its snapshot")
            if (
                settled.generated_segment_count != settlement_snapshot.generated_segment_count
                or settled.transport_confirmed_full_count
                != settlement_snapshot.transport_confirmed_full_count
            ):
                raise _EvidenceValueError("settlement counts do not equal the turn snapshot")
            if (
                settled.terminal_disposition is TerminalDisposition.COMPLETED
                and not settled.context_committed
                and (
                    settlement_snapshot.generated_segment_count != 0
                    or settlement_snapshot.queued_chunk_count != 0
                    or settlement_snapshot.started_chunk_count != 0
                    or settlement_snapshot.transport_confirmed_full_count != 0
                )
            ):
                raise _EvidenceValueError(
                    "completed turns without context require zero output activity"
                )
            turn.settlement = settled
            turn.settled = True
            if turn.kind is TurnKind.REPLAY:
                source_id = cast(str, turn.replay_of_evidence_turn_id)
                source_turn = turns.get(source_id)
                if (
                    source_turn is None
                    or source_turn.active_replay_turn_id != settled.evidence_turn_id
                ):
                    raise _EvidenceValueError("replay settlement lost its source lineage")
                source_turn.active_replay_turn_id = None
                replay_can_retry = (
                    settled.terminal_disposition is TerminalDisposition.FAILED
                    and settled.terminal_reason is TerminalReason.TRANSPORT_FAILED
                    and settlement_snapshot.queued_chunk_count
                    > settlement_snapshot.transport_confirmed_full_count
                )
                source_turn.replay_retired = not replay_can_retry
            continue
        if kind is EventKind.BINDING_CLOSED:
            closed = cast(BindingClosedPayloadV1, payload)
            if binding_closed:
                raise _EvidenceValueError("binding close is duplicated")
            if closed.binding_id != binding_opened.binding_id:
                raise _EvidenceValueError("binding close identity does not match opening")
            if closed.close_reason in _ROLLOVER_CLOSE_REASONS:
                raise _EvidenceValueError(
                    "rollover close requires the atomic rollover DTO"
                )
            if any(not turn.settled for turn in turns.values()):
                raise _EvidenceValueError("binding closes with an unsettled turn")
            binding_closed = True
            continue
        if kind is EventKind.SESSION_SEAL_REQUESTED:
            seal = cast(SessionSealRequestedPayloadV1, payload)
            if not binding_closed:
                raise _EvidenceValueError("normal seal must follow binding close")
            if any(not turn.settled for turn in turns.values()):
                raise _EvidenceValueError("session seal contains an unsettled turn")
            if (
                seal.consent_epoch_id != session_opened.consent_epoch_id
                or seal.consent_version != session_opened.consent_version
                or seal.disclosure_digest != session_opened.disclosure_digest
            ):
                raise _EvidenceValueError("session seal lineage disagrees with session opening")
            session_terminal = True
            continue
        if kind is EventKind.SESSION_TAINTED:
            session_terminal = True
            continue
        raise _EvidenceValueError("unsupported event in V1 history")


_OpaqueT = TypeVar("_OpaqueT", bound="_OpaqueCapability")


class _OpaqueCapability:
    """Identity-only, nonserializable base for process-local bearer objects."""

    __slots__ = ()

    def __init_subclass__(cls) -> None:
        if any(
            base is not _OpaqueCapability and issubclass(base, _OpaqueCapability)
            for base in cls.__bases__
        ):
            raise TypeError("opaque V1 capability classes cannot be subclassed")
        super().__init_subclass__()

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("opaque V1 capabilities can be minted only by their owner")

    def __getattribute__(self, name: str) -> Any:
        # dataclasses offers no per-class asdict/astuple opt-out for required fields.
        frame = sys._getframe(1)
        try:
            for _ in range(4):
                if (
                    frame.f_globals.get("__name__") == "dataclasses"
                    and frame.f_code.co_name
                    in {"asdict", "astuple", "_asdict_inner", "_astuple_inner"}
                ):
                    raise TypeError(
                        "opaque V1 capabilities cannot be traversed as dataclasses"
                    )
                parent = frame.f_back
                if parent is None:
                    break
                frame = parent
        finally:
            del frame
        return object.__getattribute__(self, name)

    def _validate(self) -> None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} opaque>"

    def __copy__(self) -> NoReturn:
        raise TypeError("opaque V1 capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        raise TypeError("opaque V1 capabilities cannot be deep-copied")

    def __reduce__(self) -> NoReturn:
        raise TypeError("opaque V1 capabilities cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        raise TypeError("opaque V1 capabilities cannot be serialized")

    def __getstate__(self) -> NoReturn:
        raise TypeError("opaque V1 capabilities do not disclose state")

    def __setstate__(self, state: object) -> NoReturn:
        raise TypeError("opaque V1 capabilities cannot be retargeted")


def _opaque_dataclass(model_type: type[_OpaqueT]) -> type[_OpaqueT]:
    decorated = dataclass(
        frozen=True,
        slots=True,
        eq=False,
        init=False,
        repr=False,
    )(model_type)
    type.__setattr__(decorated, "__getstate__", _OpaqueCapability.__getstate__)
    type.__setattr__(decorated, "__setstate__", _OpaqueCapability.__setstate__)
    return decorated


def _validate_authority_common(
    *,
    protocol_version: object,
    owner_generation: object,
    binding_id: object,
    binding_generation: object,
    consent_epoch_id: object,
    logical_session_id: object,
) -> None:
    _require_version(protocol_version)
    _unsigned_63(owner_generation, "owner_generation")
    validate_canonical_uuid4(binding_id, field_name="binding_id")
    _unsigned_63(binding_generation, "binding_generation")
    validate_canonical_uuid4(consent_epoch_id, field_name="consent_epoch_id")
    validate_canonical_uuid4(logical_session_id, field_name="logical_session_id")


def _validate_final_input_fields(
    *,
    utterance_id: object,
    source: object,
    input_incarnation: object,
    media_incarnation: object,
    typed_sequence: object,
) -> None:
    validate_canonical_uuid4(utterance_id, field_name="utterance_id")
    exact_source = _require_enum(source, InputSource, "source")
    _unsigned_63(input_incarnation, "input_incarnation")
    if exact_source is InputSource.MICROPHONE:
        _unsigned_63(media_incarnation, "media_incarnation")
        if typed_sequence is not None:
            raise _EvidenceValueError("microphone authority forbids typed_sequence")
    elif exact_source is InputSource.TYPED:
        if media_incarnation is not None:
            raise _EvidenceValueError("typed authority forbids media_incarnation")
        _browser_integer(typed_sequence, "typed_sequence")
    else:
        raise _EvidenceValueError("input source has no V1 authority rule")


@_opaque_dataclass
class ProjectionReservation(_OpaqueCapability):
    def _validate(self) -> None:
        return None


@_opaque_dataclass
class CreateEpochReservationV1(_OpaqueCapability):
    def _validate(self) -> None:
        return None


@_opaque_dataclass
class FinalInputAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    utterance_id: str
    source: InputSource
    input_incarnation: int
    media_incarnation: int | None
    typed_sequence: int | None

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        _validate_final_input_fields(
            utterance_id=self.utterance_id,
            source=self.source,
            input_incarnation=self.input_incarnation,
            media_incarnation=self.media_incarnation,
            typed_sequence=self.typed_sequence,
        )


@_opaque_dataclass
class CommandAdmissionAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    utterance_id: str
    source: InputSource
    input_incarnation: int
    media_incarnation: int | None
    typed_sequence: int | None
    routing_serial: int
    routing_disposition: str

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        _validate_final_input_fields(
            utterance_id=self.utterance_id,
            source=self.source,
            input_incarnation=self.input_incarnation,
            media_incarnation=self.media_incarnation,
            typed_sequence=self.typed_sequence,
        )
        _unsigned_63(self.routing_serial, "routing_serial")
        _require_literal(self.routing_disposition, "command", "routing_disposition")


@_opaque_dataclass
class UserTurnAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    utterance_id: str
    source: InputSource
    input_incarnation: int
    media_incarnation: int | None
    typed_sequence: int | None
    routing_serial: int
    routing_disposition: str

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        _validate_final_input_fields(
            utterance_id=self.utterance_id,
            source=self.source,
            input_incarnation=self.input_incarnation,
            media_incarnation=self.media_incarnation,
            typed_sequence=self.typed_sequence,
        )
        _unsigned_63(self.routing_serial, "routing_serial")
        _require_literal(self.routing_disposition, "response", "routing_disposition")


@_opaque_dataclass
class ProactiveTurnAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    proactive_invocation_serial: int

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        _unsigned_63(self.proactive_invocation_serial, "proactive_invocation_serial")


@_opaque_dataclass
class ReplayTurnAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    replay_of_evidence_turn_id: str
    replay_generation: int

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        validate_canonical_uuid4(
            self.replay_of_evidence_turn_id,
            field_name="replay_of_evidence_turn_id",
        )
        _unsigned_63(self.replay_generation, "replay_generation")


@_opaque_dataclass
class BindingCloseAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    close_reason: BindingCloseReason

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        reason = _require_enum(self.close_reason, BindingCloseReason, "close_reason")
        if reason in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError(
                "rollover close reasons require RolloverAuthorityV1"
            )


@_opaque_dataclass
class RolloverAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    predecessor_logical_session_id: str
    successor_logical_session_id: str
    successor_expires_at_utc: str
    reason: BindingCloseReason

    def _validate(self) -> None:
        _require_version(self.protocol_version)
        _unsigned_63(self.owner_generation, "owner_generation")
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        _unsigned_63(self.binding_generation, "binding_generation")
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        predecessor = validate_canonical_uuid4(
            self.predecessor_logical_session_id,
            field_name="predecessor_logical_session_id",
        )
        successor = validate_canonical_uuid4(
            self.successor_logical_session_id,
            field_name="successor_logical_session_id",
        )
        validate_canonical_utc(
            self.successor_expires_at_utc,
            field_name="successor_expires_at_utc",
        )
        reason = _require_enum(self.reason, BindingCloseReason, "reason")
        if predecessor == successor:
            raise _EvidenceValueError("rollover predecessor and successor must differ")
        if reason not in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError("rollover authority requires a rollover close reason")


@_opaque_dataclass
class LifecycleSealAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    logical_session_id: str
    close_reason: BindingCloseReason
    final_event_sequence: int
    close_epoch: bool

    def _validate(self) -> None:
        _validate_authority_common(
            protocol_version=self.protocol_version,
            owner_generation=self.owner_generation,
            binding_id=self.binding_id,
            binding_generation=self.binding_generation,
            consent_epoch_id=self.consent_epoch_id,
            logical_session_id=self.logical_session_id,
        )
        reason = _require_enum(self.close_reason, BindingCloseReason, "close_reason")
        _bounded_integer(
            self.final_event_sequence,
            "final_event_sequence",
            1,
            _MAX_EVENT_SEQUENCE,
        )
        if not _require_exact_bool(self.close_epoch, "close_epoch"):
            raise _EvidenceValueError("lifecycle seal close_epoch must be literal true")
        if reason in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError("rollover close reasons require rollover authority")


@_opaque_dataclass
class SessionExpiryAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    consent_epoch_id: str
    logical_session_id: str
    expires_at_utc: str
    deadline_admission_ordinal: int
    mode: ExpiryMode

    def _validate(self) -> None:
        _require_version(self.protocol_version)
        _unsigned_63(self.owner_generation, "owner_generation")
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        validate_canonical_uuid4(self.logical_session_id, field_name="logical_session_id")
        validate_canonical_utc(self.expires_at_utc, field_name="expires_at_utc")
        _unsigned_63(self.deadline_admission_ordinal, "deadline_admission_ordinal")
        _require_enum(self.mode, ExpiryMode, "mode")


@_opaque_dataclass
class ConsentCreateAuthorityV1(_OpaqueCapability):
    protocol_version: int
    binding_id: str
    binding_generation: int
    control_sequence: int
    control_fingerprint_hash: str
    consent_version: str
    disclosure_digest: str
    retention_hours: int
    microphone_accepted: bool
    typed_accepted: bool
    projection_reservation: ProjectionReservation
    create_epoch_reservation: CreateEpochReservationV1

    def _validate(self) -> None:
        _require_version(self.protocol_version)
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        _unsigned_63(self.binding_generation, "binding_generation")
        _browser_integer(self.control_sequence, "control_sequence")
        validate_sha256_hex(
            self.control_fingerprint_hash,
            field_name="control_fingerprint_hash",
        )
        _require_literal(self.consent_version, CONSENT_VERSION, "consent_version")
        validate_sha256_hex(self.disclosure_digest, field_name="disclosure_digest")
        _bounded_integer(self.retention_hours, "retention_hours", 1, _MAX_RETENTION_HOURS)
        microphone = _require_exact_bool(self.microphone_accepted, "microphone_accepted")
        typed = _require_exact_bool(self.typed_accepted, "typed_accepted")
        if not microphone and not typed:
            raise _EvidenceValueError("consent authority requires an accepted source")
        if type(self.projection_reservation) is not ProjectionReservation:
            raise _EvidenceTypeError(
                "projection_reservation must be an exact ProjectionReservation"
            )
        if type(self.create_epoch_reservation) is not CreateEpochReservationV1:
            raise _EvidenceTypeError(
                "create_epoch_reservation must be an exact CreateEpochReservationV1"
            )


@_opaque_dataclass
class ConsentRevokeAuthorityV1(_OpaqueCapability):
    protocol_version: int
    binding_id: str
    binding_generation: int
    consent_epoch_id: str
    revoke_gate_generation: int
    control_sequence: int
    control_fingerprint_hash: str
    projection_reservation: ProjectionReservation

    def _validate(self) -> None:
        _require_version(self.protocol_version)
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        _unsigned_63(self.binding_generation, "binding_generation")
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        _unsigned_63(self.revoke_gate_generation, "revoke_gate_generation")
        _browser_integer(self.control_sequence, "control_sequence")
        validate_sha256_hex(
            self.control_fingerprint_hash,
            field_name="control_fingerprint_hash",
        )
        if type(self.projection_reservation) is not ProjectionReservation:
            raise _EvidenceTypeError(
                "projection_reservation must be an exact ProjectionReservation"
            )


@_opaque_dataclass
class LifecycleDrainAuthorityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    final_admission_ordinal: int

    def _validate(self) -> None:
        _require_version(self.protocol_version)
        _unsigned_63(self.owner_generation, "owner_generation")
        _unsigned_63(self.final_admission_ordinal, "final_admission_ordinal")


@_opaque_dataclass
class TerminalCauseCapabilityV1(_OpaqueCapability):
    protocol_version: int
    owner_generation: int
    logical_session_id: str
    evidence_turn_id: str
    lease_serial: int
    sink_token: bytes

    def _validate(self) -> None:
        _require_version(self.protocol_version)
        _unsigned_63(self.owner_generation, "owner_generation")
        validate_canonical_uuid4(self.logical_session_id, field_name="logical_session_id")
        validate_canonical_uuid4(self.evidence_turn_id, field_name="evidence_turn_id")
        _unsigned_63(self.lease_serial, "lease_serial")
        if type(self.sink_token) is not bytes:
            raise _EvidenceTypeError("sink_token must be exact built-in bytes")
        if len(self.sink_token) != 16:
            raise _EvidenceValueError("sink_token must contain exactly 16 opaque bytes")


@dataclass(frozen=True, slots=True)
class CommandRoutingResultV1(_FinalModel):
    outcome: CommandRoutingOutcome
    command_disposition: CommandDisposition | None
    user_turn_authority: UserTurnAuthorityV1 | None

    def __post_init__(self) -> None:
        outcome = _require_enum(self.outcome, CommandRoutingOutcome, "outcome")
        if self.command_disposition is None:
            command_disposition = None
        else:
            command_disposition = _require_enum(
                self.command_disposition,
                CommandDisposition,
                "command_disposition",
            )
        if self.user_turn_authority is None:
            user_authority = None
        elif type(self.user_turn_authority) is UserTurnAuthorityV1:
            user_authority = self.user_turn_authority
            user_authority._validate()
        else:
            raise _EvidenceTypeError(
                "user_turn_authority must be an exact UserTurnAuthorityV1 or None"
            )
        if outcome is CommandRoutingOutcome.NOT_COMMAND:
            valid = command_disposition is None and user_authority is not None
        elif outcome is CommandRoutingOutcome.ACCEPTED:
            valid = command_disposition is not None and user_authority is None
        elif outcome in (
            CommandRoutingOutcome.REJECTED,
            CommandRoutingOutcome.INVALID,
        ):
            valid = command_disposition is None and user_authority is None
        else:
            raise _EvidenceValueError("command routing outcome has no V1 field rule")
        if not valid:
            raise _EvidenceValueError("command routing result fields contradict its outcome")

    def __bool__(self) -> NoReturn:
        raise TypeError("CommandRoutingResultV1 has no boolean compatibility surface")


@dataclass(frozen=True, slots=True)
class EvidenceDiagnosticsV1(_FinalModel):
    protocol_version: int
    owner_state: OwnerState
    capture_state: CaptureState
    sticky_fault: WriterFault | None
    queue_record_count: int
    queue_canonical_bytes: int
    active_lease_count: int
    pending_revoke: bool
    purge_required: bool

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        _require_enum(self.owner_state, OwnerState, "owner_state")
        _require_enum(self.capture_state, CaptureState, "capture_state")
        if self.sticky_fault is not None:
            _require_enum(self.sticky_fault, WriterFault, "sticky_fault")
        _count(self.queue_record_count, "queue_record_count", _MAX_QUEUE_RECORDS)
        _count(
            self.queue_canonical_bytes,
            "queue_canonical_bytes",
            _MAX_QUEUE_CANONICAL_BYTES,
        )
        _count(self.active_lease_count, "active_lease_count", _MAX_QUEUE_RECORDS)
        _require_exact_bool(self.pending_revoke, "pending_revoke")
        _require_exact_bool(self.purge_required, "purge_required")


@dataclass(frozen=True, slots=True)
class TerminalResolutionV1(_FinalModel):
    terminal_disposition: TerminalDisposition
    terminal_reason: TerminalReason

    def __post_init__(self) -> None:
        disposition = _require_enum(
            self.terminal_disposition,
            TerminalDisposition,
            "terminal_disposition",
        )
        reason = _require_enum(self.terminal_reason, TerminalReason, "terminal_reason")
        if not _valid_terminal_pair(disposition, reason):
            raise _EvidenceValueError("terminal resolution is not a valid V1 pair")


@dataclass(frozen=True, slots=True)
class SettledTerminalOutcomeV1(_FinalModel):
    """The terminal outcome admitted as one ordered terminal batch."""

    terminal_disposition: TerminalDisposition
    terminal_reason: TerminalReason
    context_committed: bool

    def __post_init__(self) -> None:
        disposition = _require_enum(
            self.terminal_disposition,
            TerminalDisposition,
            "terminal_disposition",
        )
        reason = _require_enum(self.terminal_reason, TerminalReason, "terminal_reason")
        _require_exact_bool(self.context_committed, "context_committed")
        if not _valid_terminal_pair(disposition, reason):
            raise _EvidenceValueError("settled terminal outcome is not a valid V1 pair")


_TERMINAL_PRIORITY: tuple[tuple[TerminalDisposition, tuple[TerminalReason, ...]], ...] = (
    (TerminalDisposition.REVOKED, (TerminalReason.CONSENT_REVOKED,)),
    (TerminalDisposition.FAILED, _FAILED_REASONS),
    (TerminalDisposition.INTERRUPTED, _INTERRUPTED_REASONS),
    (TerminalDisposition.CANCELLED, _CANCELLED_REASONS),
    (
        TerminalDisposition.COMPLETED,
        (TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,),
    ),
)


def resolve_terminal_causes(
    causes: Sequence[TerminalReason] | set[TerminalReason] | frozenset[TerminalReason],
    *,
    context_committed: bool | None = None,
    zero_output_no_op: bool = False,
) -> TerminalResolutionV1:
    """Resolve a nonempty exact cause collection by fixed V1 precedence."""

    if type(causes) not in (tuple, list, set, frozenset):
        raise _EvidenceTypeError("terminal causes must use an exact built-in collection")
    if context_committed is None:
        committed = False
    else:
        committed = _require_exact_bool(context_committed, "context_committed")
    zero_output = _require_exact_bool(zero_output_no_op, "zero_output_no_op")
    exact_causes: list[TerminalReason] = []
    for cause in causes:
        exact_causes.append(_require_enum(cause, TerminalReason, "terminal cause"))
    if not exact_causes:
        raise _EvidenceValueError("terminal cause set must not be empty")
    cause_set = set(exact_causes)
    for disposition, ordered_reasons in _TERMINAL_PRIORITY:
        for reason in ordered_reasons:
            if reason not in cause_set:
                continue
            if disposition is TerminalDisposition.COMPLETED and not (
                committed or zero_output
            ):
                raise _EvidenceValueError(
                    "completion requires context commit or explicit zero-output no-op"
                )
            return TerminalResolutionV1(
                terminal_disposition=disposition,
                terminal_reason=reason,
            )
    raise _EvidenceValueError("terminal cause set contains no supported V1 cause")


resolve_terminal_causes_v1 = resolve_terminal_causes


_PERSISTENT_EDGES = frozenset(
    {
        (PersistentSessionState.OPEN, PersistentSessionState.SEALED),
        (PersistentSessionState.OPEN, PersistentSessionState.TAINTED),
        (PersistentSessionState.SEALED, PersistentSessionState.TAINTED),
    }
)


def validate_persistent_state_transition(
    source: PersistentSessionState,
    target: PersistentSessionState,
) -> PersistentSessionState:
    exact_source = _require_enum(source, PersistentSessionState, "source")
    exact_target = _require_enum(target, PersistentSessionState, "target")
    if (exact_source, exact_target) not in _PERSISTENT_EDGES:
        raise _EvidenceValueError("persistent session transition is not a V1 edge")
    return exact_target


_RUNTIME_EDGES = frozenset(
    {
        (RuntimeSessionPhase.OPEN, RuntimeSessionPhase.EXPIRING),
        (RuntimeSessionPhase.OPEN, RuntimeSessionPhase.CLOSING),
        (RuntimeSessionPhase.OPEN, RuntimeSessionPhase.REVOKING),
        (RuntimeSessionPhase.EXPIRING, RuntimeSessionPhase.OPEN),
        (RuntimeSessionPhase.EXPIRING, RuntimeSessionPhase.REVOKING),
        (RuntimeSessionPhase.EXPIRING, RuntimeSessionPhase.STOPPED),
        (RuntimeSessionPhase.CLOSING, RuntimeSessionPhase.SEAL_QUEUED),
        (RuntimeSessionPhase.CLOSING, RuntimeSessionPhase.REVOKING),
        (RuntimeSessionPhase.CLOSING, RuntimeSessionPhase.STOPPED),
        (RuntimeSessionPhase.SEAL_QUEUED, RuntimeSessionPhase.REVOKING),
        (RuntimeSessionPhase.SEAL_QUEUED, RuntimeSessionPhase.STOPPED),
        (RuntimeSessionPhase.REVOKING, RuntimeSessionPhase.STOPPED),
    }
)


def validate_runtime_phase_transition(
    source: RuntimeSessionPhase,
    target: RuntimeSessionPhase,
) -> RuntimeSessionPhase:
    exact_source = _require_enum(source, RuntimeSessionPhase, "source")
    exact_target = _require_enum(target, RuntimeSessionPhase, "target")
    if (exact_source, exact_target) not in _RUNTIME_EDGES:
        raise _EvidenceValueError("runtime session transition is not a V1 edge")
    return exact_target


@dataclass(frozen=True, slots=True)
class EvidenceEligibilityFactsV1(_FinalModel):
    text_persisted: bool
    text_ever_persisted: bool
    purge_verified: bool
    purge_failed: bool
    turn_kind: TurnKind
    snapshot_count: int
    snapshot_valid: bool
    settlement_count: int
    settlement_valid: bool
    terminal_disposition: TerminalDisposition | None
    terminal_reason: TerminalReason | None
    context_committed: bool
    session_state: PersistentSessionState
    gap_free: bool
    hash_valid: bool
    conflict_free: bool
    epoch_state: ConsentEpochState
    unexpired: bool
    pending_erasure: bool

    def __post_init__(self) -> None:
        text_persisted = _require_exact_bool(self.text_persisted, "text_persisted")
        text_ever_persisted = _require_exact_bool(
            self.text_ever_persisted,
            "text_ever_persisted",
        )
        purge_verified = _require_exact_bool(self.purge_verified, "purge_verified")
        purge_failed = _require_exact_bool(self.purge_failed, "purge_failed")
        kind = _require_enum(self.turn_kind, TurnKind, "turn_kind")
        _count(self.snapshot_count, "snapshot_count", _MAX_EVENT_SEQUENCE)
        _require_exact_bool(self.snapshot_valid, "snapshot_valid")
        _count(self.settlement_count, "settlement_count", _MAX_EVENT_SEQUENCE)
        _require_exact_bool(self.settlement_valid, "settlement_valid")
        if self.terminal_disposition is None:
            disposition = None
        else:
            disposition = _require_enum(
                self.terminal_disposition,
                TerminalDisposition,
                "terminal_disposition",
            )
        if self.terminal_reason is None:
            reason = None
        else:
            reason = _require_enum(self.terminal_reason, TerminalReason, "terminal_reason")
        _require_exact_bool(self.context_committed, "context_committed")
        _require_enum(self.session_state, PersistentSessionState, "session_state")
        _require_exact_bool(self.gap_free, "gap_free")
        _require_exact_bool(self.conflict_free, "conflict_free")
        _require_enum(self.epoch_state, ConsentEpochState, "epoch_state")
        _require_exact_bool(self.unexpired, "unexpired")
        pending_erasure = _require_exact_bool(
            self.pending_erasure,
            "pending_erasure",
        )
        _require_exact_bool(self.hash_valid, "hash_valid")
        if text_persisted and not text_ever_persisted:
            raise _EvidenceValueError("current text requires persisted-history evidence")
        if kind is not TurnKind.USER_RESPONSE and text_ever_persisted:
            raise _EvidenceValueError("proactive and replay turns cannot persist text")
        if purge_failed and not text_ever_persisted:
            raise _EvidenceValueError("purge failure requires persisted-history evidence")
        if purge_verified and (
            text_persisted
            or not text_ever_persisted
            or purge_failed
            or pending_erasure
        ):
            raise _EvidenceValueError(
                "verified-purged facts contradict current or pending purge state"
            )
        if (disposition is None) != (reason is None):
            raise _EvidenceValueError("terminal disposition and reason must be paired")
        if (
            disposition is not None
            and reason is not None
            and not _valid_terminal_pair(disposition, reason)
        ):
            raise _EvidenceValueError("eligibility terminal facts are contradictory")


TurnEligibilityFactsV1 = EvidenceEligibilityFactsV1


def evaluate_eligibility(facts: EvidenceEligibilityFactsV1) -> EligibilityOutcome:
    if type(facts) is not EvidenceEligibilityFactsV1:
        raise _EvidenceTypeError("facts must be an exact EvidenceEligibilityFactsV1")
    checked = EvidenceEligibilityFactsV1(
        text_persisted=facts.text_persisted,
        text_ever_persisted=facts.text_ever_persisted,
        purge_verified=facts.purge_verified,
        purge_failed=facts.purge_failed,
        turn_kind=facts.turn_kind,
        snapshot_count=facts.snapshot_count,
        snapshot_valid=facts.snapshot_valid,
        settlement_count=facts.settlement_count,
        settlement_valid=facts.settlement_valid,
        terminal_disposition=facts.terminal_disposition,
        terminal_reason=facts.terminal_reason,
        context_committed=facts.context_committed,
        session_state=facts.session_state,
        gap_free=facts.gap_free,
        conflict_free=facts.conflict_free,
        epoch_state=facts.epoch_state,
        unexpired=facts.unexpired,
        pending_erasure=facts.pending_erasure,
        hash_valid=facts.hash_valid,
    )
    if checked.purge_verified:
        return EligibilityOutcome.PURGED
    if not checked.text_ever_persisted:
        return EligibilityOutcome.NEVER_PERSISTED
    if not checked.text_persisted:
        return EligibilityOutcome.PERSISTED_BUT_EXCLUDED
    if (
        checked.turn_kind is TurnKind.USER_RESPONSE
        and checked.snapshot_count == 1
        and checked.snapshot_valid
        and checked.settlement_count == 1
        and checked.settlement_valid
        and checked.terminal_disposition is TerminalDisposition.COMPLETED
        and checked.terminal_reason is TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED
        and checked.context_committed
        and checked.session_state is PersistentSessionState.SEALED
        and checked.gap_free
        and checked.hash_valid
        and checked.conflict_free
        and checked.epoch_state is ConsentEpochState.CLOSED
        and checked.unexpired
        and not checked.pending_erasure
        and not checked.purge_failed
    ):
        return EligibilityOutcome.ELIGIBLE
    return EligibilityOutcome.PERSISTED_BUT_EXCLUDED


def is_turn_eligible_v1(facts: EvidenceEligibilityFactsV1) -> bool:
    return evaluate_eligibility(facts) is EligibilityOutcome.ELIGIBLE


def writer_fault_to_control_error(
    fault: WriterFault,
    *,
    revoke_before_durable: bool = False,
) -> ControlError:
    exact_fault = _require_enum(fault, WriterFault, "fault")
    before_durable = _require_exact_bool(
        revoke_before_durable,
        "revoke_before_durable",
    )
    if before_durable:
        return ControlError.REVOKE_NOT_DURABLE
    if exact_fault in (
        WriterFault.UNSUPPORTED_PLATFORM,
        WriterFault.PATH_INVALID,
        WriterFault.ASSET_MISMATCH,
    ):
        return ControlError.CAPTURE_UNAVAILABLE
    if exact_fault in (
        WriterFault.OWNERSHIP_UNAVAILABLE,
        WriterFault.STORE_CORRUPT,
        WriterFault.PURGE_REQUIRED,
        WriterFault.SQLITE_FAULT,
        WriterFault.QUOTA_UNAVAILABLE,
        WriterFault.DRAIN_TIMEOUT,
    ):
        return ControlError.WRITER_UNAVAILABLE
    raise _EvidenceValueError("writer fault has no V1 control-error mapping")


def control_error_http_status(error: ControlError) -> int:
    exact_error = _require_enum(error, ControlError, "error")
    if exact_error is ControlError.INVALID_CONTROL:
        return 400
    if exact_error is ControlError.UNAUTHORIZED:
        return 401
    if exact_error is ControlError.FORBIDDEN:
        return 403
    if exact_error in (
        ControlError.CONTROL_SEQUENCE_CONFLICT,
        ControlError.CAPTURE_UNAVAILABLE,
        ControlError.CONTROL_PENDING,
    ):
        return 409
    if exact_error in (
        ControlError.STATUS_CAPACITY_UNAVAILABLE,
        ControlError.CONTROL_TIMEOUT,
        ControlError.WRITER_UNAVAILABLE,
        ControlError.REVOKE_NOT_DURABLE,
        ControlError.PURGE_FAILED,
    ):
        return 503
    raise _EvidenceValueError("control error has no V1 HTTP mapping")


def control_result_http_status(result: ControlResult) -> int:
    exact_result = _require_enum(result, ControlResult, "result")
    if exact_result in (ControlResult.CONSENT_ACTIVATED, ControlResult.PURGE_COMPLETED):
        return 200
    if exact_result is ControlResult.REVOKE_DURABLY_SCHEDULED:
        return 202
    raise _EvidenceValueError("control result has no V1 HTTP mapping")


@dataclass(frozen=True, slots=True)
class QueuedEvidenceRecordV1(_FinalModel):
    protocol_version: int
    snapshot: EvidenceSnapshotV1
    admission_ordinal: int
    reservation_class: QueueReservationClass
    lease_open_ordinal: int | None

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        snapshot = _copy_snapshot(self.snapshot)
        _unsigned_63(self.admission_ordinal, "admission_ordinal")
        reservation = _require_enum(
            self.reservation_class,
            QueueReservationClass,
            "reservation_class",
        )
        if self.lease_open_ordinal is None:
            lease_open = None
        else:
            lease_open = _unsigned_63(self.lease_open_ordinal, "lease_open_ordinal")
        if reservation is QueueReservationClass.TERMINAL:
            if lease_open is None:
                raise _EvidenceValueError("terminal records require lease_open_ordinal")
            if snapshot.event_kind not in (
                EventKind.TURN_SNAPSHOT,
                EventKind.TURN_SETTLED,
            ):
                raise _EvidenceValueError(
                    "terminal reservations accept only turn_snapshot or turn_settled"
                )
        else:
            if lease_open is not None:
                raise _EvidenceValueError("ordinary records forbid lease_open_ordinal")
            if snapshot.event_kind in (
                EventKind.TURN_SNAPSHOT,
                EventKind.TURN_SETTLED,
            ):
                raise _EvidenceValueError(
                    "turn terminal records require a terminal reservation"
                )
        object.__setattr__(self, "snapshot", snapshot)


@dataclass(frozen=True, slots=True)
class BindingCloseV1(_FinalModel):
    protocol_version: int
    binding_id: str
    consent_epoch_id: str
    logical_session_id: str
    admission_ordinal: int
    snapshot: EvidenceSnapshotV1

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        binding_id = validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        session_id = validate_canonical_uuid4(
            self.logical_session_id,
            field_name="logical_session_id",
        )
        _unsigned_63(self.admission_ordinal, "admission_ordinal")
        snapshot = _copy_snapshot(self.snapshot)
        if snapshot.event_kind is not EventKind.BINDING_CLOSED:
            raise _EvidenceValueError("BindingCloseV1 requires a binding_closed snapshot")
        payload = cast(BindingClosedPayloadV1, snapshot.payload)
        if payload.close_reason in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError(
                "rollover close reasons require RolloverSessionV1"
            )
        if snapshot.logical_session_id != session_id or payload.binding_id != binding_id:
            raise _EvidenceValueError("binding close snapshot lineage does not match its DTO")
        object.__setattr__(self, "snapshot", snapshot)


@dataclass(frozen=True, slots=True)
class CreateEpochV1(_FinalModel):
    protocol_version: int
    installation_id: str
    producer_instance_id: str
    consent_epoch_id: str
    logical_session_id: str
    binding_id: str
    binding_generation: int
    consent_version: str
    disclosure_digest: str
    retention_hours: int
    microphone_accepted: bool
    typed_accepted: bool
    control_sequence: int
    control_fingerprint_hash: str
    session_opened: EvidenceSnapshotV1
    binding_opened: EvidenceSnapshotV1

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        installation_id = validate_canonical_uuid4(
            self.installation_id,
            field_name="installation_id",
        )
        producer_id = validate_canonical_uuid4(
            self.producer_instance_id,
            field_name="producer_instance_id",
        )
        epoch_id = validate_canonical_uuid4(
            self.consent_epoch_id,
            field_name="consent_epoch_id",
        )
        session_id = validate_canonical_uuid4(
            self.logical_session_id,
            field_name="logical_session_id",
        )
        binding_id = validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        binding_generation = _unsigned_63(self.binding_generation, "binding_generation")
        consent_version = _require_literal(
            self.consent_version,
            CONSENT_VERSION,
            "consent_version",
        )
        digest = validate_sha256_hex(
            self.disclosure_digest,
            field_name="disclosure_digest",
        )
        retention = _bounded_integer(
            self.retention_hours,
            "retention_hours",
            1,
            _MAX_RETENTION_HOURS,
        )
        microphone = _require_exact_bool(self.microphone_accepted, "microphone_accepted")
        typed = _require_exact_bool(self.typed_accepted, "typed_accepted")
        if not microphone and not typed:
            raise _EvidenceValueError("epoch creation requires an accepted source")
        _browser_integer(self.control_sequence, "control_sequence")
        validate_sha256_hex(
            self.control_fingerprint_hash,
            field_name="control_fingerprint_hash",
        )
        session_opened = _copy_snapshot(self.session_opened)
        binding_opened = _copy_snapshot(self.binding_opened)
        if session_opened.event_kind is not EventKind.SESSION_OPENED:
            raise _EvidenceValueError("CreateEpochV1 sequence 1 must be session_opened")
        if binding_opened.event_kind is not EventKind.BINDING_OPENED:
            raise _EvidenceValueError("CreateEpochV1 sequence 2 must be binding_opened")
        if session_opened.event_sequence != 1 or binding_opened.event_sequence != 2:
            raise _EvidenceValueError("CreateEpochV1 opening sequences must be exactly 1 and 2")
        if session_opened.event_id == binding_opened.event_id:
            raise _EvidenceValueError("CreateEpochV1 opening event IDs must be distinct")
        for snapshot in (session_opened, binding_opened):
            if (
                snapshot.installation_id != installation_id
                or snapshot.producer_instance_id != producer_id
                or snapshot.logical_session_id != session_id
            ):
                raise _EvidenceValueError("CreateEpochV1 envelope lineage does not match")
        opened_payload = cast(SessionOpenedPayloadV1, session_opened.payload)
        binding_payload = cast(BindingOpenedPayloadV1, binding_opened.payload)
        if (
            opened_payload.consent_epoch_id != epoch_id
            or opened_payload.binding_id != binding_id
            or opened_payload.consent_version != consent_version
            or opened_payload.disclosure_digest != digest
            or opened_payload.retention_hours != retention
            or opened_payload.microphone_accepted is not microphone
            or opened_payload.typed_accepted is not typed
            or opened_payload.predecessor_session_id is not None
            or binding_payload.binding_id != binding_id
            or binding_payload.binding_generation != binding_generation
        ):
            raise _EvidenceValueError("CreateEpochV1 opening payload lineage does not match")
        if microphone and not binding_payload.microphone_available:
            raise _EvidenceValueError("accepted microphone source must be available")
        if typed and not binding_payload.typed_available:
            raise _EvidenceValueError("accepted typed source must be available")
        object.__setattr__(self, "session_opened", session_opened)
        object.__setattr__(self, "binding_opened", binding_opened)


@dataclass(frozen=True, slots=True)
class RolloverSessionV1(_FinalModel):
    protocol_version: int
    consent_epoch_id: str
    binding_id: str
    binding_generation: int
    predecessor_logical_session_id: str
    successor_logical_session_id: str
    predecessor_final_event_sequence: int
    successor_expires_at_utc: str
    admission_ordinal: int
    snapshots: tuple[EvidenceSnapshotV1, ...]

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        epoch_id = validate_canonical_uuid4(
            self.consent_epoch_id,
            field_name="consent_epoch_id",
        )
        binding_id = validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        binding_generation = _unsigned_63(self.binding_generation, "binding_generation")
        predecessor = validate_canonical_uuid4(
            self.predecessor_logical_session_id,
            field_name="predecessor_logical_session_id",
        )
        successor = validate_canonical_uuid4(
            self.successor_logical_session_id,
            field_name="successor_logical_session_id",
        )
        final_sequence = _bounded_integer(
            self.predecessor_final_event_sequence,
            "predecessor_final_event_sequence",
            4,
            _MAX_EVENT_SEQUENCE,
        )
        validate_canonical_utc(
            self.successor_expires_at_utc,
            field_name="successor_expires_at_utc",
        )
        _unsigned_63(self.admission_ordinal, "admission_ordinal")
        if predecessor == successor:
            raise _EvidenceValueError("rollover predecessor and successor must differ")
        if type(self.snapshots) is not tuple:
            raise _EvidenceTypeError("rollover snapshots must be an exact built-in tuple")
        if len(self.snapshots) != 4:
            raise _EvidenceValueError("rollover requires exactly four snapshots")
        snapshots = tuple(_copy_snapshot(item) for item in self.snapshots)
        expected_kinds = (
            EventKind.BINDING_CLOSED,
            EventKind.SESSION_SEAL_REQUESTED,
            EventKind.SESSION_OPENED,
            EventKind.BINDING_OPENED,
        )
        if tuple(item.event_kind for item in snapshots) != expected_kinds:
            raise _EvidenceValueError("rollover snapshot kinds or order are invalid")
        if len({item.event_id for item in snapshots}) != 4:
            raise _EvidenceValueError("rollover snapshot event IDs must be distinct")
        close_snapshot, seal_snapshot, successor_open, successor_binding = snapshots
        if (
            close_snapshot.logical_session_id != predecessor
            or seal_snapshot.logical_session_id != predecessor
            or successor_open.logical_session_id != successor
            or successor_binding.logical_session_id != successor
        ):
            raise _EvidenceValueError("rollover snapshot session lineage does not match")
        if (
            close_snapshot.event_sequence != final_sequence - 1
            or seal_snapshot.event_sequence != final_sequence
            or successor_open.event_sequence != 1
            or successor_binding.event_sequence != 2
        ):
            raise _EvidenceValueError("rollover snapshot sequences are not exact")
        first = snapshots[0]
        if any(
            item.installation_id != first.installation_id
            or item.producer_instance_id != first.producer_instance_id
            for item in snapshots[1:]
        ):
            raise _EvidenceValueError("rollover crosses installation or producer lineage")
        close_payload = cast(BindingClosedPayloadV1, close_snapshot.payload)
        seal_payload = cast(SessionSealRequestedPayloadV1, seal_snapshot.payload)
        successor_payload = cast(SessionOpenedPayloadV1, successor_open.payload)
        successor_binding_payload = cast(
            BindingOpenedPayloadV1,
            successor_binding.payload,
        )
        if close_payload.close_reason not in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError("rollover close snapshot lacks a rollover reason")
        if (
            close_payload.binding_id != binding_id
            or seal_payload.final_event_sequence != final_sequence
            or seal_payload.consent_epoch_id != epoch_id
            or successor_payload.consent_epoch_id != epoch_id
            or successor_payload.binding_id != binding_id
            or successor_payload.predecessor_session_id != predecessor
            or successor_binding_payload.binding_id != binding_id
            or successor_binding_payload.binding_generation != binding_generation
            or successor_payload.consent_version != seal_payload.consent_version
            or successor_payload.disclosure_digest != seal_payload.disclosure_digest
        ):
            raise _EvidenceValueError("rollover payload cross-links do not match")
        if (
            successor_payload.microphone_accepted
            and not successor_binding_payload.microphone_available
        ):
            raise _EvidenceValueError("accepted microphone source must be available")
        if successor_payload.typed_accepted and not successor_binding_payload.typed_available:
            raise _EvidenceValueError("accepted typed source must be available")
        object.__setattr__(self, "snapshots", snapshots)


@dataclass(frozen=True, slots=True)
class ExpireSessionV1(_FinalModel):
    protocol_version: int
    owner_generation: int
    logical_session_id: str
    consent_epoch_id: str
    expires_at_utc: str
    last_admission_ordinal: int
    erasure_request_id: str
    mode: ExpiryMode

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        _unsigned_63(self.owner_generation, "owner_generation")
        validate_canonical_uuid4(self.logical_session_id, field_name="logical_session_id")
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        validate_canonical_utc(self.expires_at_utc, field_name="expires_at_utc")
        _unsigned_63(self.last_admission_ordinal, "last_admission_ordinal")
        validate_canonical_uuid4(
            self.erasure_request_id,
            field_name="erasure_request_id",
        )
        _require_enum(self.mode, ExpiryMode, "mode")


@dataclass(frozen=True, slots=True)
class RevokeRequestV1(_FinalModel):
    protocol_version: int
    erasure_request_id: str
    consent_epoch_id: str
    control_sequence: int
    control_fingerprint_hash: str
    last_admission_ordinal: int

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        validate_canonical_uuid4(
            self.erasure_request_id,
            field_name="erasure_request_id",
        )
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        _browser_integer(self.control_sequence, "control_sequence")
        validate_sha256_hex(
            self.control_fingerprint_hash,
            field_name="control_fingerprint_hash",
        )
        _unsigned_63(self.last_admission_ordinal, "last_admission_ordinal")


@dataclass(frozen=True, slots=True)
class RevokeFinalizeV1(_FinalModel):
    protocol_version: int
    erasure_request_id: str
    consent_epoch_id: str
    control_sequence: int
    control_fingerprint_hash: str
    final_admission_ordinal: int

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        validate_canonical_uuid4(
            self.erasure_request_id,
            field_name="erasure_request_id",
        )
        validate_canonical_uuid4(self.consent_epoch_id, field_name="consent_epoch_id")
        _browser_integer(self.control_sequence, "control_sequence")
        validate_sha256_hex(
            self.control_fingerprint_hash,
            field_name="control_fingerprint_hash",
        )
        _unsigned_63(self.final_admission_ordinal, "final_admission_ordinal")


@dataclass(frozen=True, slots=True)
class SealEpochV1(_FinalModel):
    protocol_version: int
    consent_epoch_id: str
    logical_session_id: str
    binding_id: str
    final_event_sequence: int
    close_reason: BindingCloseReason
    close_epoch: bool
    admission_ordinal: int
    snapshot: EvidenceSnapshotV1

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        epoch_id = validate_canonical_uuid4(
            self.consent_epoch_id,
            field_name="consent_epoch_id",
        )
        session_id = validate_canonical_uuid4(
            self.logical_session_id,
            field_name="logical_session_id",
        )
        validate_canonical_uuid4(self.binding_id, field_name="binding_id")
        final_sequence = _bounded_integer(
            self.final_event_sequence,
            "final_event_sequence",
            1,
            _MAX_EVENT_SEQUENCE,
        )
        reason = _require_enum(self.close_reason, BindingCloseReason, "close_reason")
        if reason in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError("rollover close reasons require RolloverSessionV1")
        if not _require_exact_bool(self.close_epoch, "close_epoch"):
            raise _EvidenceValueError("SealEpochV1 close_epoch must be literal true")
        _unsigned_63(self.admission_ordinal, "admission_ordinal")
        snapshot = _copy_snapshot(self.snapshot)
        if snapshot.event_kind is not EventKind.SESSION_SEAL_REQUESTED:
            raise _EvidenceValueError("SealEpochV1 requires a seal-requested snapshot")
        payload = cast(SessionSealRequestedPayloadV1, snapshot.payload)
        if (
            snapshot.logical_session_id != session_id
            or snapshot.event_sequence != final_sequence
            or payload.final_event_sequence != final_sequence
            or payload.consent_epoch_id != epoch_id
        ):
            raise _EvidenceValueError("SealEpochV1 snapshot lineage does not match")
        object.__setattr__(self, "snapshot", snapshot)


@dataclass(frozen=True, slots=True)
class MaintenanceV1(_FinalModel):
    protocol_version: int
    erasure_reason: ErasureReason
    erasure_scope: ErasureScope
    erasure_request_id: str
    scope_id: str
    deadline_admission_ordinal: int
    artifact_manifest_version: int

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        reason = _require_enum(self.erasure_reason, ErasureReason, "erasure_reason")
        scope = _require_enum(self.erasure_scope, ErasureScope, "erasure_scope")
        validate_canonical_uuid4(
            self.erasure_request_id,
            field_name="erasure_request_id",
        )
        _unsigned_63(self.deadline_admission_ordinal, "deadline_admission_ordinal")
        _require_version(self.artifact_manifest_version, "artifact_manifest_version")
        if (reason is ErasureReason.TTL and scope is ErasureScope.SESSION) or (
            reason in (ErasureReason.REVOKED, ErasureReason.UNCLEAN_EPOCH)
            and scope is ErasureScope.CONSENT_EPOCH
        ):
            validate_canonical_uuid4(self.scope_id, field_name="scope_id")
        elif reason is ErasureReason.CLOCK_ROLLBACK and scope is ErasureScope.STORE:
            _require_literal(self.scope_id, "store", "scope_id")
        else:
            raise _EvidenceValueError("erasure reason and scope do not form a V1 pair")


@dataclass(frozen=True, slots=True)
class FullPurgeV1(_FinalModel):
    protocol_version: int
    full_purge_generation_id: str
    sentinel_state: SentinelState
    artifact_manifest_version: int

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        validate_canonical_uuid4(
            self.full_purge_generation_id,
            field_name="full_purge_generation_id",
        )
        sentinel = _require_enum(self.sentinel_state, SentinelState, "sentinel_state")
        _require_version(self.artifact_manifest_version, "artifact_manifest_version")
        if sentinel not in (
            SentinelState.FULL_PURGE_PENDING,
            SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
        ):
            raise _EvidenceValueError("full purge requires a purge-pending sentinel")


@dataclass(frozen=True, slots=True)
class DrainAndStopV1(_FinalModel):
    protocol_version: int
    owner_generation: int
    final_admission_ordinal: int

    def __post_init__(self) -> None:
        _require_version(self.protocol_version)
        _unsigned_63(self.owner_generation, "owner_generation")
        _unsigned_63(self.final_admission_ordinal, "final_admission_ordinal")


_ModelT = TypeVar("_ModelT")


def _copy_exact_model(value: object, model_type: type[_ModelT], field_name: str) -> _ModelT:
    if type(value) is not model_type:
        raise _EvidenceTypeError(f"{field_name} must be an exact {model_type.__name__}")
    values = {
        field.name: getattr(value, field.name)
        for field in fields(cast(Any, value))
    }
    return model_type(**values)


def validate_authority_composition(authority: object, command: object) -> None:
    """Validate an exact capability/DTO kind pair and all shared lineage."""

    if type(authority) is BindingCloseAuthorityV1:
        authority._validate()
        if authority.close_reason in _ROLLOVER_CLOSE_REASONS:
            raise _EvidenceValueError(
                "binding-close composition cannot perform an atomic rollover"
            )
        close_dto = _copy_exact_model(command, BindingCloseV1, "command")
        payload = cast(BindingClosedPayloadV1, close_dto.snapshot.payload)
        if (
            authority.protocol_version != close_dto.protocol_version
            or authority.binding_id != close_dto.binding_id
            or authority.consent_epoch_id != close_dto.consent_epoch_id
            or authority.logical_session_id != close_dto.logical_session_id
            or authority.close_reason is not payload.close_reason
        ):
            raise _EvidenceValueError("binding-close authority and DTO lineage disagree")
        return
    if type(authority) is RolloverAuthorityV1:
        authority._validate()
        rollover_dto = _copy_exact_model(command, RolloverSessionV1, "command")
        close_payload = cast(BindingClosedPayloadV1, rollover_dto.snapshots[0].payload)
        if (
            authority.protocol_version != rollover_dto.protocol_version
            or authority.binding_id != rollover_dto.binding_id
            or authority.binding_generation != rollover_dto.binding_generation
            or authority.consent_epoch_id != rollover_dto.consent_epoch_id
            or authority.predecessor_logical_session_id
            != rollover_dto.predecessor_logical_session_id
            or authority.successor_logical_session_id
            != rollover_dto.successor_logical_session_id
            or authority.successor_expires_at_utc
            != rollover_dto.successor_expires_at_utc
            or authority.reason is not close_payload.close_reason
        ):
            raise _EvidenceValueError("rollover authority and DTO lineage disagree")
        return
    if type(authority) is LifecycleSealAuthorityV1:
        authority._validate()
        seal_dto = _copy_exact_model(command, SealEpochV1, "command")
        if (
            authority.protocol_version != seal_dto.protocol_version
            or authority.binding_id != seal_dto.binding_id
            or authority.consent_epoch_id != seal_dto.consent_epoch_id
            or authority.logical_session_id != seal_dto.logical_session_id
            or authority.final_event_sequence != seal_dto.final_event_sequence
            or authority.close_reason is not seal_dto.close_reason
            or authority.close_epoch is not seal_dto.close_epoch
        ):
            raise _EvidenceValueError("seal authority and DTO lineage disagree")
        return
    if type(authority) is SessionExpiryAuthorityV1:
        authority._validate()
        expiry_dto = _copy_exact_model(command, ExpireSessionV1, "command")
        if (
            authority.protocol_version != expiry_dto.protocol_version
            or authority.owner_generation != expiry_dto.owner_generation
            or authority.consent_epoch_id != expiry_dto.consent_epoch_id
            or authority.logical_session_id != expiry_dto.logical_session_id
            or authority.expires_at_utc != expiry_dto.expires_at_utc
            or authority.deadline_admission_ordinal
            != expiry_dto.last_admission_ordinal
            or authority.mode is not expiry_dto.mode
        ):
            raise _EvidenceValueError("expiry authority and DTO lineage disagree")
        return
    if type(authority) is ConsentCreateAuthorityV1:
        authority._validate()
        create_dto = _copy_exact_model(command, CreateEpochV1, "command")
        if (
            authority.protocol_version != create_dto.protocol_version
            or authority.binding_id != create_dto.binding_id
            or authority.binding_generation != create_dto.binding_generation
            or authority.control_sequence != create_dto.control_sequence
            or authority.control_fingerprint_hash
            != create_dto.control_fingerprint_hash
            or authority.consent_version != create_dto.consent_version
            or authority.disclosure_digest != create_dto.disclosure_digest
            or authority.retention_hours != create_dto.retention_hours
            or authority.microphone_accepted is not create_dto.microphone_accepted
            or authority.typed_accepted is not create_dto.typed_accepted
        ):
            raise _EvidenceValueError("consent-create authority and DTO lineage disagree")
        return
    if type(authority) is ConsentRevokeAuthorityV1:
        authority._validate()
        revoke_dto = _copy_exact_model(command, RevokeRequestV1, "command")
        if (
            authority.protocol_version != revoke_dto.protocol_version
            or authority.consent_epoch_id != revoke_dto.consent_epoch_id
            or authority.control_sequence != revoke_dto.control_sequence
            or authority.control_fingerprint_hash
            != revoke_dto.control_fingerprint_hash
        ):
            raise _EvidenceValueError("consent-revoke authority and DTO lineage disagree")
        return
    if type(authority) is LifecycleDrainAuthorityV1:
        authority._validate()
        drain_dto = _copy_exact_model(command, DrainAndStopV1, "command")
        if (
            authority.protocol_version != drain_dto.protocol_version
            or authority.owner_generation != drain_dto.owner_generation
            or authority.final_admission_ordinal != drain_dto.final_admission_ordinal
        ):
            raise _EvidenceValueError("drain authority and DTO lineage disagree")
        return
    raise _EvidenceTypeError("authority and command do not form an allowed V1 kind pair")


@dataclass(frozen=True, slots=True)
class EvidenceConsentSourcesV1(_FinalModel):
    microphone: bool
    typed: bool

    def __post_init__(self) -> None:
        _require_exact_bool(self.microphone, "microphone")
        _require_exact_bool(self.typed, "typed")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvidenceConsentRequestV1(_FinalModel):
    sequence: int
    accepted: bool
    consent_version: str
    disclosure_digest: str
    retention_hours: int
    sources: EvidenceConsentSourcesV1

    def __post_init__(self) -> None:
        _browser_integer(self.sequence, "sequence")
        if not _require_exact_bool(self.accepted, "accepted"):
            raise _EvidenceValueError("evidence consent accepted must be literal true")
        _require_literal(self.consent_version, CONSENT_VERSION, "consent_version")
        validate_sha256_hex(self.disclosure_digest, field_name="disclosure_digest")
        _bounded_integer(self.retention_hours, "retention_hours", 1, _MAX_RETENTION_HOURS)
        sources = _copy_exact_model(self.sources, EvidenceConsentSourcesV1, "sources")
        if not sources.microphone and not sources.typed:
            raise _EvidenceValueError("evidence consent requires at least one source")
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True, slots=True)
class EvidenceRevokeRequestV1(_FinalModel):
    sequence: int

    def __post_init__(self) -> None:
        _browser_integer(self.sequence, "sequence")


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectionResyncRequestV1(_FinalModel):
    attempt: int
    reason: str
    request_id: str

    def __post_init__(self) -> None:
        _bounded_integer(self.attempt, "attempt", 1, 1)
        _require_literal(self.reason, "invalid_capture_status", "reason")
        validate_canonical_uuid4(self.request_id, field_name="request_id")


@dataclass(frozen=True, slots=True)
class CaptureStatusV1(_FinalModel):
    available: bool
    capture_state: CaptureState
    retention_hours: int
    consent_version: str
    disclosure_digest: str

    def __post_init__(self) -> None:
        available = _require_exact_bool(self.available, "available")
        state = _require_enum(self.capture_state, CaptureState, "capture_state")
        _bounded_integer(self.retention_hours, "retention_hours", 1, _MAX_RETENTION_HOURS)
        _require_literal(self.consent_version, CONSENT_VERSION, "consent_version")
        validate_sha256_hex(self.disclosure_digest, field_name="disclosure_digest")
        if available != (state is not CaptureState.UNAVAILABLE):
            raise _EvidenceValueError("capture status availability contradicts its state")


def _require_route_bytes(document: object) -> bytes:
    if type(document) is not bytes:
        raise _EvidenceTypeError("browser-control JSON must be exact built-in bytes")
    return document


def parse_evidence_consent_request(document: object) -> EvidenceConsentRequestV1:
    data = parse_strict_json_object(_require_route_bytes(document), max_bytes=512)
    expected = frozenset(
        {
            "sequence",
            "accepted",
            "consentVersion",
            "disclosureDigest",
            "retentionHours",
            "sources",
        }
    )
    if frozenset(data) != expected:
        raise _EvidenceValueError("consent request has a missing or unknown key")
    sources_data = data["sources"]
    if type(sources_data) is not dict:
        raise _EvidenceTypeError("consent sources must be an exact JSON object")
    sources = cast(dict[str, object], sources_data)
    if frozenset(sources) != frozenset({"microphone", "typed"}):
        raise _EvidenceValueError("consent sources have a missing or unknown key")
    return EvidenceConsentRequestV1(
        sequence=cast(int, data["sequence"]),
        accepted=cast(bool, data["accepted"]),
        consent_version=cast(str, data["consentVersion"]),
        disclosure_digest=cast(str, data["disclosureDigest"]),
        retention_hours=cast(int, data["retentionHours"]),
        sources=EvidenceConsentSourcesV1(
            microphone=cast(bool, sources["microphone"]),
            typed=cast(bool, sources["typed"]),
        ),
    )


parse_evidence_consent_request_json = parse_evidence_consent_request


def parse_evidence_revoke_request(document: object) -> EvidenceRevokeRequestV1:
    data = parse_strict_json_object(_require_route_bytes(document), max_bytes=64)
    if frozenset(data) != frozenset({"sequence"}):
        raise _EvidenceValueError("revoke request has a missing or unknown key")
    return EvidenceRevokeRequestV1(sequence=cast(int, data["sequence"]))


parse_evidence_revoke_request_json = parse_evidence_revoke_request


def parse_projection_resync_request(document: object) -> ProjectionResyncRequestV1:
    data = parse_strict_json_object(_require_route_bytes(document), max_bytes=160)
    if frozenset(data) != frozenset({"attempt", "reason", "requestId"}):
        raise _EvidenceValueError("projection-resync request has a missing or unknown key")
    return ProjectionResyncRequestV1(
        attempt=cast(int, data["attempt"]),
        reason=cast(str, data["reason"]),
        request_id=cast(str, data["requestId"]),
    )


parse_projection_resync_request_json = parse_projection_resync_request


def capture_status_to_primitive(status: CaptureStatusV1) -> dict[str, object]:
    copied = _copy_exact_model(status, CaptureStatusV1, "status")
    return {
        "available": copied.available,
        "captureState": copied.capture_state.value,
        "retentionHours": copied.retention_hours,
        "consentVersion": copied.consent_version,
        "disclosureDigest": copied.disclosure_digest,
    }


__all__ = [
    "CONSENT_VERSION",
    "AppendDisposition",
    "AssistantChunkTransportConfirmedFullPayloadV1",
    "AssistantSegmentGeneratedPayloadV1",
    "BindingCloseAuthorityV1",
    "BindingCloseReason",
    "BindingCloseV1",
    "BindingClosedPayloadV1",
    "BindingOpenedPayloadV1",
    "CaptureState",
    "CaptureStatusV1",
    "CauseDisposition",
    "CommandAdmissionAuthorityV1",
    "CommandDisposition",
    "CommandRoutedPayloadV1",
    "CommandRoutingOutcome",
    "CommandRoutingResultV1",
    "ConflictReason",
    "ConsentCreateAuthorityV1",
    "ConsentDisposition",
    "ConsentEpochState",
    "ConsentRevokeAuthorityV1",
    "ControlError",
    "ControlResult",
    "ConversationOperationKind",
    "CreateEpochReservationV1",
    "CreateEpochV1",
    "DrainAndStopV1",
    "DrainDisposition",
    "EligibilityOutcome",
    "ErasureReason",
    "ErasureScope",
    "EvidenceConsentRequestV1",
    "EvidenceConsentSourcesV1",
    "EvidenceDiagnosticsV1",
    "EvidenceEligibilityFactsV1",
    "EvidenceEventKind",
    "EvidenceModelError",
    "EvidencePayloadV1",
    "EvidenceQualificationOutcome",
    "EvidenceRevokeRequestV1",
    "EvidenceSnapshotV1",
    "EventKind",
    "ExpireSessionV1",
    "ExpiryDisposition",
    "ExpiryMode",
    "FinalInputAuthorityV1",
    "FullPurgeV1",
    "InputSource",
    "LifecycleDrainAuthorityV1",
    "LifecycleSealAuthorityV1",
    "MaintenanceV1",
    "OwnerState",
    "PersistentSessionState",
    "ProjectionReservation",
    "ProjectionResyncRequestV1",
    "ProactiveTurnAuthorityV1",
    "PurgeDisposition",
    "QueueReservationClass",
    "QueuedEvidenceRecordV1",
    "RecoveryDisposition",
    "ReplayTurnAuthorityV1",
    "RevokeDisposition",
    "RevokeFinalizeV1",
    "RevokeRequestV1",
    "RolloverAuthorityV1",
    "RolloverDisposition",
    "RolloverSessionV1",
    "RuntimeSessionPhase",
    "SettledTerminalOutcomeV1",
    "SealDisposition",
    "SealEpochV1",
    "SentinelState",
    "SessionExpiryAuthorityV1",
    "SessionExpiryMode",
    "SessionOpenedPayloadV1",
    "SessionSealRequestedPayloadV1",
    "SessionTaintedPayloadV1",
    "StoreDisposition",
    "TaintCode",
    "TerminalCauseCapabilityV1",
    "TerminalDisposition",
    "TerminalReason",
    "TerminalResolutionV1",
    "TurnEligibilityFactsV1",
    "TurnKind",
    "TurnOpenedPayloadV1",
    "TurnSettledPayloadV1",
    "TurnSnapshotPayloadV1",
    "UserFinalAcceptedPayloadV1",
    "UserTurnAuthorityV1",
    "WriterFault",
    "capture_status_to_primitive",
    "classify_event_retry",
    "control_error_http_status",
    "control_result_http_status",
    "evaluate_eligibility",
    "event_payload_to_primitive",
    "event_payload_to_primitive_v1",
    "evidence_snapshot_to_primitive",
    "evidence_snapshot_to_primitive_v1",
    "is_turn_eligible_v1",
    "parse_evidence_consent_request",
    "parse_evidence_consent_request_json",
    "parse_evidence_revoke_request",
    "parse_evidence_revoke_request_json",
    "parse_evidence_snapshot_json",
    "parse_event_payload_json",
    "parse_projection_resync_request",
    "parse_projection_resync_request_json",
    "parse_strict_json_object",
    "resolve_terminal_causes",
    "resolve_terminal_causes_v1",
    "validate_authority_composition",
    "validate_canonical_utc",
    "validate_canonical_uuid4",
    "validate_event_payload",
    "validate_event_sequence",
    "validate_evidence_text",
    "validate_persistent_state_transition",
    "validate_runtime_phase_transition",
    "validate_sha256_hex",
    "writer_fault_to_control_error",
]

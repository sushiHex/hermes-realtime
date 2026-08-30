from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import pickle
from itertools import combinations
from types import ModuleType
from typing import Any
from uuid import UUID

import pytest

REQUIRED_EXPORTS = {
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
    "EvidenceModelError",
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
    "SealDisposition",
    "SealEpochV1",
    "SentinelState",
    "SessionExpiryAuthorityV1",
    "SessionOpenedPayloadV1",
    "SessionSealRequestedPayloadV1",
    "SessionTaintedPayloadV1",
    "StoreDisposition",
    "TaintCode",
    "TerminalCauseCapabilityV1",
    "TerminalDisposition",
    "TerminalReason",
    "TerminalResolutionV1",
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
    "is_turn_eligible_v1",
    "parse_evidence_consent_request",
    "parse_evidence_consent_request_json",
    "parse_evidence_revoke_request",
    "parse_evidence_revoke_request_json",
    "parse_event_payload_json",
    "parse_evidence_snapshot_json",
    "parse_projection_resync_request",
    "parse_projection_resync_request_json",
    "parse_strict_json_object",
    "resolve_terminal_causes",
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
}


def _models() -> ModuleType:
    module = importlib.import_module("hermes_realtime.evidence.models")
    missing = sorted(REQUIRED_EXPORTS.difference(vars(module)))
    assert not missing, f"missing production exports: {', '.join(missing)}"
    return module


def _uuid(number: int) -> str:
    return str(UUID(int=number, version=4))


IDS = {
    "installation_id": _uuid(1),
    "producer_instance_id": _uuid(2),
    "consent_epoch_id": _uuid(3),
    "logical_session_id": _uuid(4),
    "binding_id": _uuid(5),
    "utterance_id": _uuid(6),
    "evidence_turn_id": _uuid(7),
    "evidence_segment_id": _uuid(8),
    "synthesis_attempt_id": _uuid(9),
    "transport_attempt_id": _uuid(10),
    "evidence_chunk_id": _uuid(11),
    "event_id": _uuid(12),
    "erasure_request_id": _uuid(13),
    "conflict_id": _uuid(14),
    "full_purge_generation_id": _uuid(15),
    "successor_logical_session_id": _uuid(16),
    "replay_event_id": _uuid(17),
}
HASH = "a" * 64
UTC = "2026-08-08T12:34:56.123456Z"
CONSENT_LITERAL = "realtime-evidence-consent-v1"


ENUM_VALUES = {
    "InputSource": {"microphone", "typed"},
    "TurnKind": {"user_response", "proactive_update", "replay"},
    "ConversationOperationKind": {"response", "proactive", "replay"},
    "CommandRoutingOutcome": {"not_command", "accepted", "rejected", "invalid"},
    "EventKind": {
        "session_opened",
        "binding_opened",
        "turn_opened",
        "user_final_accepted",
        "command_routed",
        "assistant_segment_generated",
        "assistant_chunk_transport_confirmed_full",
        "turn_snapshot",
        "turn_settled",
        "binding_closed",
        "session_seal_requested",
        "session_tainted",
    },
    "PersistentSessionState": {"open", "sealed", "tainted"},
    "RuntimeSessionPhase": {
        "open",
        "expiring",
        "closing",
        "seal_queued",
        "revoking",
        "stopped",
    },
    "ConsentEpochState": {"active", "closed", "revoked"},
    "TerminalDisposition": {"completed", "interrupted", "cancelled", "revoked", "failed"},
    "TerminalReason": {
        "authoritative_close_completed",
        "barge_in",
        "stop_speaking",
        "response_replaced",
        "binding_closed",
        "retention_expired",
        "host_shutdown",
        "caller_cancelled",
        "consent_revoked",
        "task_spawn_failed",
        "provider_failed",
        "transport_failed",
        "ledger_close_failed",
        "lifecycle_failed",
    },
    "BindingCloseReason": {
        "client_closed",
        "binding_replaced",
        "media_incarnation_replaced",
        "projection_resync",
        "capacity_rollover",
        "retention_rollover",
        "consent_revoked",
        "inactivity_expired",
        "receiver_transport_failed",
        "host_shutdown",
    },
    "TaintCode": {
        "admission_gap",
        "oversize",
        "deny_filter",
        "writer_fault",
        "quota_exceeded",
        "event_id_conflict",
        "sequence_conflict",
        "lineage_invalid",
        "terminal_missing",
        "terminal_conflict",
        "spawn_failed",
        "lease_capacity_exhausted",
        "clock_rollback",
        "ttl_expired",
        "stale_open_recovery",
        "seal_validation_failed",
        "purge_failed",
        "corrupt_store",
        "shutdown_incomplete",
    },
    "ConflictReason": {
        "event_id_envelope_mismatch",
        "cross_session_event_id",
        "session_sequence_claimed",
        "sealed_session_reuse",
    },
    "ErasureReason": {"revoked", "ttl", "clock_rollback", "unclean_epoch"},
    "ErasureScope": {"session", "consent_epoch", "store"},
    "CaptureState": {
        "unavailable",
        "idle",
        "active",
        "revoked_purging",
        "purge_failed",
        "faulted",
    },
    "OwnerState": {"absent", "recovery_only", "running", "draining", "faulted", "stopped"},
    "QueueReservationClass": {"ordinary", "terminal"},
    "WriterFault": {
        "ownership_unavailable",
        "unsupported_platform",
        "path_invalid",
        "asset_mismatch",
        "store_corrupt",
        "purge_required",
        "sqlite_fault",
        "quota_unavailable",
        "drain_timeout",
    },
    "SentinelState": {
        "clear",
        "full_purge_pending",
        "clock_rollback_purge_pending",
        "first_create_pending",
    },
    "ControlResult": {"consent_activated", "revoke_durably_scheduled", "purge_completed"},
    "ControlError": {
        "invalid_control",
        "unauthorized",
        "forbidden",
        "control_sequence_conflict",
        "capture_unavailable",
        "control_pending",
        "status_capacity_unavailable",
        "control_timeout",
        "writer_unavailable",
        "revoke_not_durable",
        "purge_failed",
    },
    "AppendDisposition": {
        "admitted",
        "disabled",
        "consent_missing",
        "invalid_authority",
        "session_closing",
        "session_tainted",
        "dropped_capacity",
        "rejected_oversize",
        "rejected_denied",
        "quota_exceeded",
        "writer_fault",
    },
    "CommandDisposition": {
        "admitted",
        "disabled",
        "consent_missing",
        "invalid_authority",
        "session_closing",
        "session_tainted",
        "dropped_capacity",
        "writer_fault",
    },
    "CauseDisposition": {
        "recorded",
        "already_recorded",
        "invalid_authority",
        "cause_set_frozen",
    },
    "RolloverDisposition": {
        "rollover_queued",
        "active_leases",
        "replayable_turns",
        "insufficient_capacity",
        "invalid_authority",
        "session_tainted",
        "writer_fault",
    },
    "ExpiryDisposition": {
        "rollover_queued",
        "erasure_durably_scheduled",
        "already_expiring",
        "writer_fault",
    },
    "ConsentDisposition": {
        "consent_activated",
        "already_activated",
        "create_pending",
        "create_failed",
        "control_timed_out",
    },
    "SealDisposition": {
        "seal_queued",
        "already_queued",
        "active_leases",
        "revoked",
        "session_tainted",
        "writer_fault",
    },
    "RevokeDisposition": {
        "closed_not_durable",
        "revoke_durably_scheduled",
        "already_scheduled",
        "purge_completed",
        "purge_failed",
        "control_timed_out",
        "writer_fault",
    },
    "PurgeDisposition": {
        "purge_completed",
        "already_absent",
        "ownership_unavailable",
        "purge_failed",
        "unsupported_platform",
    },
    "DrainDisposition": {
        "drain_queued",
        "already_queued",
        "stopped",
        "timed_out",
        "writer_fault",
    },
    "RecoveryDisposition": {
        "absent",
        "recovered",
        "purge_completed",
        "ownership_unavailable",
        "unsupported_platform",
        "faulted",
    },
    "StoreDisposition": {
        "committed",
        "idempotent",
        "conflict_tainted",
        "rejected_state",
        "faulted",
    },
    "ExpiryMode": {"rollover", "erase_stuck"},
    "EligibilityOutcome": {"eligible", "never_persisted", "persisted_but_excluded", "purged"},
}


@pytest.mark.parametrize(("name", "values"), ENUM_VALUES.items())
def test_closed_str_enum_values_are_exact(name: str, values: set[str]) -> None:
    m = _models()
    enum_type = getattr(m, name)
    assert {member.value for member in enum_type} == values
    assert all(isinstance(member, str) for member in enum_type)
    with pytest.raises(ValueError):
        enum_type("future_value")


def _future_str_enum_member(enum_type: type[Any], value: str) -> Any:
    member = str.__new__(enum_type, value)
    object.__setattr__(member, "_name_", "FUTURE_MEMBER")
    object.__setattr__(member, "_value_", value)
    assert type(member) is enum_type
    assert member not in tuple(enum_type)
    return member


def test_exact_consent_version_literal_is_independent_of_production_fixtures() -> None:
    m = _models()
    assert CONSENT_LITERAL == "realtime-evidence-consent-v1"
    assert m.CONSENT_VERSION == "realtime-evidence-consent-v1"


def _payload(m: ModuleType, kind: Any, **changes: object) -> dict[str, object]:
    payloads: dict[Any, dict[str, object]] = {
        m.EventKind.SESSION_OPENED: {
            "consent_epoch_id": IDS["consent_epoch_id"],
            "binding_id": IDS["binding_id"],
            "consent_version": CONSENT_LITERAL,
            "disclosure_digest": HASH,
            "retention_hours": 24,
            "microphone_accepted": True,
            "typed_accepted": True,
            "predecessor_session_id": None,
        },
        m.EventKind.BINDING_OPENED: {
            "binding_id": IDS["binding_id"],
            "binding_generation": 1,
            "microphone_available": True,
            "typed_available": True,
        },
        m.EventKind.TURN_OPENED: {
            "evidence_turn_id": IDS["evidence_turn_id"],
            "turn_kind": "user_response",
            "utterance_id": IDS["utterance_id"],
        },
        m.EventKind.USER_FINAL_ACCEPTED: {
            "utterance_id": IDS["utterance_id"],
            "evidence_turn_id": IDS["evidence_turn_id"],
            "source": "typed",
            "routing_disposition": "response",
            "text": "exact user text",
        },
        m.EventKind.COMMAND_ROUTED: {
            "utterance_id": IDS["utterance_id"],
            "source": "typed",
            "routing_disposition": "command",
        },
        m.EventKind.ASSISTANT_SEGMENT_GENERATED: {
            "evidence_turn_id": IDS["evidence_turn_id"],
            "evidence_segment_id": IDS["evidence_segment_id"],
            "segment_ordinal": 1,
            "text": "exact assistant text",
        },
        m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL: {
            "evidence_turn_id": IDS["evidence_turn_id"],
            "evidence_segment_id": IDS["evidence_segment_id"],
            "synthesis_attempt_id": IDS["synthesis_attempt_id"],
            "transport_attempt_id": IDS["transport_attempt_id"],
            "evidence_chunk_id": IDS["evidence_chunk_id"],
            "chunk_ordinal": 1,
            "text": "exact assistant text",
        },
        m.EventKind.TURN_SNAPSHOT: {
            "evidence_turn_id": IDS["evidence_turn_id"],
            "turn_kind": "user_response",
            "generated_segment_count": 1,
            "queued_chunk_count": 1,
            "started_chunk_count": 1,
            "transport_confirmed_full_count": 1,
            "model_context_admitted": True,
            "assistant_delivery_context_recorded": True,
        },
        m.EventKind.TURN_SETTLED: {
            "evidence_turn_id": IDS["evidence_turn_id"],
            "terminal_disposition": "completed",
            "terminal_reason": "authoritative_close_completed",
            "context_committed": True,
            "generated_segment_count": 1,
            "transport_confirmed_full_count": 1,
        },
        m.EventKind.BINDING_CLOSED: {
            "binding_id": IDS["binding_id"],
            "close_reason": "client_closed",
        },
        m.EventKind.SESSION_SEAL_REQUESTED: {
            "final_event_sequence": 9,
            "consent_epoch_id": IDS["consent_epoch_id"],
            "consent_version": CONSENT_LITERAL,
            "disclosure_digest": HASH,
        },
        m.EventKind.SESSION_TAINTED: {"taint_code": "writer_fault"},
    }
    result = payloads[kind].copy()
    result.update(changes)
    return result


def _snapshot(
    m: ModuleType,
    kind: Any,
    *,
    sequence: int = 1,
    event_id: str | None = None,
    logical_session_id: str | None = None,
    payload: object | None = None,
) -> Any:
    resolved_payload = payload if payload is not None else _payload(m, kind)
    if kind is m.EventKind.SESSION_SEAL_REQUESTED and payload is None:
        assert type(resolved_payload) is dict
        resolved_payload["final_event_sequence"] = sequence
    typed_payload = m.validate_event_payload(kind, resolved_payload)
    return m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=IDS["installation_id"],
        producer_instance_id=IDS["producer_instance_id"],
        logical_session_id=logical_session_id or IDS["logical_session_id"],
        event_id=event_id or _uuid(100 + sequence),
        event_sequence=sequence,
        event_kind=kind,
        payload=typed_payload,
    )


def test_snapshot_shape_is_frozen_slotted_and_payload_is_immutable() -> None:
    m = _models()
    snapshot = _snapshot(m, m.EventKind.COMMAND_ROUTED)
    assert [field.name for field in dataclasses.fields(snapshot)] == [
        "schema_version",
        "installation_id",
        "producer_instance_id",
        "logical_session_id",
        "event_id",
        "event_sequence",
        "event_kind",
        "payload",
    ]
    assert snapshot.__dataclass_params__.frozen is True
    assert not hasattr(snapshot, "__dict__")
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.event_sequence = 2
    with pytest.raises(TypeError):
        snapshot.payload["future"] = "value"
    with pytest.raises(TypeError):
        m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=IDS["installation_id"],
            producer_instance_id=IDS["producer_instance_id"],
            logical_session_id=IDS["logical_session_id"],
            event_id=IDS["event_id"],
            event_sequence=1,
            event_kind=m.EventKind.COMMAND_ROUTED,
            payload=_payload(m, m.EventKind.COMMAND_ROUTED),
            extension={"future": True},
        )


@pytest.mark.parametrize(
    "bad_uuid",
    [
        "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa".upper(),
        "00000000000040008000000000000001",
        "{00000000-0000-4000-8000-000000000001}",
        "00000000-0000-1000-8000-000000000001",
        "not-a-uuid",
        "",
    ],
)
def test_uuid4_validator_rejects_noncanonical_or_non_v4_values(bad_uuid: str) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        m.validate_canonical_uuid4(bad_uuid)


@pytest.mark.parametrize("variant", list("89ab"))
def test_uuid4_validator_accepts_every_rfc4122_variant_nibble(variant: str) -> None:
    m = _models()
    value = f"00000000-0000-4000-{variant}000-000000000001"
    assert m.validate_canonical_uuid4(value) == value


@pytest.mark.parametrize("version", list("012356789abcdef"))
@pytest.mark.parametrize("variant", list("01234567cdef"))
def test_uuid4_validator_rejects_every_wrong_version_and_variant_nibble(
    version: str, variant: str
) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        m.validate_canonical_uuid4(
            f"00000000-0000-{version}000-{variant}000-000000000001"
        )


def test_uuid4_validator_covers_every_identity_domain_and_hostile_subclass() -> None:
    m = _models()
    for name, value in IDS.items():
        assert m.validate_canonical_uuid4(value, field_name=name) == value

    calls = 0

    class HostileStr(str):
        def __str__(self) -> str:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile string conversion executed")

        def __eq__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile string comparison executed")

        def encode(self, *args: object, **kwargs: object) -> bytes:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile string encoding executed")

    with pytest.raises(m.EvidenceModelError):
        m.validate_canonical_uuid4(HostileStr(IDS["conflict_id"]))
    assert calls == 0


@pytest.mark.parametrize("bad_hash", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, ""])
def test_sha256_validator_is_canonical_lowercase_hex(bad_hash: str) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        m.validate_sha256_hex(bad_hash)
    assert m.validate_sha256_hex(HASH) == HASH


@pytest.mark.parametrize(
    "bad_time",
    [
        "2026-08-08T12:34:56Z",
        "2026-08-08T12:34:56.12345Z",
        "2026-08-08T12:34:56.123456+00:00",
        "2026-8-08T12:34:56.123456Z",
        "2026-02-30T12:34:56.123456Z",
        "2026-08-08t12:34:56.123456z",
        "",
    ],
)
def test_utc_validator_requires_exact_six_digit_zulu_form(bad_time: str) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        m.validate_canonical_utc(bad_time)
    assert m.validate_canonical_utc(UTC) == UTC


def test_validators_reject_str_subclasses() -> None:
    m = _models()

    calls = 0

    class HostileStr(str):
        def __str__(self) -> str:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile string conversion executed")

        def __eq__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile string comparison executed")

        def encode(self, *args: object, **kwargs: object) -> bytes:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile string encoding executed")

    with pytest.raises(m.EvidenceModelError):
        m.validate_sha256_hex(HostileStr(HASH))
    with pytest.raises(m.EvidenceModelError):
        m.validate_canonical_utc(HostileStr(UTC))
    with pytest.raises(m.EvidenceModelError):
        m.validate_evidence_text(HostileStr("text"))
    assert calls == 0


def test_text_boundary_is_codepoints_and_preserves_exact_source() -> None:
    m = _models()
    exact = "é" + "🙂" * 4094
    assert len(exact) == 4096
    assert m.validate_evidence_text(exact) == exact
    with pytest.raises(m.EvidenceModelError):
        m.validate_evidence_text(exact + "x")
    with pytest.raises(m.EvidenceModelError):
        m.validate_evidence_text("ok\ud800not-utf8")


def test_exact_built_in_integer_and_boolean_types_are_required() -> None:
    m = _models()

    calls = 0

    class HostileInt(int):
        def __lt__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile integer comparison executed")

        def __gt__(self, other: object) -> bool:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile integer comparison executed")

        def __index__(self) -> int:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile integer conversion executed")

    base = _payload(m, m.EventKind.BINDING_OPENED)
    for value in (True, HostileInt(1), 1.0):
        payload = {**base, "binding_generation": value}
        with pytest.raises(m.EvidenceModelError):
            _snapshot(m, m.EventKind.BINDING_OPENED, payload=payload)
    with pytest.raises(m.EvidenceModelError):
        _snapshot(
            m,
            m.EventKind.BINDING_OPENED,
            payload={**base, "microphone_available": 1},
        )
    assert calls == 0


def test_strict_json_rejects_duplicate_keys_nonfinite_numbers_and_wrong_roots() -> None:
    m = _models()
    hostile = [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b'{"a":1.0}',
        b'[]',
        b'null',
    ]
    for document in hostile:
        with pytest.raises(m.EvidenceModelError):
            m.parse_strict_json_object(document)
    assert m.parse_strict_json_object(b'{"a":1,"b":true}') == {"a": 1, "b": True}


def _nested_json_object(depth: int) -> bytes:
    assert depth >= 1
    array_depth = depth - 1
    return (
        b'{"value":'
        + b"[" * array_depth
        + b"null"
        + b"]" * array_depth
        + b"}"
    )


def test_strict_json_nesting_depth_is_explicitly_bounded() -> None:
    m = _models()
    depth_32 = _nested_json_object(32)
    assert m.parse_strict_json_object(depth_32) == json.loads(depth_32)
    with pytest.raises(m.EvidenceModelError, match="nesting exceeds"):
        m.parse_strict_json_object(_nested_json_object(33))


@pytest.mark.parametrize(
    "parser_name",
    [
        # The revoke parser's 64-byte ceiling cannot encode depth-32 JSON.
        "parse_event_payload_json",
        "parse_evidence_snapshot_json",
        "parse_evidence_consent_request",
        "parse_evidence_consent_request_json",
        "parse_projection_resync_request",
        "parse_projection_resync_request_json",
    ],
)
def test_schema_bound_strict_json_parsers_pin_depth_32_and_33(
    parser_name: str,
) -> None:
    m = _models()
    parser = getattr(m, parser_name)

    # Depth 32 must reach the closed schema; depth 33 must fail in strict JSON.
    with pytest.raises(m.EvidenceModelError, match="missing or unknown key"):
        if parser_name == "parse_event_payload_json":
            parser(m.EventKind.COMMAND_ROUTED, _nested_json_object(32))
        else:
            parser(_nested_json_object(32))

    with pytest.raises(m.EvidenceModelError, match="nesting exceeds"):
        if parser_name == "parse_event_payload_json":
            parser(m.EventKind.COMMAND_ROUTED, _nested_json_object(33))
        else:
            parser(_nested_json_object(33))


def test_strict_json_normalizes_recursion_failure_for_extreme_depth() -> None:
    m = _models()
    document = (
        b'{"value":' + b"[" * 2048 + b"0" + b"]" * 2048 + b"}"
    )
    with pytest.raises(m.EvidenceModelError):
        m.parse_strict_json_object(document)


def test_event_payload_json_rejects_unknown_nested_and_surrogate_values() -> None:
    m = _models()
    valid = json.dumps(_payload(m, m.EventKind.COMMAND_ROUTED), separators=(",", ":"))
    parsed = m.parse_event_payload_json(m.EventKind.COMMAND_ROUTED, valid.encode())
    assert dict(parsed) == _payload(m, m.EventKind.COMMAND_ROUTED)
    for document in (
        valid[:-1] + ',"task_id":"private"}',
        valid.replace('"typed"', '"typed\\ud800"'),
        valid.replace('"typed"', '1'),
        valid.replace('"typed"', '{"nested":"typed"}'),
    ):
        with pytest.raises(m.EvidenceModelError):
            m.parse_event_payload_json(m.EventKind.COMMAND_ROUTED, document.encode())


def test_event_payload_parser_requires_exact_bytes_before_semantic_validation() -> None:
    m = _models()
    valid = json.dumps(_payload(m, m.EventKind.COMMAND_ROUTED), separators=(",", ":"))
    with pytest.raises(m.EvidenceModelError):
        m.parse_event_payload_json(m.EventKind.COMMAND_ROUTED, valid)

    duplicate = (
        b'{"utterance_id":"'
        + IDS["utterance_id"].encode()
        + b'","source":"typed","source":"microphone",'
        b'"routing_disposition":"command"}'
    )
    with pytest.raises(m.EvidenceModelError, match="duplicate"):
        m.parse_event_payload_json(m.EventKind.COMMAND_ROUTED, duplicate)

    nonfinite = valid.replace('"typed"', "NaN").encode()
    with pytest.raises(m.EvidenceModelError, match="floating-point|nonfinite"):
        m.parse_event_payload_json(m.EventKind.COMMAND_ROUTED, nonfinite)


def test_task2_surface_excludes_canonical_record_serialization() -> None:
    m = _models()
    package = importlib.import_module("hermes_realtime.evidence")
    for module in (m, package):
        assert "canonical_json_bytes" not in vars(module)
        assert "canonical_json_bytes" not in getattr(module, "__all__", ())


def test_snapshot_json_parser_has_closed_exact_bytes_envelope() -> None:
    m = _models()
    data = {
        "schema_version": 1,
        "installation_id": IDS["installation_id"],
        "producer_instance_id": IDS["producer_instance_id"],
        "logical_session_id": IDS["logical_session_id"],
        "event_id": IDS["event_id"],
        "event_sequence": 1,
        "event_kind": "command_routed",
        "payload": _payload(m, m.EventKind.COMMAND_ROUTED),
    }
    document = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    parsed = m.parse_evidence_snapshot_json(document)
    assert parsed.event_kind is m.EventKind.COMMAND_ROUTED
    assert dict(parsed.payload) == data["payload"]
    for mutation in (
        {**data, "profile": "default"},
        {**data, "event_sequence": True},
        {**data, "event_kind": "future"},
    ):
        with pytest.raises(m.EvidenceModelError):
            m.parse_evidence_snapshot_json(
                json.dumps(mutation, separators=(",", ":")).encode()
            )


def test_snapshot_parser_mutations_reach_strict_and_nested_semantic_boundaries() -> None:
    m = _models()
    prefix = (
        b'{"schema_version":1,"installation_id":"'
        + IDS["installation_id"].encode()
        + b'","producer_instance_id":"'
        + IDS["producer_instance_id"].encode()
        + b'","logical_session_id":"'
        + IDS["logical_session_id"].encode()
        + b'","event_id":"'
        + IDS["event_id"].encode()
        + b'",'
    )
    duplicate = (
        prefix
        + b'"event_sequence":1,"event_sequence":2,"event_kind":"command_routed",'
        b'"payload":{"utterance_id":"'
        + IDS["utterance_id"].encode()
        + b'","source":"typed","routing_disposition":"command"}}'
    )
    with pytest.raises(m.EvidenceModelError, match="duplicate"):
        m.parse_evidence_snapshot_json(duplicate)

    nested_unknown = (
        prefix
        + b'"event_sequence":1,"event_kind":"command_routed","payload":{'
        b'"utterance_id":"'
        + IDS["utterance_id"].encode()
        + b'","source":"typed","routing_disposition":"command",'
        b'"metadata":{"task_id":"private"}}}'
    )
    with pytest.raises(m.EvidenceModelError, match="missing or unknown key"):
        m.parse_evidence_snapshot_json(nested_unknown)


def test_snapshot_direct_construction_requires_exact_kind_payload_dataclass() -> None:
    m = _models()
    envelope = {
        "schema_version": 1,
        "installation_id": IDS["installation_id"],
        "producer_instance_id": IDS["producer_instance_id"],
        "logical_session_id": IDS["logical_session_id"],
        "event_id": IDS["event_id"],
        "event_sequence": 1,
        "event_kind": m.EventKind.COMMAND_ROUTED,
    }
    exact_payload = m.CommandRoutedPayloadV1(
        utterance_id=IDS["utterance_id"],
        source=m.InputSource.TYPED,
        routing_disposition="command",
    )
    snapshot = m.EvidenceSnapshotV1(**envelope, payload=exact_payload)
    assert type(snapshot.payload) is m.CommandRoutedPayloadV1

    with pytest.raises(m.EvidenceModelError):
        m.EvidenceSnapshotV1(
            **envelope,
            payload=_payload(m, m.EventKind.COMMAND_ROUTED),
        )
    with pytest.raises(m.EvidenceModelError):
        m.EvidenceSnapshotV1(
            **envelope,
            payload=m.BindingClosedPayloadV1(
                binding_id=IDS["binding_id"],
                close_reason=m.BindingCloseReason.CLIENT_CLOSED,
            ),
        )


PAYLOAD_DATACLASS_FIELDS = {
    "SESSION_OPENED": (
        "SessionOpenedPayloadV1",
        [
            "consent_epoch_id",
            "binding_id",
            "consent_version",
            "disclosure_digest",
            "retention_hours",
            "microphone_accepted",
            "typed_accepted",
            "predecessor_session_id",
        ],
    ),
    "BINDING_OPENED": (
        "BindingOpenedPayloadV1",
        [
            "binding_id",
            "binding_generation",
            "microphone_available",
            "typed_available",
        ],
    ),
    "TURN_OPENED": (
        "TurnOpenedPayloadV1",
        [
            "evidence_turn_id",
            "turn_kind",
            "utterance_id",
            "replay_of_evidence_turn_id",
        ],
    ),
    "USER_FINAL_ACCEPTED": (
        "UserFinalAcceptedPayloadV1",
        [
            "utterance_id",
            "evidence_turn_id",
            "source",
            "routing_disposition",
            "text",
        ],
    ),
    "COMMAND_ROUTED": (
        "CommandRoutedPayloadV1",
        ["utterance_id", "source", "routing_disposition"],
    ),
    "ASSISTANT_SEGMENT_GENERATED": (
        "AssistantSegmentGeneratedPayloadV1",
        ["evidence_turn_id", "evidence_segment_id", "segment_ordinal", "text"],
    ),
    "ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL": (
        "AssistantChunkTransportConfirmedFullPayloadV1",
        [
            "evidence_turn_id",
            "evidence_segment_id",
            "synthesis_attempt_id",
            "transport_attempt_id",
            "evidence_chunk_id",
            "chunk_ordinal",
            "text",
        ],
    ),
    "TURN_SNAPSHOT": (
        "TurnSnapshotPayloadV1",
        [
            "evidence_turn_id",
            "turn_kind",
            "generated_segment_count",
            "queued_chunk_count",
            "started_chunk_count",
            "transport_confirmed_full_count",
            "model_context_admitted",
            "assistant_delivery_context_recorded",
        ],
    ),
    "TURN_SETTLED": (
        "TurnSettledPayloadV1",
        [
            "evidence_turn_id",
            "terminal_disposition",
            "terminal_reason",
            "context_committed",
            "generated_segment_count",
            "transport_confirmed_full_count",
        ],
    ),
    "BINDING_CLOSED": (
        "BindingClosedPayloadV1",
        ["binding_id", "close_reason"],
    ),
    "SESSION_SEAL_REQUESTED": (
        "SessionSealRequestedPayloadV1",
        [
            "final_event_sequence",
            "consent_epoch_id",
            "consent_version",
            "disclosure_digest",
        ],
    ),
    "SESSION_TAINTED": ("SessionTaintedPayloadV1", ["taint_code"]),
}


PAYLOAD_KEYS = {
    "SESSION_OPENED": {
        "consent_epoch_id",
        "binding_id",
        "consent_version",
        "disclosure_digest",
        "retention_hours",
        "microphone_accepted",
        "typed_accepted",
        "predecessor_session_id",
    },
    "BINDING_OPENED": {
        "binding_id",
        "binding_generation",
        "microphone_available",
        "typed_available",
    },
    "TURN_OPENED": {"evidence_turn_id", "turn_kind", "utterance_id"},
    "USER_FINAL_ACCEPTED": {
        "utterance_id",
        "evidence_turn_id",
        "source",
        "routing_disposition",
        "text",
    },
    "COMMAND_ROUTED": {"utterance_id", "source", "routing_disposition"},
    "ASSISTANT_SEGMENT_GENERATED": {
        "evidence_turn_id",
        "evidence_segment_id",
        "segment_ordinal",
        "text",
    },
    "TURN_SNAPSHOT": {
        "evidence_turn_id",
        "turn_kind",
        "generated_segment_count",
        "queued_chunk_count",
        "started_chunk_count",
        "transport_confirmed_full_count",
        "model_context_admitted",
        "assistant_delivery_context_recorded",
    },
    "TURN_SETTLED": {
        "evidence_turn_id",
        "terminal_disposition",
        "terminal_reason",
        "context_committed",
        "generated_segment_count",
        "transport_confirmed_full_count",
    },
    "BINDING_CLOSED": {"binding_id", "close_reason"},
    "SESSION_SEAL_REQUESTED": {
        "final_event_sequence",
        "consent_epoch_id",
        "consent_version",
        "disclosure_digest",
    },
    "SESSION_TAINTED": {"taint_code"},
}


def test_payload_dataclass_fields_are_pinned_for_every_event_kind() -> None:
    m = _models()
    assert set(PAYLOAD_DATACLASS_FIELDS) == {kind.name for kind in m.EventKind}
    for kind_name, (class_name, expected_fields) in PAYLOAD_DATACLASS_FIELDS.items():
        payload_type = getattr(m, class_name)
        assert [field.name for field in dataclasses.fields(payload_type)] == expected_fields
        payload = _snapshot(m, getattr(m.EventKind, kind_name)).payload
        assert type(payload) is payload_type


def test_closed_payload_key_sets_and_forbidden_authority_fields() -> None:
    m = _models()
    for member_name, expected in PAYLOAD_KEYS.items():
        kind = getattr(m.EventKind, member_name)
        assert set(_snapshot(m, kind).payload) == expected
    chunk_keys = set(_snapshot(m, m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL).payload)
    assert chunk_keys == {
        "evidence_turn_id",
        "evidence_segment_id",
        "synthesis_attempt_id",
        "transport_attempt_id",
        "evidence_chunk_id",
        "chunk_ordinal",
        "text",
    }
    forbidden = {
        "participant",
        "profile",
        "provider",
        "model",
        "task_id",
        "tool",
        "path",
        "url",
        "metadata",
        "exception",
        "audio",
    }
    all_keys = set().union(*PAYLOAD_KEYS.values(), chunk_keys)
    assert forbidden.isdisjoint(all_keys)


def test_payloads_reject_missing_and_unknown_keys_for_every_event_kind() -> None:
    m = _models()
    for kind in m.EventKind:
        payload = _payload(m, kind)
        for required_key in tuple(payload):
            missing = payload.copy()
            missing.pop(required_key)
            with pytest.raises(m.EvidenceModelError):
                _snapshot(m, kind, payload=missing)
        with pytest.raises(m.EvidenceModelError):
            _snapshot(m, kind, payload={**payload, "future": "value"})


@pytest.mark.parametrize(
    ("turn_kind", "present", "absent"),
    [
        ("user_response", "utterance_id", "replay_of_evidence_turn_id"),
        ("replay", "replay_of_evidence_turn_id", "utterance_id"),
        ("proactive_update", None, "utterance_id"),
    ],
)
def test_turn_opened_conditional_fields(turn_kind: str, present: str | None, absent: str) -> None:
    m = _models()
    payload: dict[str, object] = {
        "evidence_turn_id": IDS["evidence_turn_id"],
        "turn_kind": turn_kind,
    }
    if present == "utterance_id":
        payload[present] = IDS["utterance_id"]
    elif present == "replay_of_evidence_turn_id":
        payload[present] = _uuid(70)
    _snapshot(m, m.EventKind.TURN_OPENED, payload=payload)
    invalid = payload.copy()
    invalid[absent] = _uuid(71)
    with pytest.raises(m.EvidenceModelError):
        _snapshot(m, m.EventKind.TURN_OPENED, payload=invalid)


def test_chunk_payload_is_closed_by_turn_kind_and_paired_optional_fields() -> None:
    m = _models()
    user = _payload(m, m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL)
    m.validate_event_payload(
        m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
        user,
        turn_kind=m.TurnKind.USER_RESPONSE,
    )
    content_free = user.copy()
    content_free.pop("evidence_segment_id")
    content_free.pop("text")
    for kind in (m.TurnKind.PROACTIVE_UPDATE, m.TurnKind.REPLAY):
        m.validate_event_payload(
            m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
            content_free,
            turn_kind=kind,
        )
        with pytest.raises(m.EvidenceModelError):
            m.validate_event_payload(
                m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
                user,
                turn_kind=kind,
            )
    broken = user.copy()
    broken.pop("text")
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_payload(
            m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
            broken,
        )


@pytest.mark.parametrize(
    ("field", "bad_values"),
    [
        ("segment_ordinal", (0, 257, True)),
        ("chunk_ordinal", (0, 4097, True)),
        ("generated_segment_count", (-1, 257, True)),
        ("transport_confirmed_full_count", (-1, 4097, True)),
    ],
)
def test_payload_integer_boundaries(field: str, bad_values: tuple[object, ...]) -> None:
    m = _models()
    kind_by_field = {
        "segment_ordinal": m.EventKind.ASSISTANT_SEGMENT_GENERATED,
        "chunk_ordinal": m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
        "generated_segment_count": m.EventKind.TURN_SNAPSHOT,
        "transport_confirmed_full_count": m.EventKind.TURN_SETTLED,
    }
    kind = kind_by_field[field]
    for value in bad_values:
        with pytest.raises(m.EvidenceModelError):
            _snapshot(m, kind, payload=_payload(m, kind, **{field: value}))


TERMINAL_GROUPS = [
    (["consent_revoked"], "revoked", "consent_revoked"),
    (
        [
            "task_spawn_failed",
            "provider_failed",
            "transport_failed",
            "ledger_close_failed",
            "lifecycle_failed",
        ],
        "failed",
        None,
    ),
    (["barge_in", "stop_speaking", "response_replaced"], "interrupted", None),
    (
        ["binding_closed", "retention_expired", "host_shutdown", "caller_cancelled"],
        "cancelled",
        None,
    ),
    (["authoritative_close_completed"], "completed", "authoritative_close_completed"),
]


def _expected_terminal(names: set[str]) -> tuple[str, str]:
    for reasons, disposition, fixed_reason in TERMINAL_GROUPS:
        for reason in reasons:
            if reason in names:
                return disposition, fixed_reason or reason
    raise AssertionError("empty terminal cause set")


def test_terminal_resolution_is_exhaustive_and_order_independent() -> None:
    m = _models()
    reasons = list(m.TerminalReason)
    for size in range(1, len(reasons) + 1):
        for selected in combinations(reasons, size):
            expected_disposition, expected_reason = _expected_terminal(
                {reason.value for reason in selected}
            )
            forward = m.resolve_terminal_causes(selected, context_committed=True)
            reverse = m.resolve_terminal_causes(
                tuple(reversed(selected)),
                context_committed=True,
            )
            assert forward == reverse
            assert forward.terminal_disposition.value == expected_disposition
            assert forward.terminal_reason.value == expected_reason
    with pytest.raises(m.EvidenceModelError):
        m.resolve_terminal_causes((), context_committed=True)


def test_completion_resolution_never_invents_context_commitment() -> None:
    m = _models()
    completed = (m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,)

    with pytest.raises(m.EvidenceModelError):
        m.resolve_terminal_causes(completed)
    with pytest.raises(m.EvidenceModelError):
        m.resolve_terminal_causes(completed, context_committed=False)

    committed = m.resolve_terminal_causes(completed, context_committed=True)
    no_op = m.resolve_terminal_causes(completed, zero_output_no_op=True)
    assert committed.terminal_disposition is m.TerminalDisposition.COMPLETED
    assert no_op.terminal_disposition is m.TerminalDisposition.COMPLETED

    for hostile in (1, None, "true"):
        with pytest.raises(m.EvidenceModelError):
            m.resolve_terminal_causes(completed, zero_output_no_op=hostile)


def test_terminal_resolution_rejects_unregistered_future_reason_even_with_known_cause() -> None:
    m = _models()
    future = _future_str_enum_member(m.TerminalReason, "future_terminal_reason")
    with pytest.raises(m.EvidenceModelError):
        m.resolve_terminal_causes(
            (future, m.TerminalReason.BARGE_IN),
            context_committed=False,
        )


def test_terminal_payload_disposition_reason_and_commit_contradictions_fail() -> None:
    m = _models()
    kind = m.EventKind.TURN_SETTLED
    with pytest.raises(m.EvidenceModelError):
        _snapshot(
            m,
            kind,
            payload=_payload(m, kind, terminal_disposition="failed"),
        )
    with pytest.raises(m.EvidenceModelError):
        _snapshot(m, kind, payload=_payload(m, kind, context_committed=False))
    for generated, confirmed in ((1, 0), (0, 1)):
        with pytest.raises(m.EvidenceModelError):
            _snapshot(
                m,
                kind,
                payload=_payload(
                    m,
                    kind,
                    context_committed=False,
                    generated_segment_count=generated,
                    transport_confirmed_full_count=confirmed,
                ),
            )


def _valid_event_path(m: ModuleType) -> tuple[Any, ...]:
    kinds = [
        m.EventKind.SESSION_OPENED,
        m.EventKind.BINDING_OPENED,
        m.EventKind.TURN_OPENED,
        m.EventKind.USER_FINAL_ACCEPTED,
        m.EventKind.ASSISTANT_SEGMENT_GENERATED,
        m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
        m.EventKind.TURN_SNAPSHOT,
        m.EventKind.TURN_SETTLED,
        m.EventKind.BINDING_CLOSED,
        m.EventKind.SESSION_SEAL_REQUESTED,
    ]
    snapshots = []
    for sequence, kind in enumerate(kinds, 1):
        payload = _payload(m, kind)
        if kind is m.EventKind.SESSION_SEAL_REQUESTED:
            payload["final_event_sequence"] = sequence
        snapshots.append(_snapshot(m, kind, sequence=sequence, payload=payload))
    return tuple(snapshots)


def _opening_events(
    m: ModuleType,
    *,
    microphone_accepted: bool = True,
    typed_accepted: bool = True,
) -> list[Any]:
    return [
        _snapshot(
            m,
            m.EventKind.SESSION_OPENED,
            sequence=1,
            payload=_payload(
                m,
                m.EventKind.SESSION_OPENED,
                microphone_accepted=microphone_accepted,
                typed_accepted=typed_accepted,
            ),
        ),
        _snapshot(m, m.EventKind.BINDING_OPENED, sequence=2),
    ]


def _minimal_user_turn_events(
    m: ModuleType,
    *,
    start_sequence: int,
    evidence_turn_id: str,
    utterance_id: str,
    source: str = "typed",
) -> list[Any]:
    return [
        _snapshot(
            m,
            m.EventKind.TURN_OPENED,
            sequence=start_sequence,
            payload=_payload(
                m,
                m.EventKind.TURN_OPENED,
                evidence_turn_id=evidence_turn_id,
                utterance_id=utterance_id,
            ),
        ),
        _snapshot(
            m,
            m.EventKind.USER_FINAL_ACCEPTED,
            sequence=start_sequence + 1,
            payload=_payload(
                m,
                m.EventKind.USER_FINAL_ACCEPTED,
                evidence_turn_id=evidence_turn_id,
                utterance_id=utterance_id,
                source=source,
            ),
        ),
        _snapshot(
            m,
            m.EventKind.TURN_SNAPSHOT,
            sequence=start_sequence + 2,
            payload=_payload(
                m,
                m.EventKind.TURN_SNAPSHOT,
                evidence_turn_id=evidence_turn_id,
                generated_segment_count=0,
                queued_chunk_count=0,
                started_chunk_count=0,
                transport_confirmed_full_count=0,
            ),
        ),
        _snapshot(
            m,
            m.EventKind.TURN_SETTLED,
            sequence=start_sequence + 3,
            payload=_payload(
                m,
                m.EventKind.TURN_SETTLED,
                evidence_turn_id=evidence_turn_id,
                generated_segment_count=0,
                transport_confirmed_full_count=0,
            ),
        ),
    ]


def _closing_events(m: ModuleType, *, start_sequence: int) -> list[Any]:
    return [
        _snapshot(m, m.EventKind.BINDING_CLOSED, sequence=start_sequence),
        _snapshot(
            m,
            m.EventKind.SESSION_SEAL_REQUESTED,
            sequence=start_sequence + 1,
        ),
    ]


def _content_free_confirmation_payload(
    m: ModuleType,
    *,
    evidence_turn_id: str,
    chunk_ordinal: int = 1,
    evidence_chunk_id: str | None = None,
    synthesis_attempt_id: str | None = None,
    transport_attempt_id: str | None = None,
) -> dict[str, object]:
    payload = _payload(
        m,
        m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
        evidence_turn_id=evidence_turn_id,
        chunk_ordinal=chunk_ordinal,
        evidence_chunk_id=evidence_chunk_id or IDS["evidence_chunk_id"],
        synthesis_attempt_id=synthesis_attempt_id or IDS["synthesis_attempt_id"],
        transport_attempt_id=transport_attempt_id or IDS["transport_attempt_id"],
    )
    payload.pop("evidence_segment_id")
    payload.pop("text")
    return payload


def _complete_turn_kind_path(m: ModuleType, turn_kind: Any) -> tuple[Any, ...]:
    if turn_kind is m.TurnKind.USER_RESPONSE:
        return _valid_event_path(m)
    if turn_kind is m.TurnKind.PROACTIVE_UPDATE:
        events = _opening_events(m)
        events.extend(
            [
                _snapshot(
                    m,
                    m.EventKind.TURN_OPENED,
                    sequence=3,
                    payload={
                        "evidence_turn_id": IDS["evidence_turn_id"],
                        "turn_kind": "proactive_update",
                    },
                ),
                _snapshot(
                    m,
                    m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
                    sequence=4,
                    payload=_content_free_confirmation_payload(
                        m,
                        evidence_turn_id=IDS["evidence_turn_id"],
                    ),
                ),
                _snapshot(
                    m,
                    m.EventKind.TURN_SNAPSHOT,
                    sequence=5,
                    payload=_payload(
                        m,
                        m.EventKind.TURN_SNAPSHOT,
                        turn_kind="proactive_update",
                        generated_segment_count=0,
                        queued_chunk_count=1,
                        started_chunk_count=1,
                        transport_confirmed_full_count=1,
                    ),
                ),
                _snapshot(
                    m,
                    m.EventKind.TURN_SETTLED,
                    sequence=6,
                    payload=_payload(
                        m,
                        m.EventKind.TURN_SETTLED,
                        generated_segment_count=0,
                        transport_confirmed_full_count=1,
                    ),
                ),
            ]
        )
        events.extend(_closing_events(m, start_sequence=7))
        return tuple(events)

    source_turn_id = IDS["evidence_turn_id"]
    replay_turn_id = _uuid(80)
    events = _opening_events(m)
    events.extend(
        [
            _snapshot(m, m.EventKind.TURN_OPENED, sequence=3),
            _snapshot(m, m.EventKind.USER_FINAL_ACCEPTED, sequence=4),
            _snapshot(m, m.EventKind.ASSISTANT_SEGMENT_GENERATED, sequence=5),
            _snapshot(
                m,
                m.EventKind.TURN_SNAPSHOT,
                sequence=6,
                payload=_payload(
                    m,
                    m.EventKind.TURN_SNAPSHOT,
                    queued_chunk_count=1,
                    started_chunk_count=1,
                    transport_confirmed_full_count=0,
                ),
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SETTLED,
                sequence=7,
                payload=_payload(
                    m,
                    m.EventKind.TURN_SETTLED,
                    terminal_disposition="failed",
                    terminal_reason="transport_failed",
                    context_committed=False,
                    transport_confirmed_full_count=0,
                ),
            ),
            _snapshot(
                m,
                m.EventKind.TURN_OPENED,
                sequence=8,
                payload={
                    "evidence_turn_id": replay_turn_id,
                    "turn_kind": "replay",
                    "replay_of_evidence_turn_id": source_turn_id,
                },
            ),
            _snapshot(
                m,
                m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
                sequence=9,
                payload=_content_free_confirmation_payload(
                    m,
                    evidence_turn_id=replay_turn_id,
                ),
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SNAPSHOT,
                sequence=10,
                payload=_payload(
                    m,
                    m.EventKind.TURN_SNAPSHOT,
                    evidence_turn_id=replay_turn_id,
                    turn_kind="replay",
                    generated_segment_count=0,
                    queued_chunk_count=1,
                    started_chunk_count=1,
                    transport_confirmed_full_count=1,
                ),
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SETTLED,
                sequence=11,
                payload=_payload(
                    m,
                    m.EventKind.TURN_SETTLED,
                    evidence_turn_id=replay_turn_id,
                    generated_segment_count=0,
                    transport_confirmed_full_count=1,
                ),
            ),
        ]
    )
    events.extend(_closing_events(m, start_sequence=12))
    return tuple(events)


def test_event_order_accepts_closed_user_path_and_exact_counts() -> None:
    m = _models()
    events = _valid_event_path(m)
    assert m.validate_event_sequence(events) is None


@pytest.mark.parametrize(("queued_count", "started_count"), [(1, 0), (1, 1)])
def test_completed_without_context_requires_every_output_count_to_be_zero(
    queued_count: int,
    started_count: int,
) -> None:
    m = _models()
    events = _opening_events(m)
    events.extend(
        _minimal_user_turn_events(
            m,
            start_sequence=3,
            evidence_turn_id=IDS["evidence_turn_id"],
            utterance_id=IDS["utterance_id"],
        )
    )
    events[5] = _snapshot(
        m,
        m.EventKind.TURN_SETTLED,
        sequence=6,
        payload=_payload(
            m,
            m.EventKind.TURN_SETTLED,
            context_committed=False,
            generated_segment_count=0,
            transport_confirmed_full_count=0,
        ),
    )
    events.extend(_closing_events(m, start_sequence=7))
    assert m.validate_event_sequence(tuple(events)) is None

    events[4] = _snapshot(
        m,
        m.EventKind.TURN_SNAPSHOT,
        sequence=5,
        payload=_payload(
            m,
            m.EventKind.TURN_SNAPSHOT,
            generated_segment_count=0,
            queued_chunk_count=queued_count,
            started_chunk_count=started_count,
            transport_confirmed_full_count=0,
        ),
    )
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


@pytest.mark.parametrize("turn_kind_name", ["USER_RESPONSE", "PROACTIVE_UPDATE", "REPLAY"])
def test_event_state_machine_covers_each_complete_turn_kind_path(
    turn_kind_name: str,
) -> None:
    m = _models()
    turn_kind = getattr(m.TurnKind, turn_kind_name)
    assert m.validate_event_sequence(_complete_turn_kind_path(m, turn_kind)) is None


def test_event_order_rejects_gap_wrong_opening_late_event_and_count_mismatch() -> None:
    m = _models()
    events = list(_valid_event_path(m))
    contiguous_wrong_opening = (
        _snapshot(m, m.EventKind.COMMAND_ROUTED, sequence=1),
        events[1],
    )
    hostile_paths = [
        contiguous_wrong_opening,
        tuple(events[:4] + events[5:]),
        tuple(events + [_snapshot(m, m.EventKind.COMMAND_ROUTED, sequence=11)]),
    ]
    wrong_count = list(events)
    wrong_count[6] = _snapshot(
        m,
        m.EventKind.TURN_SNAPSHOT,
        sequence=7,
        payload=_payload(m, m.EventKind.TURN_SNAPSHOT, generated_segment_count=0),
    )
    hostile_paths.append(tuple(wrong_count))
    for path in hostile_paths:
        with pytest.raises(m.EvidenceModelError):
            m.validate_event_sequence(path)


@pytest.mark.parametrize(
    ("source", "session_source_changes"),
    [
        ("typed", {"microphone_accepted": True, "typed_accepted": False}),
        ("microphone", {"microphone_accepted": False, "typed_accepted": True}),
    ],
)
def test_user_final_source_requires_matching_session_acceptance_flag(
    source: str,
    session_source_changes: dict[str, bool],
) -> None:
    m = _models()
    events = list(_valid_event_path(m))
    events[0] = _snapshot(
        m,
        m.EventKind.SESSION_OPENED,
        sequence=1,
        payload=_payload(m, m.EventKind.SESSION_OPENED, **session_source_changes),
    )
    events[3] = _snapshot(
        m,
        m.EventKind.USER_FINAL_ACCEPTED,
        sequence=4,
        payload=_payload(m, m.EventKind.USER_FINAL_ACCEPTED, source=source),
    )
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


@pytest.mark.parametrize(
    ("source", "session_source_changes"),
    [
        ("typed", {"microphone_accepted": True, "typed_accepted": False}),
        ("microphone", {"microphone_accepted": False, "typed_accepted": True}),
    ],
)
def test_command_source_requires_matching_session_acceptance_flag(
    source: str,
    session_source_changes: dict[str, bool],
) -> None:
    m = _models()
    events = _opening_events(m, **session_source_changes)
    events.append(
        _snapshot(
            m,
            m.EventKind.COMMAND_ROUTED,
            sequence=3,
            payload=_payload(m, m.EventKind.COMMAND_ROUTED, source=source),
        )
    )
    events.extend(_closing_events(m, start_sequence=4))
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


@pytest.mark.parametrize(
    ("source", "availability_field"),
    [("typed", "typed_available"), ("microphone", "microphone_available")],
)
def test_user_final_source_requires_matching_binding_availability_flag(
    source: str,
    availability_field: str,
) -> None:
    m = _models()
    events = list(_valid_event_path(m))
    events[1] = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        payload=_payload(
            m,
            m.EventKind.BINDING_OPENED,
            **{availability_field: False},
        ),
    )
    events[3] = _snapshot(
        m,
        m.EventKind.USER_FINAL_ACCEPTED,
        sequence=4,
        payload=_payload(m, m.EventKind.USER_FINAL_ACCEPTED, source=source),
    )
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


@pytest.mark.parametrize(
    ("source", "availability_field"),
    [("typed", "typed_available"), ("microphone", "microphone_available")],
)
def test_command_source_requires_matching_binding_availability_flag(
    source: str,
    availability_field: str,
) -> None:
    m = _models()
    events = _opening_events(m)
    events[1] = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        payload=_payload(
            m,
            m.EventKind.BINDING_OPENED,
            **{availability_field: False},
        ),
    )
    events.append(
        _snapshot(
            m,
            m.EventKind.COMMAND_ROUTED,
            sequence=3,
            payload=_payload(m, m.EventKind.COMMAND_ROUTED, source=source),
        )
    )
    events.extend(_closing_events(m, start_sequence=4))
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


def test_utterance_id_is_unique_and_consumed_once_per_session() -> None:
    m = _models()
    shared_utterance = IDS["utterance_id"]

    duplicate_users = _opening_events(m)
    duplicate_users.extend(
        _minimal_user_turn_events(
            m,
            start_sequence=3,
            evidence_turn_id=_uuid(70),
            utterance_id=shared_utterance,
        )
    )
    duplicate_users.extend(
        _minimal_user_turn_events(
            m,
            start_sequence=7,
            evidence_turn_id=_uuid(71),
            utterance_id=shared_utterance,
        )
    )
    duplicate_users.extend(_closing_events(m, start_sequence=11))

    duplicate_commands = _opening_events(m)
    for sequence in (3, 4):
        duplicate_commands.append(
            _snapshot(
                m,
                m.EventKind.COMMAND_ROUTED,
                sequence=sequence,
                payload=_payload(
                    m,
                    m.EventKind.COMMAND_ROUTED,
                    utterance_id=shared_utterance,
                ),
            )
        )
    duplicate_commands.extend(_closing_events(m, start_sequence=5))

    for path in (duplicate_users, duplicate_commands):
        with pytest.raises(m.EvidenceModelError):
            m.validate_event_sequence(tuple(path))


def test_one_utterance_cannot_be_both_command_and_response() -> None:
    m = _models()
    events = _opening_events(m)
    events.append(_snapshot(m, m.EventKind.COMMAND_ROUTED, sequence=3))
    events.extend(
        _minimal_user_turn_events(
            m,
            start_sequence=4,
            evidence_turn_id=IDS["evidence_turn_id"],
            utterance_id=IDS["utterance_id"],
        )
    )
    events.extend(_closing_events(m, start_sequence=8))
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


def test_replay_open_rejects_nonexistent_source_turn() -> None:
    m = _models()
    events = list(_complete_turn_kind_path(m, m.TurnKind.REPLAY))
    events[7] = _snapshot(
        m,
        m.EventKind.TURN_OPENED,
        sequence=8,
        payload={
            "evidence_turn_id": _uuid(80),
            "turn_kind": "replay",
            "replay_of_evidence_turn_id": _uuid(900),
        },
    )
    with pytest.raises(m.EvidenceModelError, match="not current and replayable"):
        m.validate_event_sequence(tuple(events))


def test_replay_open_rejects_self_reference_independently() -> None:
    m = _models()
    events = list(_complete_turn_kind_path(m, m.TurnKind.REPLAY))
    events[7] = _snapshot(
        m,
        m.EventKind.TURN_OPENED,
        sequence=8,
        payload={
            "evidence_turn_id": _uuid(80),
            "turn_kind": "replay",
            "replay_of_evidence_turn_id": _uuid(80),
        },
    )
    with pytest.raises(m.EvidenceModelError, match="cannot reference itself"):
        m.validate_event_sequence(tuple(events))


def test_replay_history_rejects_foreign_binding_lineage_before_source_lookup() -> None:
    m = _models()
    events = list(_complete_turn_kind_path(m, m.TurnKind.REPLAY))
    events[1] = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        payload=_payload(
            m,
            m.EventKind.BINDING_OPENED,
            binding_id=_uuid(901),
        ),
    )
    with pytest.raises(m.EvidenceModelError, match="opening records disagree on binding"):
        m.validate_event_sequence(tuple(events))


def test_replay_history_rejects_foreign_logical_session_lineage_before_source_lookup() -> None:
    m = _models()
    events = list(_complete_turn_kind_path(m, m.TurnKind.REPLAY))
    for index in range(2, 7):
        source_event = events[index]
        events[index] = _snapshot(
            m,
            source_event.event_kind,
            sequence=source_event.event_sequence,
            event_id=source_event.event_id,
            logical_session_id=IDS["successor_logical_session_id"],
            payload=source_event.payload,
        )
    with pytest.raises(m.EvidenceModelError, match="crosses an envelope lineage"):
        m.validate_event_sequence(tuple(events))


def test_replay_open_rejects_source_that_is_no_longer_replayable() -> None:
    m = _models()
    events = list(_complete_turn_kind_path(m, m.TurnKind.REPLAY))
    events[6] = _snapshot(
        m,
        m.EventKind.TURN_SETTLED,
        sequence=7,
        payload=_payload(
            m,
            m.EventKind.TURN_SETTLED,
            terminal_disposition="completed",
            terminal_reason="authoritative_close_completed",
            context_committed=True,
            transport_confirmed_full_count=0,
        ),
    )
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


def test_replay_source_cannot_back_two_unsettled_replay_turns() -> None:
    m = _models()
    base = list(_complete_turn_kind_path(m, m.TurnKind.REPLAY))
    events = base[:8]
    second_replay_id = _uuid(81)
    events.append(
        _snapshot(
            m,
            m.EventKind.TURN_OPENED,
            sequence=9,
            payload={
                "evidence_turn_id": second_replay_id,
                "turn_kind": "replay",
                "replay_of_evidence_turn_id": IDS["evidence_turn_id"],
            },
        )
    )
    first_replay_snapshot = base[9]
    first_replay_settlement = base[10]
    events.extend(
        [
            _snapshot(
                m,
                m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
                sequence=10,
                payload=base[8].payload,
            ),
            _snapshot(
                m,
                m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL,
                sequence=11,
                payload=_content_free_confirmation_payload(
                    m,
                    evidence_turn_id=second_replay_id,
                    evidence_chunk_id=_uuid(82),
                    synthesis_attempt_id=_uuid(83),
                    transport_attempt_id=_uuid(84),
                ),
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SNAPSHOT,
                sequence=12,
                payload=first_replay_snapshot.payload,
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SETTLED,
                sequence=13,
                payload=first_replay_settlement.payload,
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SNAPSHOT,
                sequence=14,
                payload=_payload(
                    m,
                    m.EventKind.TURN_SNAPSHOT,
                    evidence_turn_id=second_replay_id,
                    turn_kind="replay",
                    generated_segment_count=0,
                    queued_chunk_count=1,
                    started_chunk_count=1,
                    transport_confirmed_full_count=1,
                ),
            ),
            _snapshot(
                m,
                m.EventKind.TURN_SETTLED,
                sequence=15,
                payload=_payload(
                    m,
                    m.EventKind.TURN_SETTLED,
                    evidence_turn_id=second_replay_id,
                    generated_segment_count=0,
                    transport_confirmed_full_count=1,
                ),
            ),
        ]
    )
    events.extend(_closing_events(m, start_sequence=16))
    with pytest.raises(m.EvidenceModelError):
        m.validate_event_sequence(tuple(events))


def test_event_retry_and_conflict_classification_is_closed() -> None:
    m = _models()
    existing = _snapshot(m, m.EventKind.COMMAND_ROUTED, event_id=IDS["event_id"])
    assert (
        m.classify_event_retry(existing, existing, target_sealed=False)
        is m.StoreDisposition.IDEMPOTENT
    )
    changed = _snapshot(
        m,
        m.EventKind.COMMAND_ROUTED,
        event_id=IDS["event_id"],
        payload=_payload(m, m.EventKind.COMMAND_ROUTED, source="microphone"),
    )
    assert (
        m.classify_event_retry(existing, changed, target_sealed=False)
        is m.ConflictReason.EVENT_ID_ENVELOPE_MISMATCH
    )
    cross_session = _snapshot(
        m,
        m.EventKind.COMMAND_ROUTED,
        event_id=IDS["event_id"],
        logical_session_id=IDS["successor_logical_session_id"],
    )
    assert (
        m.classify_event_retry(existing, cross_session, target_sealed=False)
        is m.ConflictReason.CROSS_SESSION_EVENT_ID
    )
    claimed_sequence = _snapshot(
        m,
        m.EventKind.COMMAND_ROUTED,
        event_id=IDS["replay_event_id"],
    )
    assert (
        m.classify_event_retry(existing, claimed_sequence, target_sealed=False)
        is m.ConflictReason.SESSION_SEQUENCE_CLAIMED
    )
    assert (
        m.classify_event_retry(existing, claimed_sequence, target_sealed=True)
        is m.ConflictReason.SEALED_SESSION_REUSE
    )


def test_event_retry_requires_explicit_target_sealed() -> None:
    m = _models()
    existing = _snapshot(m, m.EventKind.COMMAND_ROUTED, event_id=IDS["event_id"])
    incoming = _snapshot(m, m.EventKind.COMMAND_ROUTED, event_id=IDS["event_id"])

    with pytest.raises(TypeError):
        m.classify_event_retry(existing, incoming)


def test_runtime_phase_graph_is_exhaustive() -> None:
    m = _models()
    allowed = {
        ("open", "expiring"),
        ("open", "closing"),
        ("open", "revoking"),
        ("expiring", "open"),
        ("expiring", "stopped"),
        ("expiring", "revoking"),
        ("closing", "seal_queued"),
        ("closing", "stopped"),
        ("closing", "revoking"),
        ("seal_queued", "stopped"),
        ("seal_queued", "revoking"),
        ("revoking", "stopped"),
    }
    for source in m.RuntimeSessionPhase:
        for target in m.RuntimeSessionPhase:
            if (source.value, target.value) in allowed:
                assert m.validate_runtime_phase_transition(source, target) is target
            else:
                with pytest.raises(m.EvidenceModelError):
                    m.validate_runtime_phase_transition(source, target)


def test_persistent_state_graph_is_exhaustive() -> None:
    m = _models()
    allowed = {("open", "sealed"), ("open", "tainted"), ("sealed", "tainted")}
    for source in m.PersistentSessionState:
        for target in m.PersistentSessionState:
            if (source.value, target.value) in allowed:
                assert m.validate_persistent_state_transition(source, target) is target
            else:
                with pytest.raises(m.EvidenceModelError):
                    m.validate_persistent_state_transition(source, target)


ELIGIBILITY_FIELDS = [
    "text_persisted",
    "text_ever_persisted",
    "purge_verified",
    "purge_failed",
    "turn_kind",
    "snapshot_count",
    "snapshot_valid",
    "settlement_count",
    "settlement_valid",
    "terminal_disposition",
    "terminal_reason",
    "context_committed",
    "session_state",
    "gap_free",
    "hash_valid",
    "conflict_free",
    "epoch_state",
    "unexpired",
    "pending_erasure",
]


def _eligibility_values(m: ModuleType, **changes: object) -> dict[str, object]:
    available_fields = {
        field.name for field in dataclasses.fields(m.EvidenceEligibilityFactsV1)
    }
    values: dict[str, object] = {
        "text_persisted": True,
        "text_ever_persisted": True,
        "purge_verified": False,
        "purge_failed": False,
        "turn_kind": m.TurnKind.USER_RESPONSE,
        "snapshot_count": 1,
        "snapshot_valid": True,
        "settlement_count": 1,
        "settlement_valid": True,
        "terminal_disposition": m.TerminalDisposition.COMPLETED,
        "terminal_reason": m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
        "context_committed": True,
        "session_state": m.PersistentSessionState.SEALED,
        "gap_free": True,
        "hash_valid": True,
        "conflict_free": True,
        "epoch_state": m.ConsentEpochState.CLOSED,
        "unexpired": True,
        "pending_erasure": False,
    }
    values.update(changes)
    return {name: value for name, value in values.items() if name in available_fields}


def _eligibility_facts(m: ModuleType, **changes: object) -> Any:
    return m.EvidenceEligibilityFactsV1(**_eligibility_values(m, **changes))


def test_eligibility_facts_encode_exact_cardinalities_and_explicit_hash_fact() -> None:
    m = _models()
    assert [
        field.name for field in dataclasses.fields(m.EvidenceEligibilityFactsV1)
    ] == ELIGIBILITY_FIELDS
    facts = _eligibility_facts(m)
    assert facts.snapshot_count == 1
    assert facts.settlement_count == 1

    for field in ("snapshot_count", "settlement_count"):
        with pytest.raises(m.EvidenceModelError):
            _eligibility_facts(m, **{field: True})


def test_eligibility_hash_valid_is_required_without_a_fail_open_default() -> None:
    m = _models()
    hash_field = next(
        field
        for field in dataclasses.fields(m.EvidenceEligibilityFactsV1)
        if field.name == "hash_valid"
    )
    assert hash_field.default is dataclasses.MISSING

    without_hash = _eligibility_values(m)
    without_hash.pop("hash_valid")
    with pytest.raises(TypeError):
        m.EvidenceEligibilityFactsV1(**without_hash)


def test_eligibility_requires_every_normative_predicate() -> None:
    m = _models()
    assert m.evaluate_eligibility(_eligibility_facts(m)) is m.EligibilityOutcome.ELIGIBLE
    exclusions = [
        {"snapshot_valid": False},
        {"settlement_valid": False},
        {
            "terminal_disposition": m.TerminalDisposition.INTERRUPTED,
            "terminal_reason": m.TerminalReason.BARGE_IN,
        },
        {"context_committed": False},
        {"session_state": m.PersistentSessionState.OPEN},
        {"gap_free": False},
        {"hash_valid": False},
        {"conflict_free": False},
        {"epoch_state": m.ConsentEpochState.ACTIVE},
        {"unexpired": False},
        {"pending_erasure": True},
        {"purge_failed": True},
    ]
    fact_fields = {
        field.name for field in dataclasses.fields(m.EvidenceEligibilityFactsV1)
    }
    if "snapshot_count" in fact_fields:
        exclusions.extend(({"snapshot_count": 0}, {"snapshot_count": 2}))
    if "settlement_count" in fact_fields:
        exclusions.extend(({"settlement_count": 0}, {"settlement_count": 2}))
    for changes in exclusions:
        facts = _eligibility_facts(m, **changes)
        assert m.evaluate_eligibility(facts) is m.EligibilityOutcome.PERSISTED_BUT_EXCLUDED


def test_eligibility_distinguishes_never_persisted_excluded_and_purged() -> None:
    m = _models()
    never = _eligibility_facts(
        m,
        text_persisted=False,
        text_ever_persisted=False,
        turn_kind=m.TurnKind.PROACTIVE_UPDATE,
    )
    assert m.evaluate_eligibility(never) is m.EligibilityOutcome.NEVER_PERSISTED
    purged = _eligibility_facts(
        m,
        text_persisted=False,
        text_ever_persisted=True,
        purge_verified=True,
    )
    assert m.evaluate_eligibility(purged) is m.EligibilityOutcome.PURGED

    deletion_pending_verification = _eligibility_facts(
        m,
        text_persisted=False,
        text_ever_persisted=True,
        pending_erasure=True,
    )
    failed_purge = _eligibility_facts(
        m,
        purge_failed=True,
    )
    for facts in (deletion_pending_verification, failed_purge):
        assert (
            m.evaluate_eligibility(facts)
            is m.EligibilityOutcome.PERSISTED_BUT_EXCLUDED
        )

    with pytest.raises(m.EvidenceModelError):
        _eligibility_facts(m, text_persisted=True, purge_verified=True)
    with pytest.raises(m.EvidenceModelError):
        _eligibility_facts(m, turn_kind=m.TurnKind.REPLAY)


def test_is_turn_eligible_v1_is_an_exact_boolean_predicate() -> None:
    m = _models()
    eligible = m.is_turn_eligible_v1(_eligibility_facts(m))
    excluded = m.is_turn_eligible_v1(_eligibility_facts(m, hash_valid=False))
    assert eligible is True
    assert excluded is False


def test_public_writer_fault_and_control_http_mapping_is_exhaustive() -> None:
    m = _models()
    unavailable = {
        m.WriterFault.UNSUPPORTED_PLATFORM,
        m.WriterFault.PATH_INVALID,
        m.WriterFault.ASSET_MISMATCH,
    }
    for fault in m.WriterFault:
        expected = (
            m.ControlError.CAPTURE_UNAVAILABLE
            if fault in unavailable
            else m.ControlError.WRITER_UNAVAILABLE
        )
        assert m.writer_fault_to_control_error(fault) is expected
        assert (
            m.writer_fault_to_control_error(fault, revoke_before_durable=True)
            is m.ControlError.REVOKE_NOT_DURABLE
        )
    statuses = {
        m.ControlError.INVALID_CONTROL: 400,
        m.ControlError.UNAUTHORIZED: 401,
        m.ControlError.FORBIDDEN: 403,
        m.ControlError.CONTROL_SEQUENCE_CONFLICT: 409,
        m.ControlError.CAPTURE_UNAVAILABLE: 409,
        m.ControlError.CONTROL_PENDING: 409,
        m.ControlError.STATUS_CAPACITY_UNAVAILABLE: 503,
        m.ControlError.CONTROL_TIMEOUT: 503,
        m.ControlError.WRITER_UNAVAILABLE: 503,
        m.ControlError.REVOKE_NOT_DURABLE: 503,
        m.ControlError.PURGE_FAILED: 503,
    }
    assert {error: m.control_error_http_status(error) for error in m.ControlError} == statuses


def test_public_control_success_http_mapping_is_exact_and_exhaustive() -> None:
    m = _models()
    statuses = {
        m.ControlResult.CONSENT_ACTIVATED: 200,
        m.ControlResult.REVOKE_DURABLY_SCHEDULED: 202,
        m.ControlResult.PURGE_COMPLETED: 200,
    }
    assert {
        result: m.control_result_http_status(result) for result in m.ControlResult
    } == statuses


def test_public_fault_mapping_rejects_unregistered_future_writer_fault() -> None:
    m = _models()
    future = _future_str_enum_member(m.WriterFault, "future_writer_fault")
    with pytest.raises(m.EvidenceModelError):
        m.writer_fault_to_control_error(future)


def test_payload_source_validation_rejects_unregistered_future_input_source() -> None:
    m = _models()
    future = _future_str_enum_member(m.InputSource, "future_input_source")
    with pytest.raises(m.EvidenceModelError):
        m.UserFinalAcceptedPayloadV1(
            utterance_id=IDS["utterance_id"],
            evidence_turn_id=IDS["evidence_turn_id"],
            source=future,
            routing_disposition="response",
            text="exact source",
        )


def test_diagnostics_has_exact_content_free_shape_and_bounds() -> None:
    m = _models()
    diagnostics = m.EvidenceDiagnosticsV1(
        protocol_version=1,
        owner_state=m.OwnerState.RUNNING,
        capture_state=m.CaptureState.ACTIVE,
        sticky_fault=None,
        queue_record_count=64,
        queue_canonical_bytes=2_097_152,
        active_lease_count=64,
        pending_revoke=False,
        purge_required=False,
    )
    assert [field.name for field in dataclasses.fields(diagnostics)] == [
        "protocol_version",
        "owner_state",
        "capture_state",
        "sticky_fault",
        "queue_record_count",
        "queue_canonical_bytes",
        "active_lease_count",
        "pending_revoke",
        "purge_required",
    ]
    assert diagnostics.__dataclass_params__.frozen is True
    assert not hasattr(diagnostics, "__dict__")
    for field, value in (
        ("queue_record_count", 65),
        ("queue_canonical_bytes", 2_097_153),
        ("active_lease_count", -1),
        ("pending_revoke", 1),
    ):
        values = {
            item.name: getattr(diagnostics, item.name)
            for item in dataclasses.fields(diagnostics)
        }
        values[field] = value
        with pytest.raises(m.EvidenceModelError):
            m.EvidenceDiagnosticsV1(**values)


AUTHORITY_FIELDS = {
    "FinalInputAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "utterance_id",
        "source",
        "input_incarnation",
        "media_incarnation",
        "typed_sequence",
    ],
    "CommandAdmissionAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "utterance_id",
        "source",
        "input_incarnation",
        "media_incarnation",
        "typed_sequence",
        "routing_serial",
        "routing_disposition",
    ],
    "UserTurnAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "utterance_id",
        "source",
        "input_incarnation",
        "media_incarnation",
        "typed_sequence",
        "routing_serial",
        "routing_disposition",
    ],
    "ProactiveTurnAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "proactive_invocation_serial",
    ],
    "ReplayTurnAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "replay_of_evidence_turn_id",
        "replay_generation",
    ],
    "BindingCloseAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "close_reason",
    ],
    "RolloverAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "predecessor_logical_session_id",
        "successor_logical_session_id",
        "successor_expires_at_utc",
        "reason",
    ],
    "LifecycleSealAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "logical_session_id",
        "close_reason",
        "final_event_sequence",
        "close_epoch",
    ],
    "SessionExpiryAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "consent_epoch_id",
        "logical_session_id",
        "expires_at_utc",
        "deadline_admission_ordinal",
        "mode",
    ],
    "ConsentCreateAuthorityV1": [
        "protocol_version",
        "binding_id",
        "binding_generation",
        "control_sequence",
        "control_fingerprint_hash",
        "consent_version",
        "disclosure_digest",
        "retention_hours",
        "microphone_accepted",
        "typed_accepted",
        "projection_reservation",
        "create_epoch_reservation",
    ],
    "ConsentRevokeAuthorityV1": [
        "protocol_version",
        "binding_id",
        "binding_generation",
        "consent_epoch_id",
        "revoke_gate_generation",
        "control_sequence",
        "control_fingerprint_hash",
        "projection_reservation",
    ],
    "LifecycleDrainAuthorityV1": [
        "protocol_version",
        "owner_generation",
        "final_admission_ordinal",
    ],
    "TerminalCauseCapabilityV1": [
        "protocol_version",
        "owner_generation",
        "logical_session_id",
        "evidence_turn_id",
        "lease_serial",
        "sink_token",
    ],
}


def _structural_capability_fixture(
    m: ModuleType,
    name: str,
    values: dict[str, object] | None = None,
) -> Any:
    """Build a test-only structural fixture without defining a public mint surface."""

    cls = getattr(m, name)
    supplied = {} if values is None else values
    expected = {field.name for field in dataclasses.fields(cls)}
    assert set(supplied) == expected
    instance = object.__new__(cls)
    for field_name, value in supplied.items():
        object.__setattr__(instance, field_name, value)
    cls._validate(instance)
    return instance


def _authority_values(m: ModuleType, name: str, **changes: object) -> dict[str, object]:
    common: dict[str, object] = {
        "protocol_version": 1,
        "owner_generation": 1,
        "binding_id": IDS["binding_id"],
        "binding_generation": 1,
        "consent_epoch_id": IDS["consent_epoch_id"],
        "logical_session_id": IDS["logical_session_id"],
    }
    additions: dict[str, dict[str, object]] = {
        "FinalInputAuthorityV1": {
            "utterance_id": IDS["utterance_id"],
            "source": m.InputSource.TYPED,
            "input_incarnation": 1,
            "media_incarnation": None,
            "typed_sequence": 1,
        },
        "CommandAdmissionAuthorityV1": {
            "utterance_id": IDS["utterance_id"],
            "source": m.InputSource.TYPED,
            "input_incarnation": 1,
            "media_incarnation": None,
            "typed_sequence": 1,
            "routing_serial": 1,
            "routing_disposition": "command",
        },
        "UserTurnAuthorityV1": {
            "utterance_id": IDS["utterance_id"],
            "source": m.InputSource.TYPED,
            "input_incarnation": 1,
            "media_incarnation": None,
            "typed_sequence": 1,
            "routing_serial": 1,
            "routing_disposition": "response",
        },
        "ProactiveTurnAuthorityV1": {"proactive_invocation_serial": 1},
        "ReplayTurnAuthorityV1": {
            "replay_of_evidence_turn_id": IDS["evidence_turn_id"],
            "replay_generation": 1,
        },
        "BindingCloseAuthorityV1": {"close_reason": m.BindingCloseReason.CLIENT_CLOSED},
        "RolloverAuthorityV1": {
            "predecessor_logical_session_id": IDS["logical_session_id"],
            "successor_logical_session_id": IDS["successor_logical_session_id"],
            "successor_expires_at_utc": UTC,
            "reason": m.BindingCloseReason.CAPACITY_ROLLOVER,
        },
        "LifecycleSealAuthorityV1": {
            "close_reason": m.BindingCloseReason.CLIENT_CLOSED,
            "final_event_sequence": 4,
            "close_epoch": True,
        },
        "SessionExpiryAuthorityV1": {
            "expires_at_utc": UTC,
            "deadline_admission_ordinal": 2,
            "mode": m.ExpiryMode.ERASE_STUCK,
        },
        "ConsentCreateAuthorityV1": {
            "control_sequence": 1,
            "control_fingerprint_hash": HASH,
            "consent_version": CONSENT_LITERAL,
            "disclosure_digest": HASH,
            "retention_hours": 24,
            "microphone_accepted": True,
            "typed_accepted": True,
            "projection_reservation": _structural_capability_fixture(
                m,
                "ProjectionReservation",
            ),
            "create_epoch_reservation": _structural_capability_fixture(
                m,
                "CreateEpochReservationV1",
            ),
        },
        "ConsentRevokeAuthorityV1": {
            "revoke_gate_generation": 1,
            "control_sequence": 1,
            "control_fingerprint_hash": HASH,
            "projection_reservation": _structural_capability_fixture(
                m,
                "ProjectionReservation",
            ),
        },
        "LifecycleDrainAuthorityV1": {"final_admission_ordinal": 2},
        "TerminalCauseCapabilityV1": {
            "evidence_turn_id": IDS["evidence_turn_id"],
            "lease_serial": 1,
            "sink_token": b"0123456789abcdef",
        },
    }
    allowed = set(AUTHORITY_FIELDS[name])
    values = {**common, **additions[name]}
    values = {key: value for key, value in values.items() if key in allowed}
    values.update(changes)
    return values


def _authority(m: ModuleType, name: str, **changes: object) -> Any:
    return _structural_capability_fixture(
        m,
        name,
        _authority_values(m, name, **changes),
    )


@pytest.mark.parametrize(("name", "expected_fields"), AUTHORITY_FIELDS.items())
def test_authority_shapes_are_exact_private_frozen_and_slotted(
    name: str,
    expected_fields: list[str],
) -> None:
    m = _models()
    cls = getattr(m, name)
    assert [field.name for field in dataclasses.fields(cls)] == expected_fields
    with pytest.raises(TypeError):
        cls(**_authority_values(m, name))
    authority = _authority(m, name)
    assert authority.__dataclass_params__.frozen is True
    assert not hasattr(authority, "__dict__")


def test_capability_classes_expose_no_publicly_callable_mint() -> None:
    m = _models()
    capability_names = tuple(AUTHORITY_FIELDS) + (
        "ProjectionReservation",
        "CreateEpochReservationV1",
    )
    for name in capability_names:
        cls = getattr(m, name)
        assert not callable(getattr(cls, "_mint", None)), name


def test_runtime_capabilities_use_identity_and_reject_repr_copy_pickle_and_json() -> None:
    m = _models()
    first = _authority(m, "FinalInputAuthorityV1")
    second = _authority(m, "FinalInputAuthorityV1")
    assert first is not second
    assert first != second
    assert IDS["binding_id"] not in repr(first)
    assert IDS["utterance_id"] not in repr(first)
    assert "0123456789abcdef" not in repr(
        _authority(m, "TerminalCauseCapabilityV1")
    )
    assert isinstance(hash(first), int)
    for operation in (copy.copy, copy.deepcopy, pickle.dumps, json.dumps):
        with pytest.raises(TypeError):
            operation(first)


def test_capability_dataclass_traversal_does_not_disclose_bearers() -> None:
    m = _models()
    authority = _authority(m, "FinalInputAuthorityV1")
    for operation in (dataclasses.asdict, dataclasses.astuple):
        with pytest.raises(TypeError):
            operation(authority)


def test_final_input_and_routing_authorities_enforce_source_and_literal_kind() -> None:
    m = _models()
    _authority(
        m,
        "FinalInputAuthorityV1",
        source=m.InputSource.MICROPHONE,
        media_incarnation=1,
        typed_sequence=None,
    )
    contradictions = [
        {"source": m.InputSource.MICROPHONE, "media_incarnation": None, "typed_sequence": 1},
        {"source": m.InputSource.TYPED, "media_incarnation": 1, "typed_sequence": None},
        {"typed_sequence": 2**53},
    ]
    for changes in contradictions:
        with pytest.raises(m.EvidenceModelError):
            _authority(m, "FinalInputAuthorityV1", **changes)
    with pytest.raises(m.EvidenceModelError):
        _authority(m, "CommandAdmissionAuthorityV1", routing_disposition="response")
    with pytest.raises(m.EvidenceModelError):
        _authority(m, "UserTurnAuthorityV1", routing_disposition="command")


def test_final_input_authority_rejects_unregistered_future_source_member() -> None:
    m = _models()
    future = _future_str_enum_member(m.InputSource, "future_input_source")
    with pytest.raises(m.EvidenceModelError):
        _authority(
            m,
            "FinalInputAuthorityV1",
            source=future,
            media_incarnation=None,
            typed_sequence=1,
        )


def test_rollover_seal_expiry_and_consent_authority_contradictions_fail() -> None:
    m = _models()
    invalid = [
        ("RolloverAuthorityV1", {"reason": m.BindingCloseReason.CLIENT_CLOSED}),
        ("LifecycleSealAuthorityV1", {"close_epoch": False}),
        ("SessionExpiryAuthorityV1", {"expires_at_utc": UTC.lower()}),
        ("ConsentCreateAuthorityV1", {"microphone_accepted": False, "typed_accepted": False}),
        ("ConsentCreateAuthorityV1", {"projection_reservation": object()}),
        ("TerminalCauseCapabilityV1", {"sink_token": b"short"}),
    ]
    for name, changes in invalid:
        with pytest.raises(m.EvidenceModelError):
            _authority(m, name, **changes)


def test_command_routing_result_has_closed_cross_field_combinations() -> None:
    m = _models()
    user = _authority(m, "UserTurnAuthorityV1")
    not_command = m.CommandRoutingResultV1(
        outcome=m.CommandRoutingOutcome.NOT_COMMAND,
        command_disposition=None,
        user_turn_authority=user,
    )
    accepted = m.CommandRoutingResultV1(
        outcome=m.CommandRoutingOutcome.ACCEPTED,
        command_disposition=m.CommandDisposition.ADMITTED,
        user_turn_authority=None,
    )
    assert [field.name for field in dataclasses.fields(not_command)] == [
        "outcome",
        "command_disposition",
        "user_turn_authority",
    ]
    assert accepted.outcome is m.CommandRoutingOutcome.ACCEPTED
    for result in (not_command, accepted):
        with pytest.raises(TypeError):
            bool(result)
    for values in (
        (m.CommandRoutingOutcome.NOT_COMMAND, None, None),
        (m.CommandRoutingOutcome.ACCEPTED, None, user),
        (m.CommandRoutingOutcome.REJECTED, m.CommandDisposition.ADMITTED, None),
        (m.CommandRoutingOutcome.INVALID, None, user),
    ):
        with pytest.raises(m.EvidenceModelError):
            m.CommandRoutingResultV1(
                outcome=values[0],
                command_disposition=values[1],
                user_turn_authority=values[2],
            )


DTO_FIELDS = {
    "QueuedEvidenceRecordV1": [
        "protocol_version",
        "snapshot",
        "admission_ordinal",
        "reservation_class",
        "lease_open_ordinal",
    ],
    "BindingCloseV1": [
        "protocol_version",
        "binding_id",
        "consent_epoch_id",
        "logical_session_id",
        "admission_ordinal",
        "snapshot",
    ],
    "CreateEpochV1": [
        "protocol_version",
        "installation_id",
        "producer_instance_id",
        "consent_epoch_id",
        "logical_session_id",
        "binding_id",
        "binding_generation",
        "consent_version",
        "disclosure_digest",
        "retention_hours",
        "microphone_accepted",
        "typed_accepted",
        "control_sequence",
        "control_fingerprint_hash",
        "session_opened",
        "binding_opened",
    ],
    "RolloverSessionV1": [
        "protocol_version",
        "consent_epoch_id",
        "binding_id",
        "binding_generation",
        "predecessor_logical_session_id",
        "successor_logical_session_id",
        "predecessor_final_event_sequence",
        "successor_expires_at_utc",
        "admission_ordinal",
        "snapshots",
    ],
    "ExpireSessionV1": [
        "protocol_version",
        "owner_generation",
        "logical_session_id",
        "consent_epoch_id",
        "expires_at_utc",
        "last_admission_ordinal",
        "erasure_request_id",
        "mode",
    ],
    "RevokeRequestV1": [
        "protocol_version",
        "erasure_request_id",
        "consent_epoch_id",
        "control_sequence",
        "control_fingerprint_hash",
        "last_admission_ordinal",
    ],
    "RevokeFinalizeV1": [
        "protocol_version",
        "erasure_request_id",
        "consent_epoch_id",
        "control_sequence",
        "control_fingerprint_hash",
        "final_admission_ordinal",
    ],
    "SealEpochV1": [
        "protocol_version",
        "consent_epoch_id",
        "logical_session_id",
        "binding_id",
        "final_event_sequence",
        "close_reason",
        "close_epoch",
        "admission_ordinal",
        "snapshot",
    ],
    "MaintenanceV1": [
        "protocol_version",
        "erasure_reason",
        "erasure_scope",
        "erasure_request_id",
        "scope_id",
        "deadline_admission_ordinal",
        "artifact_manifest_version",
    ],
    "FullPurgeV1": [
        "protocol_version",
        "full_purge_generation_id",
        "sentinel_state",
        "artifact_manifest_version",
    ],
    "DrainAndStopV1": ["protocol_version", "owner_generation", "final_admission_ordinal"],
}


def _create_epoch_snapshots(m: ModuleType) -> tuple[Any, Any]:
    return (
        _snapshot(m, m.EventKind.SESSION_OPENED, sequence=1),
        _snapshot(m, m.EventKind.BINDING_OPENED, sequence=2),
    )


def _rollover_snapshots(
    m: ModuleType,
    *,
    final_sequence: int = 4,
    reason: str = "capacity_rollover",
    event_ids: tuple[str, str, str, str] | None = None,
) -> tuple[Any, Any, Any, Any]:
    resolved_event_ids = event_ids or tuple(_uuid(300 + offset) for offset in range(4))
    close_snapshot = _snapshot(
        m,
        m.EventKind.BINDING_CLOSED,
        sequence=final_sequence - 1,
        event_id=resolved_event_ids[0],
        payload=_payload(m, m.EventKind.BINDING_CLOSED, close_reason=reason),
    )
    seal_snapshot = _snapshot(
        m,
        m.EventKind.SESSION_SEAL_REQUESTED,
        sequence=final_sequence,
        event_id=resolved_event_ids[1],
        payload=_payload(
            m,
            m.EventKind.SESSION_SEAL_REQUESTED,
            final_event_sequence=final_sequence,
        ),
    )
    successor_open = _snapshot(
        m,
        m.EventKind.SESSION_OPENED,
        sequence=1,
        event_id=resolved_event_ids[2],
        logical_session_id=IDS["successor_logical_session_id"],
        payload=_payload(
            m,
            m.EventKind.SESSION_OPENED,
            predecessor_session_id=IDS["logical_session_id"],
        ),
    )
    successor_binding = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        event_id=resolved_event_ids[3],
        logical_session_id=IDS["successor_logical_session_id"],
    )
    return close_snapshot, seal_snapshot, successor_open, successor_binding


def _dto_values(m: ModuleType, name: str, **changes: object) -> dict[str, object]:
    session_opened, binding_opened = _create_epoch_snapshots(m)
    close_snapshot = _snapshot(
        m,
        m.EventKind.BINDING_CLOSED,
        sequence=3,
    )
    seal_snapshot = _snapshot(
        m,
        m.EventKind.SESSION_SEAL_REQUESTED,
        sequence=4,
        payload=_payload(m, m.EventKind.SESSION_SEAL_REQUESTED, final_event_sequence=4),
    )
    successor_open = _snapshot(
        m,
        m.EventKind.SESSION_OPENED,
        sequence=1,
        logical_session_id=IDS["successor_logical_session_id"],
        payload=_payload(
            m,
            m.EventKind.SESSION_OPENED,
            predecessor_session_id=IDS["logical_session_id"],
        ),
    )
    successor_binding = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        logical_session_id=IDS["successor_logical_session_id"],
    )
    rollover_close = _snapshot(
        m,
        m.EventKind.BINDING_CLOSED,
        sequence=3,
        payload=_payload(
            m,
            m.EventKind.BINDING_CLOSED,
            close_reason="capacity_rollover",
        ),
    )
    values: dict[str, dict[str, object]] = {
        "QueuedEvidenceRecordV1": {
            "protocol_version": 1,
            "snapshot": _snapshot(m, m.EventKind.COMMAND_ROUTED),
            "admission_ordinal": 1,
            "reservation_class": m.QueueReservationClass.ORDINARY,
            "lease_open_ordinal": None,
        },
        "BindingCloseV1": {
            "protocol_version": 1,
            "binding_id": IDS["binding_id"],
            "consent_epoch_id": IDS["consent_epoch_id"],
            "logical_session_id": IDS["logical_session_id"],
            "admission_ordinal": 3,
            "snapshot": close_snapshot,
        },
        "CreateEpochV1": {
            "protocol_version": 1,
            "installation_id": IDS["installation_id"],
            "producer_instance_id": IDS["producer_instance_id"],
            "consent_epoch_id": IDS["consent_epoch_id"],
            "logical_session_id": IDS["logical_session_id"],
            "binding_id": IDS["binding_id"],
            "binding_generation": 1,
            "consent_version": CONSENT_LITERAL,
            "disclosure_digest": HASH,
            "retention_hours": 24,
            "microphone_accepted": True,
            "typed_accepted": True,
            "control_sequence": 1,
            "control_fingerprint_hash": HASH,
            "session_opened": session_opened,
            "binding_opened": binding_opened,
        },
        "RolloverSessionV1": {
            "protocol_version": 1,
            "consent_epoch_id": IDS["consent_epoch_id"],
            "binding_id": IDS["binding_id"],
            "binding_generation": 1,
            "predecessor_logical_session_id": IDS["logical_session_id"],
            "successor_logical_session_id": IDS["successor_logical_session_id"],
            "predecessor_final_event_sequence": 4,
            "successor_expires_at_utc": UTC,
            "admission_ordinal": 3,
            "snapshots": (rollover_close, seal_snapshot, successor_open, successor_binding),
        },
        "ExpireSessionV1": {
            "protocol_version": 1,
            "owner_generation": 1,
            "logical_session_id": IDS["logical_session_id"],
            "consent_epoch_id": IDS["consent_epoch_id"],
            "expires_at_utc": UTC,
            "last_admission_ordinal": 2,
            "erasure_request_id": IDS["erasure_request_id"],
            "mode": m.ExpiryMode.ERASE_STUCK,
        },
        "RevokeRequestV1": {
            "protocol_version": 1,
            "erasure_request_id": IDS["erasure_request_id"],
            "consent_epoch_id": IDS["consent_epoch_id"],
            "control_sequence": 1,
            "control_fingerprint_hash": HASH,
            "last_admission_ordinal": 2,
        },
        "RevokeFinalizeV1": {
            "protocol_version": 1,
            "erasure_request_id": IDS["erasure_request_id"],
            "consent_epoch_id": IDS["consent_epoch_id"],
            "control_sequence": 1,
            "control_fingerprint_hash": HASH,
            "final_admission_ordinal": 3,
        },
        "SealEpochV1": {
            "protocol_version": 1,
            "consent_epoch_id": IDS["consent_epoch_id"],
            "logical_session_id": IDS["logical_session_id"],
            "binding_id": IDS["binding_id"],
            "final_event_sequence": 4,
            "close_reason": m.BindingCloseReason.CLIENT_CLOSED,
            "close_epoch": True,
            "admission_ordinal": 4,
            "snapshot": seal_snapshot,
        },
        "MaintenanceV1": {
            "protocol_version": 1,
            "erasure_reason": m.ErasureReason.TTL,
            "erasure_scope": m.ErasureScope.SESSION,
            "erasure_request_id": IDS["erasure_request_id"],
            "scope_id": IDS["logical_session_id"],
            "deadline_admission_ordinal": 2,
            "artifact_manifest_version": 1,
        },
        "FullPurgeV1": {
            "protocol_version": 1,
            "full_purge_generation_id": IDS["full_purge_generation_id"],
            "sentinel_state": m.SentinelState.FULL_PURGE_PENDING,
            "artifact_manifest_version": 1,
        },
        "DrainAndStopV1": {
            "protocol_version": 1,
            "owner_generation": 1,
            "final_admission_ordinal": 2,
        },
    }
    result = values[name]
    result.update(changes)
    return result


def _dto(m: ModuleType, name: str, **changes: object) -> Any:
    return getattr(m, name)(**_dto_values(m, name, **changes))


@pytest.mark.parametrize(("name", "expected_fields"), DTO_FIELDS.items())
def test_dto_shapes_are_exact_frozen_slotted_and_versioned(
    name: str,
    expected_fields: list[str],
) -> None:
    m = _models()
    dto = _dto(m, name)
    assert [field.name for field in dataclasses.fields(dto)] == expected_fields
    assert dto.__dataclass_params__.frozen is True
    assert not hasattr(dto, "__dict__")
    with pytest.raises(m.EvidenceModelError):
        _dto(m, name, protocol_version=2)


@pytest.mark.parametrize(
    ("name", "changes"),
    [
        pytest.param("ExpireSessionV1", {"owner_generation": True}, id="expire-owner"),
        pytest.param(
            "ExpireSessionV1",
            {"logical_session_id": "not-a-uuid"},
            id="expire-session",
        ),
        pytest.param(
            "ExpireSessionV1",
            {"consent_epoch_id": "not-a-uuid"},
            id="expire-epoch",
        ),
        pytest.param(
            "ExpireSessionV1",
            {"expires_at_utc": "2026-08-08T12:34:56Z"},
            id="expire-time",
        ),
        pytest.param(
            "ExpireSessionV1",
            {"last_admission_ordinal": -1},
            id="expire-watermark",
        ),
        pytest.param(
            "ExpireSessionV1",
            {"erasure_request_id": "not-a-uuid"},
            id="expire-request",
        ),
        pytest.param("ExpireSessionV1", {"mode": "rollover"}, id="expire-mode"),
        pytest.param(
            "RevokeRequestV1",
            {"erasure_request_id": "not-a-uuid"},
            id="revoke-request-id",
        ),
        pytest.param(
            "RevokeRequestV1",
            {"consent_epoch_id": "not-a-uuid"},
            id="revoke-request-epoch",
        ),
        pytest.param(
            "RevokeRequestV1",
            {"control_sequence": 0},
            id="revoke-request-sequence",
        ),
        pytest.param(
            "RevokeRequestV1",
            {"control_fingerprint_hash": "A" * 64},
            id="revoke-request-fingerprint",
        ),
        pytest.param(
            "RevokeRequestV1",
            {"last_admission_ordinal": True},
            id="revoke-request-watermark",
        ),
        pytest.param(
            "RevokeFinalizeV1",
            {"erasure_request_id": "not-a-uuid"},
            id="revoke-finalize-id",
        ),
        pytest.param(
            "RevokeFinalizeV1",
            {"consent_epoch_id": "not-a-uuid"},
            id="revoke-finalize-epoch",
        ),
        pytest.param(
            "RevokeFinalizeV1",
            {"control_sequence": True},
            id="revoke-finalize-sequence",
        ),
        pytest.param(
            "RevokeFinalizeV1",
            {"control_fingerprint_hash": "A" * 64},
            id="revoke-finalize-fingerprint",
        ),
        pytest.param(
            "RevokeFinalizeV1",
            {"final_admission_ordinal": -1},
            id="revoke-finalize-watermark",
        ),
        pytest.param(
            "DrainAndStopV1",
            {"owner_generation": -1},
            id="drain-owner",
        ),
        pytest.param(
            "DrainAndStopV1",
            {"final_admission_ordinal": True},
            id="drain-watermark",
        ),
    ],
)
def test_transport_dtos_reject_malformed_fields(
    name: str,
    changes: dict[str, object],
) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        _dto(m, name, **changes)


def test_queue_record_terminal_reservation_accepts_only_turn_terminals() -> None:
    m = _models()
    terminal_kinds = {m.EventKind.TURN_SNAPSHOT, m.EventKind.TURN_SETTLED}
    assert m.EventKind.COMMAND_ROUTED not in terminal_kinds
    for kind in m.EventKind:
        changes = {
            "snapshot": _snapshot(m, kind),
            "reservation_class": m.QueueReservationClass.TERMINAL,
            "lease_open_ordinal": 1,
        }
        if kind in terminal_kinds:
            _dto(m, "QueuedEvidenceRecordV1", **changes)
        else:
            with pytest.raises(m.EvidenceModelError):
                _dto(m, "QueuedEvidenceRecordV1", **changes)

    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "QueuedEvidenceRecordV1",
            snapshot=_snapshot(m, m.EventKind.TURN_SETTLED),
            reservation_class=m.QueueReservationClass.TERMINAL,
            lease_open_ordinal=None,
        )
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "QueuedEvidenceRecordV1", lease_open_ordinal=1)


@pytest.mark.parametrize("kind_name", ["TURN_SNAPSHOT", "TURN_SETTLED"])
def test_queue_record_ordinary_reservation_rejects_turn_terminal_kind(
    kind_name: str,
) -> None:
    m = _models()
    terminal_kind = getattr(m.EventKind, kind_name)

    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "QueuedEvidenceRecordV1",
            snapshot=_snapshot(m, terminal_kind),
            reservation_class=m.QueueReservationClass.ORDINARY,
            lease_open_ordinal=None,
        )


def test_create_epoch_rejects_snapshot_lineage_and_source_contradictions() -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "CreateEpochV1", producer_instance_id=_uuid(999))
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "CreateEpochV1", typed_accepted=False, microphone_accepted=False)
    wrong_open = _snapshot(
        m,
        m.EventKind.SESSION_OPENED,
        sequence=1,
        payload=_payload(m, m.EventKind.SESSION_OPENED, disclosure_digest="b" * 64),
    )
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "CreateEpochV1", session_opened=wrong_open)


@pytest.mark.parametrize("availability_field", ["microphone_available", "typed_available"])
def test_create_epoch_accepted_source_must_be_available(
    availability_field: str,
) -> None:
    m = _models()
    unavailable_binding = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        payload=_payload(
            m,
            m.EventKind.BINDING_OPENED,
            **{availability_field: False},
        ),
    )
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "CreateEpochV1", binding_opened=unavailable_binding)


def test_create_epoch_compound_requires_distinct_opening_event_ids() -> None:
    m = _models()
    session_opened, _ = _create_epoch_snapshots(m)
    duplicate_binding_opened = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        event_id=session_opened.event_id,
    )
    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "CreateEpochV1",
            session_opened=session_opened,
            binding_opened=duplicate_binding_opened,
        )


def test_rollover_requires_exact_four_ordered_cross_linked_snapshots() -> None:
    m = _models()
    valid = _dto_values(m, "RolloverSessionV1")["snapshots"]
    assert isinstance(valid, tuple)
    for hostile in (valid[:3], tuple(reversed(valid)), list(valid)):
        with pytest.raises(m.EvidenceModelError):
            _dto(m, "RolloverSessionV1", snapshots=hostile)
    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "RolloverSessionV1",
            successor_logical_session_id=IDS["logical_session_id"],
        )


@pytest.mark.parametrize("final_sequence", [2, 3])
def test_rollover_predecessor_final_sequence_starts_at_four(
    final_sequence: int,
) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "RolloverSessionV1",
            predecessor_final_event_sequence=final_sequence,
            snapshots=_rollover_snapshots(m, final_sequence=final_sequence),
        )


def test_rollover_compound_requires_four_distinct_event_ids() -> None:
    m = _models()
    repeated = (IDS["event_id"],) * 4
    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "RolloverSessionV1",
            snapshots=_rollover_snapshots(m, event_ids=repeated),
        )


@pytest.mark.parametrize("availability_field", ["microphone_available", "typed_available"])
def test_rollover_successor_accepted_source_must_be_available(
    availability_field: str,
) -> None:
    m = _models()
    snapshots = list(_rollover_snapshots(m))
    snapshots[3] = _snapshot(
        m,
        m.EventKind.BINDING_OPENED,
        sequence=2,
        event_id=snapshots[3].event_id,
        logical_session_id=IDS["successor_logical_session_id"],
        payload=_payload(
            m,
            m.EventKind.BINDING_OPENED,
            **{availability_field: False},
        ),
    )
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "RolloverSessionV1", snapshots=tuple(snapshots))


@pytest.mark.parametrize(
    "reason_name",
    ["CAPACITY_ROLLOVER", "RETENTION_ROLLOVER"],
)
def test_binding_close_authority_forbids_rollover_reasons(reason_name: str) -> None:
    m = _models()
    reason = getattr(m.BindingCloseReason, reason_name)
    with pytest.raises(m.EvidenceModelError):
        _authority(m, "BindingCloseAuthorityV1", close_reason=reason)


@pytest.mark.parametrize(
    "reason_name",
    ["CAPACITY_ROLLOVER", "RETENTION_ROLLOVER"],
)
def test_binding_close_dto_forbids_rollover_reasons(reason_name: str) -> None:
    m = _models()
    reason = getattr(m.BindingCloseReason, reason_name)
    rollover_close = _snapshot(
        m,
        m.EventKind.BINDING_CLOSED,
        sequence=3,
        payload=_payload(m, m.EventKind.BINDING_CLOSED, close_reason=reason.value),
    )
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "BindingCloseV1", snapshot=rollover_close)


def test_seal_binding_close_and_maintenance_cross_field_rules_are_closed() -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "SealEpochV1", close_epoch=False)
    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "SealEpochV1",
            close_reason=m.BindingCloseReason.CAPACITY_ROLLOVER,
        )
    with pytest.raises(m.EvidenceModelError):
        _dto(m, "BindingCloseV1", binding_id=_uuid(500))
    valid_pairs = [
        (m.ErasureReason.TTL, m.ErasureScope.SESSION, IDS["logical_session_id"]),
        (
            m.ErasureReason.REVOKED,
            m.ErasureScope.CONSENT_EPOCH,
            IDS["consent_epoch_id"],
        ),
        (
            m.ErasureReason.UNCLEAN_EPOCH,
            m.ErasureScope.CONSENT_EPOCH,
            IDS["consent_epoch_id"],
        ),
        (m.ErasureReason.CLOCK_ROLLBACK, m.ErasureScope.STORE, "store"),
    ]
    for reason, scope, scope_id in valid_pairs:
        _dto(
            m,
            "MaintenanceV1",
            erasure_reason=reason,
            erasure_scope=scope,
            scope_id=scope_id,
        )
    with pytest.raises(m.EvidenceModelError):
        _dto(
            m,
            "MaintenanceV1",
            erasure_reason=m.ErasureReason.TTL,
            erasure_scope=m.ErasureScope.STORE,
            scope_id="store",
        )


def test_full_purge_accepts_only_two_pending_sentinel_states() -> None:
    m = _models()
    _dto(
        m,
        "FullPurgeV1",
        sentinel_state=m.SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
    )
    for state in (m.SentinelState.CLEAR, m.SentinelState.FIRST_CREATE_PENDING):
        with pytest.raises(m.EvidenceModelError):
            _dto(m, "FullPurgeV1", sentinel_state=state)


def test_authority_dto_kind_pairs_compose_when_lineage_matches() -> None:
    m = _models()
    valid_pairs = [
        (_authority(m, "BindingCloseAuthorityV1"), _dto(m, "BindingCloseV1")),
        (_authority(m, "RolloverAuthorityV1"), _dto(m, "RolloverSessionV1")),
        (_authority(m, "LifecycleSealAuthorityV1"), _dto(m, "SealEpochV1")),
        (_authority(m, "SessionExpiryAuthorityV1"), _dto(m, "ExpireSessionV1")),
        (_authority(m, "ConsentCreateAuthorityV1"), _dto(m, "CreateEpochV1")),
        (_authority(m, "ConsentRevokeAuthorityV1"), _dto(m, "RevokeRequestV1")),
        (_authority(m, "LifecycleDrainAuthorityV1"), _dto(m, "DrainAndStopV1")),
    ]
    for authority, dto in valid_pairs:
        assert m.validate_authority_composition(authority, dto) is None


def test_authority_dto_kind_separation_is_closed() -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        m.validate_authority_composition(
            _authority(m, "BindingCloseAuthorityV1"),
            _dto(m, "SealEpochV1"),
        )


@pytest.mark.parametrize(
    ("authority_name", "dto_name", "authority_changes"),
    [
        pytest.param(
            "BindingCloseAuthorityV1",
            "BindingCloseV1",
            {"consent_epoch_id": _uuid(901)},
            id="binding-close",
        ),
        pytest.param(
            "RolloverAuthorityV1",
            "RolloverSessionV1",
            {"successor_logical_session_id": _uuid(902)},
            id="rollover",
        ),
        pytest.param(
            "LifecycleSealAuthorityV1",
            "SealEpochV1",
            {"final_event_sequence": 5},
            id="seal",
        ),
        pytest.param(
            "SessionExpiryAuthorityV1",
            "ExpireSessionV1",
            {"owner_generation": 2},
            id="expiry",
        ),
        pytest.param(
            "ConsentCreateAuthorityV1",
            "CreateEpochV1",
            {"retention_hours": 25},
            id="create",
        ),
        pytest.param(
            "ConsentRevokeAuthorityV1",
            "RevokeRequestV1",
            {"control_sequence": 2},
            id="revoke",
        ),
        pytest.param(
            "LifecycleDrainAuthorityV1",
            "DrainAndStopV1",
            {"owner_generation": 2},
            id="drain",
        ),
    ],
)
def test_each_authority_dto_pair_rejects_its_own_lineage_mismatch(
    authority_name: str,
    dto_name: str,
    authority_changes: dict[str, object],
) -> None:
    m = _models()
    with pytest.raises(m.EvidenceModelError):
        m.validate_authority_composition(
            _authority(m, authority_name, **authority_changes),
            _dto(m, dto_name),
        )


def test_authority_kind_separation_prevents_text_or_input_lineage_on_proactive_replay() -> None:
    m = _models()
    for name in ("ProactiveTurnAuthorityV1", "ReplayTurnAuthorityV1"):
        field_names = {field.name for field in dataclasses.fields(getattr(m, name))}
        assert "utterance_id" not in field_names
        assert "source" not in field_names
        assert "text" not in field_names
        assert "typed_sequence" not in field_names
    command_fields = {field.name for field in dataclasses.fields(m.CommandAdmissionAuthorityV1)}
    assert "command_name" not in command_fields
    assert "arguments" not in command_fields
    assert "ticket" not in command_fields
    assert "result" not in command_fields


CONTROL_DTO_FIELDS = {
    "EvidenceConsentSourcesV1": ["microphone", "typed"],
    "EvidenceConsentRequestV1": [
        "sequence",
        "accepted",
        "consent_version",
        "disclosure_digest",
        "retention_hours",
        "sources",
    ],
    "EvidenceRevokeRequestV1": ["sequence"],
    "ProjectionResyncRequestV1": ["attempt", "reason", "request_id"],
    "CaptureStatusV1": [
        "available",
        "capture_state",
        "retention_hours",
        "consent_version",
        "disclosure_digest",
    ],
}


def _consent_sources(m: ModuleType, **changes: object) -> Any:
    values = {"microphone": True, "typed": True}
    values.update(changes)
    return m.EvidenceConsentSourcesV1(**values)


def _consent_request(m: ModuleType, **changes: object) -> Any:
    values = {
        "sequence": 1,
        "accepted": True,
        "consent_version": CONSENT_LITERAL,
        "disclosure_digest": HASH,
        "retention_hours": 24,
        "sources": _consent_sources(m),
    }
    values.update(changes)
    return m.EvidenceConsentRequestV1(**values)


def _capture_status(m: ModuleType, **changes: object) -> Any:
    values = {
        "available": True,
        "capture_state": m.CaptureState.IDLE,
        "retention_hours": 24,
        "consent_version": CONSENT_LITERAL,
        "disclosure_digest": HASH,
    }
    values.update(changes)
    return m.CaptureStatusV1(**values)


def _json_document(value: dict[str, object]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _consent_document(**changes: object) -> bytes:
    values: dict[str, object] = {
        "sequence": 1,
        "accepted": True,
        "consentVersion": CONSENT_LITERAL,
        "disclosureDigest": HASH,
        "retentionHours": 24,
        "sources": {"microphone": True, "typed": True},
    }
    values.update(changes)
    return _json_document(values)


def _resync_document(**changes: object) -> bytes:
    values: dict[str, object] = {
        "attempt": 1,
        "reason": "invalid_capture_status",
        "requestId": _uuid(600),
    }
    values.update(changes)
    return _json_document(values)


def test_public_control_dtos_have_exact_frozen_slotted_shapes() -> None:
    m = _models()
    instances = {
        "EvidenceConsentSourcesV1": _consent_sources(m),
        "EvidenceConsentRequestV1": _consent_request(m),
        "EvidenceRevokeRequestV1": m.EvidenceRevokeRequestV1(sequence=1),
        "ProjectionResyncRequestV1": m.ProjectionResyncRequestV1(
            attempt=1,
            reason="invalid_capture_status",
            request_id=_uuid(600),
        ),
        "CaptureStatusV1": _capture_status(m),
    }
    for name, instance in instances.items():
        assert [field.name for field in dataclasses.fields(instance)] == CONTROL_DTO_FIELDS[name]
        assert instance.__dataclass_params__.frozen is True
        assert not hasattr(instance, "__dict__")

    with pytest.raises(TypeError):
        m.EvidenceRevokeRequestV1(sequence=1, future=True)
    with pytest.raises(TypeError):
        _consent_request(m, future=True)
    with pytest.raises(TypeError):
        _capture_status(m, transcript="forbidden")


def test_evidence_consent_request_constructor_requires_explicit_accepted() -> None:
    m = _models()

    with pytest.raises(TypeError):
        m.EvidenceConsentRequestV1(
            sequence=1,
            consent_version=CONSENT_LITERAL,
            disclosure_digest=HASH,
            retention_hours=24,
            sources=_consent_sources(m),
        )


def test_projection_resync_request_constructor_requires_explicit_attempt() -> None:
    m = _models()

    with pytest.raises(TypeError):
        m.ProjectionResyncRequestV1(
            reason="invalid_capture_status",
            request_id=_uuid(600),
        )


def test_projection_resync_request_constructor_requires_explicit_reason() -> None:
    m = _models()

    with pytest.raises(TypeError):
        m.ProjectionResyncRequestV1(
            attempt=1,
            request_id=_uuid(600),
        )


def test_public_control_dto_bounds_literals_and_cross_fields_are_closed() -> None:
    m = _models()
    for sequence in (0, 2**53, True):
        with pytest.raises(m.EvidenceModelError):
            _consent_request(m, sequence=sequence)
        with pytest.raises(m.EvidenceModelError):
            m.EvidenceRevokeRequestV1(sequence=sequence)

    for retention in (0, 169, True):
        with pytest.raises(m.EvidenceModelError):
            _consent_request(m, retention_hours=retention)
        with pytest.raises(m.EvidenceModelError):
            _capture_status(m, retention_hours=retention)

    for accepted in (False, 1, "true"):
        with pytest.raises(m.EvidenceModelError):
            _consent_request(m, accepted=accepted)
    with pytest.raises(m.EvidenceModelError):
        _consent_request(m, consent_version="future-consent")
    with pytest.raises(m.EvidenceModelError):
        _consent_request(m, disclosure_digest="A" * 64)
    with pytest.raises(m.EvidenceModelError):
        _consent_request(m, sources=_consent_sources(m, microphone=False, typed=False))
    with pytest.raises(m.EvidenceModelError):
        _consent_request(m, sources=_consent_sources(m, typed=1))

    for attempt in (0, 2, True):
        with pytest.raises(m.EvidenceModelError):
            m.ProjectionResyncRequestV1(
                attempt=attempt,
                reason="invalid_capture_status",
                request_id=_uuid(600),
            )
    with pytest.raises(m.EvidenceModelError):
        m.ProjectionResyncRequestV1(
            attempt=1,
            reason="future_reason",
            request_id=_uuid(600),
        )
    with pytest.raises(m.EvidenceModelError):
        m.ProjectionResyncRequestV1(
            attempt=1,
            reason="invalid_capture_status",
            request_id=_uuid(0xABCDEF).upper(),
        )

    with pytest.raises(m.EvidenceModelError):
        _capture_status(m, available=False, capture_state=m.CaptureState.IDLE)
    with pytest.raises(m.EvidenceModelError):
        _capture_status(m, available=True, capture_state=m.CaptureState.UNAVAILABLE)
    unavailable = _capture_status(
        m,
        available=False,
        capture_state=m.CaptureState.UNAVAILABLE,
    )
    assert unavailable.available is False


def test_public_control_parsers_accept_only_exact_bounded_key_sets() -> None:
    m = _models()
    consent = m.parse_evidence_consent_request(_consent_document())
    revoke = m.parse_evidence_revoke_request(b'{"sequence":1}')
    resync = m.parse_projection_resync_request(_resync_document())
    assert consent.sequence == 1
    assert revoke.sequence == 1
    assert resync.request_id == _uuid(600)

    hostile_documents = [
        (m.parse_evidence_consent_request, _consent_document(future=True)),
        (
            m.parse_evidence_consent_request,
            _consent_document(
                sources={"microphone": True, "typed": True, "future": False}
            ),
        ),
        (m.parse_evidence_revoke_request, b'{"sequence":1,"future":true}'),
        (m.parse_projection_resync_request, _resync_document(future=True)),
    ]
    for parser, document in hostile_documents:
        with pytest.raises(m.EvidenceModelError):
            parser(document)

    duplicate_consent = _consent_document().replace(
        b'{"sequence":1,',
        b'{"sequence":1,"sequence":2,',
        1,
    )
    with pytest.raises(m.EvidenceModelError, match="duplicate"):
        m.parse_evidence_consent_request(duplicate_consent)

    for parser, oversized in (
        (m.parse_evidence_consent_request, b" " * 513),
        (m.parse_evidence_revoke_request, b" " * 65),
        (m.parse_projection_resync_request, b" " * 161),
    ):
        with pytest.raises(m.EvidenceModelError):
            parser(oversized)


@pytest.mark.parametrize(
    "parser_name",
    [
        "parse_strict_json_object",
        "parse_event_payload_json",
        "parse_evidence_snapshot_json",
        "parse_evidence_consent_request",
        "parse_evidence_consent_request_json",
        "parse_evidence_revoke_request",
        "parse_evidence_revoke_request_json",
        "parse_projection_resync_request",
        "parse_projection_resync_request_json",
    ],
)
def test_every_strict_json_parser_requires_exact_bytes(parser_name: str) -> None:
    m = _models()
    command_payload = _json_document(_payload(m, m.EventKind.COMMAND_ROUTED))
    snapshot_document = _json_document(
        {
            "schema_version": 1,
            "installation_id": IDS["installation_id"],
            "producer_instance_id": IDS["producer_instance_id"],
            "logical_session_id": IDS["logical_session_id"],
            "event_id": IDS["event_id"],
            "event_sequence": 1,
            "event_kind": "command_routed",
            "payload": _payload(m, m.EventKind.COMMAND_ROUTED),
        }
    )
    documents = {
        "parse_strict_json_object": b'{"value":1}',
        "parse_event_payload_json": command_payload,
        "parse_evidence_snapshot_json": snapshot_document,
        "parse_evidence_consent_request": _consent_document(),
        "parse_evidence_consent_request_json": _consent_document(),
        "parse_evidence_revoke_request": b'{"sequence":1}',
        "parse_evidence_revoke_request_json": b'{"sequence":1}',
        "parse_projection_resync_request": _resync_document(),
        "parse_projection_resync_request_json": _resync_document(),
    }
    parser = getattr(m, parser_name)
    document = documents[parser_name]
    with pytest.raises(m.EvidenceModelError):
        if parser_name == "parse_event_payload_json":
            parser(m.EventKind.COMMAND_ROUTED, document.decode("utf-8"))
        else:
            parser(document.decode("utf-8"))


def test_exact_bytes_subclasses_are_rejected_without_running_hostile_methods() -> None:
    m = _models()
    calls = 0

    class HostileBytes(bytes):
        def decode(self, *args: object, **kwargs: object) -> str:
            nonlocal calls
            calls += 1
            raise AssertionError("hostile bytes decoder executed")

    document = HostileBytes(b'{"value":1}')
    with pytest.raises(m.EvidenceModelError):
        m.parse_strict_json_object(document)
    with pytest.raises(m.EvidenceModelError):
        m.parse_event_payload_json(m.EventKind.COMMAND_ROUTED, document)
    with pytest.raises(m.EvidenceModelError):
        m.parse_evidence_snapshot_json(document)
    with pytest.raises(m.EvidenceModelError):
        m.parse_evidence_consent_request(document)
    with pytest.raises(m.EvidenceModelError):
        m.parse_evidence_revoke_request(document)
    with pytest.raises(m.EvidenceModelError):
        m.parse_projection_resync_request(document)
    assert calls == 0


def test_capture_status_public_shape_and_primitive_keys_are_exact() -> None:
    m = _models()
    status = _capture_status(m, capture_state=m.CaptureState.ACTIVE)
    assert m.capture_status_to_primitive(status) == {
        "available": True,
        "captureState": "active",
        "retentionHours": 24,
        "consentVersion": "realtime-evidence-consent-v1",
        "disclosureDigest": HASH,
    }

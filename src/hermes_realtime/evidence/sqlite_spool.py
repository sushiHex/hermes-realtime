"""Single-connection SQLite implementation of the private V1 writer transport.

It intentionally contains no conversation imports: foreground code only queues
validated models, while this module serializes durable mutations. This module
is the only place in the package that may import :mod:`sqlite3`.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import sqlite3
import struct
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, TypeVar, cast
from uuid import UUID, uuid4

from .models import (
    CONSENT_VERSION,
    BindingCloseReason,
    BindingCloseV1,
    CaptureState,
    ConflictReason,
    CreateEpochV1,
    DrainAndStopV1,
    DrainDisposition,
    EventKind,
    EvidenceDiagnosticsV1,
    EvidenceModelError,
    EvidenceSnapshotV1,
    ExpireSessionV1,
    FullPurgeV1,
    MaintenanceV1,
    OwnerState,
    PurgeDisposition,
    QueuedEvidenceRecordV1,
    RecoveryDisposition,
    RevokeDisposition,
    RevokeFinalizeV1,
    RevokeRequestV1,
    RolloverSessionV1,
    SealEpochV1,
    SentinelState,
    StoreDisposition,
    TaintCode,
    classify_event_retry,
    evidence_snapshot_to_primitive,
    parse_evidence_snapshot_json,
    parse_strict_json_object,
    validate_canonical_utc,
    validate_canonical_uuid4,
    validate_event_sequence,
    validate_sha256_hex,
)
from .storage_security import (
    MANIFEST_V1,
    SENTINEL_SLOT_OFFSETS,
    SENTINEL_SLOT_SIZE,
    ActivationTemporaryV1,
    EvidenceArtifactManifestV1,
    EvidenceRootHandleV1,
    EvidenceStorageError,
    EvidenceStorageProbeV1,
    RepositoryBoundaryV1,
    ResolvedEvidenceManifestV1,
    SentinelHandleV1,
    ValidatedEvidenceRootV1,
    WindowsEvidenceRootHandleV1,
    WindowsStorageProbeV1,
    WriterFault,
    audit_root_occupancy,
    check_maintenance_headroom,
    create_activation_temporary,
    decode_sentinel_image,
    encode_root_marker,
    initial_sentinel_image,
    next_sentinel_slot,
    parse_root_marker,
    resolve_for_containment,
    validate_evidence_database_path,
)

_ResultT = TypeVar("_ResultT")

HRE1_MAGIC = b"HRE1"
HRE1_SCHEMA_VERSION = 1
MAX_EVENT_SEQUENCE = 9_216
MAX_CANONICAL_PAYLOAD_BYTES = 32 * 1024
_MAX_JS_INTEGER = 9_007_199_254_740_991
_MAX_SQLITE_INTEGER = 2**63 - 1
_ZERO_OFFSET = timedelta(0)


class EvidenceFrameError(ValueError):
    """A noncanonical canonical-JSON value or HRE1 frame field."""


def _store_corrupt(message: str) -> EvidenceStorageError:
    return EvidenceStorageError(WriterFault.STORE_CORRUPT, message)


def _require_canonical_json_value(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise EvidenceFrameError("canonical JSON object keys must be exact strings")
            _require_canonical_json_value(item)
        return
    if type(value) is list:
        for item in value:
            _require_canonical_json_value(item)
        return
    if type(value) not in (str, int, bool, type(None)):
        raise EvidenceFrameError("canonical JSON admits no float or foreign type")


def canonical_json_bytes(value: object) -> bytes:
    """Encode strict UTF-8 canonical JSON: sorted keys, ``(',', ':')``, finite integers."""

    _require_canonical_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def format_canonical_utc(moment: datetime) -> str:
    """Render the exact ``YYYY-MM-DDTHH:MM:SS.ffffffZ`` spelling."""

    if type(moment) is not datetime or moment.utcoffset() != _ZERO_OFFSET:
        raise EvidenceFrameError("canonical evidence time must be an aware UTC datetime")
    return f"{moment:%Y-%m-%dT%H:%M:%S}.{moment.microsecond:06d}Z"


def _frame_field(validate: object, value: object, name: str) -> str:
    assert callable(validate)
    try:
        return str(validate(value, field_name=name))
    except EvidenceModelError as exc:
        # Static by policy: the chained model error names the field for the owner log.
        raise EvidenceFrameError("an HRE1 frame field is not its canonical spelling") from exc


def hre1_record_frame(
    *,
    installation_id: str,
    producer_instance_id: str,
    event_id: str,
    logical_session_id: str,
    event_sequence: int,
    event_kind: str,
    recorded_at_utc: str,
    payload_hash: str,
    previous_hash: str | None,
) -> bytes:
    """Build the exact §5.3 record frame; no other field or padding exists."""

    identifiers = (
        _frame_field(validate_canonical_uuid4, installation_id, "installation_id"),
        _frame_field(validate_canonical_uuid4, producer_instance_id, "producer_instance_id"),
        _frame_field(validate_canonical_uuid4, event_id, "event_id"),
        _frame_field(validate_canonical_uuid4, logical_session_id, "logical_session_id"),
    )
    if type(event_sequence) is not int or not 1 <= event_sequence <= MAX_EVENT_SEQUENCE:
        raise EvidenceFrameError("event_sequence is outside its closed 1..9216 range")
    if type(event_kind) is not str or event_kind not in tuple(EventKind):
        raise EvidenceFrameError("event_kind is not a closed V1 event kind")
    moment = _frame_field(validate_canonical_utc, recorded_at_utc, "recorded_at_utc")
    digest = _frame_field(validate_sha256_hex, payload_hash, "payload_hash")
    kind_bytes = event_kind.encode("utf-8")
    moment_bytes = moment.encode("utf-8")
    frame = (
        HRE1_MAGIC
        + struct.pack(">I", HRE1_SCHEMA_VERSION)
        + b"".join(UUID(item).bytes for item in identifiers)
        + struct.pack(">Q", event_sequence)
        + struct.pack(">I", len(kind_bytes))
        + kind_bytes
        + struct.pack(">I", len(moment_bytes))
        + moment_bytes
        + bytes.fromhex(digest)
    )
    if previous_hash is None:
        return frame + b"\x00"
    previous = _frame_field(validate_sha256_hex, previous_hash, "previous_hash")
    return frame + b"\x01" + bytes.fromhex(previous)


def hre1_record_hash(
    *,
    installation_id: str,
    producer_instance_id: str,
    event_id: str,
    logical_session_id: str,
    event_sequence: int,
    event_kind: str,
    recorded_at_utc: str,
    payload_hash: str,
    previous_hash: str | None,
) -> str:
    """``SHA256`` of the exact frame bytes."""

    return hashlib.sha256(
        hre1_record_frame(
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            event_id=event_id,
            logical_session_id=logical_session_id,
            event_sequence=event_sequence,
            event_kind=event_kind,
            recorded_at_utc=recorded_at_utc,
            payload_hash=payload_hash,
            previous_hash=previous_hash,
        )
    ).hexdigest()


APPLICATION_ID = 0x48524531
USER_VERSION = 1
MAX_MAIN_DB_PAGE_COUNT = 24_576
MAIN_DB_PAGE_SIZE = 4_096
MAX_MAIN_DB_BYTES = MAX_MAIN_DB_PAGE_COUNT * MAIN_DB_PAGE_SIZE

REQUIRED_PRAGMAS: tuple[tuple[str, object], ...] = (
    ("journal_mode", "delete"),
    ("synchronous", 3),
    ("encoding", "UTF-8"),
    ("page_size", MAIN_DB_PAGE_SIZE),
    ("max_page_count", MAX_MAIN_DB_PAGE_COUNT),
    ("foreign_keys", 1),
    ("busy_timeout", 0),
    ("trusted_schema", 0),
    ("secure_delete", 1),
    ("temp_store", 2),
)

# Reusable named-check fragments; whitespace is normalized away before hashing.
_UUID4 = (
    "GLOB '????????-????-4???-[89ab]???-????????????' AND length({0})=36 "
    "AND {0} NOT GLOB '*[^0-9a-f-]*'"
)
_SHA256 = "length({0})=64 AND {0} NOT GLOB '*[^0-9a-f]*'"
_UTC = (
    "length({0})=27 AND {0} GLOB "
    "'[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T[0-9][0-9]:[0-9][0-9]:[0-9][0-9]."
    "[0-9][0-9][0-9][0-9][0-9][0-9]Z'"
)

SCHEMA_DDL_V1 = """
CREATE TABLE producer_installation (
    singleton INTEGER PRIMARY KEY
        CONSTRAINT installation_is_singleton CHECK (singleton = 1),
    installation_id TEXT NOT NULL UNIQUE
        CONSTRAINT installation_id_is_canonical CHECK (
            installation_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(installation_id) = 36
            AND installation_id NOT GLOB '*[^0-9a-f-]*'),
    created_at_utc TEXT NOT NULL
        CONSTRAINT installation_created_at_is_canonical CHECK (
            length(created_at_utc) = 27
            AND created_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    clock_high_water_utc TEXT NOT NULL
        CONSTRAINT installation_high_water_is_canonical CHECK (
            length(clock_high_water_utc) = 27
            AND clock_high_water_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    purge_required INTEGER NOT NULL DEFAULT 0
        CONSTRAINT installation_purge_required_is_boolean CHECK (purge_required IN (0, 1)),
    purge_reason TEXT,
    purge_scope TEXT,
    CONSTRAINT installation_purge_is_closed CHECK (
        (purge_required = 0 AND purge_reason IS NULL AND purge_scope IS NULL)
        OR (purge_required = 1 AND purge_reason = 'clock_rollback' AND purge_scope = 'store'))
) STRICT;

CREATE TABLE consent_epochs (
    consent_epoch_id TEXT PRIMARY KEY
        CONSTRAINT epoch_id_is_canonical CHECK (
            consent_epoch_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(consent_epoch_id) = 36
            AND consent_epoch_id NOT GLOB '*[^0-9a-f-]*'),
    producer_instance_id TEXT NOT NULL
        CONSTRAINT epoch_producer_is_canonical CHECK (
            producer_instance_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(producer_instance_id) = 36
            AND producer_instance_id NOT GLOB '*[^0-9a-f-]*'),
    state TEXT NOT NULL
        CONSTRAINT epoch_state_is_closed CHECK (state IN ('active', 'closed', 'revoked')),
    opened_at_utc TEXT NOT NULL
        CONSTRAINT epoch_opened_at_is_canonical CHECK (
            length(opened_at_utc) = 27
            AND opened_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    closed_at_utc TEXT
        CONSTRAINT epoch_closed_at_is_canonical CHECK (
            closed_at_utc IS NULL
            OR (length(closed_at_utc) = 27
                AND closed_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                    || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z')),
    CONSTRAINT epoch_close_time_matches_state CHECK (
        (state = 'active' AND closed_at_utc IS NULL)
        OR (state IN ('closed', 'revoked') AND closed_at_utc IS NOT NULL))
) STRICT;

CREATE TABLE evidence_sessions (
    logical_session_id TEXT PRIMARY KEY
        CONSTRAINT session_id_is_canonical CHECK (
            logical_session_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(logical_session_id) = 36
            AND logical_session_id NOT GLOB '*[^0-9a-f-]*'),
    consent_epoch_id TEXT NOT NULL REFERENCES consent_epochs(consent_epoch_id),
    producer_instance_id TEXT NOT NULL
        CONSTRAINT session_producer_is_canonical CHECK (
            producer_instance_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(producer_instance_id) = 36
            AND producer_instance_id NOT GLOB '*[^0-9a-f-]*'),
    opened_at_utc TEXT NOT NULL
        CONSTRAINT session_opened_at_is_canonical CHECK (
            length(opened_at_utc) = 27
            AND opened_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    expires_at_utc TEXT NOT NULL
        CONSTRAINT session_expires_at_is_canonical CHECK (
            length(expires_at_utc) = 27
            AND expires_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    consent_version TEXT NOT NULL
        CONSTRAINT session_consent_version_is_exact CHECK (
            consent_version = 'realtime-evidence-consent-v1'),
    state TEXT NOT NULL
        CONSTRAINT session_state_is_closed CHECK (state IN ('open', 'sealed', 'tainted')),
    final_event_sequence INTEGER
        CONSTRAINT session_final_sequence_in_range CHECK (
            final_event_sequence IS NULL OR final_event_sequence BETWEEN 1 AND 9216),
    head_hash TEXT
        CONSTRAINT session_head_hash_is_canonical CHECK (
            head_hash IS NULL
            OR (length(head_hash) = 64 AND head_hash NOT GLOB '*[^0-9a-f]*')),
    canonical_bytes INTEGER NOT NULL DEFAULT 0
        CONSTRAINT session_canonical_bytes_in_range CHECK (
            canonical_bytes BETWEEN 0 AND 41943040),
    event_count INTEGER NOT NULL DEFAULT 0
        CONSTRAINT session_event_count_in_range CHECK (event_count BETWEEN 0 AND 9216),
    taint_code TEXT
        CONSTRAINT session_taint_code_is_closed CHECK (
            taint_code IS NULL OR taint_code IN (
                'admission_gap', 'oversize', 'deny_filter', 'writer_fault', 'quota_exceeded',
                'event_id_conflict', 'sequence_conflict', 'lineage_invalid', 'terminal_missing',
                'terminal_conflict', 'spawn_failed', 'lease_capacity_exhausted', 'clock_rollback',
                'ttl_expired', 'stale_open_recovery', 'seal_validation_failed', 'purge_failed',
                'corrupt_store', 'shutdown_incomplete')),
    CONSTRAINT session_final_sequence_matches_event_count CHECK (
        final_event_sequence IS NULL OR final_event_sequence = event_count),
    CONSTRAINT session_state_is_consistent CHECK (
        (state = 'open' AND final_event_sequence IS NULL AND head_hash IS NULL
            AND taint_code IS NULL)
        OR (state = 'sealed' AND final_event_sequence IS NOT NULL AND head_hash IS NOT NULL
            AND taint_code IS NULL)
        OR (state = 'tainted' AND taint_code IS NOT NULL
            AND ((final_event_sequence IS NULL AND head_hash IS NULL)
                OR (final_event_sequence IS NOT NULL AND head_hash IS NOT NULL))))
) STRICT;

CREATE TABLE evidence_events (
    event_id TEXT PRIMARY KEY
        CONSTRAINT event_id_is_canonical CHECK (
            event_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(event_id) = 36
            AND event_id NOT GLOB '*[^0-9a-f-]*'),
    logical_session_id TEXT NOT NULL
        REFERENCES evidence_sessions(logical_session_id) ON DELETE CASCADE,
    event_sequence INTEGER NOT NULL
        CONSTRAINT event_sequence_in_range CHECK (event_sequence BETWEEN 1 AND 9216),
    event_kind TEXT NOT NULL
        CONSTRAINT event_kind_is_closed CHECK (event_kind IN (
            'session_opened', 'binding_opened', 'turn_opened', 'user_final_accepted',
            'command_routed', 'assistant_segment_generated',
            'assistant_chunk_transport_confirmed_full', 'turn_snapshot', 'turn_settled',
            'binding_closed', 'session_seal_requested', 'session_tainted')),
    recorded_at_utc TEXT NOT NULL
        CONSTRAINT event_recorded_at_is_canonical CHECK (
            length(recorded_at_utc) = 27
            AND recorded_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    canonical_payload TEXT NOT NULL,
    payload_hash TEXT NOT NULL
        CONSTRAINT event_payload_hash_is_canonical CHECK (
            length(payload_hash) = 64 AND payload_hash NOT GLOB '*[^0-9a-f]*'),
    previous_hash TEXT
        CONSTRAINT event_previous_hash_is_canonical CHECK (
            previous_hash IS NULL
            OR (length(previous_hash) = 64 AND previous_hash NOT GLOB '*[^0-9a-f]*')),
    record_hash TEXT NOT NULL
        CONSTRAINT event_record_hash_is_canonical CHECK (
            length(record_hash) = 64 AND record_hash NOT GLOB '*[^0-9a-f]*'),
    canonical_bytes INTEGER NOT NULL
        CONSTRAINT event_canonical_bytes_is_exact CHECK (
            canonical_bytes = length(CAST(canonical_payload AS BLOB))
            AND canonical_bytes <= 32768),
    CONSTRAINT event_first_record_opens_the_chain CHECK (
        (event_sequence = 1 AND previous_hash IS NULL)
        OR (event_sequence > 1 AND previous_hash IS NOT NULL)),
    UNIQUE (logical_session_id, event_sequence)
) STRICT;

CREATE TRIGGER evidence_events_are_append_only BEFORE UPDATE ON evidence_events BEGIN
    SELECT RAISE(ABORT, 'evidence_events is append-only');
END;

CREATE TABLE evidence_conflicts (
    conflict_id TEXT PRIMARY KEY
        CONSTRAINT conflict_id_is_canonical CHECK (
            conflict_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(conflict_id) = 36
            AND conflict_id NOT GLOB '*[^0-9a-f-]*'),
    logical_session_id TEXT NOT NULL
        REFERENCES evidence_sessions(logical_session_id) ON DELETE CASCADE,
    claimed_event_id TEXT NOT NULL
        CONSTRAINT conflict_claimed_event_is_canonical CHECK (
            claimed_event_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(claimed_event_id) = 36
            AND claimed_event_id NOT GLOB '*[^0-9a-f-]*'),
    reason_code TEXT NOT NULL
        CONSTRAINT conflict_reason_is_closed CHECK (reason_code IN (
            'event_id_envelope_mismatch', 'cross_session_event_id',
            'session_sequence_claimed', 'sealed_session_reuse')),
    recorded_at_utc TEXT NOT NULL
        CONSTRAINT conflict_recorded_at_is_canonical CHECK (
            length(recorded_at_utc) = 27
            AND recorded_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    UNIQUE (logical_session_id, claimed_event_id)
) STRICT;

CREATE TABLE erasure_requests (
    erasure_request_id TEXT PRIMARY KEY
        CONSTRAINT request_id_is_canonical CHECK (
            erasure_request_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(erasure_request_id) = 36
            AND erasure_request_id NOT GLOB '*[^0-9a-f-]*'),
    scope_kind TEXT NOT NULL
        CONSTRAINT request_scope_kind_is_closed CHECK (
            scope_kind IN ('session', 'consent_epoch', 'store')),
    scope_key TEXT NOT NULL,
    requested_at_utc TEXT NOT NULL
        CONSTRAINT request_requested_at_is_canonical CHECK (
            length(requested_at_utc) = 27
            AND requested_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    reason_code TEXT NOT NULL
        CONSTRAINT request_reason_is_closed CHECK (
            reason_code IN ('revoked', 'ttl', 'clock_rollback', 'unclean_epoch')),
    state TEXT NOT NULL
        CONSTRAINT request_state_is_closed CHECK (
            state IN ('pending', 'logical_deleted', 'purge_failed')),
    control_sequence INTEGER
        CONSTRAINT request_control_sequence_in_range CHECK (
            control_sequence IS NULL OR control_sequence BETWEEN 1 AND 9007199254740991),
    control_fingerprint_hash TEXT
        CONSTRAINT request_control_fingerprint_is_canonical CHECK (
            control_fingerprint_hash IS NULL
            OR (length(control_fingerprint_hash) = 64
                AND control_fingerprint_hash NOT GLOB '*[^0-9a-f]*')),
    ttl_consent_epoch_id TEXT
        CONSTRAINT request_ttl_epoch_is_canonical CHECK (
            ttl_consent_epoch_id IS NULL
            OR (ttl_consent_epoch_id GLOB '????????-????-4???-[89ab]???-????????????'
                AND length(ttl_consent_epoch_id) = 36
                AND ttl_consent_epoch_id NOT GLOB '*[^0-9a-f-]*')),
    ttl_expires_at_utc TEXT
        CONSTRAINT request_ttl_expiry_is_canonical CHECK (
            ttl_expires_at_utc IS NULL
            OR (length(ttl_expires_at_utc) = 27
                AND ttl_expires_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                    || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z')),
    last_admission_ordinal INTEGER NOT NULL
        CONSTRAINT request_last_admission_ordinal_in_range CHECK (
            last_admission_ordinal BETWEEN 1 AND 9223372036854775807),
    final_admission_ordinal INTEGER
        CONSTRAINT request_final_admission_ordinal_in_range CHECK (
            final_admission_ordinal IS NULL
            OR final_admission_ordinal BETWEEN 1 AND 9223372036854775807),
    resume_state TEXT
        CONSTRAINT request_resume_state_is_closed CHECK (
            resume_state IS NULL OR resume_state IN ('pending', 'logical_deleted')),
    erased_session_count INTEGER,
    erased_event_count INTEGER,
    UNIQUE (scope_kind, scope_key, reason_code),
    CONSTRAINT request_state_is_consistent CHECK (
        (state = 'pending' AND resume_state IS NULL
            AND erased_session_count IS NULL AND erased_event_count IS NULL)
        OR (state = 'logical_deleted' AND resume_state IS NULL
            AND erased_session_count IS NOT NULL AND erased_event_count IS NOT NULL
            AND erased_session_count >= 0 AND erased_event_count >= 0)
        OR (state = 'purge_failed' AND resume_state IS NOT NULL
            AND ((resume_state = 'pending' AND erased_session_count IS NULL
                    AND erased_event_count IS NULL)
                OR (resume_state = 'logical_deleted' AND erased_session_count IS NOT NULL
                    AND erased_event_count IS NOT NULL
                    AND erased_session_count >= 0 AND erased_event_count >= 0)))),
    CONSTRAINT request_reason_matches_scope CHECK (
        (reason_code = 'ttl' AND scope_kind = 'session' AND scope_key <> 'store')
        OR (reason_code IN ('revoked', 'unclean_epoch') AND scope_kind = 'consent_epoch'
            AND scope_key <> 'store')
        OR (reason_code = 'clock_rollback' AND scope_kind = 'store' AND scope_key = 'store')),
    CONSTRAINT request_revoke_authority_is_conditional CHECK (
        (reason_code = 'revoked'
            AND control_sequence IS NOT NULL
            AND control_fingerprint_hash IS NOT NULL
            AND ttl_consent_epoch_id IS NULL
            AND ttl_expires_at_utc IS NULL
            AND last_admission_ordinal IS NOT NULL
            AND (final_admission_ordinal IS NULL
                OR final_admission_ordinal >= last_admission_ordinal))
        OR (reason_code = 'ttl'
            AND control_sequence IS NULL
            AND control_fingerprint_hash IS NOT NULL
            AND ttl_consent_epoch_id IS NOT NULL
            AND ttl_expires_at_utc IS NOT NULL
            AND final_admission_ordinal IS NULL)
        OR (reason_code IN ('unclean_epoch', 'clock_rollback')
            AND control_sequence IS NULL
            AND control_fingerprint_hash IS NULL
            AND ttl_consent_epoch_id IS NULL
            AND ttl_expires_at_utc IS NULL
            AND final_admission_ordinal IS NULL))
) STRICT;

CREATE TABLE erasure_tombstones (
    erasure_request_id TEXT PRIMARY KEY
        CONSTRAINT tombstone_id_is_canonical CHECK (
            erasure_request_id GLOB '????????-????-4???-[89ab]???-????????????'
            AND length(erasure_request_id) = 36
            AND erasure_request_id NOT GLOB '*[^0-9a-f-]*'),
    scope_kind TEXT NOT NULL
        CONSTRAINT tombstone_scope_kind_is_closed CHECK (
            scope_kind IN ('session', 'consent_epoch', 'store')),
    scope_key TEXT NOT NULL,
    reason_code TEXT NOT NULL
        CONSTRAINT tombstone_reason_is_closed CHECK (
            reason_code IN ('revoked', 'ttl', 'clock_rollback', 'unclean_epoch')),
    control_sequence INTEGER
        CONSTRAINT tombstone_control_sequence_in_range CHECK (
            control_sequence IS NULL OR control_sequence BETWEEN 1 AND 9007199254740991),
    control_fingerprint_hash TEXT
        CONSTRAINT tombstone_control_fingerprint_is_canonical CHECK (
            control_fingerprint_hash IS NULL
            OR (length(control_fingerprint_hash) = 64
                AND control_fingerprint_hash NOT GLOB '*[^0-9a-f]*')),
    ttl_consent_epoch_id TEXT
        CONSTRAINT tombstone_ttl_epoch_is_canonical CHECK (
            ttl_consent_epoch_id IS NULL
            OR (ttl_consent_epoch_id GLOB '????????-????-4???-[89ab]???-????????????'
                AND length(ttl_consent_epoch_id) = 36
                AND ttl_consent_epoch_id NOT GLOB '*[^0-9a-f-]*')),
    ttl_expires_at_utc TEXT
        CONSTRAINT tombstone_ttl_expiry_is_canonical CHECK (
            ttl_expires_at_utc IS NULL
            OR (length(ttl_expires_at_utc) = 27
                AND ttl_expires_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                    || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z')),
    last_admission_ordinal INTEGER NOT NULL
        CONSTRAINT tombstone_last_admission_ordinal_in_range CHECK (
            last_admission_ordinal BETWEEN 1 AND 9223372036854775807),
    final_admission_ordinal INTEGER
        CONSTRAINT tombstone_final_admission_ordinal_in_range CHECK (
            final_admission_ordinal IS NULL
            OR final_admission_ordinal BETWEEN 1 AND 9223372036854775807),
    erased_at_utc TEXT NOT NULL
        CONSTRAINT tombstone_erased_at_is_canonical CHECK (
            length(erased_at_utc) = 27
            AND erased_at_utc GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                || '[0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9][0-9][0-9][0-9][0-9]Z'),
    erased_session_count INTEGER NOT NULL
        CONSTRAINT tombstone_session_count_is_nonnegative CHECK (erased_session_count >= 0),
    erased_event_count INTEGER NOT NULL
        CONSTRAINT tombstone_event_count_is_nonnegative CHECK (erased_event_count >= 0),
    UNIQUE (scope_kind, scope_key, reason_code),
    CONSTRAINT tombstone_reason_matches_scope CHECK (
        (reason_code = 'ttl' AND scope_kind = 'session' AND scope_key <> 'store')
        OR (reason_code IN ('revoked', 'unclean_epoch') AND scope_kind = 'consent_epoch'
            AND scope_key <> 'store')
        OR (reason_code = 'clock_rollback' AND scope_kind = 'store' AND scope_key = 'store')),
    CONSTRAINT tombstone_revoke_authority_is_conditional CHECK (
        (reason_code = 'revoked'
            AND control_sequence IS NOT NULL
            AND control_fingerprint_hash IS NOT NULL
            AND ttl_consent_epoch_id IS NULL
            AND ttl_expires_at_utc IS NULL
            AND final_admission_ordinal IS NOT NULL
            AND final_admission_ordinal >= last_admission_ordinal)
        OR (reason_code = 'ttl'
            AND control_sequence IS NULL
            AND control_fingerprint_hash IS NOT NULL
            AND ttl_consent_epoch_id IS NOT NULL
            AND ttl_expires_at_utc IS NOT NULL
            AND erased_at_utc >= ttl_expires_at_utc
            AND final_admission_ordinal IS NULL)
        OR (reason_code IN ('unclean_epoch', 'clock_rollback')
            AND control_sequence IS NULL
            AND control_fingerprint_hash IS NULL
            AND ttl_consent_epoch_id IS NULL
            AND ttl_expires_at_utc IS NULL
            AND final_admission_ordinal IS NULL))
) STRICT;

CREATE TRIGGER erasure_request_authority_is_immutable
BEFORE UPDATE OF
    erasure_request_id,
    scope_kind,
    scope_key,
    requested_at_utc,
    reason_code,
    control_sequence,
    control_fingerprint_hash,
    ttl_consent_epoch_id,
    ttl_expires_at_utc,
    last_admission_ordinal
ON erasure_requests
BEGIN
    SELECT RAISE(ABORT, 'erasure request authority is immutable');
END;

CREATE TRIGGER erasure_request_final_watermark_sets_once
BEFORE UPDATE OF final_admission_ordinal ON erasure_requests
WHEN NOT (OLD.final_admission_ordinal IS NULL AND NEW.final_admission_ordinal IS NOT NULL)
BEGIN
    SELECT RAISE(ABORT, 'erasure request final watermark sets once');
END;

CREATE TRIGGER erasure_tombstones_are_immutable
BEFORE UPDATE ON erasure_tombstones
BEGIN
    SELECT RAISE(ABORT, 'erasure tombstones are immutable');
END;

CREATE TRIGGER erasure_tombstones_are_append_only
BEFORE DELETE ON erasure_tombstones
BEGIN
    SELECT RAISE(ABORT, 'erasure tombstones are append-only');
END;

CREATE INDEX evidence_sessions_by_expiry ON evidence_sessions(expires_at_utc);

CREATE INDEX erasure_requests_by_state ON erasure_requests(state);
"""


def normalize_schema_ddl(text: str) -> str:
    """Collapse every whitespace run so formatting can never change the digest."""

    return " ".join(text.split())


NORMALIZED_SCHEMA_DDL_V1 = normalize_schema_ddl(SCHEMA_DDL_V1)
SCHEMA_DDL_SHA256 = hashlib.sha256(NORMALIZED_SCHEMA_DDL_V1.encode("utf-8")).hexdigest()


def read_store_pragmas(connection: sqlite3.Connection) -> dict[str, object]:
    """Read exactly the required pragmas from a live connection."""

    return {
        name: connection.execute(f"PRAGMA {name}").fetchone()[0] for name, _ in REQUIRED_PRAGMAS
    }


def _apply_required_pragmas(connection: sqlite3.Connection) -> None:
    for name, value in REQUIRED_PRAGMAS:
        sql_value = f"'{value}'" if type(value) is str else str(value)
        connection.execute(f"PRAGMA {name}={sql_value}")


def applied_schema_digest(connection: sqlite3.Connection) -> str:
    """Digest the normalized SQL the database itself reports."""

    rows = connection.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type, name"
    ).fetchall()
    joined = "\n".join(f"{row[0]}:{row[1]}:{normalize_schema_ddl(str(row[2]))}" for row in rows)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@cache
def applied_schema_reference_digest() -> str:
    """The digest a database built from the checked-in DDL must report."""

    connection = sqlite3.connect(":memory:", isolation_level=None)
    try:
        connection.executescript(SCHEMA_DDL_V1)
        return applied_schema_digest(connection)
    finally:
        connection.close()


def validate_store_schema(connection: sqlite3.Connection) -> None:
    """Fail closed unless identity, pragmas, and applied schema are exactly V1."""

    if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
        raise _store_corrupt("the database does not carry the HRE1 application identity")
    if connection.execute("PRAGMA user_version").fetchone()[0] != USER_VERSION:
        raise _store_corrupt("the database does not carry schema user_version 1")
    if read_store_pragmas(connection) != dict(REQUIRED_PRAGMAS):
        raise _store_corrupt("the connection does not carry the required sole-writer pragmas")
    if applied_schema_digest(connection) != applied_schema_reference_digest():
        raise _store_corrupt("the applied schema is not the checked-in normalized V1 DDL")


def create_store_database(
    path: Path,
    *,
    factory: type[sqlite3.Connection] = sqlite3.Connection,
) -> sqlite3.Connection:
    """Create the V1 store at ``path`` and return its single owning connection."""

    connection = sqlite3.connect(path, isolation_level=None, factory=factory)
    try:
        _apply_required_pragmas(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.executescript(SCHEMA_DDL_V1)
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute(f"PRAGMA user_version={USER_VERSION}")
        connection.commit()
        _apply_required_pragmas(connection)
    except BaseException:
        connection.close()
        raise
    return connection


_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"
_SQLITE_HEADER_BYTES = 100
_SQLITE_TEXT_ENCODING_UTF8 = 1


def database_read_only_uri(path: Path) -> str:
    """Return a standards-correct read-only URI naming the exact literal file.

    ``Path.as_uri`` percent-encodes the path, so a literal ``%20`` or ``%41`` in a
    directory name survives SQLite's URI decoding intact instead of silently
    aliasing a different -- possibly valid -- neighbouring store.
    """

    if not path.is_absolute():
        raise _store_corrupt("an evidence database URI requires an absolute path")
    return f"{path.as_uri()}?mode=ro&immutable=0"


def hot_journal_path(database: Path) -> Path:
    """The rollback journal sibling of this exact literal database file."""

    return MANIFEST_V1.child(database.parent, MANIFEST_V1.database_journal)


def hot_journal_present(database: Path) -> bool:
    """Whether an owned, nonempty hot journal sits beside the literal database."""

    journal = hot_journal_path(database)
    try:
        return journal.is_file() and journal.stat().st_size > 0
    except OSError:  # pragma: no cover - a vanished journal is simply absent
        return False


def validate_database_header(path: Path) -> None:
    """Validate identity from the literal file's first 100 header bytes."""

    try:
        with path.open("rb") as handle:
            header = handle.read(_SQLITE_HEADER_BYTES)
    except OSError as exc:
        raise _store_corrupt("the evidence database header cannot be read") from exc
    if len(header) < _SQLITE_HEADER_BYTES or header[:16] != _SQLITE_HEADER_MAGIC:
        raise _store_corrupt("the evidence database has no SQLite header")
    declared = int.from_bytes(header[16:18], "big")
    page_size = 65_536 if declared == 1 else declared
    if page_size != MAIN_DB_PAGE_SIZE:
        raise _store_corrupt("the database does not use the required page size")
    if int.from_bytes(header[56:60], "big") != _SQLITE_TEXT_ENCODING_UTF8:
        raise _store_corrupt("the database is not UTF-8 encoded")
    if int.from_bytes(header[60:64], "big") != USER_VERSION:
        raise _store_corrupt("the database does not carry schema user_version 1")
    if int.from_bytes(header[68:72], "big") != APPLICATION_ID:
        raise _store_corrupt("the database does not carry the HRE1 application identity")


def open_store_database(
    path: Path,
    *,
    factory: type[sqlite3.Connection] = sqlite3.Connection,
) -> sqlite3.Connection:
    """Open an existing validated store and return its single owning connection."""

    connection = sqlite3.connect(path, isolation_level=None, factory=factory)
    try:
        _apply_required_pragmas(connection)
        validate_store_schema(connection)
    except BaseException:
        connection.close()
        raise
    return connection


def preflight_store_database(path: Path) -> None:
    """Read-only preflight for identity, schema, encoding, and page size."""

    if not path.is_file():
        raise _store_corrupt("the evidence database is absent")
    try:
        connection = sqlite3.connect(
            database_read_only_uri(path),
            uri=True,
            isolation_level=None,
        )
    except sqlite3.Error as exc:
        raise _store_corrupt("the evidence database cannot be opened read-only") from exc
    try:
        _validate_open_store_identity(connection)
    except (EvidenceStorageError, sqlite3.DatabaseError) as exc:
        # §5.1: an owned hot journal may legitimately block read-only validation, so
        # identity falls back to this same literal file's header and the later
        # mutable open performs SQLite's normal hot-journal crash recovery.
        if not hot_journal_present(path):
            if isinstance(exc, EvidenceStorageError):
                raise
            raise _store_corrupt("the evidence database header or schema is unreadable") from exc
        validate_database_header(path)
    finally:
        connection.close()


def _validate_open_store_identity(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
        raise _store_corrupt("the database does not carry the HRE1 application identity")
    if connection.execute("PRAGMA user_version").fetchone()[0] != USER_VERSION:
        raise _store_corrupt("the database does not carry schema user_version 1")
    if connection.execute("PRAGMA encoding").fetchone()[0] != "UTF-8":
        raise _store_corrupt("the database is not UTF-8 encoded")
    if connection.execute("PRAGMA page_size").fetchone()[0] != MAIN_DB_PAGE_SIZE:
        raise _store_corrupt("the database does not use the required page size")
    if applied_schema_digest(connection) != applied_schema_reference_digest():
        raise _store_corrupt("the applied schema is not the checked-in normalized V1 DDL")


DENY_FILTER_RESOURCE = "deny_filter_v1.json"
DENY_FILTER_ENGINE = "python-re-search-v1"
DENY_FILTER_VERSION = 1
DENY_FILTER_MAX_BYTES = 16_384
DENY_FILTER_CANONICAL_BYTES = (
    files("hermes_realtime.evidence").joinpath(DENY_FILTER_RESOURCE).read_bytes()
)
DENY_FILTER_RESOURCE_SHA256 = hashlib.sha256(DENY_FILTER_CANONICAL_BYTES).hexdigest()


@dataclass(frozen=True, slots=True)
class DenyFilterV1:
    """The compiled package deny resource; a pass makes no nonsensitivity claim."""

    engine: str
    version: int
    patterns: tuple[tuple[str, re.Pattern[str]], ...]

    @classmethod
    def from_document(cls, document: object) -> DenyFilterV1:
        """Exact-type parse the decoded object, order-independently by key."""

        if type(document) is not dict or set(document) != {
            "engine",
            "flags",
            "patterns",
            "version",
        }:
            raise EvidenceFrameError("the deny resource has a missing or unknown key")
        if document["engine"] != DENY_FILTER_ENGINE:
            raise EvidenceFrameError("the deny resource declares an unsupported engine")
        if type(document["version"]) is not int or document["version"] != DENY_FILTER_VERSION:
            raise EvidenceFrameError("the deny resource declares an unsupported version")
        if type(document["flags"]) is not list or document["flags"] != ["ASCII"]:
            raise EvidenceFrameError("the deny resource declares unsupported flags")
        raw_patterns = document["patterns"]
        if type(raw_patterns) is not list or not raw_patterns:
            raise EvidenceFrameError("the deny resource patterns must be a nonempty list")
        compiled: list[tuple[str, re.Pattern[str]]] = []
        for entry in raw_patterns:
            if (
                type(entry) is not dict
                or set(entry) != {"id", "expression"}
                or type(entry["id"]) is not str
                or type(entry["expression"]) is not str
            ):
                raise EvidenceFrameError("a deny pattern is not an exact id/expression object")
            try:
                compiled.append((entry["id"], re.compile(entry["expression"], re.ASCII)))
            except re.error as exc:
                raise EvidenceFrameError("a deny expression does not compile") from exc
        if len({identifier for identifier, _ in compiled}) != len(compiled):
            raise EvidenceFrameError("deny pattern identifiers must be unique")
        return cls(
            engine=DENY_FILTER_ENGINE,
            version=DENY_FILTER_VERSION,
            patterns=tuple(compiled),
        )

    @classmethod
    def load(cls) -> DenyFilterV1:
        """Load and validate the checked-in package resource bytes."""

        raw = DENY_FILTER_CANONICAL_BYTES
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1 or b"\r" in raw or b"\x00" in raw:
            raise EvidenceFrameError("the deny resource is not one LF-terminated line")
        try:
            document = parse_strict_json_object(raw[:-1], max_bytes=DENY_FILTER_MAX_BYTES)
        except EvidenceModelError as exc:
            raise EvidenceFrameError("the deny resource is not strict JSON") from exc
        return cls.from_document(document)

    def first_match(self, text: str) -> str | None:
        """Search each exact source string independently, in listed order."""

        if type(text) is not str:
            raise EvidenceFrameError("the deny filter searches exact built-in strings only")
        for identifier, pattern in self.patterns:
            if pattern.search(text) is not None:
                return identifier
        return None


MAX_SESSION_EVENTS = 9_216
MAX_SESSION_CANONICAL_BYTES = 40 * 1024 * 1024
MAX_GLOBAL_LIVE_CANONICAL_BYTES = 80 * 1024 * 1024

_CONFLICT_TAINTS: dict[ConflictReason, TaintCode] = {
    ConflictReason.EVENT_ID_ENVELOPE_MISMATCH: TaintCode.EVENT_ID_CONFLICT,
    ConflictReason.CROSS_SESSION_EVENT_ID: TaintCode.EVENT_ID_CONFLICT,
    ConflictReason.SESSION_SEQUENCE_CLAIMED: TaintCode.SEQUENCE_CONFLICT,
    ConflictReason.SEALED_SESSION_REUSE: TaintCode.EVENT_ID_CONFLICT,
}

_EVENT_COLUMNS = (
    "INSERT INTO evidence_events (event_id, logical_session_id, event_sequence, event_kind,"
    " recorded_at_utc, canonical_payload, payload_hash, previous_hash, record_hash,"
    " canonical_bytes) VALUES (?,?,?,?,?,?,?,?,?,?)"
)


class _Refused(Exception):  # noqa: N818 - an internal control-flow signal, never public
    """An internal refusal carrying its exact closed disposition and optional taint."""

    __slots__ = ("disposition", "taint")

    def __init__(self, disposition: StoreDisposition, taint: TaintCode | None = None) -> None:
        super().__init__(disposition.value)
        self.disposition = disposition
        self.taint = taint


class _ClockRolledBack(Exception):  # noqa: N818 - an internal control-flow signal
    """Sampled UTC fell below the durable high water; the sentinel is already latched."""


@dataclass(frozen=True, slots=True)
class _AppendPlan:
    """Everything one validated append needs, computed before its transaction opens."""

    snapshot: EvidenceSnapshotV1
    payload: bytes
    payload_hash: str
    previous_hash: str | None
    record_hash: str
    recorded_at_utc: str
    event_count: int
    canonical_bytes: int


def check_canonical_payload_size(payload: bytes) -> None:
    """Reject a canonical payload larger than the closed 32 KiB bound."""

    if len(payload) > MAX_CANONICAL_PAYLOAD_BYTES:
        raise EvidenceFrameError("canonical payload exceeds its closed 32 KiB bound")


def _walk_payload_text(value: object) -> tuple[str, ...]:
    """Collect exactly the closed ``text`` values a payload may carry."""

    if type(value) is dict:
        found: list[str] = []
        for key, item in value.items():
            if key == "text" and type(item) is str:
                found.append(item)
            else:
                found.extend(_walk_payload_text(item))
        return tuple(found)
    if type(value) is list:
        collected: list[str] = []
        for item in value:
            collected.extend(_walk_payload_text(item))
        return tuple(collected)
    return ()


def _write_new_file(path: Path, data: bytes) -> ActivationTemporaryV1:
    """Create and flush ``path`` while retaining its exclusive descriptor."""

    try:
        retained = create_activation_temporary(path)
        descriptor = retained.require_descriptor()
        written = 0
        while written < len(data):
            written += os.write(descriptor, data[written:])
        os.fsync(descriptor)
        return retained
    except BaseException:
        if "retained" in locals():
            retained.close()
        raise


class SQLiteEvidenceSpool:
    """One owner, one connection, explicit transactions, sticky content-free faults.

    Task 4 owns store creation, the append/seal/rollover transactions, conflict
    quarantine, quota preflight, the sentinel-first clock latch, and diagnostics.
    Task 5A owns exact physical full-purge recovery. Task 5B1 durably schedules an
    exact revoke request and fences later evidence mutation; Task 5B2 owns shared
    logical deletion, bounded physical maintenance, tombstoning, and restart replay.
    Runtime revoke finalization remains closed until its later checkpoint.
    """

    protocol_version: Literal[1] = 1

    def __init__(
        self,
        database: Path,
        *,
        clock: Callable[[], datetime],
        uuid_factory: Callable[[], str] = lambda: str(uuid4()),
        probe: EvidenceStorageProbeV1 | None = None,
        manifest: EvidenceArtifactManifestV1 = MANIFEST_V1,
        owner_generation: int = 1,
        connection_factory: type[sqlite3.Connection] = sqlite3.Connection,
        sentinel_opener: Callable[[Path], SentinelHandleV1] = SentinelHandleV1,
        repository_boundary: RepositoryBoundaryV1 | None = None,
        path_resolver: Callable[[Path], Path] = resolve_for_containment,
        root_handle_opener: Callable[[Path], EvidenceRootHandleV1] = WindowsEvidenceRootHandleV1,
    ) -> None:
        self._database = database
        self._clock = clock
        self._uuid_factory = uuid_factory
        self._probe: EvidenceStorageProbeV1 = probe or WindowsStorageProbeV1()
        self._manifest = manifest
        self._owner_generation = owner_generation
        self._factory = connection_factory
        self._root: ValidatedEvidenceRootV1 | None = None
        self._connection: sqlite3.Connection | None = None
        self._sentinel_opener = sentinel_opener
        self._repository_boundary = repository_boundary
        self._path_resolver = path_resolver
        self._root_handle_opener = root_handle_opener
        self._sentinel_handle: SentinelHandleV1 | None = None
        self._sticky_fault: WriterFault | None = None
        self._owner_state = OwnerState.ABSENT
        self._capture_state = CaptureState.UNAVAILABLE
        self._installation_id: str | None = None
        self._producer_instance_id: str | None = None
        self._current_session_id: str | None = None
        # Process-local accepted-command identity. These DTOs carry no text and are
        # never persisted; recovery deliberately leaves them unset so a post-restart
        # repeat fails closed instead of claiming an unprovable idempotent result.
        self._accepted_create: CreateEpochV1 | None = None
        self._accepted_seal: SealEpochV1 | None = None
        self._accepted_rollover: RolloverSessionV1 | None = None
        self._last_admission_ordinal = 0

    # -- owner-only accessors -----------------------------------------------------

    @property
    def owner_generation(self) -> int:
        """The exact generation this owner accepts on its drain lane."""

        return self._owner_generation

    @property
    def database(self) -> Path:
        return self._database

    @property
    def probe(self) -> EvidenceStorageProbeV1:
        return self._probe

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise _store_corrupt("the evidence store is not open")
        return self._connection

    def close(self) -> None:
        """Release the connection and the sentinel lease without claiming a drain."""

        if self._connection is not None:
            with suppress(sqlite3.Error):  # closing is best effort
                self._connection.close()
            self._connection = None
        self._release_sentinel()
        # The parent authority is released last, after every database operation.
        self._release_root_authority()

    def _close_for_drain(self) -> bool:
        """Close in authority order and report only confirmed lease release."""

        if self._connection is not None:
            try:
                self._connection.close()
            except sqlite3.Error:
                return False
            self._connection = None
        if not self._release_sentinel():
            return False
        self._release_root_authority()
        return True

    # -- fault and state latches --------------------------------------------------

    def _latch(self, fault: WriterFault) -> None:
        """Latch a sticky, content-free writer fault."""

        if self._sticky_fault is None:
            self._sticky_fault = fault
        self._owner_state = OwnerState.FAULTED
        self._capture_state = CaptureState.FAULTED

    def _rollback(self) -> None:
        if self._connection is not None and self._connection.in_transaction:
            with suppress(sqlite3.Error):  # the fault latch is the durable outcome
                self._connection.rollback()

    # -- sentinel and ownership ---------------------------------------------------

    def _release_sentinel(self) -> bool:
        handle = self._sentinel_handle
        if handle is None:
            return True
        try:
            handle.close()
        except OSError:
            return False
        self._sentinel_handle = None
        return True

    def _acquire_sentinel(self, path: Path) -> None:
        """Open the final sentinel and take its nonblocking single-owner lease."""

        if self._sentinel_handle is None:
            self._sentinel_handle = self._sentinel_opener(path)

    def _read_sentinel(self) -> bytes:
        handle = self._sentinel_handle
        if handle is None:
            raise _store_corrupt("the sentinel lease is not held")
        return handle.read_image()

    def _transition_sentinel(
        self,
        state: SentinelState,
        *,
        state_generation_id: str | None = None,
    ) -> None:
        """Write and flush only the inactive slot, then reselect the new generation."""

        handle = self._sentinel_handle
        if handle is None:
            raise _store_corrupt("the sentinel lease is not held")
        before = handle.read_image()
        offset, slot = next_sentinel_slot(before, state, state_generation_id=state_generation_id)
        expected = decode_sentinel_image(before).active.generation + 1
        handle.write_slot(offset, slot)
        after = handle.read_image()
        predecessor = SENTINEL_SLOT_OFFSETS[0 if offset == SENTINEL_SLOT_OFFSETS[1] else 1]
        if (
            after[:16] != before[:16]
            or after[predecessor : predecessor + SENTINEL_SLOT_SIZE]
            != before[predecessor : predecessor + SENTINEL_SLOT_SIZE]
        ):
            raise _store_corrupt("a sentinel transition disturbed its durable predecessor")
        selected = decode_sentinel_image(after)
        if selected.active.generation != expected or selected.active.state is not state:
            raise _store_corrupt("the sentinel transition did not become the selected slot")

    def _ensure_root(self) -> ValidatedEvidenceRootV1:
        if self._root is None:
            self._root = validate_evidence_database_path(
                self._database,
                probe=self._probe,
                manifest=self._manifest,
                boundary=self._repository_boundary,
                resolver=self._path_resolver,
                root_handle_opener=self._root_handle_opener,
            )
        return self._root

    @property
    def resolved_manifest(self) -> ResolvedEvidenceManifestV1:
        """The manifest resolved once against this owner's retained parent authority."""

        if self._root is None:
            raise _store_corrupt("the evidence root authority is not held")
        return self._root.require_authority()[1]

    def _release_root_authority(self) -> None:
        root, self._root = self._root, None
        if root is not None and root.authority is not None:
            root.authority.close()

    def _activate(
        self,
        root: ValidatedEvidenceRootV1,
        temporary: Path,
        final: Path,
        *,
        retained_temporary: ActivationTemporaryV1,
        expected_temporary: bytes,
    ) -> None:
        """Activate one exact temporary through the retained authority."""

        try:
            authority, _resolved = root.require_authority()
            # §§3.7/5.1: the retained handle must still name the same object, and the
            # configured path must still resolve to it, before any activation rename.
            authority.revalidate()
            authority.compare_fresh()
            authority.activate_exact_temporary(
                temporary,
                final,
                retained_temporary,
                expected_temporary,
            )
        finally:
            retained_temporary.close()

    def _ensure_root_marker(self, root: ValidatedEvidenceRootV1) -> None:
        _authority, resolved = root.require_authority()
        if resolved.root_marker.is_file():
            parse_root_marker(resolved.root_marker.read_bytes())
            return
        image = encode_root_marker(self._uuid_factory())
        retained_temporary = _write_new_file(resolved.root_marker_init, image)
        self._activate(
            root,
            resolved.root_marker_init,
            resolved.root_marker,
            retained_temporary=retained_temporary,
            expected_temporary=image,
        )
        if parse_root_marker(resolved.root_marker.read_bytes()) is None:  # pragma: no cover
            raise _store_corrupt("the activated root marker did not validate")

    def _ensure_sentinel(self, root: ValidatedEvidenceRootV1) -> None:
        _authority, resolved = root.require_authority()
        if not resolved.sentinel.is_file():
            image = initial_sentinel_image(self._uuid_factory())
            retained_temporary = _write_new_file(resolved.sentinel_init, image)
            self._activate(
                root,
                resolved.sentinel_init,
                resolved.sentinel,
                retained_temporary=retained_temporary,
                expected_temporary=image,
            )
            if resolved.sentinel.read_bytes() != image:
                raise _store_corrupt("the activated sentinel did not validate")
        self._acquire_sentinel(resolved.sentinel)

    # -- store lifecycle ----------------------------------------------------------

    def _validate_erasure_requests(self) -> None:
        """Validate every durable live authority before recovery can report success."""

        connection = self.connection
        requests = connection.execute(
            "SELECT erasure_request_id, scope_kind, scope_key, requested_at_utc,"
            " reason_code, state, control_sequence, control_fingerprint_hash,"
            " ttl_consent_epoch_id, ttl_expires_at_utc, last_admission_ordinal,"
            " final_admission_ordinal, resume_state,"
            " erased_session_count, erased_event_count"
            " FROM erasure_requests ORDER BY erasure_request_id"
        ).fetchall()
        try:
            for row in requests:
                (
                    request_id,
                    scope_kind,
                    scope_key,
                    requested_at_utc,
                    reason_code,
                    state,
                    control_sequence,
                    control_fingerprint_hash,
                    ttl_consent_epoch_id,
                    ttl_expires_at_utc,
                    last_admission_ordinal,
                    final_admission_ordinal,
                    resume_state,
                    erased_session_count,
                    erased_event_count,
                ) = row
                if not all(
                    type(value) is str
                    for value in (
                        request_id,
                        scope_kind,
                        scope_key,
                        requested_at_utc,
                        reason_code,
                        state,
                    )
                ):
                    raise _store_corrupt("an erasure request has invalid scalar authority")
                validate_canonical_uuid4(request_id, field_name="erasure_request_id")
                validate_canonical_utc(requested_at_utc, field_name="requested_at_utc")
                expected_scope = {
                    "ttl": "session",
                    "revoked": "consent_epoch",
                    "unclean_epoch": "consent_epoch",
                    "clock_rollback": "store",
                }.get(reason_code)
                if scope_kind != expected_scope:
                    raise _store_corrupt("an erasure request has incompatible scope authority")
                if scope_kind in ("session", "consent_epoch"):
                    validate_canonical_uuid4(scope_key, field_name="scope_key")
                elif scope_kind != "store" or scope_key != "store":
                    raise _store_corrupt("an erasure request has an invalid scope")
                if (
                    type(last_admission_ordinal) is not int
                    or not 1 <= last_admission_ordinal <= _MAX_SQLITE_INTEGER
                ):
                    raise _store_corrupt("an erasure request has an invalid admission watermark")
                if reason_code == "revoked":
                    if (
                        type(control_sequence) is not int
                        or not 1 <= control_sequence <= _MAX_JS_INTEGER
                        or type(control_fingerprint_hash) is not str
                        or ttl_consent_epoch_id is not None
                        or ttl_expires_at_utc is not None
                    ):
                        raise _store_corrupt("a revoke request has incomplete control authority")
                    validate_sha256_hex(
                        control_fingerprint_hash,
                        field_name="control_fingerprint_hash",
                    )
                    if final_admission_ordinal is not None and (
                        type(final_admission_ordinal) is not int
                        or not last_admission_ordinal
                        <= final_admission_ordinal
                        <= _MAX_SQLITE_INTEGER
                    ):
                        raise _store_corrupt("a revoke request has an invalid final watermark")
                elif reason_code == "ttl":
                    if (
                        control_sequence is not None
                        or type(control_fingerprint_hash) is not str
                        or type(ttl_consent_epoch_id) is not str
                        or type(ttl_expires_at_utc) is not str
                        or final_admission_ordinal is not None
                    ):
                        raise _store_corrupt("a TTL request has incomplete exact authority")
                    validate_sha256_hex(
                        control_fingerprint_hash,
                        field_name="control_fingerprint_hash",
                    )
                    validate_canonical_uuid4(
                        ttl_consent_epoch_id,
                        field_name="ttl_consent_epoch_id",
                    )
                    validate_canonical_utc(
                        ttl_expires_at_utc,
                        field_name="ttl_expires_at_utc",
                    )
                    if requested_at_utc < ttl_expires_at_utc:
                        raise _store_corrupt("a TTL request predates its persisted expiry")
                elif any(
                    value is not None
                    for value in (
                        control_sequence,
                        control_fingerprint_hash,
                        ttl_consent_epoch_id,
                        ttl_expires_at_utc,
                        final_admission_ordinal,
                    )
                ):
                    raise _store_corrupt("an epoch/store request carries foreign authority")

                if state == "pending":
                    coherent_state = (
                        resume_state is None
                        and erased_session_count is None
                        and erased_event_count is None
                    )
                    durable_phase = "pending"
                elif state == "logical_deleted":
                    coherent_state = (
                        resume_state is None
                        and type(erased_session_count) is int
                        and type(erased_event_count) is int
                        and 0 <= erased_session_count <= _MAX_SQLITE_INTEGER
                        and 0 <= erased_event_count <= _MAX_SQLITE_INTEGER
                    )
                    durable_phase = "logical_deleted"
                elif state == "purge_failed" and resume_state == "pending":
                    coherent_state = (
                        erased_session_count is None and erased_event_count is None
                    )
                    durable_phase = "pending"
                elif state == "purge_failed" and resume_state == "logical_deleted":
                    coherent_state = (
                        type(erased_session_count) is int
                        and type(erased_event_count) is int
                        and 0 <= erased_session_count <= _MAX_SQLITE_INTEGER
                        and 0 <= erased_event_count <= _MAX_SQLITE_INTEGER
                    )
                    durable_phase = "logical_deleted"
                else:
                    coherent_state = False
                    durable_phase = "invalid"
                if not coherent_state:
                    raise _store_corrupt("an erasure request has an incoherent durable state")
                if (
                    reason_code == "revoked"
                    and final_admission_ordinal is None
                    and state != "pending"
                ):
                    raise _store_corrupt(
                        "an unfinalized revoke has progressed beyond its pending state"
                    )
                if (
                    scope_kind == "session"
                    and durable_phase == "logical_deleted"
                    and erased_session_count != 1
                ):
                    raise _store_corrupt("a session erasure has no exact deleted-session count")

                if scope_kind == "session":
                    remaining = connection.execute(
                        "SELECT COUNT(*) FROM evidence_sessions WHERE logical_session_id=?",
                        (scope_key,),
                    ).fetchone()[0]
                elif scope_kind == "consent_epoch":
                    remaining = connection.execute(
                        "SELECT COUNT(*) FROM consent_epochs WHERE consent_epoch_id=?",
                        (scope_key,),
                    ).fetchone()[0]
                else:
                    if durable_phase == "pending":
                        remaining = connection.execute(
                            "SELECT COUNT(*) FROM producer_installation WHERE singleton=1"
                        ).fetchone()[0]
                    else:
                        remaining = connection.execute(
                            "SELECT (SELECT COUNT(*) FROM consent_epochs)"
                            " + (SELECT COUNT(*) FROM evidence_sessions)"
                        ).fetchone()[0]
                expected_remaining = 1 if durable_phase == "pending" else 0
                if remaining != expected_remaining:
                    raise _store_corrupt("an erasure request contradicts its durable scope phase")
        except (EvidenceModelError, TypeError, ValueError) as exc:
            raise _store_corrupt("an erasure request carries invalid durable authority") from exc

    def _validate_erasure_tombstones(self) -> None:
        """Validate complete terminal authority and its required relational absence."""

        connection = self.connection
        tombstones = connection.execute(
            "SELECT erasure_request_id, scope_kind, scope_key, reason_code,"
            " control_sequence, control_fingerprint_hash, ttl_consent_epoch_id,"
            " ttl_expires_at_utc, last_admission_ordinal, final_admission_ordinal,"
            " erased_at_utc, erased_session_count,"
            " erased_event_count FROM erasure_tombstones ORDER BY erasure_request_id"
        ).fetchall()
        try:
            for row in tombstones:
                (
                    request_id,
                    scope_kind,
                    scope_key,
                    reason_code,
                    control_sequence,
                    control_fingerprint_hash,
                    ttl_consent_epoch_id,
                    ttl_expires_at_utc,
                    last_admission_ordinal,
                    final_admission_ordinal,
                    erased_at_utc,
                    erased_session_count,
                    erased_event_count,
                ) = row
                if not all(
                    type(value) is str
                    for value in (
                        request_id,
                        scope_kind,
                        scope_key,
                        reason_code,
                        erased_at_utc,
                    )
                ):
                    raise _store_corrupt("an erasure tombstone has invalid scalar authority")
                validate_canonical_uuid4(request_id, field_name="erasure_request_id")
                validate_canonical_utc(erased_at_utc, field_name="erased_at_utc")
                if scope_kind in ("session", "consent_epoch"):
                    validate_canonical_uuid4(scope_key, field_name="scope_key")
                elif scope_kind != "store" or scope_key != "store":
                    raise _store_corrupt("an erasure tombstone has an invalid scope")
                if reason_code not in ("ttl", "revoked", "unclean_epoch", "clock_rollback"):
                    raise _store_corrupt("an erasure tombstone has an invalid reason")
                expected_scope = {
                    "ttl": "session",
                    "revoked": "consent_epoch",
                    "unclean_epoch": "consent_epoch",
                    "clock_rollback": "store",
                }[reason_code]
                if scope_kind != expected_scope:
                    raise _store_corrupt("an erasure tombstone has incompatible scope authority")
                if (
                    type(last_admission_ordinal) is not int
                    or not 1 <= last_admission_ordinal <= _MAX_SQLITE_INTEGER
                    or type(erased_session_count) is not int
                    or not 0 <= erased_session_count <= _MAX_SQLITE_INTEGER
                    or type(erased_event_count) is not int
                    or not 0 <= erased_event_count <= _MAX_SQLITE_INTEGER
                ):
                    raise _store_corrupt("an erasure tombstone has invalid ordinal or counts")
                if scope_kind == "session" and erased_session_count != 1:
                    raise _store_corrupt("a session tombstone has no exact erased-session count")
                if reason_code == "revoked":
                    if (
                        type(control_sequence) is not int
                        or not 1 <= control_sequence <= _MAX_JS_INTEGER
                        or type(control_fingerprint_hash) is not str
                        or ttl_consent_epoch_id is not None
                        or ttl_expires_at_utc is not None
                        or type(final_admission_ordinal) is not int
                        or not last_admission_ordinal
                        <= final_admission_ordinal
                        <= _MAX_SQLITE_INTEGER
                    ):
                        raise _store_corrupt("a revoke tombstone has incomplete control authority")
                    validate_sha256_hex(
                        control_fingerprint_hash,
                        field_name="control_fingerprint_hash",
                    )
                elif reason_code == "ttl":
                    if (
                        control_sequence is not None
                        or type(control_fingerprint_hash) is not str
                        or type(ttl_consent_epoch_id) is not str
                        or type(ttl_expires_at_utc) is not str
                        or final_admission_ordinal is not None
                    ):
                        raise _store_corrupt("a TTL tombstone has incomplete exact authority")
                    validate_sha256_hex(
                        control_fingerprint_hash,
                        field_name="control_fingerprint_hash",
                    )
                    validate_canonical_uuid4(
                        ttl_consent_epoch_id,
                        field_name="ttl_consent_epoch_id",
                    )
                    validate_canonical_utc(
                        ttl_expires_at_utc,
                        field_name="ttl_expires_at_utc",
                    )
                    if erased_at_utc < ttl_expires_at_utc:
                        raise _store_corrupt("a TTL tombstone predates its durable expiry")
                elif any(
                    value is not None
                    for value in (
                        control_sequence,
                        control_fingerprint_hash,
                        ttl_consent_epoch_id,
                        ttl_expires_at_utc,
                        final_admission_ordinal,
                    )
                ):
                    raise _store_corrupt("an epoch/store tombstone carries foreign authority")

                if scope_kind == "session":
                    remaining = connection.execute(
                        "SELECT COUNT(*) FROM evidence_sessions WHERE logical_session_id=?",
                        (scope_key,),
                    ).fetchone()[0]
                elif scope_kind == "consent_epoch":
                    remaining = connection.execute(
                        "SELECT (SELECT COUNT(*) FROM consent_epochs WHERE consent_epoch_id=?)"
                        " + (SELECT COUNT(*) FROM evidence_sessions WHERE consent_epoch_id=?)",
                        (scope_key, scope_key),
                    ).fetchone()[0]
                else:
                    remaining = connection.execute(
                        "SELECT (SELECT COUNT(*) FROM consent_epochs)"
                        " + (SELECT COUNT(*) FROM evidence_sessions)"
                    ).fetchone()[0]
                if remaining != 0:
                    raise _store_corrupt("an erasure tombstone coexists with its live scope")
        except (EvidenceModelError, TypeError, ValueError) as exc:
            raise _store_corrupt("an erasure tombstone carries invalid durable authority") from exc

        overlap = connection.execute(
            "SELECT 1 FROM erasure_requests AS r JOIN erasure_tombstones AS t"
            " ON r.erasure_request_id=t.erasure_request_id"
            " OR (r.scope_kind=t.scope_kind AND r.scope_key=t.scope_key"
            " AND r.reason_code=t.reason_code) LIMIT 1"
        ).fetchone()
        if overlap is not None:
            raise _store_corrupt("live and terminal erasure authority overlap")

    def _load_store_state(self) -> None:
        connection = self.connection
        self._validate_erasure_requests()
        self._validate_erasure_tombstones()
        self._validate_durable_evidence()
        installation = connection.execute(
            "SELECT installation_id FROM producer_installation WHERE singleton=1"
        ).fetchone()
        self._installation_id = None if installation is None else str(installation[0])
        open_sessions = connection.execute(
            "SELECT logical_session_id, producer_instance_id FROM evidence_sessions"
            " WHERE state='open'"
        ).fetchall()
        pending_erasures = connection.execute(
            "SELECT reason_code, state FROM erasure_requests ORDER BY erasure_request_id"
        ).fetchall()
        if len(open_sessions) == 1:
            self._current_session_id = str(open_sessions[0][0])
            self._producer_instance_id = str(open_sessions[0][1])
            self._capture_state = CaptureState.ACTIVE
        else:
            self._current_session_id = None
            self._capture_state = CaptureState.IDLE
        if pending_erasures:
            if any(str(row[1]) == "purge_failed" for row in pending_erasures):
                self._capture_state = CaptureState.PURGE_FAILED
            elif all(str(row[0]) == "revoked" for row in pending_erasures):
                self._capture_state = CaptureState.REVOKED_PURGING
            else:
                self._capture_state = CaptureState.FAULTED
            self._owner_state = OwnerState.RECOVERY_ONLY
            return
        self._owner_state = OwnerState.RUNNING

    def _open_owned_store(self, *, for_create: bool) -> None:
        """Validate the root, take ownership, then open or create the database."""

        root = self._ensure_root()
        audit_root_occupancy(root.root, manifest=self._manifest)
        self._ensure_root_marker(root)
        self._ensure_sentinel(root)
        image = decode_sentinel_image(self._read_sentinel())
        if root.database.is_file():
            if image.active.state is not SentinelState.CLEAR:
                raise EvidenceStorageError(
                    WriterFault.PURGE_REQUIRED,
                    "a durable purge is pending for this store",
                )
            preflight_store_database(root.database)
            self._connection = open_store_database(root.database, factory=self._factory)
            self._load_store_state()
            return
        if not for_create:
            return
        check_maintenance_headroom(root.root, probe=self._probe, manifest=self._manifest)
        if image.active.state is SentinelState.CLEAR:
            self._transition_sentinel(
                SentinelState.FIRST_CREATE_PENDING,
                state_generation_id=self._uuid_factory(),
            )
        elif image.active.state is not SentinelState.FIRST_CREATE_PENDING:
            raise EvidenceStorageError(
                WriterFault.PURGE_REQUIRED,
                "a durable purge is pending for this store",
            )
        self._connection = create_store_database(root.database, factory=self._factory)
        self._owner_state = OwnerState.RUNNING

    # -- clock --------------------------------------------------------------------

    def _high_water(self) -> str | None:
        row = self.connection.execute(
            "SELECT clock_high_water_utc FROM producer_installation WHERE singleton=1"
        ).fetchone()
        return None if row is None else str(row[0])

    def _latch_clock_rollback(self) -> None:
        """Write and flush the sentinel latch *before* any SQLite work."""

        self._transition_sentinel(
            SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
            state_generation_id=self._uuid_factory(),
        )
        try:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE producer_installation SET purge_required=1,"
                " purge_reason='clock_rollback', purge_scope='store' WHERE singleton=1"
            )
            connection.commit()
        except (sqlite3.Error, EvidenceStorageError):
            self._rollback()
        self._latch(WriterFault.PURGE_REQUIRED)

    def _sample_now(self) -> datetime:
        """Sample UTC and enforce the sentinel-first rollback protocol."""

        moment = self._clock()
        text = format_canonical_utc(moment)
        high_water = self._high_water()
        if high_water is not None and text < high_water:
            self._latch_clock_rollback()
            raise _ClockRolledBack()
        return moment

    # -- snapshot helpers ---------------------------------------------------------

    def _exact_snapshot(self, snapshot: object) -> tuple[EvidenceSnapshotV1, bytes]:
        """Round-trip a snapshot through the closed models and return its payload."""

        try:
            primitive = evidence_snapshot_to_primitive(cast(EvidenceSnapshotV1, snapshot))
            reconstructed = parse_evidence_snapshot_json(canonical_json_bytes(primitive))
            payload = canonical_json_bytes(primitive["payload"])
        except (EvidenceModelError, EvidenceFrameError, TypeError, ValueError) as exc:
            raise _Refused(StoreDisposition.REJECTED_STATE) from exc
        return reconstructed, payload

    def _stored_snapshot(
        self,
        row: tuple[object, ...],
        *,
        installation_id: str | None = None,
        producer_instance_id: str | None = None,
    ) -> EvidenceSnapshotV1:
        """Rebuild the exact snapshot a stored row represents."""

        stored_installation_id = (
            self._installation_id if installation_id is None else installation_id
        )
        stored_producer_instance_id = (
            self._producer_instance_id
            if producer_instance_id is None
            else producer_instance_id
        )
        return parse_evidence_snapshot_json(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "installation_id": stored_installation_id,
                    "producer_instance_id": stored_producer_instance_id,
                    "logical_session_id": str(row[0]),
                    "event_id": str(row[1]),
                    "event_sequence": int(cast(int, row[2])),
                    "event_kind": str(row[3]),
                    "payload": json.loads(str(row[4])),
                }
            )
        )

    def _session_history(self, session_id: str) -> tuple[EvidenceSnapshotV1, ...]:
        rows = self.connection.execute(
            "SELECT logical_session_id, event_id, event_sequence, event_kind, canonical_payload"
            " FROM evidence_events WHERE logical_session_id=? ORDER BY event_sequence",
            (session_id,),
        ).fetchall()
        return tuple(self._stored_snapshot(row) for row in rows)

    def _validated_session_history(self, session_id: str) -> tuple[EvidenceSnapshotV1, ...]:
        """Recompute canonical payload and HRE1 chain identity from durable rows."""

        authority = self.connection.execute(
            "SELECT s.producer_instance_id, p.installation_id FROM evidence_sessions AS s"
            " CROSS JOIN producer_installation AS p WHERE s.logical_session_id=?"
            " AND p.singleton=1",
            (session_id,),
        ).fetchone()
        if authority is None:
            raise EvidenceFrameError("stored session authority is missing")
        producer_instance_id, installation_id = map(str, authority)
        rows = self.connection.execute(
            "SELECT logical_session_id, event_id, event_sequence, event_kind, canonical_payload,"
            " payload_hash, previous_hash, record_hash, recorded_at_utc, canonical_bytes"
            " FROM evidence_events WHERE logical_session_id=? ORDER BY event_sequence",
            (session_id,),
        ).fetchall()
        snapshots: list[EvidenceSnapshotV1] = []
        previous_hash: str | None = None
        for row in rows:
            snapshot = self._stored_snapshot(
                tuple(row[:5]),
                installation_id=installation_id,
                producer_instance_id=producer_instance_id,
            )
            validate_canonical_utc(str(row[8]), field_name="recorded_at_utc")
            payload = canonical_json_bytes(json.loads(str(row[4])))
            if payload != str(row[4]).encode("utf-8") or len(payload) != int(row[9]):
                raise EvidenceFrameError("stored canonical payload identity is invalid")
            payload_hash, record_hash = self._record_hash(
                snapshot,
                recorded_at=str(row[8]),
                payload=payload,
                previous_hash=previous_hash,
            )
            if (
                str(row[5]) != payload_hash
                or row[6] != previous_hash
                or str(row[7]) != record_hash
            ):
                raise EvidenceFrameError("stored evidence hash chain is invalid")
            snapshots.append(snapshot)
            previous_hash = record_hash
        return tuple(snapshots)

    def _validate_durable_evidence(self) -> None:
        """Revalidate every durable session aggregate, sequence, and HRE1 chain."""

        try:
            if self.connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise EvidenceFrameError("durable evidence contains a foreign-key violation")
            orphan_events = self.connection.execute(
                "SELECT COUNT(*) FROM evidence_events AS e LEFT JOIN evidence_sessions AS s"
                " ON s.logical_session_id=e.logical_session_id"
                " WHERE s.logical_session_id IS NULL"
            ).fetchone()
            orphan_conflicts = self.connection.execute(
                "SELECT COUNT(*) FROM evidence_conflicts AS c LEFT JOIN evidence_sessions AS s"
                " ON s.logical_session_id=c.logical_session_id"
                " WHERE s.logical_session_id IS NULL"
            ).fetchone()
            if orphan_events != (0,) or orphan_conflicts != (0,):
                raise EvidenceFrameError("durable evidence contains orphan child rows")
            conflicts_rows = self.connection.execute(
                "SELECT conflict_id, logical_session_id, claimed_event_id, reason_code,"
                " recorded_at_utc FROM evidence_conflicts ORDER BY conflict_id"
            ).fetchall()
            for conflict in conflicts_rows:
                validate_canonical_uuid4(str(conflict[0]), field_name="conflict_id")
                validate_canonical_uuid4(
                    str(conflict[1]), field_name="logical_session_id"
                )
                validate_canonical_uuid4(str(conflict[2]), field_name="claimed_event_id")
                ConflictReason(str(conflict[3]))
                validate_canonical_utc(str(conflict[4]), field_name="recorded_at_utc")
            sessions = self.connection.execute(
                "SELECT logical_session_id, consent_epoch_id, producer_instance_id, state,"
                " final_event_sequence, head_hash, canonical_bytes, event_count, taint_code,"
                " opened_at_utc, expires_at_utc, consent_version"
                " FROM evidence_sessions ORDER BY logical_session_id"
            ).fetchall()
            for row in sessions:
                (
                    session_id,
                    epoch_id,
                    producer_instance_id,
                    state,
                    final_event_sequence,
                    head_hash,
                    canonical_bytes,
                    event_count,
                    taint_code,
                    opened_at_utc,
                    expires_at_utc,
                    consent_version,
                ) = row
                validate_canonical_uuid4(str(session_id), field_name="logical_session_id")
                validate_canonical_uuid4(str(epoch_id), field_name="consent_epoch_id")
                validate_canonical_uuid4(
                    str(producer_instance_id), field_name="producer_instance_id"
                )
                validate_canonical_utc(str(opened_at_utc), field_name="opened_at_utc")
                validate_canonical_utc(str(expires_at_utc), field_name="expires_at_utc")
                if str(expires_at_utc) <= str(opened_at_utc):
                    raise EvidenceFrameError("durable session expiry is not after opening")
                if str(consent_version) != CONSENT_VERSION:
                    raise EvidenceFrameError("durable session consent version is unsupported")
                state_text = str(state)
                if state_text not in {"open", "sealed", "tainted"}:
                    raise EvidenceFrameError("durable session state is unsupported")
                if state_text == "open" and (
                    final_event_sequence is not None
                    or head_hash is not None
                    or taint_code is not None
                ):
                    raise EvidenceFrameError("open session terminal authority is inconsistent")
                if state_text == "sealed" and (
                    final_event_sequence is None
                    or head_hash is None
                    or taint_code is not None
                ):
                    raise EvidenceFrameError("sealed session terminal authority is inconsistent")
                if state_text == "tainted" and (
                    taint_code is None
                    or ((final_event_sequence is None) != (head_hash is None))
                ):
                    raise EvidenceFrameError("tainted session authority is inconsistent")
                if taint_code is not None:
                    TaintCode(str(taint_code))
                if head_hash is not None:
                    validate_sha256_hex(str(head_hash), field_name="head_hash")
                history = self._validated_session_history(str(session_id))
                if not history:
                    raise EvidenceFrameError("a durable session has no opening records")
                validation_history = history
                if (
                    state_text == "sealed"
                    and len(history) >= 2
                    and history[-2].event_kind is EventKind.BINDING_CLOSED
                ):
                    close_primitive = evidence_snapshot_to_primitive(history[-2])
                    close_payload = cast(dict[str, object], close_primitive["payload"])
                    if close_payload.get("close_reason") in {
                        BindingCloseReason.CAPACITY_ROLLOVER.value,
                        BindingCloseReason.RETENTION_ROLLOVER.value,
                    }:
                        close_payload["close_reason"] = BindingCloseReason.HOST_SHUTDOWN.value
                        normalized_close = parse_evidence_snapshot_json(
                            canonical_json_bytes(close_primitive)
                        )
                        validation_history = (
                            *history[:-2],
                            normalized_close,
                            history[-1],
                        )
                validate_event_sequence(validation_history)
                if tuple(snapshot.event_sequence for snapshot in history) != tuple(
                    range(1, len(history) + 1)
                ):
                    raise EvidenceFrameError("a durable session sequence is not contiguous")
                aggregate = self.connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(canonical_bytes),0),"
                    " (SELECT record_hash FROM evidence_events WHERE logical_session_id=?"
                    " ORDER BY event_sequence DESC LIMIT 1)"
                    " FROM evidence_events WHERE logical_session_id=?",
                    (session_id, session_id),
                ).fetchone()
                if aggregate is None or (
                    int(aggregate[0]) != int(event_count)
                    or int(aggregate[1]) != int(canonical_bytes)
                    or int(event_count) != len(history)
                ):
                    raise EvidenceFrameError("durable session aggregates do not match its events")
                if final_event_sequence is not None and (
                    int(final_event_sequence) != len(history)
                    or head_hash is None
                    or str(head_hash) != str(aggregate[2])
                ):
                    raise EvidenceFrameError("durable terminal session authority is inconsistent")
                conflicts = int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM evidence_conflicts WHERE logical_session_id=?",
                        (session_id,),
                    ).fetchone()[0]
                )
                if conflicts and str(state) != "tainted":
                    raise EvidenceFrameError("an untainted durable session carries conflicts")
                epoch = self.connection.execute(
                    "SELECT producer_instance_id, state FROM consent_epochs"
                    " WHERE consent_epoch_id=?",
                    (epoch_id,),
                ).fetchone()
                if epoch is None or str(epoch[0]) != str(producer_instance_id):
                    raise EvidenceFrameError("durable session epoch authority is inconsistent")
                if state_text == "sealed":
                    epoch_state = str(epoch[1])
                    if epoch_state not in {"active", "closed", "revoked"}:
                        raise EvidenceFrameError(
                            "a sealed session has unsupported epoch authority"
                        )
                    if epoch_state == "active":
                        rollover_successors = self.connection.execute(
                            "SELECT COUNT(*) FROM evidence_sessions"
                            " WHERE consent_epoch_id=? AND producer_instance_id=?"
                            " AND state='open' AND logical_session_id<>?"
                            " AND opened_at_utc>=?",
                            (epoch_id, producer_instance_id, session_id, opened_at_utc),
                        ).fetchone()
                        if rollover_successors != (1,):
                            raise EvidenceFrameError(
                                "an active-epoch sealed session has no unique rollover successor"
                            )
        except (EvidenceFrameError, EvidenceModelError, TypeError, ValueError) as exc:
            raise _store_corrupt("durable evidence integrity validation failed") from exc

    def _previous_hash(self, session_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT record_hash FROM evidence_events WHERE logical_session_id=?"
            " ORDER BY event_sequence DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return None if row is None else str(row[0])

    def _record_hash(self, snapshot: EvidenceSnapshotV1, *, recorded_at: str, payload: bytes,
                     previous_hash: str | None) -> tuple[str, str]:
        payload_hash = hashlib.sha256(payload).hexdigest()
        return payload_hash, hre1_record_hash(
            installation_id=snapshot.installation_id,
            producer_instance_id=snapshot.producer_instance_id,
            event_id=snapshot.event_id,
            logical_session_id=snapshot.logical_session_id,
            event_sequence=snapshot.event_sequence,
            event_kind=snapshot.event_kind.value,
            recorded_at_utc=recorded_at,
            payload_hash=payload_hash,
            previous_hash=previous_hash,
        )

    # -- taints and conflicts -----------------------------------------------------

    def _taint_session(self, session_id: str, code: TaintCode) -> None:
        connection = self.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE evidence_sessions SET state='tainted', taint_code=?"
                " WHERE logical_session_id=? AND state<>'tainted'",
                (code.value, session_id),
            )
            connection.commit()
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)

    def _record_conflict(self, session_id: str, claimed_event_id: str, reason: ConflictReason,
                         recorded_at: str) -> None:
        """Store one content-free conflict row and quarantine the incoming session."""

        connection = self.connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO evidence_conflicts (conflict_id, logical_session_id,"
                " claimed_event_id, reason_code, recorded_at_utc) VALUES (?,?,?,?,?)",
                (self._uuid_factory(), session_id, claimed_event_id, reason.value, recorded_at),
            )
            connection.execute(
                "UPDATE evidence_sessions SET state='tainted', taint_code=?"
                " WHERE logical_session_id=? AND state<>'tainted'",
                (_CONFLICT_TAINTS[reason].value, session_id),
            )
            connection.commit()
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)

    # -- quotas -------------------------------------------------------------------

    def live_canonical_bytes(self) -> int:
        """The canonical bytes every live session currently holds."""

        row = self.connection.execute(
            "SELECT COALESCE(SUM(canonical_bytes), 0) FROM evidence_sessions"
        ).fetchone()
        return int(row[0])

    def _check_quotas(self, *, event_count: int, canonical_bytes: int, payload_length: int) -> None:
        connection = self.connection
        if event_count + 1 > MAX_SESSION_EVENTS:
            raise _Refused(StoreDisposition.REJECTED_STATE, TaintCode.QUOTA_EXCEEDED)
        if canonical_bytes + payload_length > MAX_SESSION_CANONICAL_BYTES:
            raise _Refused(StoreDisposition.REJECTED_STATE, TaintCode.QUOTA_EXCEEDED)
        if self.live_canonical_bytes() + payload_length > MAX_GLOBAL_LIVE_CANONICAL_BYTES:
            raise _Refused(StoreDisposition.REJECTED_STATE, TaintCode.QUOTA_EXCEEDED)
        page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
        ceiling = min(
            int(connection.execute("PRAGMA max_page_count").fetchone()[0]),
            MAX_MAIN_DB_PAGE_COUNT,
        )
        needed = -(-payload_length // MAIN_DB_PAGE_SIZE) + 2
        if page_count + needed > ceiling:
            raise _Refused(StoreDisposition.REJECTED_STATE, TaintCode.QUOTA_EXCEEDED)

    # -- the append pipeline ------------------------------------------------------

    def _plan_append(self, snapshot: object) -> _AppendPlan:
        connection = self.connection
        exact, payload = self._exact_snapshot(snapshot)
        try:
            check_canonical_payload_size(payload)
        except EvidenceFrameError as exc:
            raise _Refused(StoreDisposition.REJECTED_STATE, TaintCode.OVERSIZE) from exc
        deny = DenyFilterV1.load()
        for text in _walk_payload_text(json.loads(payload.decode("utf-8"))):
            if deny.first_match(text) is not None:
                raise _Refused(StoreDisposition.REJECTED_STATE, TaintCode.DENY_FILTER)
        if (
            exact.installation_id != self._installation_id
            or exact.producer_instance_id != self._producer_instance_id
        ):
            raise _store_corrupt("the incoming envelope crosses installation or producer lineage")

        # §5.3: the event-id lookup strictly precedes any timestamp allocation.
        existing = connection.execute(
            "SELECT logical_session_id, event_id, event_sequence, event_kind, canonical_payload"
            " FROM evidence_events WHERE event_id=?",
            (exact.event_id,),
        ).fetchone()
        session_row = connection.execute(
            "SELECT state, event_count, canonical_bytes FROM evidence_sessions"
            " WHERE logical_session_id=?",
            (exact.logical_session_id,),
        ).fetchone()
        if session_row is None or exact.logical_session_id != self._current_session_id:
            raise _store_corrupt("the incoming envelope names an unknown or noncurrent session")
        state, event_count, canonical_bytes = (
            str(session_row[0]),
            int(session_row[1]),
            int(session_row[2]),
        )
        if existing is not None:
            classification = classify_event_retry(
                self._stored_snapshot(existing),
                exact,
                target_sealed=state == "sealed",
            )
            if classification is StoreDisposition.IDEMPOTENT:
                raise _Refused(StoreDisposition.IDEMPOTENT)
            if isinstance(classification, ConflictReason):
                self._conflict(exact, classification)
                raise _Refused(StoreDisposition.CONFLICT_TAINTED)
        if state == "tainted":
            raise _Refused(StoreDisposition.REJECTED_STATE)
        if state == "sealed":
            self._conflict(exact, ConflictReason.SEALED_SESSION_REUSE)
            raise _Refused(StoreDisposition.CONFLICT_TAINTED)
        claimed = connection.execute(
            "SELECT event_id FROM evidence_events WHERE logical_session_id=? AND event_sequence=?",
            (exact.logical_session_id, exact.event_sequence),
        ).fetchone()
        if claimed is not None:
            self._conflict(exact, ConflictReason.SESSION_SEQUENCE_CLAIMED)
            raise _Refused(StoreDisposition.CONFLICT_TAINTED)
        self._check_quotas(
            event_count=event_count,
            canonical_bytes=canonical_bytes,
            payload_length=len(payload),
        )
        if exact.event_sequence != event_count + 1:
            # §3.4: a gap latches the session row without fabricating a taint event.
            raise _Refused(StoreDisposition.CONFLICT_TAINTED, TaintCode.ADMISSION_GAP)
        try:
            validate_event_sequence((*self._session_history(exact.logical_session_id), exact))
        except EvidenceModelError as exc:
            raise _Refused(
                StoreDisposition.CONFLICT_TAINTED,
                TaintCode.LINEAGE_INVALID,
            ) from exc

        recorded_at = format_canonical_utc(self._sample_now())
        previous_hash = self._previous_hash(exact.logical_session_id)
        payload_hash, record_hash = self._record_hash(
            exact,
            recorded_at=recorded_at,
            payload=payload,
            previous_hash=previous_hash,
        )
        return _AppendPlan(
            snapshot=exact,
            payload=payload,
            payload_hash=payload_hash,
            previous_hash=previous_hash,
            record_hash=record_hash,
            recorded_at_utc=recorded_at,
            event_count=event_count,
            canonical_bytes=canonical_bytes,
        )

    def _conflict(self, snapshot: EvidenceSnapshotV1, reason: ConflictReason) -> None:
        self._record_conflict(
            snapshot.logical_session_id,
            snapshot.event_id,
            reason,
            format_canonical_utc(self._sample_now()),
        )

    def _insert_event(self, plan: _AppendPlan) -> None:
        self.connection.execute(
            _EVENT_COLUMNS,
            (
                plan.snapshot.event_id,
                plan.snapshot.logical_session_id,
                plan.snapshot.event_sequence,
                plan.snapshot.event_kind.value,
                plan.recorded_at_utc,
                plan.payload.decode("utf-8"),
                plan.payload_hash,
                plan.previous_hash,
                plan.record_hash,
                len(plan.payload),
            ),
        )
        self.connection.execute(
            "UPDATE evidence_sessions SET event_count=?, canonical_bytes=?"
            " WHERE logical_session_id=?",
            (
                plan.event_count + 1,
                plan.canonical_bytes + len(plan.payload),
                plan.snapshot.logical_session_id,
            ),
        )

    def _advance_high_water(self, recorded_at: str) -> None:
        self.connection.execute(
            "UPDATE producer_installation SET clock_high_water_utc=?"
            " WHERE singleton=1 AND clock_high_water_utc<?",
            (recorded_at, recorded_at),
        )

    def _append(self, snapshot: object) -> StoreDisposition:
        if self._sticky_fault is not None:
            return StoreDisposition.FAULTED
        if self._connection is None:
            return StoreDisposition.FAULTED
        if self._current_session_id is None:
            return StoreDisposition.REJECTED_STATE
        if (
            self._owner_state is OwnerState.RECOVERY_ONLY
            or self._capture_state in {CaptureState.REVOKED_PURGING, CaptureState.PURGE_FAILED}
        ):
            return StoreDisposition.REJECTED_STATE
        try:
            plan = self._plan_append(snapshot)
        except _Refused as refusal:
            if refusal.taint is not None and self._current_session_id is not None:
                self._taint_session(self._current_session_id, refusal.taint)
            return refusal.disposition
        except _ClockRolledBack:
            return StoreDisposition.FAULTED
        except EvidenceStorageError as exc:
            self._latch(exc.fault)
            return StoreDisposition.FAULTED
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        try:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            self._insert_event(plan)
            self._advance_high_water(plan.recorded_at_utc)
            connection.commit()
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        return StoreDisposition.COMMITTED

    # -- transport surface: physical full-purge recovery --------------------------

    def _reset_purged_process_state(self, *, clear_sticky_fault: bool = False) -> None:
        """Forget every value whose authority died with the six owned artifacts."""

        if clear_sticky_fault:
            self._sticky_fault = None
        self._installation_id = None
        self._producer_instance_id = None
        self._current_session_id = None
        self._accepted_create = None
        self._accepted_seal = None
        self._accepted_rollover = None
        self._last_admission_ordinal = 0
        self._owner_state = OwnerState.ABSENT
        self._capture_state = CaptureState.UNAVAILABLE

    def _purge_pending_artifacts(
        self,
        root: ValidatedEvidenceRootV1,
        *,
        state: SentinelState,
        generation_id: str,
    ) -> bool:
        """Erase only the retained manifest's six paths, keeping its pending latch."""

        authority, resolved = root.require_authority()
        image = decode_sentinel_image(self._read_sentinel())
        if image.active.state is not state or image.active.state_generation_id != generation_id:
            return False
        try:
            # SQLite must release all journal and sidecar handles before any deletion.
            # A pathological connection factory can still make close fail; that is a
            # retryable purge failure, never an exception crossing the transport.
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            for artifact in resolved.deletable:
                # The parent capability is checked immediately before each exact child
                # mutation; no scan, pattern, or recursive filesystem authority exists.
                authority.revalidate()
                authority.compare_fresh()
                artifact.unlink(missing_ok=True)
            for artifact in resolved.deletable:
                if os.path.lexists(artifact):
                    return False
            # State is reset before the durable clear, so a clear sentinel never
            # leaves this process carrying an old installation/session authority.
            # Keep a complete rollback image: a failed clear must not erase a sticky
            # fault or make an unsuccessful purge look like a fresh owner.
            previous = (
                self._sticky_fault,
                self._installation_id,
                self._producer_instance_id,
                self._current_session_id,
                self._accepted_create,
                self._accepted_seal,
                self._accepted_rollover,
                self._owner_state,
                self._capture_state,
            )
            self._reset_purged_process_state(clear_sticky_fault=True)
            try:
                authority.revalidate()
                authority.compare_fresh()
                self._transition_sentinel(SentinelState.CLEAR)
            except (EvidenceStorageError, OSError, sqlite3.Error):
                (
                    self._sticky_fault,
                    self._installation_id,
                    self._producer_instance_id,
                    self._current_session_id,
                    self._accepted_create,
                    self._accepted_seal,
                    self._accepted_rollover,
                    self._owner_state,
                    self._capture_state,
                ) = previous
                raise
        except (EvidenceStorageError, OSError, sqlite3.Error):
            # The selected pending sentinel remains durable and retryable.
            return False
        return True

    def _remove_temporary_after_validation(
        self,
        root: ValidatedEvidenceRootV1,
        temporary: Path,
        validate: Callable[[bytes], object],
    ) -> bool:
        """Lease, validate, and remove one exact activation temporary."""

        authority, _resolved = root.require_authority()
        try:
            self._acquire_sentinel(temporary)
        except (EvidenceStorageError, OSError):
            return False
        validated = False
        try:
            validate(self._read_sentinel())
            authority.revalidate()
            authority.compare_fresh()
            validated = True
        except (EvidenceStorageError, OSError):
            pass
        if not self._release_sentinel():
            return False
        if not validated:
            return False
        try:
            authority.revalidate()
            authority.compare_fresh()
            temporary.unlink()
            return not os.path.lexists(temporary)
        except (EvidenceStorageError, OSError):
            return False

    def _recover_sentinel_initialization(self, root: ValidatedEvidenceRootV1) -> bool:
        """Remove only a leased initial sentinel and the exact six debris leaves."""

        authority, resolved = root.require_authority()

        def validate_initial_image() -> bool:
            image = self._read_sentinel()
            active = decode_sentinel_image(image).active
            return (
                active.state is SentinelState.FIRST_CREATE_PENDING
                and active.state_generation_id is not None
                and image == initial_sentinel_image(active.state_generation_id)
            )

        # Prove the temporary lease can be cleanly released before any mutation. A
        # failed close remains retained in ``_sentinel_handle`` for an exact retry.
        try:
            self._acquire_sentinel(resolved.sentinel_init)
            validated = validate_initial_image()
            authority.revalidate()
            authority.compare_fresh()
        except (EvidenceStorageError, OSError):
            validated = False
        if not self._release_sentinel() or not validated:
            return False

        # Reacquire and revalidate the exact temporary around debris deletion.
        try:
            self._acquire_sentinel(resolved.sentinel_init)
        except (EvidenceStorageError, OSError):
            return False
        recovered = False
        try:
            if validate_initial_image():
                recovered = True
                for artifact in resolved.deletable:
                    authority.revalidate()
                    authority.compare_fresh()
                    artifact.unlink(missing_ok=True)
                    if os.path.lexists(artifact):
                        recovered = False
                        break
        except (EvidenceStorageError, OSError):
            recovered = False
        if not self._release_sentinel():
            return False
        if not recovered:
            return False
        try:
            authority.revalidate()
            authority.compare_fresh()
            resolved.sentinel_init.unlink()
            return not os.path.lexists(resolved.sentinel_init)
        except (EvidenceStorageError, OSError):
            return False

    def _finish_detected_clock_rollback(self) -> RecoveryDisposition:
        """Complete a sentinel-authorized rollback purge in this recovery call."""

        try:
            root = self._ensure_root()
            active = decode_sentinel_image(self._read_sentinel()).active
            generation_id = active.state_generation_id
            result = RecoveryDisposition.FAULTED
            if (
                active.state is SentinelState.CLOCK_ROLLBACK_PURGE_PENDING
                and generation_id is not None
                and self._purge_pending_artifacts(
                    root,
                    state=SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
                    generation_id=generation_id,
                )
            ):
                result = RecoveryDisposition.PURGE_COMPLETED
            return result if self._release_sentinel() else RecoveryDisposition.FAULTED
        except (EvidenceStorageError, OSError, sqlite3.Error):
            return RecoveryDisposition.FAULTED

    def recover_existing(self) -> RecoveryDisposition:
        """Complete a durable physical purge before opening any SQLite database."""

        if self._sticky_fault is not None:
            return RecoveryDisposition.FAULTED
        if self._connection is not None:
            try:
                validate_store_schema(self.connection)
                self._load_store_state()
                if not self._resume_recoverable_erasures():
                    return RecoveryDisposition.FAULTED
                if not self._schedule_recovery_privacy_erasures(cold_start=False):
                    return RecoveryDisposition.FAULTED
                return RecoveryDisposition.RECOVERED
            except _ClockRolledBack:
                return self._finish_detected_clock_rollback()
            except EvidenceStorageError as exc:
                self._latch(exc.fault)
                return RecoveryDisposition.FAULTED
            except sqlite3.Error:
                self._latch(WriterFault.SQLITE_FAULT)
                return RecoveryDisposition.FAULTED
        if not self._probe.platform_is_supported():
            return RecoveryDisposition.UNSUPPORTED_PLATFORM
        try:
            root = self._ensure_root()
            audit_root_occupancy(root.root, manifest=self._manifest)
            _authority, resolved = root.require_authority()
            if not resolved.root_marker.is_file():
                if os.path.lexists(resolved.root_marker_init):
                    if not self._remove_temporary_after_validation(
                        root,
                        resolved.root_marker_init,
                        lambda raw: parse_root_marker(raw),
                    ):
                        return RecoveryDisposition.FAULTED
                    return RecoveryDisposition.ABSENT
                if (
                    os.path.lexists(resolved.sentinel)
                    or os.path.lexists(resolved.sentinel_init)
                    or any(os.path.lexists(path) for path in resolved.deletable)
                ):
                    return RecoveryDisposition.FAULTED
                return RecoveryDisposition.ABSENT
            parse_root_marker(resolved.root_marker.read_bytes())
            if not resolved.sentinel.is_file():
                if os.path.lexists(resolved.root_marker_init):
                    return RecoveryDisposition.FAULTED
                has_temporary = os.path.lexists(resolved.sentinel_init)
                has_debris = tuple(os.path.lexists(path) for path in resolved.deletable)
                if has_temporary:
                    return (
                        RecoveryDisposition.ABSENT
                        if self._recover_sentinel_initialization(root)
                        else RecoveryDisposition.FAULTED
                    )
                if any(has_debris):
                    return RecoveryDisposition.FAULTED
                return RecoveryDisposition.ABSENT
            self._acquire_sentinel(resolved.sentinel)
            image = decode_sentinel_image(self._read_sentinel())
            active = image.active
            if active.state in (
                SentinelState.FIRST_CREATE_PENDING,
                SentinelState.FULL_PURGE_PENDING,
                SentinelState.CLOCK_ROLLBACK_PURGE_PENDING,
            ):
                result = RecoveryDisposition.FAULTED
                try:
                    generation_id = active.state_generation_id
                    if generation_id is not None and self._purge_pending_artifacts(
                        root, state=active.state, generation_id=generation_id
                    ):
                        result = RecoveryDisposition.PURGE_COMPLETED
                finally:
                    released = self._release_sentinel()
                if not released:
                    return RecoveryDisposition.FAULTED
                return result
            if active.state is not SentinelState.CLEAR:
                return RecoveryDisposition.FAULTED
            if not root.database.is_file():
                return RecoveryDisposition.ABSENT
            preflight_store_database(root.database)
            self._connection = open_store_database(root.database, factory=self._factory)
            self._load_store_state()
            if not self._resume_recoverable_erasures():
                return RecoveryDisposition.FAULTED
            if not self._schedule_recovery_privacy_erasures(cold_start=True):
                return RecoveryDisposition.FAULTED
        except _ClockRolledBack:
            return self._finish_detected_clock_rollback()
        except EvidenceStorageError as exc:
            if exc.fault is WriterFault.OWNERSHIP_UNAVAILABLE:
                return RecoveryDisposition.OWNERSHIP_UNAVAILABLE
            self._latch(exc.fault)
            return RecoveryDisposition.FAULTED
        except sqlite3.Error:
            self._latch(WriterFault.SQLITE_FAULT)
            return RecoveryDisposition.FAULTED
        except OSError:
            return RecoveryDisposition.FAULTED
        return RecoveryDisposition.RECOVERED

    def create_epoch(self, command: CreateEpochV1) -> StoreDisposition:
        """Create or open the owned store and persist the epoch plus both openings."""

        if self._sticky_fault is not None:
            return StoreDisposition.FAULTED
        if type(command) is not CreateEpochV1:
            return StoreDisposition.REJECTED_STATE
        try:
            if self._connection is None:
                self._open_owned_store(for_create=True)
        except EvidenceStorageError as exc:
            self._latch(exc.fault)
            return StoreDisposition.FAULTED
        except sqlite3.Error:
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED

        connection = self.connection
        try:
            erased_epoch = connection.execute(
                "SELECT 1 FROM erasure_tombstones WHERE scope_kind='consent_epoch'"
                " AND scope_key=? LIMIT 1",
                (command.consent_epoch_id,),
            ).fetchone()
            if erased_epoch is not None:
                return StoreDisposition.REJECTED_STATE
            existing = connection.execute(
                "SELECT consent_epoch_id FROM consent_epochs"
            ).fetchall()
            if existing:
                return self._existing_epoch_disposition(command, existing)
            self._installation_id = command.installation_id
            self._producer_instance_id = command.producer_instance_id
            opened, opened_payload = self._exact_snapshot(command.session_opened)
            binding, binding_payload = self._exact_snapshot(command.binding_opened)
            for payload in (opened_payload, binding_payload):
                check_canonical_payload_size(payload)
            moment = self._sample_now()
            opened_at = format_canonical_utc(moment)
            expires_at = format_canonical_utc(
                moment + timedelta(hours=command.retention_hours)
            )
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO producer_installation (singleton, installation_id, created_at_utc,"
                " clock_high_water_utc, purge_required, purge_reason, purge_scope)"
                " VALUES (1,?,?,?,0,NULL,NULL)",
                (command.installation_id, opened_at, opened_at),
            )
            connection.execute(
                "INSERT INTO consent_epochs (consent_epoch_id, producer_instance_id, state,"
                " opened_at_utc, closed_at_utc) VALUES (?,?, 'active', ?, NULL)",
                (command.consent_epoch_id, command.producer_instance_id, opened_at),
            )
            connection.execute(
                "INSERT INTO evidence_sessions (logical_session_id, consent_epoch_id,"
                " producer_instance_id, opened_at_utc, expires_at_utc, consent_version, state,"
                " final_event_sequence, head_hash, canonical_bytes, event_count, taint_code)"
                " VALUES (?,?,?,?,?,?, 'open', NULL, NULL, 0, 0, NULL)",
                (
                    command.logical_session_id,
                    command.consent_epoch_id,
                    command.producer_instance_id,
                    opened_at,
                    expires_at,
                    command.consent_version,
                ),
            )
            self._write_chain(
                (opened, opened_payload),
                (binding, binding_payload),
                recorded_at=opened_at,
                session_id=command.logical_session_id,
            )
            connection.commit()
        except _Refused:
            self._rollback()
            return StoreDisposition.REJECTED_STATE
        except _ClockRolledBack:
            return StoreDisposition.FAULTED
        except (EvidenceModelError, EvidenceFrameError):
            self._rollback()
            return StoreDisposition.REJECTED_STATE
        except EvidenceStorageError as exc:
            self._rollback()
            self._latch(exc.fault)
            return StoreDisposition.FAULTED
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        self._current_session_id = command.logical_session_id
        self._accepted_create = command
        self._owner_state = OwnerState.RUNNING
        self._capture_state = CaptureState.ACTIVE
        self._last_admission_ordinal = max(
            command.session_opened.event_sequence,
            command.binding_opened.event_sequence,
        )
        try:
            self._transition_sentinel(SentinelState.CLEAR)
        except EvidenceStorageError as exc:  # pragma: no cover - device level failure
            self._latch(exc.fault)
            return StoreDisposition.FAULTED
        return StoreDisposition.COMMITTED

    def _stored_matches_snapshot(self, snapshot: EvidenceSnapshotV1) -> bool:
        """Whether the committed row for this event id is byte-exactly this snapshot."""

        row = self.connection.execute(
            "SELECT logical_session_id, event_id, event_sequence, event_kind, canonical_payload"
            " FROM evidence_events WHERE event_id=?",
            (snapshot.event_id,),
        ).fetchone()
        if row is None:
            return False
        try:
            stored = self._stored_snapshot(row)
        except EvidenceModelError:  # pragma: no cover - a stored row is already validated
            return False
        return (
            classify_event_retry(stored, snapshot, target_sealed=False)
            is StoreDisposition.IDEMPOTENT
        )

    def _existing_epoch_disposition(
        self,
        command: CreateEpochV1,
        existing: list[tuple[object, ...]],
    ) -> StoreDisposition:
        """Only a provably exact repeat is idempotent; anything else refuses."""

        retained = self._accepted_create
        if retained is None or command != retained:
            # Two control fields are not reconstructible from the schema, and the
            # plan forbids persisting them merely for retries, so an unproven repeat
            # fails closed rather than claiming idempotency.
            return StoreDisposition.REJECTED_STATE
        if [str(row[0]) for row in existing] != [command.consent_epoch_id]:
            return StoreDisposition.REJECTED_STATE
        installation = self.connection.execute(
            "SELECT installation_id FROM producer_installation WHERE singleton=1"
        ).fetchone()
        if installation is None or str(installation[0]) != command.installation_id:
            return StoreDisposition.REJECTED_STATE
        session = self.connection.execute(
            "SELECT consent_epoch_id, producer_instance_id, consent_version"
            " FROM evidence_sessions WHERE logical_session_id=?",
            (command.logical_session_id,),
        ).fetchone()
        if session is None or (str(session[0]), str(session[1]), str(session[2])) != (
            command.consent_epoch_id,
            command.producer_instance_id,
            command.consent_version,
        ):
            return StoreDisposition.REJECTED_STATE
        if not all(
            self._stored_matches_snapshot(snapshot)
            for snapshot in (command.session_opened, command.binding_opened)
        ):
            return StoreDisposition.REJECTED_STATE
        return StoreDisposition.IDEMPOTENT

    def _sealed_epoch_disposition(
        self,
        command: SealEpochV1,
        sealed: tuple[object, ...],
    ) -> StoreDisposition:
        """Only a provably exact seal repeat is idempotent."""

        retained = self._accepted_seal
        if retained is None or command != retained:
            return StoreDisposition.REJECTED_STATE
        if int(cast(int, sealed[1])) != command.final_event_sequence:
            return StoreDisposition.REJECTED_STATE
        if not self._stored_matches_snapshot(command.snapshot):
            return StoreDisposition.REJECTED_STATE
        head = self.connection.execute(
            "SELECT head_hash FROM evidence_sessions WHERE logical_session_id=?",
            (command.logical_session_id,),
        ).fetchone()
        record = self.connection.execute(
            "SELECT record_hash FROM evidence_events WHERE event_id=?",
            (command.snapshot.event_id,),
        ).fetchone()
        if head is None or record is None or str(head[0]) != str(record[0]):
            return StoreDisposition.REJECTED_STATE
        epoch = self.connection.execute(
            "SELECT state FROM consent_epochs WHERE consent_epoch_id=?",
            (command.consent_epoch_id,),
        ).fetchone()
        if epoch is None or str(epoch[0]) != "closed":
            return StoreDisposition.REJECTED_STATE
        preceding = self.connection.execute(
            "SELECT canonical_payload FROM evidence_events WHERE logical_session_id=?"
            " AND event_sequence=? AND event_kind=?",
            (
                command.logical_session_id,
                command.final_event_sequence - 1,
                EventKind.BINDING_CLOSED.value,
            ),
        ).fetchone()
        if preceding is None:
            return StoreDisposition.REJECTED_STATE
        closed = json.loads(str(preceding[0]))
        if type(closed) is not dict or closed.get("binding_id") != command.binding_id:
            return StoreDisposition.REJECTED_STATE
        return StoreDisposition.IDEMPOTENT

    def _existing_rollover_disposition(self, command: RolloverSessionV1) -> StoreDisposition:
        """Only a provably exact rollover repeat is idempotent."""

        retained = self._accepted_rollover
        if retained is None or command != retained:
            return StoreDisposition.REJECTED_STATE
        predecessor = self.connection.execute(
            "SELECT state, final_event_sequence, head_hash FROM evidence_sessions"
            " WHERE logical_session_id=?",
            (command.predecessor_logical_session_id,),
        ).fetchone()
        if predecessor is None or str(predecessor[0]) != "sealed":
            return StoreDisposition.REJECTED_STATE
        if int(cast(int, predecessor[1])) != command.predecessor_final_event_sequence:
            return StoreDisposition.REJECTED_STATE
        successor = self.connection.execute(
            "SELECT consent_epoch_id, expires_at_utc, state, producer_instance_id"
            " FROM evidence_sessions WHERE logical_session_id=?",
            (command.successor_logical_session_id,),
        ).fetchone()
        if successor is None or (
            str(successor[0]),
            str(successor[1]),
            str(successor[2]),
        ) != (
            command.consent_epoch_id,
            command.successor_expires_at_utc,
            "open",
        ):
            return StoreDisposition.REJECTED_STATE
        if not all(self._stored_matches_snapshot(snapshot) for snapshot in command.snapshots):
            return StoreDisposition.REJECTED_STATE
        seal_record = self.connection.execute(
            "SELECT record_hash FROM evidence_events WHERE event_id=?",
            (command.snapshots[1].event_id,),
        ).fetchone()
        if seal_record is None or str(predecessor[2]) != str(seal_record[0]):
            return StoreDisposition.REJECTED_STATE
        return StoreDisposition.IDEMPOTENT

    def _write_chain(
        self,
        *snapshots: tuple[EvidenceSnapshotV1, bytes],
        recorded_at: str,
        session_id: str,
    ) -> str | None:
        """Insert a contiguous run of records and return the last record hash."""

        previous = self._previous_hash(session_id)
        total = 0
        count = 0
        for snapshot, payload in snapshots:
            payload_hash, record_hash = self._record_hash(
                snapshot,
                recorded_at=recorded_at,
                payload=payload,
                previous_hash=previous,
            )
            self.connection.execute(
                _EVENT_COLUMNS,
                (
                    snapshot.event_id,
                    snapshot.logical_session_id,
                    snapshot.event_sequence,
                    snapshot.event_kind.value,
                    recorded_at,
                    payload.decode("utf-8"),
                    payload_hash,
                    previous,
                    record_hash,
                    len(payload),
                ),
            )
            previous = record_hash
            total += len(payload)
            count += 1
        self.connection.execute(
            "UPDATE evidence_sessions SET event_count=event_count+?,"
            " canonical_bytes=canonical_bytes+? WHERE logical_session_id=?",
            (count, total, session_id),
        )
        return previous

    def append_record(self, item: QueuedEvidenceRecordV1) -> StoreDisposition:
        """Append one queued ordinary or terminal record."""

        if type(item) is not QueuedEvidenceRecordV1:
            return (
                StoreDisposition.FAULTED
                if self._sticky_fault is not None
                else StoreDisposition.REJECTED_STATE
            )
        result = self._append(item.snapshot)
        if result in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT):
            self._last_admission_ordinal = max(
                self._last_admission_ordinal,
                item.admission_ordinal,
            )
        return result

    def append_binding_close(self, command: BindingCloseV1) -> StoreDisposition:
        """Append the single current binding's close record."""

        if type(command) is not BindingCloseV1:
            return (
                StoreDisposition.FAULTED
                if self._sticky_fault is not None
                else StoreDisposition.REJECTED_STATE
            )
        result = self._append(command.snapshot)
        if result in (StoreDisposition.COMMITTED, StoreDisposition.IDEMPOTENT):
            self._last_admission_ordinal = max(
                self._last_admission_ordinal,
                command.admission_ordinal,
            )
        return result

    def seal_epoch(self, command: SealEpochV1) -> StoreDisposition:
        """Append the seal record and atomically seal the session and epoch."""

        if self._sticky_fault is not None:
            return StoreDisposition.FAULTED
        if type(command) is not SealEpochV1 or self._connection is None:
            return StoreDisposition.REJECTED_STATE
        if (
            self._owner_state is OwnerState.RECOVERY_ONLY
            or self._capture_state in {CaptureState.REVOKED_PURGING, CaptureState.PURGE_FAILED}
        ):
            return StoreDisposition.REJECTED_STATE
        connection = self.connection
        try:
            sealed = connection.execute(
                "SELECT state, final_event_sequence FROM evidence_sessions"
                " WHERE logical_session_id=?",
                (command.logical_session_id,),
            ).fetchone()
            if sealed is not None and str(sealed[0]) == "sealed":
                return self._sealed_epoch_disposition(command, sealed)
            preceding = connection.execute(
                "SELECT event_kind FROM evidence_events WHERE logical_session_id=? AND"
                " event_sequence=?",
                (command.logical_session_id, command.final_event_sequence - 1),
            ).fetchone()
            if preceding is None or str(preceding[0]) != EventKind.BINDING_CLOSED.value:
                return StoreDisposition.REJECTED_STATE
            # §3.4: the seal sequence must equal pre-append event_count + 1. Checking
            # it here keeps a caller-shape mismatch a plain refusal instead of letting
            # the relational CHECK turn it into a sticky writer fault.
            counts = connection.execute(
                "SELECT state, event_count FROM evidence_sessions WHERE logical_session_id=?",
                (command.logical_session_id,),
            ).fetchone()
            if (
                counts is None
                or str(counts[0]) != "open"
                or int(counts[1]) + 1 != command.final_event_sequence
            ):
                return StoreDisposition.REJECTED_STATE
            plan = self._plan_append(command.snapshot)
        except _Refused as refusal:
            if refusal.taint is not None and self._current_session_id is not None:
                self._taint_session(self._current_session_id, refusal.taint)
            return refusal.disposition
        except _ClockRolledBack:
            return StoreDisposition.FAULTED
        except EvidenceStorageError as exc:
            self._latch(exc.fault)
            return StoreDisposition.FAULTED
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        try:
            connection.execute("BEGIN IMMEDIATE")
            validate_store_schema(connection)
            epoch = connection.execute(
                "SELECT state FROM consent_epochs WHERE consent_epoch_id=?",
                (command.consent_epoch_id,),
            ).fetchone()
            current = connection.execute(
                "SELECT state, event_count, canonical_bytes, expires_at_utc"
                " FROM evidence_sessions WHERE logical_session_id=? AND consent_epoch_id=?",
                (command.logical_session_id, command.consent_epoch_id),
            ).fetchone()
            aggregate = connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(canonical_bytes),0) FROM evidence_events"
                " WHERE logical_session_id=?",
                (command.logical_session_id,),
            ).fetchone()
            conflicts = connection.execute(
                "SELECT COUNT(*) FROM evidence_conflicts WHERE logical_session_id=?",
                (command.logical_session_id,),
            ).fetchone()
            pending = connection.execute(
                "SELECT COUNT(*) FROM erasure_requests WHERE"
                " (scope_kind='session' AND scope_key=?)"
                " OR (scope_kind='consent_epoch' AND scope_key=?)",
                (command.logical_session_id, command.consent_epoch_id),
            ).fetchone()
            if (
                epoch != ("active",)
                or current is None
                or str(current[0]) != "open"
                or int(current[1]) + 1 != command.final_event_sequence
                or aggregate != (int(current[1]), int(current[2]))
                or conflicts != (0,)
                or pending != (0,)
                or plan.recorded_at_utc >= str(current[3])
            ):
                connection.rollback()
                return StoreDisposition.REJECTED_STATE
            validate_event_sequence(
                (*self._validated_session_history(command.logical_session_id), command.snapshot)
            )
            self._check_quotas(
                event_count=int(current[1]),
                canonical_bytes=int(current[2]),
                payload_length=len(plan.payload),
            )
            self._insert_event(plan)
            connection.execute(
                "UPDATE evidence_sessions SET state='sealed', final_event_sequence=?,"
                " head_hash=? WHERE logical_session_id=?",
                (command.final_event_sequence, plan.record_hash, command.logical_session_id),
            )
            connection.execute(
                "UPDATE consent_epochs SET state='closed', closed_at_utc=?"
                " WHERE consent_epoch_id=?",
                (plan.recorded_at_utc, command.consent_epoch_id),
            )
            self._advance_high_water(plan.recorded_at_utc)
            connection.commit()
        except _Refused as refusal:
            self._rollback()
            return refusal.disposition
        except (EvidenceModelError, EvidenceFrameError, TypeError, ValueError):
            self._rollback()
            return StoreDisposition.REJECTED_STATE
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        self._accepted_seal = command
        self._last_admission_ordinal = max(
            self._last_admission_ordinal,
            command.admission_ordinal,
        )
        self._capture_state = CaptureState.IDLE
        return StoreDisposition.COMMITTED

    def rollover_session(self, command: RolloverSessionV1) -> StoreDisposition:
        """Atomically seal the predecessor and open the successor in one transaction."""

        if self._sticky_fault is not None:
            return StoreDisposition.FAULTED
        if type(command) is not RolloverSessionV1 or self._connection is None:
            return StoreDisposition.REJECTED_STATE
        if (
            self._owner_state is OwnerState.RECOVERY_ONLY
            or self._capture_state in {CaptureState.REVOKED_PURGING, CaptureState.PURGE_FAILED}
        ):
            return StoreDisposition.REJECTED_STATE
        connection = self.connection
        try:
            successor = connection.execute(
                "SELECT state FROM evidence_sessions WHERE logical_session_id=?",
                (command.successor_logical_session_id,),
            ).fetchone()
            if successor is not None:
                return self._existing_rollover_disposition(command)
            predecessor = connection.execute(
                "SELECT state, event_count, canonical_bytes FROM evidence_sessions"
                " WHERE logical_session_id=?",
                (command.predecessor_logical_session_id,),
            ).fetchone()
            if (
                predecessor is None
                or str(predecessor[0]) != "open"
                or command.predecessor_logical_session_id != self._current_session_id
            ):
                return StoreDisposition.REJECTED_STATE
            close, seal, opened, binding = command.snapshots
            if close.event_sequence != int(predecessor[1]) + 1:
                return StoreDisposition.REJECTED_STATE
            prepared = [self._exact_snapshot(item) for item in command.snapshots]
            for _snapshot, payload in prepared:
                check_canonical_payload_size(payload)
            self._check_quotas(
                event_count=int(predecessor[1]) + 1,
                canonical_bytes=int(predecessor[2]),
                payload_length=sum(len(payload) for _snapshot, payload in prepared),
            )
            recorded_at = format_canonical_utc(self._sample_now())
        except _Refused as refusal:
            if refusal.taint is not None:
                self._taint_session(command.predecessor_logical_session_id, refusal.taint)
            return refusal.disposition
        except _ClockRolledBack:
            return StoreDisposition.FAULTED
        except (EvidenceModelError, EvidenceFrameError):
            return StoreDisposition.REJECTED_STATE
        except EvidenceStorageError as exc:
            self._latch(exc.fault)
            return StoreDisposition.FAULTED
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        try:
            connection.execute("BEGIN IMMEDIATE")
            seal_hash = self._write_chain(
                prepared[0],
                prepared[1],
                recorded_at=recorded_at,
                session_id=command.predecessor_logical_session_id,
            )
            connection.execute(
                "UPDATE evidence_sessions SET state='sealed', final_event_sequence=?,"
                " head_hash=? WHERE logical_session_id=?",
                (
                    command.predecessor_final_event_sequence,
                    seal_hash,
                    command.predecessor_logical_session_id,
                ),
            )
            connection.execute(
                "INSERT INTO evidence_sessions (logical_session_id, consent_epoch_id,"
                " producer_instance_id, opened_at_utc, expires_at_utc, consent_version, state,"
                " final_event_sequence, head_hash, canonical_bytes, event_count, taint_code)"
                " VALUES (?,?,?,?,?,?, 'open', NULL, NULL, 0, 0, NULL)",
                (
                    command.successor_logical_session_id,
                    command.consent_epoch_id,
                    opened.producer_instance_id,
                    recorded_at,
                    command.successor_expires_at_utc,
                    CONSENT_VERSION,
                ),
            )
            self._write_chain(
                prepared[2],
                prepared[3],
                recorded_at=recorded_at,
                session_id=command.successor_logical_session_id,
            )
            self._advance_high_water(recorded_at)
            connection.commit()
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return StoreDisposition.FAULTED
        self._current_session_id = command.successor_logical_session_id
        self._accepted_rollover = command
        self._last_admission_ordinal = max(
            self._last_admission_ordinal,
            command.admission_ordinal,
        )
        del seal, binding
        return StoreDisposition.COMMITTED

    def drain_and_close(self, command: DrainAndStopV1) -> DrainDisposition:
        """Close this owner's connection and release its sentinel lease exactly once."""

        if type(command) is not DrainAndStopV1:
            return DrainDisposition.ALREADY_QUEUED
        if command.owner_generation != self._owner_generation:
            # The drain lane belongs to another owner generation; not accepted here.
            return DrainDisposition.ALREADY_QUEUED
        if self._connection is not None:
            try:
                pending_revoke = self.connection.execute(
                    "SELECT 1 FROM erasure_requests WHERE reason_code='revoked' LIMIT 1"
                ).fetchone()
                if (
                    command.final_admission_ordinal < self._last_admission_ordinal
                    or pending_revoke is not None
                ):
                    return DrainDisposition.ALREADY_QUEUED
            except sqlite3.Error:
                self._latch(WriterFault.SQLITE_FAULT)
                return DrainDisposition.ALREADY_QUEUED
        if not self._close_for_drain():
            return DrainDisposition.ALREADY_QUEUED
        self._owner_state = OwnerState.STOPPED
        if self._capture_state is not CaptureState.FAULTED:
            self._capture_state = CaptureState.UNAVAILABLE
        return DrainDisposition.STOPPED

    def active_session_expiry(self, logical_session_id: str) -> str | None:
        """Return the exact persisted expiry anchor for one active owned session."""

        try:
            validate_canonical_uuid4(logical_session_id, field_name="logical_session_id")
            row = self.connection.execute(
                "SELECT expires_at_utc FROM evidence_sessions"
                " WHERE logical_session_id=? AND state='open'",
                (logical_session_id,),
            ).fetchone()
        except (sqlite3.Error, EvidenceModelError, EvidenceStorageError):
            return None
        return None if row is None else str(row[0])

    def close_owner_marker(self) -> bool:
        """Provide a content-free successful job before owner-thread shutdown."""

        return True

    def diagnostics(self) -> EvidenceDiagnosticsV1:
        """Report exact closed owner state; never a path, identifier, or text."""

        pending_revoke = False
        purge_required = self._sticky_fault is WriterFault.PURGE_REQUIRED
        if self._connection is not None:
            try:
                pending_revoke = bool(
                    self._connection.execute(
                        "SELECT COUNT(*) FROM erasure_requests WHERE reason_code='revoked'"
                        " AND state<>'logical_deleted'"
                    ).fetchone()[0]
                )
                purge_required = purge_required or bool(
                    self._connection.execute(
                        "SELECT COALESCE(MAX(purge_required), 0) FROM producer_installation"
                    ).fetchone()[0]
                )
            except sqlite3.Error:  # pragma: no cover - diagnostics never raises
                self._latch(WriterFault.SQLITE_FAULT)
        return EvidenceDiagnosticsV1(
            protocol_version=1,
            owner_state=self._owner_state,
            capture_state=self._capture_state,
            sticky_fault=self._sticky_fault,
            queue_record_count=0,
            queue_canonical_bytes=0,
            active_lease_count=0,
            pending_revoke=pending_revoke,
            purge_required=purge_required,
        )

    # -- transport surface: intentionally Task-5, narrow and fail-closed ----------

    @staticmethod
    def _authority_fingerprint(authority: dict[str, object]) -> str:
        return hashlib.sha256(canonical_json_bytes(authority)).hexdigest()

    def _expire_authority_fingerprint(self, command: ExpireSessionV1) -> str:
        return self._authority_fingerprint(
            {
                "authority_kind": "expire_session_v1",
                "consent_epoch_id": command.consent_epoch_id,
                "erasure_request_id": command.erasure_request_id,
                "expires_at_utc": command.expires_at_utc,
                "last_admission_ordinal": command.last_admission_ordinal,
                "logical_session_id": command.logical_session_id,
                "mode": command.mode.value,
                "owner_generation": command.owner_generation,
                "protocol_version": command.protocol_version,
            }
        )

    def _maintenance_authority_fingerprint(
        self,
        command: MaintenanceV1,
        *,
        consent_epoch_id: str,
        expires_at_utc: str,
    ) -> str:
        return self._authority_fingerprint(
            {
                "artifact_manifest_version": command.artifact_manifest_version,
                "authority_kind": "maintenance_v1",
                "consent_epoch_id": consent_epoch_id,
                "deadline_admission_ordinal": command.deadline_admission_ordinal,
                "erasure_reason": command.erasure_reason.value,
                "erasure_request_id": command.erasure_request_id,
                "erasure_scope": command.erasure_scope.value,
                "expires_at_utc": expires_at_utc,
                "protocol_version": command.protocol_version,
                "scope_id": command.scope_id,
            }
        )

    def _schedule_recovery_privacy_erasures(self, *, cold_start: bool) -> bool:
        """Schedule every missing unclean/TTL authority before recovery availability."""

        del cold_start  # Public recovery is an explicit privacy boundary on either open path.
        try:
            now = format_canonical_utc(self._sample_now())
            connection = self.connection
            active_epochs = connection.execute(
                "SELECT consent_epoch_id FROM consent_epochs WHERE closed_at_utc IS NULL"
                " ORDER BY consent_epoch_id"
            ).fetchall()
            for (epoch_id,) in active_epochs:
                covered = connection.execute(
                    "SELECT 1 FROM erasure_requests WHERE scope_kind='consent_epoch'"
                    " AND scope_key=? AND reason_code='unclean_epoch' LIMIT 1",
                    (epoch_id,),
                ).fetchone()
                if covered is not None:
                    continue
                watermark = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(event_count),1) FROM evidence_sessions"
                        " WHERE consent_epoch_id=?",
                        (epoch_id,),
                    ).fetchone()[0]
                )
                request_id = self._uuid_factory()
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                    " requested_at_utc, reason_code, state, control_sequence,"
                    " control_fingerprint_hash, last_admission_ordinal,"
                    " final_admission_ordinal, resume_state, erased_session_count,"
                    " erased_event_count) VALUES"
                    " (?, 'consent_epoch', ?, ?, 'unclean_epoch', 'pending', NULL, NULL, ?,"
                    " NULL, NULL, NULL, NULL)",
                    (request_id, epoch_id, now, max(1, watermark)),
                )
                self._advance_high_water(now)
                connection.commit()
                if self._progress_erasure(request_id) not in (
                    PurgeDisposition.PURGE_COMPLETED,
                    PurgeDisposition.ALREADY_ABSENT,
                ):
                    return False

            candidates = connection.execute(
                "SELECT logical_session_id, consent_epoch_id, expires_at_utc, event_count"
                " FROM evidence_sessions ORDER BY logical_session_id"
            ).fetchall()
            for session_id, epoch_id, expires_at_utc, event_count in candidates:
                if now < str(expires_at_utc):
                    continue
                covered = connection.execute(
                    "SELECT 1 FROM erasure_requests WHERE scope_kind='session'"
                    " AND scope_key=? AND reason_code='ttl' LIMIT 1",
                    (session_id,),
                ).fetchone()
                if covered is not None:
                    continue
                request_id = self._uuid_factory()
                watermark = max(1, int(event_count))
                fingerprint = self._authority_fingerprint(
                    {
                        "authority_kind": "recovery_ttl_v1",
                        "consent_epoch_id": str(epoch_id),
                        "erasure_request_id": request_id,
                        "expires_at_utc": str(expires_at_utc),
                        "last_admission_ordinal": watermark,
                        "logical_session_id": str(session_id),
                    }
                )
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                    " requested_at_utc, reason_code, state, control_sequence,"
                    " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
                    " last_admission_ordinal, final_admission_ordinal, resume_state,"
                    " erased_session_count, erased_event_count)"
                    " VALUES (?, 'session', ?, ?, 'ttl', 'pending', NULL, ?, ?, ?, ?, NULL,"
                    " NULL, NULL, NULL)",
                    (
                        request_id,
                        session_id,
                        now,
                        fingerprint,
                        epoch_id,
                        expires_at_utc,
                        watermark,
                    ),
                )
                self._advance_high_water(now)
                connection.commit()
                if self._progress_erasure(request_id) not in (
                    PurgeDisposition.PURGE_COMPLETED,
                    PurgeDisposition.ALREADY_ABSENT,
                ):
                    return False
            unfinalized_revoke = connection.execute(
                "SELECT 1 FROM erasure_requests WHERE reason_code='revoked'"
                " AND final_admission_ordinal IS NULL LIMIT 1"
            ).fetchone()
            return unfinalized_revoke is None
        except _ClockRolledBack:
            self._rollback()
            raise
        except (EvidenceStorageError, sqlite3.Error, TypeError, ValueError):
            self._rollback()
            return False

    def _resume_recoverable_erasures(self) -> bool:
        """Settle every storage-complete checkpoint before recovery advertises availability.

        A scheduled revoke without its Task-5B3 final watermark is not deletion
        authority. Leave it pending; the separate unclean-epoch discovery must still
        settle process-death privacy before recovery returns.
        """

        try:
            candidates = self.connection.execute(
                "SELECT erasure_request_id, reason_code, final_admission_ordinal"
                " FROM erasure_requests ORDER BY requested_at_utc, erasure_request_id"
            ).fetchall()
        except sqlite3.Error:
            self._latch(WriterFault.SQLITE_FAULT)
            return False
        for request_id, reason_code, final_admission_ordinal in candidates:
            if str(reason_code) == "revoked" and final_admission_ordinal is None:
                continue
            result = self._progress_erasure(str(request_id))
            if result not in (
                PurgeDisposition.PURGE_COMPLETED,
                PurgeDisposition.ALREADY_ABSENT,
            ):
                return False
        return True

    def _mark_erasure_failed(self, erasure_request_id: str) -> None:
        """Persist a retryable checkpoint without weakening immutable authority."""

        self._rollback()
        try:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, resume_state FROM erasure_requests"
                " WHERE erasure_request_id=?",
                (erasure_request_id,),
            ).fetchone()
            if row is not None:
                state, resume_state = str(row[0]), row[1]
                if state == "pending":
                    connection.execute(
                        "UPDATE erasure_requests SET state='purge_failed',"
                        " resume_state='pending' WHERE erasure_request_id=?",
                        (erasure_request_id,),
                    )
                elif state == "logical_deleted":
                    connection.execute(
                        "UPDATE erasure_requests SET state='purge_failed',"
                        " resume_state='logical_deleted' WHERE erasure_request_id=?",
                        (erasure_request_id,),
                    )
                elif state != "purge_failed" or resume_state not in (
                    "pending",
                    "logical_deleted",
                ):
                    raise _store_corrupt("an erasure failure has no valid durable checkpoint")
            connection.commit()
            self._load_store_state()
        except (EvidenceStorageError, sqlite3.Error, TypeError, ValueError):
            self._rollback()

    def _progress_erasure(self, erasure_request_id: str) -> PurgeDisposition:
        """Complete one already-authorized logical erasure and durable tombstone.

        This private Task-5B primitive deliberately has no scheduler policy: callers
        must first persist an exact request.  The request row is the sole authority,
        and its immutable identity is copied verbatim into one terminal tombstone.
        """

        authority_validated = False
        try:
            validate_canonical_uuid4(erasure_request_id, field_name="erasure_request_id")
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            validate_store_schema(connection)
            request = connection.execute(
                "SELECT scope_kind, scope_key, requested_at_utc, reason_code, state,"
                " control_sequence, control_fingerprint_hash, ttl_consent_epoch_id,"
                " ttl_expires_at_utc, last_admission_ordinal, final_admission_ordinal,"
                " resume_state, erased_session_count,"
                " erased_event_count FROM erasure_requests WHERE erasure_request_id=?",
                (erasure_request_id,),
            ).fetchone()
            if request is None:
                tombstone = connection.execute(
                    "SELECT 1 FROM erasure_tombstones WHERE erasure_request_id=?",
                    (erasure_request_id,),
                ).fetchone()
                connection.commit()
                return (
                    PurgeDisposition.ALREADY_ABSENT
                    if tombstone is not None
                    else PurgeDisposition.PURGE_FAILED
                )

            (
                scope_kind,
                scope_key,
                requested_at_utc,
                reason_code,
                state,
                control_sequence,
                control_fingerprint_hash,
                ttl_consent_epoch_id,
                ttl_expires_at_utc,
                last_admission_ordinal,
                final_admission_ordinal,
                resume_state,
                erased_session_count,
                erased_event_count,
            ) = request
            if not all(
                type(value) is str
                for value in (scope_kind, scope_key, requested_at_utc, reason_code, state)
            ):
                raise _store_corrupt("an erasure request carries non-text authority")
            validate_canonical_utc(str(requested_at_utc))
            scope_kind = str(scope_kind)
            scope_key = str(scope_key)
            reason_code = str(reason_code)
            state = str(state)
            expected_scope = {
                "ttl": "session",
                "revoked": "consent_epoch",
                "unclean_epoch": "consent_epoch",
                "clock_rollback": "store",
            }.get(reason_code)
            if expected_scope != scope_kind:
                raise _store_corrupt("an erasure reason does not match its scope")
            if scope_kind == "store":
                if scope_key != "store":
                    raise _store_corrupt("a store erasure has a noncanonical scope key")
            else:
                validate_canonical_uuid4(scope_key, field_name="scope_key")
            if (
                type(last_admission_ordinal) is not int
                or not 1 <= last_admission_ordinal <= _MAX_SQLITE_INTEGER
            ):
                raise _store_corrupt("an erasure request has an invalid admission watermark")
            authorized_at: str | None = None
            if reason_code == "revoked":
                if (
                    type(control_sequence) is not int
                    or not 1 <= control_sequence <= _MAX_JS_INTEGER
                    or type(control_fingerprint_hash) is not str
                    or ttl_consent_epoch_id is not None
                    or ttl_expires_at_utc is not None
                ):
                    raise _store_corrupt("a revoke request has invalid control authority")
                validate_sha256_hex(
                    control_fingerprint_hash,
                    field_name="control_fingerprint_hash",
                )
                if final_admission_ordinal is not None and (
                    type(final_admission_ordinal) is not int
                    or not last_admission_ordinal <= final_admission_ordinal <= _MAX_SQLITE_INTEGER
                ):
                    raise _store_corrupt("a revoke request has an invalid final watermark")
            elif reason_code == "ttl":
                if (
                    control_sequence is not None
                    or type(control_fingerprint_hash) is not str
                    or type(ttl_consent_epoch_id) is not str
                    or type(ttl_expires_at_utc) is not str
                    or final_admission_ordinal is not None
                ):
                    raise _store_corrupt("a TTL request has invalid exact authority")
                validate_sha256_hex(
                    control_fingerprint_hash,
                    field_name="control_fingerprint_hash",
                )
                validate_canonical_uuid4(
                    ttl_consent_epoch_id,
                    field_name="ttl_consent_epoch_id",
                )
                validate_canonical_utc(
                    ttl_expires_at_utc,
                    field_name="ttl_expires_at_utc",
                )
                authorized_at = format_canonical_utc(self._sample_now())
                if (
                    requested_at_utc < ttl_expires_at_utc
                    or authorized_at < ttl_expires_at_utc
                ):
                    raise _store_corrupt("a TTL erasure lacks elapsed durable expiry authority")
                pending_phase = state == "pending" or (
                    state == "purge_failed" and resume_state == "pending"
                )
                if pending_phase:
                    target = connection.execute(
                        "SELECT consent_epoch_id, expires_at_utc FROM evidence_sessions"
                        " WHERE logical_session_id=?",
                        (scope_key,),
                    ).fetchone()
                    if target != (ttl_consent_epoch_id, ttl_expires_at_utc):
                        raise _store_corrupt("a TTL request does not match its durable target")
            elif (
                control_sequence is not None
                or control_fingerprint_hash is not None
                or ttl_consent_epoch_id is not None
                or ttl_expires_at_utc is not None
                or final_admission_ordinal is not None
            ):
                raise _store_corrupt("an epoch/store request carries foreign authority")
            if state == "pending":
                coherent_state = (
                    resume_state is None
                    and erased_session_count is None
                    and erased_event_count is None
                )
            elif state == "logical_deleted":
                coherent_state = (
                    resume_state is None
                    and type(erased_session_count) is int
                    and type(erased_event_count) is int
                    and erased_session_count >= 0
                    and erased_event_count >= 0
                )
            elif state == "purge_failed":
                coherent_state = (
                    resume_state == "pending"
                    and erased_session_count is None
                    and erased_event_count is None
                ) or (
                    resume_state == "logical_deleted"
                    and type(erased_session_count) is int
                    and type(erased_event_count) is int
                    and erased_session_count >= 0
                    and erased_event_count >= 0
                )
            else:
                coherent_state = False
            if not coherent_state:
                raise _store_corrupt("an erasure request has an incoherent durable state")
            authority_validated = True
            terminal_overlap = connection.execute(
                "SELECT 1 FROM erasure_tombstones WHERE erasure_request_id=?"
                " OR (scope_kind=? AND scope_key=? AND reason_code=?) LIMIT 1",
                (erasure_request_id, scope_kind, scope_key, reason_code),
            ).fetchone()
            if terminal_overlap is not None:
                connection.rollback()
                return PurgeDisposition.PURGE_FAILED
            if reason_code == "revoked" and final_admission_ordinal is None:
                # Task 5B3 owns final-watermark authority.  A merely scheduled
                # revoke is not deletion authority and remains byte-for-byte pending.
                connection.rollback()
                return PurgeDisposition.PURGE_FAILED

            if state == "purge_failed":
                if resume_state not in ("pending", "logical_deleted"):
                    raise _store_corrupt("an erasure request has no valid resume state")
                state = str(resume_state)
                connection.execute(
                    "UPDATE erasure_requests SET state=?, resume_state=NULL"
                    " WHERE erasure_request_id=?",
                    (state, erasure_request_id),
                )

            if state == "pending":
                if scope_kind == "session":
                    erased_session_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM evidence_sessions WHERE logical_session_id=?",
                            (scope_key,),
                        ).fetchone()[0]
                    )
                    if erased_session_count != 1:
                        raise _store_corrupt(
                            "a pending session erasure has no exact relational target"
                        )
                    erased_event_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM evidence_events WHERE logical_session_id=?",
                            (scope_key,),
                        ).fetchone()[0]
                    )
                    connection.execute(
                        "DELETE FROM evidence_sessions WHERE logical_session_id=?",
                        (scope_key,),
                    )
                elif scope_kind == "consent_epoch":
                    epoch_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM consent_epochs WHERE consent_epoch_id=?",
                            (scope_key,),
                        ).fetchone()[0]
                    )
                    if epoch_count != 1:
                        raise _store_corrupt(
                            "a pending epoch erasure has no exact relational target"
                        )
                    erased_session_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM evidence_sessions WHERE consent_epoch_id=?",
                            (scope_key,),
                        ).fetchone()[0]
                    )
                    erased_event_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM evidence_events WHERE logical_session_id IN"
                            " (SELECT logical_session_id FROM evidence_sessions"
                            " WHERE consent_epoch_id=?)",
                            (scope_key,),
                        ).fetchone()[0]
                    )
                    connection.execute(
                        "DELETE FROM evidence_sessions WHERE consent_epoch_id=?",
                        (scope_key,),
                    )
                    connection.execute(
                        "DELETE FROM consent_epochs WHERE consent_epoch_id=?",
                        (scope_key,),
                    )
                elif scope_kind == "store" and scope_key == "store":
                    installation_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM producer_installation"
                        ).fetchone()[0]
                    )
                    if installation_count != 1:
                        raise _store_corrupt(
                            "a pending store erasure has no exact relational target"
                        )
                    erased_session_count = int(
                        connection.execute("SELECT COUNT(*) FROM evidence_sessions").fetchone()[0]
                    )
                    erased_event_count = int(
                        connection.execute("SELECT COUNT(*) FROM evidence_events").fetchone()[0]
                    )
                    connection.execute("DELETE FROM evidence_sessions")
                    connection.execute("DELETE FROM consent_epochs")
                else:
                    raise _store_corrupt("an erasure request carries an invalid scope")
                connection.execute(
                    "UPDATE erasure_requests SET state='logical_deleted', resume_state=NULL,"
                    " erased_session_count=?, erased_event_count=?"
                    " WHERE erasure_request_id=?",
                    (erased_session_count, erased_event_count, erasure_request_id),
                )
                connection.commit()
            elif state == "logical_deleted":
                if type(erased_session_count) is not int or type(erased_event_count) is not int:
                    raise _store_corrupt("a logically deleted request has no immutable counts")
                connection.commit()
            else:
                raise _store_corrupt("an erasure request has an invalid progression state")

            # VACUUM is intentionally outside every transaction.  Revalidate the
            # retained parent capability and the exact V1 schema around maintenance.
            root = self._ensure_root()
            authority, _resolved = root.require_authority()
            authority.revalidate()
            authority.compare_fresh()
            check_maintenance_headroom(
                root.root,
                probe=self._probe,
                manifest=self._manifest,
            )
            connection.execute("VACUUM")
            authority.revalidate()
            authority.compare_fresh()
            audit_root_occupancy(root.root, manifest=self._manifest)
            validate_store_schema(connection)
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise _store_corrupt("logical erasure left a foreign-key violation")
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise _store_corrupt("logical erasure failed SQLite integrity validation")

            if scope_kind == "session":
                remaining = connection.execute(
                    "SELECT COUNT(*) FROM evidence_sessions WHERE logical_session_id=?",
                    (scope_key,),
                ).fetchone()[0]
            elif scope_kind == "consent_epoch":
                remaining = connection.execute(
                    "SELECT (SELECT COUNT(*) FROM consent_epochs WHERE consent_epoch_id=?)"
                    " + (SELECT COUNT(*) FROM evidence_sessions WHERE consent_epoch_id=?)",
                    (scope_key, scope_key),
                ).fetchone()[0]
            else:
                remaining = connection.execute(
                    "SELECT (SELECT COUNT(*) FROM consent_epochs)"
                    " + (SELECT COUNT(*) FROM evidence_sessions)"
                    " + (SELECT COUNT(*) FROM evidence_events)"
                ).fetchone()[0]
            if int(remaining) != 0:
                raise _store_corrupt("logical erasure left rows inside its authorized scope")

            erased_at = authorized_at or format_canonical_utc(self._sample_now())
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT state, resume_state, erased_session_count, erased_event_count"
                " FROM erasure_requests WHERE erasure_request_id=?",
                (erasure_request_id,),
            ).fetchone()
            if current != ("logical_deleted", None, erased_session_count, erased_event_count):
                raise _store_corrupt("erasure authority changed before tombstoning")
            connection.execute(
                "INSERT INTO erasure_tombstones (erasure_request_id, scope_kind, scope_key,"
                " reason_code, control_sequence, control_fingerprint_hash,"
                " ttl_consent_epoch_id, ttl_expires_at_utc, last_admission_ordinal,"
                " final_admission_ordinal, erased_at_utc, erased_session_count,"
                " erased_event_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    erasure_request_id,
                    scope_kind,
                    scope_key,
                    reason_code,
                    control_sequence,
                    control_fingerprint_hash,
                    ttl_consent_epoch_id,
                    ttl_expires_at_utc,
                    last_admission_ordinal,
                    final_admission_ordinal,
                    erased_at,
                    erased_session_count,
                    erased_event_count,
                ),
            )
            connection.execute(
                "DELETE FROM erasure_requests WHERE erasure_request_id=?",
                (erasure_request_id,),
            )
            self._advance_high_water(erased_at)
            connection.commit()
            self._load_store_state()
            return PurgeDisposition.PURGE_COMPLETED
        except (
            _ClockRolledBack,
            EvidenceModelError,
            EvidenceStorageError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ):
            if authority_validated:
                self._mark_erasure_failed(erasure_request_id)
            else:
                self._rollback()
                self._latch(WriterFault.STORE_CORRUPT)
            return PurgeDisposition.PURGE_FAILED

    def expire_session(self, command: ExpireSessionV1) -> PurgeDisposition:
        """Persist exact TTL authority, then complete its shared durable erasure."""

        if (
            type(command) is not ExpireSessionV1
            or self._sticky_fault is not None
            or self._connection is None
            or command.owner_generation != self._owner_generation
            or command.mode.value != "erase_stuck"
        ):
            return PurgeDisposition.PURGE_FAILED
        try:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            fingerprint = self._expire_authority_fingerprint(command)
            expected = (
                "session",
                command.logical_session_id,
                "ttl",
                command.last_admission_ordinal,
                fingerprint,
                command.consent_epoch_id,
                command.expires_at_utc,
            )
            existing = connection.execute(
                "SELECT scope_kind, scope_key, reason_code, last_admission_ordinal,"
                " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc"
                " FROM erasure_requests WHERE erasure_request_id=?",
                (command.erasure_request_id,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                if tuple(existing) != expected:
                    return PurgeDisposition.PURGE_FAILED
            else:
                completed = connection.execute(
                    "SELECT scope_kind, scope_key, reason_code, last_admission_ordinal,"
                    " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc"
                    " FROM erasure_tombstones WHERE erasure_request_id=?",
                    (command.erasure_request_id,),
                ).fetchone()
                if completed is not None:
                    connection.commit()
                    return (
                        PurgeDisposition.ALREADY_ABSENT
                        if tuple(completed) == expected
                        else PurgeDisposition.PURGE_FAILED
                    )
                target = connection.execute(
                    "SELECT consent_epoch_id, expires_at_utc FROM evidence_sessions"
                    " WHERE logical_session_id=?",
                    (command.logical_session_id,),
                ).fetchone()
                if target is None or tuple(target) != (
                    command.consent_epoch_id,
                    command.expires_at_utc,
                ):
                    connection.rollback()
                    return PurgeDisposition.PURGE_FAILED
                requested_at = format_canonical_utc(self._sample_now())
                if requested_at < command.expires_at_utc:
                    connection.rollback()
                    return PurgeDisposition.PURGE_FAILED
                connection.execute(
                    "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                    " requested_at_utc, reason_code, state, control_sequence,"
                    " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
                    " last_admission_ordinal, final_admission_ordinal, resume_state,"
                    " erased_session_count, erased_event_count)"
                    " VALUES (?, 'session', ?, ?, 'ttl', 'pending', NULL, ?, ?, ?, ?, NULL,"
                    " NULL, NULL, NULL)",
                    (
                        command.erasure_request_id,
                        command.logical_session_id,
                        requested_at,
                        fingerprint,
                        command.consent_epoch_id,
                        command.expires_at_utc,
                        command.last_admission_ordinal,
                    ),
                )
                self._advance_high_water(requested_at)
                connection.commit()
        except _ClockRolledBack:
            self._rollback()
            return PurgeDisposition.PURGE_FAILED
        except (EvidenceModelError, EvidenceStorageError, sqlite3.Error, TypeError, ValueError):
            self._rollback()
            return PurgeDisposition.PURGE_FAILED
        return self._progress_erasure(command.erasure_request_id)

    def commit_revoke_request(self, command: RevokeRequestV1) -> RevokeDisposition:
        """Durably schedule one exact revoke request without erasing evidence."""

        if type(command) is not RevokeRequestV1:
            return RevokeDisposition.WRITER_FAULT
        if self._sticky_fault is not None or self._connection is None:
            return RevokeDisposition.WRITER_FAULT
        try:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT scope_kind, scope_key, reason_code, state, control_sequence,"
                " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
                " erased_session_count, erased_event_count, resume_state"
                " FROM erasure_requests WHERE erasure_request_id=?",
                (command.erasure_request_id,),
            ).fetchone()
            expected = (
                "consent_epoch",
                command.consent_epoch_id,
                "revoked",
                "pending",
                command.control_sequence,
                command.control_fingerprint_hash,
                command.last_admission_ordinal,
                None,
                None,
                None,
                None,
            )
            if existing is not None:
                connection.commit()
                return (
                    RevokeDisposition.ALREADY_SCHEDULED
                    if tuple(existing) == expected
                    else RevokeDisposition.WRITER_FAULT
                )
            epoch = connection.execute(
                "SELECT state FROM consent_epochs WHERE consent_epoch_id=?",
                (command.consent_epoch_id,),
            ).fetchone()
            if epoch is None or str(epoch[0]) != "active":
                connection.rollback()
                return RevokeDisposition.WRITER_FAULT
            requested_at = format_canonical_utc(self._sample_now())
            connection.execute(
                "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                " requested_at_utc, reason_code, state, control_sequence,"
                " control_fingerprint_hash, last_admission_ordinal, final_admission_ordinal,"
                " resume_state, erased_session_count, erased_event_count)"
                " VALUES (?, 'consent_epoch', ?, ?, 'revoked', 'pending', ?, ?, ?, NULL,"
                " NULL, NULL, NULL)",
                (
                    command.erasure_request_id,
                    command.consent_epoch_id,
                    requested_at,
                    command.control_sequence,
                    command.control_fingerprint_hash,
                    command.last_admission_ordinal,
                ),
            )
            connection.execute(
                "UPDATE consent_epochs SET state='revoked', closed_at_utc=?"
                " WHERE consent_epoch_id=? AND state='active'",
                (requested_at, command.consent_epoch_id),
            )
            self._advance_high_water(requested_at)
            connection.commit()
        except _ClockRolledBack:
            self._rollback()
            return RevokeDisposition.WRITER_FAULT
        except (sqlite3.Error, EvidenceModelError, EvidenceStorageError):
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return RevokeDisposition.WRITER_FAULT
        self._capture_state = CaptureState.REVOKED_PURGING
        return RevokeDisposition.REVOKE_DURABLY_SCHEDULED

    def finalize_revoke(self, command: RevokeFinalizeV1) -> RevokeDisposition:
        """Persist exact final authority, then progress the durable revoke request."""

        if type(command) is not RevokeFinalizeV1:
            return RevokeDisposition.WRITER_FAULT
        if self._sticky_fault is not None or self._connection is None:
            return RevokeDisposition.WRITER_FAULT
        try:
            if (
                type(command.protocol_version) is not int
                or command.protocol_version != 1
                or type(command.erasure_request_id) is not str
                or type(command.consent_epoch_id) is not str
                or type(command.control_sequence) is not int
                or not 1 <= command.control_sequence <= _MAX_JS_INTEGER
                or type(command.control_fingerprint_hash) is not str
                or type(command.final_admission_ordinal) is not int
                or not 1 <= command.final_admission_ordinal <= _MAX_SQLITE_INTEGER
            ):
                return RevokeDisposition.WRITER_FAULT
            validate_canonical_uuid4(
                command.erasure_request_id,
                field_name="erasure_request_id",
            )
            validate_canonical_uuid4(
                command.consent_epoch_id,
                field_name="consent_epoch_id",
            )
            validate_sha256_hex(
                command.control_fingerprint_hash,
                field_name="control_fingerprint_hash",
            )
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            validate_store_schema(connection)
            self._validate_erasure_requests()
            request = connection.execute(
                "SELECT scope_kind, scope_key, reason_code, control_sequence,"
                " control_fingerprint_hash, last_admission_ordinal,"
                " final_admission_ordinal FROM erasure_requests"
                " WHERE erasure_request_id=?",
                (command.erasure_request_id,),
            ).fetchone()
            if request is None:
                self._validate_erasure_tombstones()
                tombstone = connection.execute(
                    "SELECT scope_kind, scope_key, reason_code, control_sequence,"
                    " control_fingerprint_hash, final_admission_ordinal"
                    " FROM erasure_tombstones WHERE erasure_request_id=?",
                    (command.erasure_request_id,),
                ).fetchone()
                connection.commit()
                expected_terminal = (
                    "consent_epoch",
                    command.consent_epoch_id,
                    "revoked",
                    command.control_sequence,
                    command.control_fingerprint_hash,
                    command.final_admission_ordinal,
                )
                return (
                    RevokeDisposition.PURGE_COMPLETED
                    if tombstone is not None and tuple(tombstone) == expected_terminal
                    else RevokeDisposition.WRITER_FAULT
                )
            (
                scope_kind,
                scope_key,
                reason_code,
                control_sequence,
                control_fingerprint_hash,
                last_admission_ordinal,
                final_admission_ordinal,
            ) = request
            expected_authority = (
                "consent_epoch",
                command.consent_epoch_id,
                "revoked",
                command.control_sequence,
                command.control_fingerprint_hash,
            )
            if (
                (
                    scope_kind,
                    scope_key,
                    reason_code,
                    control_sequence,
                    control_fingerprint_hash,
                )
                != expected_authority
                or type(last_admission_ordinal) is not int
                or command.final_admission_ordinal < last_admission_ordinal
                or (
                    final_admission_ordinal is not None
                    and final_admission_ordinal != command.final_admission_ordinal
                )
            ):
                connection.rollback()
                return RevokeDisposition.WRITER_FAULT
            if final_admission_ordinal is None:
                connection.execute(
                    "UPDATE erasure_requests SET final_admission_ordinal=?"
                    " WHERE erasure_request_id=? AND final_admission_ordinal IS NULL",
                    (command.final_admission_ordinal, command.erasure_request_id),
                )
            connection.commit()
        except (EvidenceModelError, TypeError, ValueError):
            self._rollback()
            return RevokeDisposition.WRITER_FAULT
        except EvidenceStorageError as exc:
            self._rollback()
            self._latch(exc.fault)
            return RevokeDisposition.WRITER_FAULT
        except sqlite3.Error:
            self._rollback()
            self._latch(WriterFault.SQLITE_FAULT)
            return RevokeDisposition.WRITER_FAULT

        result = self._progress_erasure(command.erasure_request_id)
        return (
            RevokeDisposition.PURGE_COMPLETED
            if result in (PurgeDisposition.PURGE_COMPLETED, PurgeDisposition.ALREADY_ABSENT)
            else RevokeDisposition.PURGE_FAILED
        )

    def run_maintenance(self, command: MaintenanceV1) -> PurgeDisposition:
        """Schedule and settle one elapsed TTL erasure through the shared primitive."""

        if (
            type(command) is not MaintenanceV1
            or self._sticky_fault is not None
            or self._connection is None
            or command.artifact_manifest_version != self._manifest.version
            or command.erasure_reason.value != "ttl"
            or command.erasure_scope.value != "session"
        ):
            return PurgeDisposition.PURGE_FAILED
        try:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT scope_kind, scope_key, reason_code, last_admission_ordinal,"
                " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc"
                " FROM erasure_requests WHERE erasure_request_id=?",
                (command.erasure_request_id,),
            ).fetchone()
            if existing is not None:
                fingerprint = self._maintenance_authority_fingerprint(
                    command,
                    consent_epoch_id=str(existing[5]),
                    expires_at_utc=str(existing[6]),
                )
                expected = (
                    "session",
                    command.scope_id,
                    "ttl",
                    command.deadline_admission_ordinal,
                    fingerprint,
                    existing[5],
                    existing[6],
                )
                connection.commit()
                if tuple(existing) != expected:
                    return PurgeDisposition.PURGE_FAILED
            else:
                completed = connection.execute(
                    "SELECT scope_kind, scope_key, reason_code, last_admission_ordinal,"
                    " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc"
                    " FROM erasure_tombstones WHERE erasure_request_id=?",
                    (command.erasure_request_id,),
                ).fetchone()
                if completed is not None:
                    fingerprint = self._maintenance_authority_fingerprint(
                        command,
                        consent_epoch_id=str(completed[5]),
                        expires_at_utc=str(completed[6]),
                    )
                    expected = (
                        "session",
                        command.scope_id,
                        "ttl",
                        command.deadline_admission_ordinal,
                        fingerprint,
                        completed[5],
                        completed[6],
                    )
                    connection.commit()
                    return (
                        PurgeDisposition.ALREADY_ABSENT
                        if tuple(completed) == expected
                        else PurgeDisposition.PURGE_FAILED
                    )
                target = connection.execute(
                    "SELECT consent_epoch_id, expires_at_utc FROM evidence_sessions"
                    " WHERE logical_session_id=?",
                    (command.scope_id,),
                ).fetchone()
                if target is None:
                    connection.rollback()
                    return PurgeDisposition.PURGE_FAILED
                requested_at = format_canonical_utc(self._sample_now())
                if requested_at < str(target[1]):
                    connection.rollback()
                    return PurgeDisposition.PURGE_FAILED
                fingerprint = self._maintenance_authority_fingerprint(
                    command,
                    consent_epoch_id=str(target[0]),
                    expires_at_utc=str(target[1]),
                )
                connection.execute(
                    "INSERT INTO erasure_requests (erasure_request_id, scope_kind, scope_key,"
                    " requested_at_utc, reason_code, state, control_sequence,"
                    " control_fingerprint_hash, ttl_consent_epoch_id, ttl_expires_at_utc,"
                    " last_admission_ordinal, final_admission_ordinal, resume_state,"
                    " erased_session_count, erased_event_count)"
                    " VALUES (?, 'session', ?, ?, 'ttl', 'pending', NULL, ?, ?, ?, ?, NULL,"
                    " NULL, NULL, NULL)",
                    (
                        command.erasure_request_id,
                        command.scope_id,
                        requested_at,
                        fingerprint,
                        target[0],
                        target[1],
                        command.deadline_admission_ordinal,
                    ),
                )
                self._advance_high_water(requested_at)
                connection.commit()
        except _ClockRolledBack:
            self._rollback()
            return PurgeDisposition.PURGE_FAILED
        except (EvidenceModelError, EvidenceStorageError, sqlite3.Error, TypeError, ValueError):
            self._rollback()
            return PurgeDisposition.PURGE_FAILED
        return self._progress_erasure(command.erasure_request_id)

    def _purge_full_store_with_lease(
        self, command: FullPurgeV1
    ) -> tuple[PurgeDisposition, bool]:
        """Attempt one full purge and report whether the command gained authority."""

        purge_authorized = False
        try:
            root = self._ensure_root()
            audit_root_occupancy(root.root, manifest=self._manifest)
            authority, resolved = root.require_authority()
            if not resolved.root_marker.is_file():
                return PurgeDisposition.PURGE_FAILED, purge_authorized
            parse_root_marker(resolved.root_marker.read_bytes())
            if not resolved.sentinel.is_file():
                return PurgeDisposition.PURGE_FAILED, purge_authorized
            self._acquire_sentinel(resolved.sentinel)
            active = decode_sentinel_image(self._read_sentinel()).active
            if (
                active.state is SentinelState.CLEAR
                and not any(os.path.lexists(path) for path in resolved.deletable)
            ):
                return PurgeDisposition.ALREADY_ABSENT, purge_authorized
            if active.state is SentinelState.CLEAR:
                if command.sentinel_state is not SentinelState.FULL_PURGE_PENDING:
                    return PurgeDisposition.PURGE_FAILED, purge_authorized
                purge_authorized = True
                authority.revalidate()
                authority.compare_fresh()
                self._transition_sentinel(
                    SentinelState.FULL_PURGE_PENDING,
                    state_generation_id=command.full_purge_generation_id,
                )
                active = decode_sentinel_image(self._read_sentinel()).active
            if (
                active.state is not command.sentinel_state
                or active.state_generation_id != command.full_purge_generation_id
            ):
                return PurgeDisposition.PURGE_FAILED, purge_authorized
            purge_authorized = True
            result = (
                PurgeDisposition.PURGE_COMPLETED
                if self._purge_pending_artifacts(
                    root,
                    state=command.sentinel_state,
                    generation_id=command.full_purge_generation_id,
                )
                else PurgeDisposition.PURGE_FAILED
            )
            return result, purge_authorized
        except EvidenceStorageError as exc:
            if exc.fault is WriterFault.OWNERSHIP_UNAVAILABLE:
                return PurgeDisposition.OWNERSHIP_UNAVAILABLE, purge_authorized
            return PurgeDisposition.PURGE_FAILED, purge_authorized
        except (OSError, sqlite3.Error):
            return PurgeDisposition.PURGE_FAILED, purge_authorized

    def purge_full_store(self, command: FullPurgeV1) -> PurgeDisposition:
        """Physically erase exactly a selected, lease-held V1 artifact manifest."""

        if not self._probe.platform_is_supported():
            return PurgeDisposition.UNSUPPORTED_PLATFORM
        if (
            type(command) is not FullPurgeV1
            or command.artifact_manifest_version != self._manifest.version
        ):
            return PurgeDisposition.PURGE_FAILED
        lease_was_held = self._sentinel_handle is not None
        result, purge_authorized = self._purge_full_store_with_lease(command)
        if (not lease_was_held or purge_authorized) and not self._release_sentinel():
            return PurgeDisposition.PURGE_FAILED
        return result


class _WriterJob:
    """One queued operation and the caller's synchronous result slot."""

    __slots__ = ("method", "argument", "done", "result", "failed", "close_after")

    def __init__(self, method: str, argument: object, *, close_after: bool = False) -> None:
        self.method = method
        self.argument = argument
        self.done = threading.Event()
        self.result: object = None
        self.failed = False
        self.close_after = close_after


class SQLiteEvidenceWriterDaemonV1:
    """The one owned writer thread; the only runtime owner of a spool.

    Every public protocol call submits an exact typed operation through a bounded
    private queue and blocks for the exact typed result. The owned
    :class:`SQLiteEvidenceSpool` -- and therefore its SQLite connection, sentinel
    lease, and parent authority -- is constructed and touched only on this thread.
    Callers never reach it.
    """

    protocol_version: Literal[1] = 1
    THREAD_NAME = "hermes-evidence-writer"
    #: Ordinary command capacity, counted exactly under the dispatch lock. The
    #: owner-only drain lane is reserved independently so one valid drain can always
    #: queue behind every previously accepted ordinary job.
    ORDINARY_CAPACITY = 64

    def __init__(self, spool_factory: Callable[[], SQLiteEvidenceSpool]) -> None:
        self._spool_factory = spool_factory
        self._queue: queue.Queue[_WriterJob] = queue.Queue()
        self._dispatch_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._stopped = threading.Event()
        self._ready = threading.Event()
        self._started = False
        self._owner_generation: int | None = None
        self._ordinary_count = 0
        self._drain_pending: _WriterJob | None = None
        self._spool: SQLiteEvidenceSpool | None = None
        self._owner_thread_id: int | None = None
        self._sticky_fault: WriterFault | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=self.THREAD_NAME,
            daemon=True,
        )

    # -- owner-only observation ---------------------------------------------------

    @property
    def owner_thread_id(self) -> int | None:
        return self._owner_thread_id

    @property
    def is_running(self) -> bool:
        return self._started and self._thread.is_alive() and not self._stopped.is_set()

    @property
    def pending_ordinary_count(self) -> int:
        """Accepted-but-unfinished ordinary jobs; never derived from ``qsize()``."""

        with self._dispatch_lock:
            return self._ordinary_count

    @property
    def drain_is_pending(self) -> bool:
        with self._dispatch_lock:
            return self._drain_pending is not None

    @property
    def owner_generation(self) -> int | None:
        """The generation published by the owned spool once startup completed."""

        return self._owner_generation

    @property
    def spool_for_test(self) -> SQLiteEvidenceSpool:
        """The owned spool, for owner-thread assertions only; never call it here."""

        if self._spool is None:
            raise _store_corrupt("the writer daemon owns no spool")
        return self._spool

    def join(self, timeout: float | None = None) -> bool:
        if not self._started:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    # -- the owned thread ---------------------------------------------------------

    def _ensure_started(self) -> None:
        with self._start_lock:
            if not self._started:
                self._started = True
                self._thread.start()

    def _run(self) -> None:
        self._owner_thread_id = threading.get_ident()
        try:
            try:
                self._spool = self._spool_factory()
                # Publish the authoritative generation the spool itself declares; the
                # daemon never keeps an unverified copy of its own.
                self._owner_generation = self._spool.owner_generation
            finally:
                # Readiness is signalled on every path, including BaseException, so a
                # waiting caller can never deadlock on a failed startup.
                self._ready.set()
        except BaseException:
            # A store that cannot even be constructed leaves no thread or handle
            # behind: latch content-free, refuse every caller, and exit.
            self._sticky_fault = WriterFault.STORE_CORRUPT
            self._shutdown()
            return
        try:
            while True:
                job = self._queue.get()
                stopping = False
                try:
                    self._execute(job)
                    if job.close_after:
                        with self._dispatch_lock:
                            self._ordinary_count = max(0, self._ordinary_count - 1)
                        stopping = True
                    elif job.method == "drain_and_close":
                        with self._dispatch_lock:
                            self._drain_pending = None
                        # Exit only when the spool itself reports an exact STOPPED.
                        # A refused or foreign drain leaves this owner fully alive.
                        stopping = not job.failed and job.result is DrainDisposition.STOPPED
                    else:
                        with self._dispatch_lock:
                            self._ordinary_count = max(0, self._ordinary_count - 1)
                    if stopping:
                        # Close before signalling, so no caller can observe STOPPED
                        # while this owner is still running.
                        self._shutdown()
                finally:
                    job.done.set()
                if stopping:
                    return
        finally:
            self._shutdown()

    def _execute(self, job: _WriterJob) -> None:
        spool = self._spool
        try:
            if spool is None:  # pragma: no cover - the run loop guarantees a spool
                raise _store_corrupt("the writer daemon owns no spool")
            method = getattr(spool, job.method)
            job.result = method() if job.argument is None else method(job.argument)
        except Exception:
            # Content-free isolation: the caller receives its typed failure result and
            # the owner latches a sticky fault. KeyboardInterrupt/SystemExit propagate.
            job.failed = True
            if self._sticky_fault is None:
                self._sticky_fault = WriterFault.STORE_CORRUPT

    def _shutdown(self) -> None:
        """Idempotently stop dispatch, close the spool, and signal every waiter."""

        self._ready.set()
        with self._dispatch_lock:
            already_stopped = self._stopped.is_set()
            self._stopped.set()
            self._drain_pending = None
            self._ordinary_count = 0
        if not already_stopped and self._spool is not None:
            with suppress(Exception):  # closing is best effort
                self._spool.close()
        while True:
            try:
                pending = self._queue.get_nowait()
            except queue.Empty:
                return
            pending.failed = True
            pending.done.set()

    # -- synchronous submission ---------------------------------------------------

    def _submit(
        self,
        method: str,
        argument: object,
        refusal: Callable[[], _ResultT],
    ) -> _ResultT:
        """Submit one ordinary operation against the exact 64-slot counter.

        The refusal is produced lazily, after startup has been observed, so a
        refusal snapshot can never describe a pre-startup state.
        """

        self._ensure_started()
        self._ready.wait()
        if self._sticky_fault is not None:
            # A latched owner refuses every further ordinary operation.
            return refusal()
        job = _WriterJob(method, argument)
        with self._dispatch_lock:
            if self._stopped.is_set() or self._drain_pending is not None:
                # Once a valid drain is accepted no further ordinary work is enqueued.
                return refusal()
            if self._ordinary_count >= self.ORDINARY_CAPACITY:
                return refusal()
            self._ordinary_count += 1
            self._queue.put_nowait(job)
        job.done.wait()
        if job.failed:
            return refusal()
        return cast("_ResultT", job.result)

    def _submit_one_shot_and_close(
        self,
        method: str,
        argument: object,
        refusal: Callable[[], _ResultT],
    ) -> _ResultT:
        """Run one recovery-only operation, closing before its result is visible."""

        self._ensure_started()
        self._ready.wait()
        if self._sticky_fault is not None:
            return refusal()
        job = _WriterJob(method, argument, close_after=True)
        with self._dispatch_lock:
            if (
                self._stopped.is_set()
                or self._drain_pending is not None
                or self._ordinary_count != 0
            ):
                return refusal()
            self._ordinary_count = 1
            self._queue.put_nowait(job)
        job.done.wait()
        if job.failed:
            return refusal()
        return cast("_ResultT", job.result)

    # -- transport surface --------------------------------------------------------

    def recover_existing(self) -> RecoveryDisposition:
        return self._submit("recover_existing", None, lambda: RecoveryDisposition.FAULTED)

    def recover_existing_and_close(self) -> RecoveryDisposition:
        """Perform recovery-only inspection and release every owned handle."""

        return self._submit_one_shot_and_close(
            "recover_existing",
            None,
            lambda: RecoveryDisposition.FAULTED,
        )

    def close(self) -> bool:
        """Stop this owner without claiming a durable drain acknowledgment."""

        if self._stopped.is_set():
            return self.join(2.0)
        result = self._submit_one_shot_and_close("close_owner_marker", None, lambda: False)
        return bool(result) and self.join(2.0)

    def create_epoch(self, command: CreateEpochV1) -> StoreDisposition:
        return self._submit("create_epoch", command, lambda: StoreDisposition.FAULTED)

    def active_session_expiry(self, logical_session_id: str) -> str | None:
        """Observe one content-free persisted deadline on the owner thread."""

        return self._submit("active_session_expiry", logical_session_id, lambda: None)

    def append_record(self, item: QueuedEvidenceRecordV1) -> StoreDisposition:
        return self._submit("append_record", item, lambda: StoreDisposition.FAULTED)

    def append_binding_close(self, command: BindingCloseV1) -> StoreDisposition:
        return self._submit("append_binding_close", command, lambda: StoreDisposition.FAULTED)

    def rollover_session(self, command: RolloverSessionV1) -> StoreDisposition:
        return self._submit("rollover_session", command, lambda: StoreDisposition.FAULTED)

    def expire_session(self, command: ExpireSessionV1) -> PurgeDisposition:
        return self._submit("expire_session", command, lambda: PurgeDisposition.PURGE_FAILED)

    def commit_revoke_request(self, command: RevokeRequestV1) -> RevokeDisposition:
        return self._submit(
            "commit_revoke_request",
            command,
            lambda: RevokeDisposition.WRITER_FAULT,
        )

    def finalize_revoke(self, command: RevokeFinalizeV1) -> RevokeDisposition:
        return self._submit("finalize_revoke", command, lambda: RevokeDisposition.WRITER_FAULT)

    def seal_epoch(self, command: SealEpochV1) -> StoreDisposition:
        return self._submit("seal_epoch", command, lambda: StoreDisposition.FAULTED)

    def run_maintenance(self, command: MaintenanceV1) -> PurgeDisposition:
        return self._submit("run_maintenance", command, lambda: PurgeDisposition.PURGE_FAILED)

    def purge_full_store(self, command: FullPurgeV1) -> PurgeDisposition:
        return self._submit("purge_full_store", command, lambda: PurgeDisposition.PURGE_FAILED)

    def purge_full_store_and_close(self, command: FullPurgeV1) -> PurgeDisposition:
        """Perform one full purge and release every owned handle before returning."""

        return self._submit_one_shot_and_close(
            "purge_full_store",
            command,
            lambda: PurgeDisposition.PURGE_FAILED,
        )

    def drain_and_close(self, command: DrainAndStopV1) -> DrainDisposition:
        """Process every prior operation, then close only on an exact ``STOPPED``.

        The owner-only lane is reserved independently of ordinary capacity, so a
        valid drain always queues behind previously accepted work. It holds exactly
        one item: a concurrent request while one is pending is refused as
        ``ALREADY_QUEUED`` and can never cancel or replace the accepted drain.
        """

        self._ensure_started()
        # Wait only for startup readiness, never for queued work.
        self._ready.wait()
        if self._stopped.is_set():
            # Truthful: this owner is already closed.
            return DrainDisposition.STOPPED
        generation = self._owner_generation
        if (
            type(command) is not DrainAndStopV1
            or generation is None
            or command.owner_generation != generation
        ):
            # Authority is checked before any reservation, so a foreign or wrong-type
            # request never seizes the lane, refuses ordinary work, enters the queue,
            # or reaches the owner thread.
            return DrainDisposition.ALREADY_QUEUED
        job = _WriterJob("drain_and_close", command)
        with self._dispatch_lock:
            if self._stopped.is_set():
                return DrainDisposition.STOPPED
            if self._drain_pending is not None:
                return DrainDisposition.ALREADY_QUEUED
            self._drain_pending = job
            self._queue.put_nowait(job)
        job.done.wait()
        if job.failed:
            return (
                DrainDisposition.STOPPED
                if self._stopped.is_set()
                else DrainDisposition.ALREADY_QUEUED
            )
        return cast("DrainDisposition", job.result)

    def diagnostics(self) -> EvidenceDiagnosticsV1:
        return self._submit("diagnostics", None, self._stopped_diagnostics)

    def _stopped_diagnostics(self) -> EvidenceDiagnosticsV1:
        """Content-free diagnostics for a daemon that is stopped or never started."""

        faulted = self._sticky_fault is not None
        return EvidenceDiagnosticsV1(
            protocol_version=1,
            owner_state=OwnerState.FAULTED if faulted else OwnerState.STOPPED,
            capture_state=CaptureState.FAULTED if faulted else CaptureState.UNAVAILABLE,
            sticky_fault=self._sticky_fault,
            queue_record_count=0,
            queue_canonical_bytes=0,
            active_lease_count=0,
            pending_revoke=False,
            purge_required=False,
        )

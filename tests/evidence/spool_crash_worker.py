"""Test-only process harness for the closed Task 5D crash matrix.

This module contains no recovery implementation. It drives the real SQLite evidence
spool and stops the process at named durable boundaries selected by the parent test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hermes_realtime.evidence.models import (
        BindingCloseV1,
        CreateEpochV1,
        EventKind,
        EvidenceSnapshotV1,
        RevokeFinalizeV1,
        RevokeRequestV1,
        SealEpochV1,
    )
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

REPOSITORY = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPOSITORY / "src"))

EXIT_MODES = (197, 198)

FAILPOINTS = (
    "before_begin",
    "after_event_insert_before_commit",
    "after_event_commit",
    "before_seal_commit",
    "after_seal_commit_before_ack",
    "after_revoke_request_commit",
    "after_logical_purge_before_vacuum",
    "after_vacuum_before_ack",
    "after_drain_ack_before_exit",
    "after_root_marker_init_create",
    "after_root_marker_init_partial_write",
    "after_root_marker_init_full_write",
    "after_root_marker_init_flush",
    "after_root_marker_activation",
    "after_sentinel_init_create",
    "after_sentinel_init_partial_write",
    "after_sentinel_init_full_write",
    "after_sentinel_init_flush",
    "after_sentinel_activation",
    "after_first_db_create",
    "after_first_schema_commit",
    "after_first_epoch_commit",
    "after_first_marker_clear",
    "after_recreate_pending_fsync",
    "after_recreate_db_create",
    "after_recreate_schema_commit",
    "after_recreate_epoch_commit",
    "after_recreate_marker_clear",
    "before_rollback_sentinel_write",
    "after_rollback_sentinel_fsync",
    "before_rollback_db_latch",
    "after_rollback_db_latch",
    "after_full_purge_marker_fsync",
    "after_full_purge_db_delete",
    "after_full_purge_journal_delete",
    "after_full_purge_wal_delete",
    "after_full_purge_shm_delete",
    "after_full_purge_vacuum_delete",
    "after_full_purge_tmp_delete",
    "after_full_purge_absence_verify",
    "after_full_purge_marker_clear",
)


def case_ids() -> tuple[str, ...]:
    """Return the exact closed failpoint × exit-mode case IDs in execution order."""

    return tuple(
        f"{failpoint}@exit{exit_mode}" for failpoint in FAILPOINTS for exit_mode in EXIT_MODES
    )


INSTALLATION_ID = "10000000-0000-4000-8000-000000000001"
PRODUCER_ID = "10000000-0000-4000-8000-000000000002"
EPOCH_ID = "10000000-0000-4000-8000-000000000003"
SESSION_ID = "10000000-0000-4000-8000-000000000004"
BINDING_ID = "10000000-0000-4000-8000-000000000006"
CURRENT_EPOCH_ID = "20000000-0000-4000-8000-000000000003"
CURRENT_SESSION_ID = "20000000-0000-4000-8000-000000000004"
CURRENT_BINDING_ID = "20000000-0000-4000-8000-000000000006"
DISCLOSURE_DIGEST = "cd" * 32
CONTROL_FINGERPRINT = "ab" * 32
START = datetime(2026, 8, 8, tzinfo=UTC)

_ACTIVE_FAILPOINT: str | None = None
_ACTIVE_OPERATION: str | None = None
_ACTIVE_EXIT_MODE = 197
_EVENT_WRITTEN = False
_ERASURE_REQUEST_WRITTEN = False
_LOGICAL_DELETE_WRITTEN = False
_CREATE_COMMIT_COUNT = 0
_ROLLBACK_UPDATE_WRITTEN = False
_VACUUM_RETURNED = False


def _arm(failpoint: str, operation: str, exit_mode: int) -> None:
    global _ACTIVE_EXIT_MODE, _ACTIVE_FAILPOINT, _ACTIVE_OPERATION
    global _CREATE_COMMIT_COUNT, _ERASURE_REQUEST_WRITTEN, _EVENT_WRITTEN
    global _LOGICAL_DELETE_WRITTEN, _ROLLBACK_UPDATE_WRITTEN, _VACUUM_RETURNED
    _ACTIVE_FAILPOINT = failpoint
    _ACTIVE_OPERATION = operation
    _ACTIVE_EXIT_MODE = exit_mode
    _EVENT_WRITTEN = False
    _ERASURE_REQUEST_WRITTEN = False
    _LOGICAL_DELETE_WRITTEN = False
    _CREATE_COMMIT_COUNT = 0
    _ROLLBACK_UPDATE_WRITTEN = False
    _VACUUM_RETURNED = False


def _trip() -> None:
    if _ACTIVE_FAILPOINT is None:  # pragma: no cover - guarded by every caller
        raise RuntimeError("no crashpoint is armed")
    _crashpoint(_ACTIVE_FAILPOINT, _ACTIVE_EXIT_MODE)


class _CrashConnection(sqlite3.Connection):
    """Real SQLite connection with test-only barriers around exact durable calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if (
            _ACTIVE_OPERATION == "first_create" and _ACTIVE_FAILPOINT == "after_first_db_create"
        ) or (_ACTIVE_OPERATION == "recreate" and _ACTIVE_FAILPOINT == "after_recreate_db_create"):
            _trip()

    def execute(self, sql: str, parameters: Any = (), /) -> sqlite3.Cursor:
        global _ERASURE_REQUEST_WRITTEN, _EVENT_WRITTEN, _LOGICAL_DELETE_WRITTEN
        global _ROLLBACK_UPDATE_WRITTEN, _VACUUM_RETURNED
        normalized = " ".join(sql.split())
        if (
            _ACTIVE_FAILPOINT == "before_begin"
            and _ACTIVE_OPERATION == "append"
            and normalized == "BEGIN IMMEDIATE"
        ):
            _trip()
        if (
            _ACTIVE_OPERATION == "rollback"
            and _ACTIVE_FAILPOINT == "before_rollback_db_latch"
            and normalized == "BEGIN IMMEDIATE"
        ):
            _trip()
        result = super().execute(sql, parameters)
        if _ACTIVE_OPERATION in {"append", "seal"} and normalized.startswith(
            "INSERT INTO evidence_events"
        ):
            _EVENT_WRITTEN = True
            if _ACTIVE_FAILPOINT == "after_event_insert_before_commit":
                _trip()
        if _ACTIVE_OPERATION == "revoke" and normalized.startswith("INSERT INTO erasure_requests"):
            _ERASURE_REQUEST_WRITTEN = True
        if _ACTIVE_OPERATION == "finalize" and "SET state='logical_deleted'" in normalized:
            _LOGICAL_DELETE_WRITTEN = True
        if (
            _ACTIVE_OPERATION == "rollback"
            and "SET purge_required=1" in normalized
            and "purge_reason='clock_rollback'" in normalized
        ):
            _ROLLBACK_UPDATE_WRITTEN = True
        if _ACTIVE_OPERATION == "finalize" and normalized == "VACUUM":
            # Preserve the temporal contract: `after_vacuum_before_ack` is
            # reachable only after sqlite3 has returned from the real VACUUM.
            _VACUUM_RETURNED = True
            if _ACTIVE_FAILPOINT == "after_vacuum_before_ack":
                _trip()
        return result

    def commit(self) -> None:
        global _CREATE_COMMIT_COUNT
        if _ACTIVE_OPERATION in {"first_create", "recreate"}:
            super().commit()
            _CREATE_COMMIT_COUNT += 1
            schema_failpoint = {
                "first_create": "after_first_schema_commit",
                "recreate": "after_recreate_schema_commit",
            }[_ACTIVE_OPERATION]
            epoch_failpoint = {
                "first_create": "after_first_epoch_commit",
                "recreate": "after_recreate_epoch_commit",
            }[_ACTIVE_OPERATION]
            if _CREATE_COMMIT_COUNT == 1 and schema_failpoint == _ACTIVE_FAILPOINT:
                _trip()
            if _CREATE_COMMIT_COUNT == 2 and epoch_failpoint == _ACTIVE_FAILPOINT:
                _trip()
            return
        if _ACTIVE_OPERATION == "rollback":
            super().commit()
            if _ROLLBACK_UPDATE_WRITTEN and _ACTIVE_FAILPOINT == "after_rollback_db_latch":
                _trip()
            return
        if _ACTIVE_OPERATION == "seal" and _EVENT_WRITTEN:
            if _ACTIVE_FAILPOINT == "before_seal_commit":
                _trip()
            super().commit()
            if _ACTIVE_FAILPOINT == "after_seal_commit_before_ack":
                _trip()
            return
        super().commit()
        if (
            _ACTIVE_OPERATION == "append"
            and _EVENT_WRITTEN
            and _ACTIVE_FAILPOINT == "after_event_commit"
        ):
            _trip()
        if (
            _ACTIVE_OPERATION == "revoke"
            and _ERASURE_REQUEST_WRITTEN
            and _ACTIVE_FAILPOINT == "after_revoke_request_commit"
        ):
            _trip()
        if (
            _ACTIVE_OPERATION == "finalize"
            and _LOGICAL_DELETE_WRITTEN
            and _ACTIVE_FAILPOINT == "after_logical_purge_before_vacuum"
        ):
            _trip()


class _Clock:
    def __init__(self, start: datetime = START) -> None:
        self._next = start

    def __call__(self) -> datetime:
        current = self._next
        self._next += timedelta(seconds=1)
        return current

    def set(self, moment: datetime) -> None:
        self._next = moment


class _Uuids:
    def __init__(self) -> None:
        self._issued = 0

    def __call__(self) -> str:
        self._issued += 1
        return f"50000000-0000-4000-8000-{self._issued:012d}"


class _StorageProbe:
    def platform_is_supported(self) -> bool:
        return True

    def volume_is_fixed_local(self, root: Path) -> bool:
        return True

    def path_has_reparse_point(self, path: Path) -> bool:
        return False

    def path_has_alternate_data_streams(self, path: Path) -> bool:
        return False

    def path_grants_only_owner(self, path: Path) -> bool:
        return True

    def allocated_bytes(self, path: Path) -> int:
        return 0

    def volume_free_bytes(self, root: Path) -> int:
        return 1 << 40


def _event_id(index: int) -> str:
    return f"40000000-0000-4000-8000-{index:012d}"


def _make_spool(case_root: Path, *, clock: _Clock | None = None) -> SQLiteEvidenceSpool:
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

    root = case_root / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    return SQLiteEvidenceSpool(
        root / "capture-v1.sqlite3",
        clock=clock or _Clock(),
        uuid_factory=_Uuids(),
        probe=_StorageProbe(),
        connection_factory=_CrashConnection,
    )


def _make_create_epoch(
    *,
    epoch_id: str = EPOCH_ID,
    session_id: str = SESSION_ID,
    binding_id: str = BINDING_ID,
    control_sequence: int = 1,
    event_offset: int = 0,
) -> CreateEpochV1:
    from hermes_realtime.evidence.models import (
        BindingOpenedPayloadV1,
        CreateEpochV1,
        EventKind,
        EvidenceSnapshotV1,
        SessionOpenedPayloadV1,
    )

    opened = EvidenceSnapshotV1(
        schema_version=1,
        installation_id=INSTALLATION_ID,
        producer_instance_id=PRODUCER_ID,
        logical_session_id=session_id,
        event_id=_event_id(event_offset + 1),
        event_sequence=1,
        event_kind=EventKind.SESSION_OPENED,
        payload=SessionOpenedPayloadV1(
            consent_epoch_id=epoch_id,
            binding_id=binding_id,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE_DIGEST,
            retention_hours=24,
            microphone_accepted=False,
            typed_accepted=True,
            predecessor_session_id=None,
        ),
    )
    binding = EvidenceSnapshotV1(
        schema_version=1,
        installation_id=INSTALLATION_ID,
        producer_instance_id=PRODUCER_ID,
        logical_session_id=session_id,
        event_id=_event_id(event_offset + 2),
        event_sequence=2,
        event_kind=EventKind.BINDING_OPENED,
        payload=BindingOpenedPayloadV1(
            binding_id=binding_id,
            binding_generation=1,
            microphone_available=False,
            typed_available=True,
        ),
    )
    return CreateEpochV1(
        protocol_version=1,
        installation_id=INSTALLATION_ID,
        producer_instance_id=PRODUCER_ID,
        consent_epoch_id=epoch_id,
        logical_session_id=session_id,
        binding_id=binding_id,
        binding_generation=1,
        consent_version="realtime-evidence-consent-v1",
        disclosure_digest=DISCLOSURE_DIGEST,
        retention_hours=24,
        microphone_accepted=False,
        typed_accepted=True,
        control_sequence=control_sequence,
        control_fingerprint_hash=CONTROL_FINGERPRINT,
        session_opened=opened,
        binding_opened=binding,
    )


def _snapshot(
    kind: EventKind,
    sequence: int,
    payload: Any,
    event_index: int,
    *,
    session_id: str = SESSION_ID,
) -> EvidenceSnapshotV1:
    from hermes_realtime.evidence.models import EvidenceSnapshotV1

    return EvidenceSnapshotV1(
        schema_version=1,
        installation_id=INSTALLATION_ID,
        producer_instance_id=PRODUCER_ID,
        logical_session_id=session_id,
        event_id=_event_id(event_index),
        event_sequence=sequence,
        event_kind=kind,
        payload=payload,
    )


def _ordinary_record(*, session_id: str = SESSION_ID, event_index: int = 10) -> Any:
    from hermes_realtime.evidence.models import (
        EventKind,
        QueuedEvidenceRecordV1,
        QueueReservationClass,
        TurnKind,
        TurnOpenedPayloadV1,
    )

    snapshot = _snapshot(
        EventKind.TURN_OPENED,
        3,
        TurnOpenedPayloadV1(
            evidence_turn_id="10000000-0000-4000-8000-000000000007",
            turn_kind=TurnKind.USER_RESPONSE,
            utterance_id="10000000-0000-4000-8000-000000000008",
            replay_of_evidence_turn_id=None,
        ),
        event_index,
        session_id=session_id,
    )
    return QueuedEvidenceRecordV1(
        protocol_version=1,
        snapshot=snapshot,
        admission_ordinal=3,
        reservation_class=QueueReservationClass.ORDINARY,
        lease_open_ordinal=None,
    )


def _binding_close() -> BindingCloseV1:
    from hermes_realtime.evidence.models import (
        BindingClosedPayloadV1,
        BindingCloseReason,
        BindingCloseV1,
        EventKind,
    )

    snapshot = _snapshot(
        EventKind.BINDING_CLOSED,
        3,
        BindingClosedPayloadV1(
            binding_id=BINDING_ID,
            close_reason=BindingCloseReason.CLIENT_CLOSED,
        ),
        20,
    )
    return BindingCloseV1(
        protocol_version=1,
        binding_id=BINDING_ID,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        admission_ordinal=3,
        snapshot=snapshot,
    )


def _seal_command() -> SealEpochV1:
    from hermes_realtime.evidence.models import (
        BindingCloseReason,
        EventKind,
        SealEpochV1,
        SessionSealRequestedPayloadV1,
    )

    snapshot = _snapshot(
        EventKind.SESSION_SEAL_REQUESTED,
        4,
        SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=EPOCH_ID,
            consent_version="realtime-evidence-consent-v1",
            disclosure_digest=DISCLOSURE_DIGEST,
        ),
        21,
    )
    return SealEpochV1(
        protocol_version=1,
        consent_epoch_id=EPOCH_ID,
        logical_session_id=SESSION_ID,
        binding_id=BINDING_ID,
        final_event_sequence=4,
        close_reason=BindingCloseReason.HOST_SHUTDOWN,
        close_epoch=True,
        admission_ordinal=4,
        snapshot=snapshot,
    )


def _revoke_commands() -> tuple[RevokeRequestV1, RevokeFinalizeV1]:
    from hermes_realtime.evidence.models import RevokeFinalizeV1, RevokeRequestV1

    request = RevokeRequestV1(
        protocol_version=1,
        erasure_request_id=_event_id(90),
        consent_epoch_id=EPOCH_ID,
        control_sequence=7,
        control_fingerprint_hash=CONTROL_FINGERPRINT,
        last_admission_ordinal=2,
    )
    finalize = RevokeFinalizeV1(
        protocol_version=1,
        erasure_request_id=request.erasure_request_id,
        consent_epoch_id=request.consent_epoch_id,
        control_sequence=request.control_sequence,
        control_fingerprint_hash=request.control_fingerprint_hash,
        final_admission_ordinal=2,
    )
    return request, finalize


def _crashpoint(failpoint: str, exit_mode: int) -> None:
    import hermes_realtime

    provenance = Path(hermes_realtime.__file__).resolve()
    if not provenance.is_relative_to((REPOSITORY / "src").resolve()):
        raise RuntimeError("worker import escaped the candidate source")
    payload = {
        "caseId": f"{failpoint}@exit{exit_mode}",
        "importProvenance": "candidate-source",
        "pid": os.getpid(),
        "protocolVersion": 1,
        "state": "READY",
    }
    if failpoint == "after_vacuum_before_ack":
        if not _VACUUM_RETURNED:
            raise RuntimeError("VACUUM failpoint was reached before sqlite3 returned")
        payload["vacuumReturned"] = True
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    sys.stdout.flush()
    if exit_mode == 197:
        os._exit(197)
    threading.Event().wait()
    raise AssertionError("the parent must terminate exit198 workers")  # pragma: no cover


def _run_ordinary(case_root: Path, failpoint: str, exit_mode: int) -> None:
    from hermes_realtime.evidence.models import DrainAndStopV1

    owned = _make_spool(case_root)
    if owned.create_epoch(_make_create_epoch()).value != "committed":
        raise RuntimeError("failed to establish the ordinary crash store")
    if failpoint in {
        "before_begin",
        "after_event_insert_before_commit",
        "after_event_commit",
    }:
        _arm(failpoint, "append", exit_mode)
        owned.append_record(_ordinary_record())
    elif failpoint in {"before_seal_commit", "after_seal_commit_before_ack"}:
        if owned.append_binding_close(_binding_close()).value != "committed":
            raise RuntimeError("failed to close the crash-test binding")
        _arm(failpoint, "seal", exit_mode)
        owned.seal_epoch(_seal_command())
    elif failpoint == "after_revoke_request_commit":
        request, _ = _revoke_commands()
        _arm(failpoint, "revoke", exit_mode)
        owned.commit_revoke_request(request)
    elif failpoint in {
        "after_logical_purge_before_vacuum",
        "after_vacuum_before_ack",
    }:
        request, finalize = _revoke_commands()
        if owned.commit_revoke_request(request).value != "revoke_durably_scheduled":
            raise RuntimeError("failed to schedule the crash-test revoke")
        _arm(failpoint, "finalize", exit_mode)
        owned.finalize_revoke(finalize)
    elif failpoint == "after_drain_ack_before_exit":
        if owned.append_binding_close(_binding_close()).value != "committed":
            raise RuntimeError("failed to close the drain-test binding")
        if owned.seal_epoch(_seal_command()).value != "committed":
            raise RuntimeError("failed to seal the drain-test epoch")
        result = owned.drain_and_close(
            DrainAndStopV1(
                protocol_version=1,
                owner_generation=owned.owner_generation,
                final_admission_ordinal=4,
            )
        )
        if result.value != "stopped":
            raise RuntimeError("failed to receive the real drain acknowledgement")
        _crashpoint(failpoint, exit_mode)
    raise RuntimeError(f"ordinary failpoint returned without crashing: {failpoint}")


def _install_first_create_hooks(failpoint: str, exit_mode: int) -> None:
    from hermes_realtime.evidence import sqlite_spool

    original_write = sqlite_spool._write_new_file
    original_activate = sqlite_spool.SQLiteEvidenceSpool._activate
    original_transition = sqlite_spool.SQLiteEvidenceSpool._transition_sentinel

    def crash_write(path: Path, data: bytes):  # type: ignore[no-untyped-def]
        target = (
            path.name == ".hermes-realtime-evidence-root-v1.init"
            and failpoint.startswith("after_root_marker_init_")
        ) or (path.name == "capture-v1.owner.init" and failpoint.startswith("after_sentinel_init_"))
        if not target:
            return original_write(path, data)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        if failpoint.endswith("_create"):
            _crashpoint(failpoint, exit_mode)
        if failpoint.endswith("_partial_write"):
            os.write(descriptor, data[: max(1, len(data) // 2)])
            _crashpoint(failpoint, exit_mode)
        os.write(descriptor, data)
        if failpoint.endswith("_full_write"):
            _crashpoint(failpoint, exit_mode)
        os.fsync(descriptor)
        _crashpoint(failpoint, exit_mode)

    def crash_activate(  # type: ignore[no-untyped-def]
        self,
        root,
        temporary: Path,
        final: Path,
        **kwargs,
    ) -> None:
        original_activate(self, root, temporary, final, **kwargs)
        if (
            failpoint == "after_root_marker_activation"
            and final.name == ".hermes-realtime-evidence-root-v1"
        ) or (failpoint == "after_sentinel_activation" and final.name == "capture-v1.owner"):
            _crashpoint(failpoint, exit_mode)

    def crash_transition(self, state, **kwargs):  # type: ignore[no-untyped-def]
        original_transition(self, state, **kwargs)
        if failpoint == "after_first_marker_clear" and state.value == "clear":
            _crashpoint(failpoint, exit_mode)
        if failpoint == "after_recreate_pending_fsync" and state.value == "first_create_pending":
            _crashpoint(failpoint, exit_mode)
        if failpoint == "after_recreate_marker_clear" and state.value == "clear":
            _crashpoint(failpoint, exit_mode)

    sqlite_spool._write_new_file = crash_write
    sqlite_spool.SQLiteEvidenceSpool._activate = crash_activate  # type: ignore[method-assign]
    sqlite_spool.SQLiteEvidenceSpool._transition_sentinel = crash_transition  # type: ignore[method-assign]


def _run_first_create(case_root: Path, failpoint: str, exit_mode: int) -> None:
    root = case_root / "evidence"
    root.mkdir(parents=True, exist_ok=True)
    _install_first_create_hooks(failpoint, exit_mode)
    _arm(failpoint, "first_create", exit_mode)
    owned = _make_spool(case_root)
    owned.create_epoch(_make_create_epoch())
    raise RuntimeError(f"first-create failpoint returned without crashing: {failpoint}")


def _run_recreate(case_root: Path, failpoint: str, exit_mode: int) -> None:
    from hermes_realtime.evidence.models import FullPurgeV1, SentinelState

    owned = _make_spool(case_root)
    if owned.create_epoch(_make_create_epoch()).value != "committed":
        raise RuntimeError("failed to establish the recreation predecessor")
    decoy = case_root / "evidence" / "recreation-decoy.bin"
    decoy.write_bytes(b"not-owned")
    if (
        owned.purge_full_store(
            FullPurgeV1(
                protocol_version=1,
                full_purge_generation_id=_event_id(95),
                sentinel_state=SentinelState.FULL_PURGE_PENDING,
                artifact_manifest_version=1,
            )
        ).value
        != "purge_completed"
    ):
        raise RuntimeError("failed to complete the recreation predecessor purge")
    _install_first_create_hooks(failpoint, exit_mode)
    _arm(failpoint, "recreate", exit_mode)
    owned.create_epoch(_make_create_epoch())
    raise RuntimeError(f"recreation failpoint returned without crashing: {failpoint}")


def _install_rollback_hooks(failpoint: str, exit_mode: int) -> None:
    from hermes_realtime.evidence import sqlite_spool

    original_transition = sqlite_spool.SQLiteEvidenceSpool._transition_sentinel

    def crash_transition(self, state, **kwargs):  # type: ignore[no-untyped-def]
        is_rollback = state.value == "clock_rollback_purge_pending"
        if is_rollback and failpoint == "before_rollback_sentinel_write":
            _crashpoint(failpoint, exit_mode)
        original_transition(self, state, **kwargs)
        if is_rollback and failpoint == "after_rollback_sentinel_fsync":
            _crashpoint(failpoint, exit_mode)

    sqlite_spool.SQLiteEvidenceSpool._transition_sentinel = crash_transition  # type: ignore[method-assign]


def _seed_closed_older_epoch(owned: SQLiteEvidenceSpool, case_root: Path) -> None:
    import shutil

    seed_root = case_root / "closed-older-seed"
    older = _make_spool(seed_root)
    if older.create_epoch(_make_create_epoch()).value != "committed":
        raise RuntimeError("failed to create the seeded older epoch")
    if older.append_binding_close(_binding_close()).value != "committed":
        raise RuntimeError("failed to close the seeded older binding")
    if older.seal_epoch(_seal_command()).value != "committed":
        raise RuntimeError("failed to seal the seeded older epoch")
    older.close()

    connection = owned.connection
    connection.execute(
        "ATTACH DATABASE ? AS closed_older",
        (str(seed_root / "evidence" / "capture-v1.sqlite3"),),
    )
    try:
        connection.execute("INSERT INTO consent_epochs SELECT * FROM closed_older.consent_epochs")
        connection.execute(
            "INSERT INTO evidence_sessions SELECT * FROM closed_older.evidence_sessions"
        )
        connection.execute("INSERT INTO evidence_events SELECT * FROM closed_older.evidence_events")
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE closed_older")
    shutil.rmtree(seed_root)


def _record_rollback_baseline(owned: SQLiteEvidenceSpool, case_root: Path) -> None:
    database_sha256 = hashlib.sha256(
        "\n".join(owned.connection.iterdump()).encode("utf-8")
    ).hexdigest()
    sentinel_bytes = owned._read_sentinel()
    (case_root / "rollback-baseline.json").write_text(
        json.dumps(
            {
                "databaseSha256": database_sha256,
                "sentinelSha256": hashlib.sha256(sentinel_bytes).hexdigest(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _run_rollback(case_root: Path, failpoint: str, exit_mode: int) -> None:
    clock = _Clock()
    owned = _make_spool(case_root, clock=clock)
    if (
        owned.create_epoch(
            _make_create_epoch(
                epoch_id=CURRENT_EPOCH_ID,
                session_id=CURRENT_SESSION_ID,
                binding_id=CURRENT_BINDING_ID,
                event_offset=100,
            )
        ).value
        != "committed"
    ):
        raise RuntimeError("failed to establish the rollback current epoch")
    _seed_closed_older_epoch(owned, case_root)
    _record_rollback_baseline(owned, case_root)
    (case_root / "evidence" / "rollback-decoy.bin").write_bytes(b"not-owned")
    _install_rollback_hooks(failpoint, exit_mode)
    _arm(failpoint, "rollback", exit_mode)
    clock.set(START - timedelta(hours=2))
    owned.append_record(_ordinary_record(session_id=CURRENT_SESSION_ID, event_index=110))
    raise RuntimeError(f"rollback failpoint returned without crashing: {failpoint}")


def _install_full_purge_hooks(failpoint: str, exit_mode: int) -> None:
    from hermes_realtime.evidence import sqlite_spool

    original_transition = sqlite_spool.SQLiteEvidenceSpool._transition_sentinel
    original_unlink = Path.unlink
    delete_failpoints = {
        "capture-v1.sqlite3": "after_full_purge_db_delete",
        "capture-v1.sqlite3-journal": "after_full_purge_journal_delete",
        "capture-v1.sqlite3-wal": "after_full_purge_wal_delete",
        "capture-v1.sqlite3-shm": "after_full_purge_shm_delete",
        "capture-v1.sqlite3-vacuum": "after_full_purge_vacuum_delete",
        "capture-v1.sqlite3-tmp": "after_full_purge_tmp_delete",
    }

    def crash_unlink(path: Path, *args: object, **kwargs: object) -> None:
        original_unlink(path, *args, **kwargs)  # type: ignore[arg-type]
        if delete_failpoints.get(path.name) == failpoint:
            _crashpoint(failpoint, exit_mode)

    def crash_transition(self, state, **kwargs):  # type: ignore[no-untyped-def]
        if state.value == "clear" and failpoint == "after_full_purge_absence_verify":
            _crashpoint(failpoint, exit_mode)
        original_transition(self, state, **kwargs)
        if (
            state.value == "full_purge_pending" and failpoint == "after_full_purge_marker_fsync"
        ) or (state.value == "clear" and failpoint == "after_full_purge_marker_clear"):
            _crashpoint(failpoint, exit_mode)

    Path.unlink = crash_unlink  # type: ignore[assignment]
    sqlite_spool.SQLiteEvidenceSpool._transition_sentinel = crash_transition  # type: ignore[method-assign]


def _run_full_purge(case_root: Path, failpoint: str, exit_mode: int) -> None:
    from hermes_realtime.evidence.models import FullPurgeV1, SentinelState

    owned = _make_spool(case_root)
    if owned.create_epoch(_make_create_epoch()).value != "committed":
        raise RuntimeError("failed to establish the full-purge predecessor")
    root = case_root / "evidence"
    (root / "purge-decoy.bin").write_bytes(b"not-owned")
    for name in (
        "capture-v1.sqlite3-journal",
        "capture-v1.sqlite3-wal",
        "capture-v1.sqlite3-shm",
        "capture-v1.sqlite3-vacuum",
        "capture-v1.sqlite3-tmp",
    ):
        (root / name).write_bytes(b"owned-sidecar")
    _install_full_purge_hooks(failpoint, exit_mode)
    owned.purge_full_store(
        FullPurgeV1(
            protocol_version=1,
            full_purge_generation_id=_event_id(96),
            sentinel_state=SentinelState.FULL_PURGE_PENDING,
            artifact_manifest_version=1,
        )
    )
    raise RuntimeError(f"full-purge failpoint returned without crashing: {failpoint}")


def _emit_contender_frame(payload: dict[str, object]) -> None:
    """Emit one bounded, content-free contender protocol frame."""

    sys.stdout.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
    sys.stdout.flush()


def _wait_for_shared_start(start_event_name: str) -> None:
    """Wait at one Windows kernel event shared by both contender processes."""

    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    wait_object_0 = 0
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_event = kernel32.OpenEventW
    open_event.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
    open_event.restype = wintypes.HANDLE
    wait_for_single_object = kernel32.WaitForSingleObject
    wait_for_single_object.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    wait_for_single_object.restype = wintypes.DWORD
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    handle = open_event(synchronize, False, start_event_name)
    if not handle:
        raise RuntimeError("shared start event is unavailable")
    try:
        if wait_for_single_object(handle, 10_000) != wait_object_0:
            raise RuntimeError("shared start event deadline expired")
    finally:
        if not close_handle(handle):
            raise RuntimeError("shared start event handle did not close")


def _run_marker_only_contender(
    case_root: Path,
    contender_id: int,
    start_event_name: str,
) -> None:
    """Drive the real create path; the parent owns both barriers and recovery setup."""

    from hermes_realtime.evidence.models import StoreDisposition, WriterFault

    owned = _make_spool(case_root, clock=_Clock(START + timedelta(hours=2)))
    try:
        _emit_contender_frame(
            {
                "contenderId": contender_id,
                "protocolVersion": 1,
                "state": "ARMED",
            }
        )
        _wait_for_shared_start(start_event_name)
        result = owned.create_epoch(
            _make_create_epoch(
                epoch_id=CURRENT_EPOCH_ID,
                session_id=CURRENT_SESSION_ID,
                binding_id=CURRENT_BINDING_ID,
            )
        )
        diagnostics = owned.diagnostics()
        ownership_refusal = (
            result is StoreDisposition.FAULTED
            and diagnostics.sticky_fault is WriterFault.OWNERSHIP_UNAVAILABLE
        )
        _emit_contender_frame(
            {
                "contenderId": contender_id,
                "disposition": result.value,
                "ownershipRefusal": ownership_refusal,
                "protocolVersion": 1,
                "state": "TERMINAL",
            }
        )
        if sys.stdin.readline() != "CLOSE\n":
            raise RuntimeError("contender close protocol failed")
    finally:
        owned.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5D evidence spool crash worker")
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--failpoint", choices=FAILPOINTS)
    parser.add_argument("--exit-mode", choices=EXIT_MODES, type=int)
    parser.add_argument("--marker-only-contender", type=int, choices=(0, 1))
    parser.add_argument("--start-event")
    args = parser.parse_args(argv)
    crash = args.failpoint is not None or args.exit_mode is not None
    if args.marker_only_contender is None and not (
        args.failpoint is not None and args.exit_mode is not None
    ):
        parser.error("crash workers require both --failpoint and --exit-mode")
    if args.marker_only_contender is not None and crash:
        parser.error("marker-only contenders cannot select crash arguments")
    if (args.marker_only_contender is None) != (args.start_event is None):
        parser.error("marker-only contenders require one shared start event")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.marker_only_contender is not None:
        _run_marker_only_contender(args.case_root, args.marker_only_contender, args.start_event)
        return 0
    args.case_root.mkdir(parents=True, exist_ok=False)
    if args.failpoint in FAILPOINTS[:9]:
        _run_ordinary(args.case_root, args.failpoint, args.exit_mode)
        return 0  # pragma: no cover - crashpoint never returns
    if args.failpoint in FAILPOINTS[9:23]:
        _run_first_create(args.case_root, args.failpoint, args.exit_mode)
        return 0  # pragma: no cover - crashpoint never returns
    if args.failpoint in FAILPOINTS[23:28]:
        _run_recreate(args.case_root, args.failpoint, args.exit_mode)
        return 0  # pragma: no cover - crashpoint never returns
    if args.failpoint in FAILPOINTS[28:32]:
        _run_rollback(args.case_root, args.failpoint, args.exit_mode)
        return 0  # pragma: no cover - crashpoint never returns
    if args.failpoint in FAILPOINTS[32:]:
        _run_full_purge(args.case_root, args.failpoint, args.exit_mode)
        return 0  # pragma: no cover - crashpoint never returns
    raise RuntimeError(f"failpoint is not implemented: {args.failpoint}")


if __name__ == "__main__":
    raise SystemExit(main())

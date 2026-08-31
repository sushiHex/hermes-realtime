from __future__ import annotations

import copy
import dataclasses
import pickle
from queue import Empty, Full
from threading import Barrier, Event, Lock, Thread
from types import MethodType, ModuleType
from typing import Any
from uuid import UUID

import pytest

HASH = "a" * 64


def _uuid(value: int) -> str:
    return str(UUID(int=value, version=4))


def _create_epoch(m: ModuleType, *, seed: int = 1) -> object:
    installation_id = _uuid(seed)
    producer_instance_id = _uuid(seed + 1)
    consent_epoch_id = _uuid(seed + 2)
    logical_session_id = _uuid(seed + 3)
    binding_id = _uuid(seed + 4)
    session_opened = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=installation_id,
        producer_instance_id=producer_instance_id,
        logical_session_id=logical_session_id,
        event_id=_uuid(seed + 5),
        event_sequence=1,
        event_kind=m.EventKind.SESSION_OPENED,
        payload=m.SessionOpenedPayloadV1(
            consent_epoch_id=consent_epoch_id,
            binding_id=binding_id,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest=HASH,
            retention_hours=24,
            microphone_accepted=True,
            typed_accepted=True,
            predecessor_session_id=None,
        ),
    )
    binding_opened = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=installation_id,
        producer_instance_id=producer_instance_id,
        logical_session_id=logical_session_id,
        event_id=_uuid(seed + 6),
        event_sequence=2,
        event_kind=m.EventKind.BINDING_OPENED,
        payload=m.BindingOpenedPayloadV1(
            binding_id=binding_id,
            binding_generation=1,
            microphone_available=True,
            typed_available=True,
        ),
    )
    return m.CreateEpochV1(
        protocol_version=1,
        installation_id=installation_id,
        producer_instance_id=producer_instance_id,
        consent_epoch_id=consent_epoch_id,
        logical_session_id=logical_session_id,
        binding_id=binding_id,
        binding_generation=1,
        consent_version=m.CONSENT_VERSION,
        disclosure_digest=HASH,
        retention_hours=24,
        microphone_accepted=True,
        typed_accepted=True,
        control_sequence=1,
        control_fingerprint_hash=HASH,
        session_opened=session_opened,
        binding_opened=binding_opened,
    )


def _capability(cls: type[Any], **values: object) -> Any:
    assert {field.name for field in dataclasses.fields(cls)} == set(values)
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    cls._validate(instance)
    return instance


def _turn_authority(
    m: ModuleType,
    name: str,
    create: Any,
    *,
    owner_generation: int,
    serial: int = 1,
) -> Any:
    common = {
        "protocol_version": 1,
        "owner_generation": owner_generation,
        "binding_id": create.binding_id,
        "binding_generation": create.binding_generation,
        "consent_epoch_id": create.consent_epoch_id,
        "logical_session_id": create.logical_session_id,
    }
    if name == "UserTurnAuthorityV1":
        values = {
            **common,
            "utterance_id": _uuid(1_000 + serial),
            "source": m.InputSource.TYPED,
            "input_incarnation": serial,
            "media_incarnation": None,
            "typed_sequence": serial,
            "routing_serial": serial,
            "routing_disposition": "response",
        }
    elif name == "ProactiveTurnAuthorityV1":
        values = {**common, "proactive_invocation_serial": serial}
    elif name == "ReplayTurnAuthorityV1":
        values = {
            **common,
            "replay_of_evidence_turn_id": _uuid(2_000 + serial),
            "replay_generation": serial,
        }
    elif name == "CommandAdmissionAuthorityV1":
        values = {
            **common,
            "utterance_id": _uuid(3_000 + serial),
            "source": m.InputSource.TYPED,
            "input_incarnation": serial,
            "media_incarnation": None,
            "typed_sequence": serial,
            "routing_serial": serial,
            "routing_disposition": "command",
        }
    else:
        raise AssertionError(f"unknown fixture authority {name}")
    return _capability(getattr(m, name), **values)


def _active_admission(
    a: ModuleType,
    m: ModuleType,
    *,
    owner_generation: int = 101,
    writer: Any | None = None,
    deny_filter: Any | None = None,
    quota_check: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    sink = a.BoundedEvidenceWriterQueueV1() if writer is None else writer
    operations = a.ConversationOperationScheduler(
        owner_generation=owner_generation,
        max_operations=64,
    )
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=owner_generation,
        writer_sink=sink,
        operation_scheduler=operations,
        deny_filter=deny_filter,
        quota_check=quota_check,
    )
    create = _create_epoch(m, seed=400)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    item = admission.try_enqueue_create_epoch(reservation)
    assert item is not None
    assert sink.get_nowait() is item
    retained = admission.complete_create_epoch(
        item,
        disposition=m.StoreDisposition.COMMITTED,
        binding_current=True,
    )
    assert retained is not None
    return admission, operations, sink, create


def _activate_next_epoch(
    m: ModuleType,
    admission: Any,
    writer: Any,
    *,
    seed: int,
) -> Any:
    """Complete one fresh durable create through the same admission owner."""

    create = _create_epoch(m, seed=seed)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    item = admission.try_enqueue_create_epoch(reservation)
    assert item is not None
    assert writer.get_nowait() is item
    retained = admission.complete_create_epoch(
        item,
        disposition=m.StoreDisposition.COMMITTED,
        binding_current=True,
    )
    assert retained is not None
    return create


def _owner_drain_authority(
    m: ModuleType,
    create: Any,
    admission: Any,
    *,
    owner_generation: int,
) -> Any:
    from hermes_realtime.evidence.lifecycle import EvidenceLifecycleOwner

    owner = EvidenceLifecycleOwner(owner_generation=owner_generation)
    owner.activate_binding(
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
    )
    owner.close_binding(m.BindingCloseReason.CLIENT_CLOSED)
    return owner.mint_drain_authority(
        final_admission_ordinal=admission.final_admission_ordinal,
    )


def _command_snapshot(m: ModuleType, create: Any, authority: Any, sequence: int) -> Any:
    return m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(4_000 + sequence),
        event_sequence=sequence,
        event_kind=m.EventKind.COMMAND_ROUTED,
        payload=m.CommandRoutedPayloadV1(
            utterance_id=authority.utterance_id,
            source=authority.source,
            routing_disposition="command",
        ),
    )


def _turn_snapshot(
    m: ModuleType,
    create: Any,
    lease: Any,
    sequence: int,
    event_kind: Any,
    payload: Any,
) -> Any:
    return m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(5_000 + sequence),
        event_sequence=sequence,
        event_kind=event_kind,
        payload=payload,
    )


def _projection_reservation(m: ModuleType) -> Any:
    return _capability(m.ProjectionReservation)


def _revoke_authority(
    m: ModuleType,
    create: Any,
    *,
    control_sequence: int = 2,
) -> Any:
    return _capability(
        m.ConsentRevokeAuthorityV1,
        protocol_version=1,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        revoke_gate_generation=1,
        control_sequence=control_sequence,
        control_fingerprint_hash=HASH,
        projection_reservation=_projection_reservation(m),
    )


def _consent_authority(m: ModuleType, create: Any, reservation: Any) -> Any:
    return _capability(
        m.ConsentCreateAuthorityV1,
        protocol_version=1,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        control_sequence=create.control_sequence,
        control_fingerprint_hash=create.control_fingerprint_hash,
        consent_version=create.consent_version,
        disclosure_digest=create.disclosure_digest,
        retention_hours=create.retention_hours,
        microphone_accepted=create.microphone_accepted,
        typed_accepted=create.typed_accepted,
        projection_reservation=_projection_reservation(m),
        create_epoch_reservation=reservation,
    )


def _binding_close_authority(m: ModuleType, create: Any, owner_generation: int) -> Any:
    return _capability(
        m.BindingCloseAuthorityV1,
        protocol_version=1,
        owner_generation=owner_generation,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
    )


def _binding_close_snapshot(m: ModuleType, create: Any, sequence: int = 3) -> Any:
    return m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(6_000 + sequence),
        event_sequence=sequence,
        event_kind=m.EventKind.BINDING_CLOSED,
        payload=m.BindingClosedPayloadV1(
            binding_id=create.binding_id,
            close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        ),
    )


def _close_and_seal_epoch(
    m: ModuleType,
    admission: Any,
    writer: Any,
    create: Any,
    *,
    owner_generation: int,
    close_sequence: int = 3,
) -> tuple[Any, Any, Any]:
    """Take one active epoch through its ordinary binding-close/seal terminal path."""

    close_authority = _binding_close_authority(m, create, owner_generation)
    close_snapshot = _binding_close_snapshot(m, create, sequence=close_sequence)
    assert (
        admission.try_close_binding(close_authority, close_snapshot)
        is m.AppendDisposition.ADMITTED
    )
    admission.complete_ordered_item(writer.get_nowait())

    final_event_sequence = close_sequence + 1
    seal_snapshot = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(6_500 + final_event_sequence),
        event_sequence=final_event_sequence,
        event_kind=m.EventKind.SESSION_SEAL_REQUESTED,
        payload=m.SessionSealRequestedPayloadV1(
            final_event_sequence=final_event_sequence,
            consent_epoch_id=create.consent_epoch_id,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest=HASH,
        ),
    )
    seal_authority = _capability(
        m.LifecycleSealAuthorityV1,
        protocol_version=1,
        owner_generation=owner_generation,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        final_event_sequence=final_event_sequence,
        close_epoch=True,
    )
    seal = m.SealEpochV1(
        protocol_version=1,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        binding_id=create.binding_id,
        final_event_sequence=final_event_sequence,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        close_epoch=True,
        admission_ordinal=admission.final_admission_ordinal + 1,
        snapshot=seal_snapshot,
    )
    assert admission.request_seal(seal_authority, seal) is m.SealDisposition.SEAL_QUEUED
    admission.complete_ordered_item(writer.get_nowait())
    return close_authority, seal_authority, seal


def _rollover_pair(
    m: ModuleType,
    create: Any,
    *,
    owner_generation: int,
    admission_ordinal: int = 3,
    seed: int = 7_000,
) -> tuple[Any, Any]:
    successor = _uuid(seed)
    expires = "2030-01-02T03:04:05.000000Z"
    predecessor_close = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(seed + 1),
        event_sequence=3,
        event_kind=m.EventKind.BINDING_CLOSED,
        payload=m.BindingClosedPayloadV1(
            binding_id=create.binding_id,
            close_reason=m.BindingCloseReason.CAPACITY_ROLLOVER,
        ),
    )
    predecessor_seal = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(seed + 2),
        event_sequence=4,
        event_kind=m.EventKind.SESSION_SEAL_REQUESTED,
        payload=m.SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=create.consent_epoch_id,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest=HASH,
        ),
    )
    successor_open = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=successor,
        event_id=_uuid(seed + 3),
        event_sequence=1,
        event_kind=m.EventKind.SESSION_OPENED,
        payload=m.SessionOpenedPayloadV1(
            consent_epoch_id=create.consent_epoch_id,
            binding_id=create.binding_id,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest=HASH,
            retention_hours=create.retention_hours,
            microphone_accepted=create.microphone_accepted,
            typed_accepted=create.typed_accepted,
            predecessor_session_id=create.logical_session_id,
        ),
    )
    successor_binding = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=successor,
        event_id=_uuid(seed + 4),
        event_sequence=2,
        event_kind=m.EventKind.BINDING_OPENED,
        payload=m.BindingOpenedPayloadV1(
            binding_id=create.binding_id,
            binding_generation=create.binding_generation,
            microphone_available=True,
            typed_available=True,
        ),
    )
    authority = _capability(
        m.RolloverAuthorityV1,
        protocol_version=1,
        owner_generation=owner_generation,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        predecessor_logical_session_id=create.logical_session_id,
        successor_logical_session_id=successor,
        successor_expires_at_utc=expires,
        reason=m.BindingCloseReason.CAPACITY_ROLLOVER,
    )
    command = m.RolloverSessionV1(
        protocol_version=1,
        consent_epoch_id=create.consent_epoch_id,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        predecessor_logical_session_id=create.logical_session_id,
        successor_logical_session_id=successor,
        predecessor_final_event_sequence=4,
        successor_expires_at_utc=expires,
        admission_ordinal=admission_ordinal,
        snapshots=(
            predecessor_close,
            predecessor_seal,
            successor_open,
            successor_binding,
        ),
    )
    return authority, command


def _admission() -> ModuleType:
    from hermes_realtime.evidence import admission

    return admission


def test_conversation_operation_reservations_are_exactly_bounded_and_shared() -> None:
    a = _admission()
    from hermes_realtime.evidence import ConversationOperationKind

    owner = a.ConversationOperationScheduler(owner_generation=7, max_operations=3)
    response = owner.try_reserve(ConversationOperationKind.RESPONSE)
    proactive = owner.try_reserve(ConversationOperationKind.PROACTIVE)
    replay = owner.try_reserve(ConversationOperationKind.REPLAY)

    assert response is not None
    assert proactive is not None
    assert replay is not None
    assert owner.try_reserve(ConversationOperationKind.RESPONSE) is None
    assert owner.active_count == 3
    assert [field.name for field in dataclasses.fields(response)] == [
        "owner_generation",
        "operation_serial",
        "kind",
    ]
    assert response.__dataclass_params__.frozen is True
    assert not hasattr(response, "__dict__")
    assert "7" not in repr(response)
    for copier in (copy.copy, copy.deepcopy, pickle.dumps):
        with pytest.raises(TypeError):
            copier(response)


def test_conversation_operation_owner_rejects_wrong_kind_owner_and_reuse() -> None:
    a = _admission()
    from hermes_realtime.evidence import ConversationOperationKind

    owner = a.ConversationOperationScheduler(owner_generation=11, max_operations=2)
    other = a.ConversationOperationScheduler(owner_generation=12, max_operations=2)
    reservation = owner.try_reserve(ConversationOperationKind.RESPONSE)
    assert reservation is not None

    with pytest.raises(a.ReservationError):
        other.consume(reservation, ConversationOperationKind.RESPONSE)
    with pytest.raises(a.ReservationError):
        owner.consume(reservation, ConversationOperationKind.PROACTIVE)

    owner.consume(reservation, ConversationOperationKind.RESPONSE)
    with pytest.raises(a.ReservationError):
        owner.consume(reservation, ConversationOperationKind.RESPONSE)
    owner.release(reservation)
    assert owner.active_count == 0
    with pytest.raises(a.ReservationError):
        owner.release(reservation)

    stale = owner.try_reserve(ConversationOperationKind.REPLAY)
    assert stale is not None
    owner.close()
    assert owner.active_count == 0
    with pytest.raises(a.ReservationError):
        owner.consume(stale, ConversationOperationKind.REPLAY)
    with pytest.raises(a.ReservationError):
        owner.release(stale)
    assert owner.try_reserve(ConversationOperationKind.RESPONSE) is None


def test_operation_bearer_and_injected_scheduler_owner_are_revalidated() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    operations = a.ConversationOperationScheduler(owner_generation=18, max_operations=1)
    reservation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert reservation is not None
    object.__setattr__(reservation, "operation_serial", reservation.operation_serial + 1)

    with pytest.raises(a.ReservationError, match="stale|tampered"):
        operations.validate(reservation, m.ConversationOperationKind.RESPONSE)
    assert operations.active_count == 1

    with pytest.raises(ValueError, match="owner generation"):
        a.EvidenceAdmissionControllerV1(
            enabled=True,
            owner_generation=19,
            writer_sink=a.BoundedEvidenceWriterQueueV1(),
            operation_scheduler=operations,
        )

    class StructuralSink:
        protocol_version = 1

        def put_nowait(self, item: Any) -> None:
            del item

    with pytest.raises(TypeError, match="exact BoundedEvidenceWriterQueueV1"):
        a.EvidenceAdmissionControllerV1(
            enabled=True,
            owner_generation=18,
            writer_sink=StructuralSink(),
        )


def test_idle_proactive_transfer_is_atomic_and_replay_cannot_be_source() -> None:
    a = _admission()
    from hermes_realtime.evidence import ConversationOperationKind

    owner = a.ConversationOperationScheduler(owner_generation=21, max_operations=1)
    idle = owner.try_reserve(ConversationOperationKind.PROACTIVE)
    assert idle is not None

    interrupt = owner.transfer_idle_proactive(idle)
    assert interrupt.kind is ConversationOperationKind.PROACTIVE
    assert interrupt.operation_serial == idle.operation_serial + 1
    assert owner.active_count == 1
    with pytest.raises(a.ReservationError):
        owner.release(idle)
    owner.release(interrupt)

    replay = owner.try_reserve(ConversationOperationKind.REPLAY)
    assert replay is not None
    with pytest.raises(a.ReservationError):
        owner.transfer_idle_proactive(replay)
    owner.release(replay)

    consumed = owner.try_reserve(ConversationOperationKind.PROACTIVE)
    assert consumed is not None
    owner.consume(consumed, ConversationOperationKind.PROACTIVE)
    with pytest.raises(a.ReservationError, match="consumed"):
        owner.transfer_idle_proactive(consumed)
    owner.release(consumed)


def test_create_epoch_reserves_five_credits_and_releases_or_transfers_exactly_once() -> None:
    a = _admission()
    from hermes_realtime.evidence import StoreDisposition
    from hermes_realtime.evidence import models as m

    writer = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=31,
        writer_sink=writer,
    )
    create = _create_epoch(m)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    assert admission.diagnostics().queue_record_count == 5
    assert admission.diagnostics().queue_canonical_bytes == 163_840

    item = admission.try_enqueue_create_epoch(reservation)
    assert item is not None
    assert writer.get_nowait() is item
    retained = admission.complete_create_epoch(
        item,
        disposition=StoreDisposition.COMMITTED,
        binding_current=True,
    )
    assert retained is not None
    assert retained.cleanup_only is False
    assert admission.diagnostics().queue_record_count == 3
    assert admission.diagnostics().queue_canonical_bytes == 98_304
    with pytest.raises(a.ReservationError):
        admission.complete_create_epoch(
            item,
            disposition=StoreDisposition.COMMITTED,
            binding_current=True,
        )

    stale_writer = a.BoundedEvidenceWriterQueueV1()
    stale_admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=32,
        writer_sink=stale_writer,
    )
    stale_reservation = stale_admission.try_reserve_create_epoch(_create_epoch(m, seed=20))
    assert stale_reservation is not None
    stale_item = stale_admission.try_enqueue_create_epoch(stale_reservation)
    assert stale_item is not None
    assert stale_writer.get_nowait() is stale_item
    cleanup = stale_admission.complete_create_epoch(
        stale_item,
        disposition=StoreDisposition.IDEMPOTENT,
        binding_current=False,
    )
    assert cleanup is not None
    assert cleanup.cleanup_only is True
    assert stale_admission.diagnostics().queue_record_count == 3
    stale_admission.release_session_control(cleanup)
    assert stale_admission.diagnostics().queue_record_count == 0
    assert stale_admission.diagnostics().queue_canonical_bytes == 0
    with pytest.raises(a.ReservationError):
        stale_admission.release_session_control(cleanup)

    failed_writer = a.BoundedEvidenceWriterQueueV1()
    failed_admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=33,
        writer_sink=failed_writer,
    )
    failed_reservation = failed_admission.try_reserve_create_epoch(
        _create_epoch(m, seed=40)
    )
    assert failed_reservation is not None
    failed_item = failed_admission.try_enqueue_create_epoch(failed_reservation)
    assert failed_item is not None
    assert failed_writer.get_nowait() is failed_item
    assert (
        failed_admission.complete_create_epoch(
            failed_item,
            disposition=StoreDisposition.FAULTED,
            binding_current=True,
        )
        is None
    )
    assert failed_admission.diagnostics().queue_record_count == 0
    assert failed_admission.diagnostics().queue_canonical_bytes == 0


def test_begin_consent_consumes_only_scheduler_owned_create_capability_and_signals_once() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    writer = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=34,
        writer_sink=writer,
    )
    create = _create_epoch(m, seed=50)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    authority = _consent_authority(m, create, reservation)

    ticket = admission.begin_consent(authority)
    assert ticket.disposition is m.ConsentDisposition.CREATE_PENDING
    assert not ticket.durability_event.is_set()
    assert admission.begin_consent(authority) is ticket
    item = writer.get_nowait()
    retained = admission.complete_create_epoch(
        item,
        disposition=m.StoreDisposition.COMMITTED,
        binding_current=True,
    )
    assert retained is not None
    assert ticket.disposition is m.ConsentDisposition.CONSENT_ACTIVATED
    assert ticket.durability_event.is_set()

    wrong_owner = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=35,
        writer_sink=a.BoundedEvidenceWriterQueueV1(),
    )
    with pytest.raises(a.ReservationError):
        wrong_owner.begin_consent(authority)


def test_active_epoch_rejects_second_create_before_any_credit_or_writer_mutation() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, _ = _active_admission(a, m, owner_generation=36)
    before = admission.diagnostics()

    assert admission.try_reserve_create_epoch(_create_epoch(m, seed=90)) is None
    after = admission.diagnostics()
    assert after.queue_record_count == before.queue_record_count == 3
    assert after.queue_canonical_bytes == before.queue_canonical_bytes == 98_304
    assert writer.ordered_count == 0


def test_complete_create_epoch_resets_epoch_latches_after_a_prior_seal() -> None:
    """A sealed epoch cannot leave close/seal or consumed-bearer state behind."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, first = _active_admission(a, m, owner_generation=37)
    consumed_command = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        first,
        owner_generation=37,
    )
    assert (
        admission.try_admit_command(
            consumed_command,
            _command_snapshot(m, first, consumed_command, sequence=3),
        )
        is m.CommandDisposition.ADMITTED
    )
    admission.complete_ordered_item(writer.get_nowait())
    old_close, old_seal_authority, old_seal = _close_and_seal_epoch(
        m,
        admission,
        writer,
        first,
        owner_generation=37,
        close_sequence=4,
    )
    assert admission.diagnostics().owner_state is m.OwnerState.STOPPED

    second = _activate_next_epoch(m, admission, writer, seed=500)

    assert admission.diagnostics().capture_state is m.CaptureState.ACTIVE
    assert admission._binding_close_admitted is False
    assert admission._seal_ticket_disposition is None
    assert admission._expiry_authority is None
    assert admission._revoke_authority is None
    assert admission._revoke_ticket is None
    assert admission._revoke_item is None
    assert admission._revoke_request_completed is False
    assert admission._session_tainted is False
    assert admission._taint_code is None
    assert admission._used_authorities == set()
    assert admission._evidence_turn_ids == set()
    assert admission._replayable_turn_count == 0

    assert (
        admission.try_close_binding(
            old_close,
            _binding_close_snapshot(m, first, sequence=4),
        )
        is m.AppendDisposition.INVALID_AUTHORITY
    )
    assert admission.request_seal(old_seal_authority, old_seal) is m.SealDisposition.WRITER_FAULT
    assert admission.diagnostics().capture_state is m.CaptureState.ACTIVE

    _close_and_seal_epoch(
        m,
        admission,
        writer,
        second,
        owner_generation=37,
    )
    assert admission.diagnostics().owner_state is m.OwnerState.STOPPED


def test_complete_create_epoch_recreates_cleanly_after_expiry_and_successful_purge() -> None:
    """Expiry and revoke tickets are epoch-scoped and cannot control later creates."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, first = _active_admission(a, m, owner_generation=38)
    old_expiry = _capability(
        m.SessionExpiryAuthorityV1,
        protocol_version=1,
        owner_generation=38,
        consent_epoch_id=first.consent_epoch_id,
        logical_session_id=first.logical_session_id,
        expires_at_utc="2030-01-02T03:04:05.000000Z",
        deadline_admission_ordinal=admission.final_admission_ordinal,
        mode=m.ExpiryMode.ERASE_STUCK,
    )
    assert admission.begin_expiry(old_expiry) is m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
    admission.complete_ordered_item(writer.get_nowait())
    assert admission.diagnostics().owner_state is m.OwnerState.STOPPED

    second = _activate_next_epoch(m, admission, writer, seed=600)
    assert admission._expiry_authority is None
    assert admission.begin_expiry(old_expiry) is m.ExpiryDisposition.WRITER_FAULT
    assert admission.diagnostics().capture_state is m.CaptureState.ACTIVE

    tainted_authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        second,
        owner_generation=38,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    lease_result = admission.try_reserve_user_turn(tainted_authority, operation)
    assert lease_result.lease is not None
    assert (
        admission.try_admit_user_final(
            lease_result.lease,
            tainted_authority,
            "x" * 4_097,
        )
        is m.AppendDisposition.REJECTED_OVERSIZE
    )
    assert admission.discard_unopened_user_turn(lease_result.lease, tainted_authority)
    assert admission._session_tainted is True

    old_revoke_authority = _revoke_authority(m, second)
    old_revoke_ticket = admission.begin_revoke(old_revoke_authority)
    request = writer.get_nowait()
    admission.complete_revoke_request(
        request,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )
    finalize = admission.try_enqueue_revoke_finalize(old_revoke_ticket)
    assert finalize is not None
    assert writer.get_nowait() is finalize
    admission.complete_revoke_finalize(finalize, m.RevokeDisposition.PURGE_COMPLETED)
    assert old_revoke_ticket.terminal_event.is_set()
    assert admission.diagnostics().owner_state is m.OwnerState.STOPPED

    third = _activate_next_epoch(m, admission, writer, seed=700)
    assert admission.diagnostics().capture_state is m.CaptureState.ACTIVE
    assert admission._session_tainted is False
    assert admission._taint_code is None
    assert admission._revoke_authority is None
    assert admission._revoke_ticket is None
    assert admission._revoke_item is None
    assert admission._revoke_request_completed is False
    with pytest.raises(a.ReservationError, match="different revoke|stale"):
        admission.begin_revoke(old_revoke_authority)
    with pytest.raises(a.ReservationError, match="stale"):
        admission.try_enqueue_revoke_finalize(old_revoke_ticket)

    fresh_ticket = admission.begin_revoke(_revoke_authority(m, third))
    assert fresh_ticket is not old_revoke_ticket
    request = writer.get_nowait()
    admission.complete_revoke_request(
        request,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )
    finalize = admission.try_enqueue_revoke_finalize(fresh_ticket)
    assert finalize is not None
    assert writer.get_nowait() is finalize
    admission.complete_revoke_finalize(finalize, m.RevokeDisposition.PURGE_COMPLETED)
    assert fresh_ticket.terminal_event.is_set()


def test_terminal_and_create_reservations_use_exact_count_and_byte_limits_atomically() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=41,
        writer_sink=a.BoundedEvidenceWriterQueueV1(),
    )
    terminals = [admission.try_reserve_terminal() for _ in range(32)]
    assert all(item is not None for item in terminals)
    assert admission.try_reserve_terminal() is None
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 64
    assert diagnostics.queue_canonical_bytes == 2_097_152
    assert admission.try_reserve_create_epoch(_create_epoch(m, seed=60)) is None

    first = terminals[0]
    assert first is not None
    admission.release_terminal_reservation(first)
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 62
    assert diagnostics.queue_canonical_bytes == 2_031_616
    assert admission.try_reserve_create_epoch(_create_epoch(m, seed=80)) is None

    second = terminals[1]
    assert second is not None
    admission.release_terminal_reservation(second)
    assert admission.try_reserve_create_epoch(_create_epoch(m, seed=100)) is None

    third = terminals[2]
    assert third is not None
    admission.release_terminal_reservation(third)
    create_reservation = admission.try_reserve_create_epoch(_create_epoch(m, seed=100))
    assert create_reservation is not None
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 63
    assert diagnostics.queue_canonical_bytes == 2_064_384

    admission.release_create_epoch_reservation(create_reservation)
    for terminal in terminals[3:]:
        assert terminal is not None
        admission.release_terminal_reservation(terminal)
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0


@pytest.mark.parametrize(
    ("authority_name", "operation_kind", "expected_turn_kind"),
    [
        ("UserTurnAuthorityV1", "RESPONSE", "USER_RESPONSE"),
        ("ProactiveTurnAuthorityV1", "PROACTIVE", "PROACTIVE_UPDATE"),
        ("ReplayTurnAuthorityV1", "REPLAY", "REPLAY"),
    ],
)
def test_turn_reservation_consumes_exact_matching_operation_and_reserves_terminals(
    authority_name: str,
    operation_kind: str,
    expected_turn_kind: str,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, _, create = _active_admission(a, m, owner_generation=201)
    authority = _turn_authority(
        m,
        authority_name,
        create,
        owner_generation=201,
    )
    kind = getattr(m.ConversationOperationKind, operation_kind)
    operation = operations.try_reserve(kind)
    assert operation is not None

    reserve = getattr(
        admission,
        {
            "UserTurnAuthorityV1": "try_reserve_user_turn",
            "ProactiveTurnAuthorityV1": "try_reserve_proactive_turn",
            "ReplayTurnAuthorityV1": "try_reserve_replay_turn",
        }[authority_name],
    )
    result = reserve(authority, operation)
    assert result.disposition is m.AppendDisposition.ADMITTED
    assert result.lease is not None
    assert result.lease.turn_kind is getattr(m.TurnKind, expected_turn_kind)
    assert result.lease.operation_serial == operation.operation_serial
    assert result.lease.owner_generation == 201
    assert admission.diagnostics().active_lease_count == 1
    assert admission.diagnostics().queue_record_count == 5
    operations.validate(operation, kind)
    with pytest.raises(a.ReservationError):
        operations.consume(operation, kind)


def test_wrong_operation_kind_is_invalid_without_consuming_either_bearer() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, _, create = _active_admission(a, m, owner_generation=211)
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=211,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None

    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is None
    assert result.disposition is m.AppendDisposition.INVALID_AUTHORITY
    operations.validate(
        operation,
        m.ConversationOperationKind.RESPONSE,
        require_unconsumed=True,
    )


def test_evidence_terminal_budget_failure_keeps_conversation_reservation_live() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, _, create = _active_admission(a, m, owner_generation=221)
    blockers = [admission.try_reserve_terminal() for _ in range(30)]
    assert all(item is not None for item in blockers)
    assert admission.diagnostics().queue_record_count == 63
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=221,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None

    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is None
    assert result.disposition is m.AppendDisposition.DROPPED_CAPACITY
    operations.validate(
        operation,
        m.ConversationOperationKind.RESPONSE,
        require_unconsumed=True,
    )
    assert operations.active_count == 1
    assert admission.diagnostics().capture_state is m.CaptureState.FAULTED


def test_command_admission_is_content_free_fifo_and_releases_exact_byte_credit() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=231)
    first_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=231,
        serial=1,
    )
    second_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=231,
        serial=2,
    )
    first_snapshot = _command_snapshot(m, create, first_authority, 3)
    second_snapshot = _command_snapshot(m, create, second_authority, 4)
    first_bytes = a.canonical_record_bytes(first_snapshot)
    second_bytes = a.canonical_record_bytes(second_snapshot)

    assert (
        admission.try_admit_command(first_authority, first_snapshot)
        is m.CommandDisposition.ADMITTED
    )
    assert (
        admission.try_admit_command(second_authority, second_snapshot)
        is m.CommandDisposition.ADMITTED
    )
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 5
    assert diagnostics.queue_canonical_bytes == 98_304 + first_bytes + second_bytes

    first_item = writer.get_nowait()
    second_item = writer.get_nowait()
    assert first_item.payload.snapshot.event_sequence == 3
    assert second_item.payload.snapshot.event_sequence == 4
    admission.complete_ordered_item(first_item)
    assert admission.diagnostics().queue_canonical_bytes == 98_304 + second_bytes
    admission.complete_ordered_item(second_item)
    assert admission.diagnostics().queue_record_count == 3
    assert admission.diagnostics().queue_canonical_bytes == 98_304
    with pytest.raises(a.ReservationError):
        admission.complete_ordered_item(first_item)
    assert (
        admission.try_admit_command(first_authority, first_snapshot)
        is m.CommandDisposition.INVALID_AUTHORITY
    )


@pytest.mark.parametrize("remaining_ordinals", (0, 1))
def test_command_ordinal_exhaustion_faults_before_charging_capacity(
    remaining_ordinals: int,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=23101)
    authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=23101,
    )
    snapshot = _command_snapshot(m, create, authority, 3)
    before = admission.diagnostics()
    physical_items_before = admission._ordered_physical_items
    admission._next_admission_ordinal = a._MAX_UNSIGNED_63 + 1 - remaining_ordinals

    expected = (
        m.CommandDisposition.WRITER_FAULT
        if remaining_ordinals == 0
        else m.CommandDisposition.ADMITTED
    )
    assert admission.try_admit_command(authority, snapshot) is expected
    after = admission.diagnostics()
    if remaining_ordinals == 0:
        with pytest.raises(Empty):
            writer.get_nowait()
        assert after.queue_record_count == before.queue_record_count
        assert after.queue_canonical_bytes == before.queue_canonical_bytes
        assert admission._ordered_physical_items == physical_items_before
        assert after.sticky_fault is m.WriterFault.SQLITE_FAULT
    else:
        assert writer.get_nowait() is not None


def test_capacity_drop_does_not_consume_command_authority() -> None:
    """A rejected publication remains independently authorized after real release."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=2311)
    for serial in range(1, 62):
        authority = _turn_authority(
            m,
            "CommandAdmissionAuthorityV1",
            create,
            owner_generation=2311,
            serial=serial,
        )
        assert (
            admission.try_admit_command(
                authority,
                _command_snapshot(m, create, authority, serial + 2),
            )
            is m.CommandDisposition.ADMITTED
        )

    retry_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=2311,
        serial=62,
    )
    retry_snapshot = _command_snapshot(m, create, retry_authority, 64)
    assert (
        admission.try_admit_command(retry_authority, retry_snapshot)
        is m.CommandDisposition.DROPPED_CAPACITY
    )

    admission.complete_ordered_item(writer.get_nowait())

    assert (
        admission.try_admit_command(retry_authority, retry_snapshot)
        is m.CommandDisposition.ADMITTED
    )


def test_dequeue_credit_release_never_takes_the_foreground_admission_lock() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=232)
    authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=232,
    )
    assert (
        admission.try_admit_command(
            authority,
            _command_snapshot(m, create, authority, 3),
        )
        is m.CommandDisposition.ADMITTED
    )
    item = writer.get_nowait()
    completed = Event()
    errors: list[BaseException] = []

    def release_credit() -> None:
        try:
            admission.complete_ordered_item(item)
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    admission._admission_lock.acquire()
    try:
        consumer = Thread(target=release_credit)
        consumer.start()
        assert completed.wait(timeout=1)
    finally:
        admission._admission_lock.release()
    consumer.join(timeout=1)
    assert not consumer.is_alive()
    assert errors == []
    assert admission.diagnostics().queue_record_count == 3


def test_blocked_writer_cannot_delay_or_raise_into_a_foreground_admission() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=233)
    first_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=233,
        serial=1,
    )
    second_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=233,
        serial=2,
    )
    assert (
        admission.try_admit_command(
            first_authority,
            _command_snapshot(m, create, first_authority, 3),
        )
        is m.CommandDisposition.ADMITTED
    )
    writer_blocked = Event()
    release_writer = Event()
    worker_done = Event()

    def blocked_writer() -> None:
        item = writer.get_nowait()
        writer_blocked.set()
        release_writer.wait()
        admission.complete_ordered_item(item)
        worker_done.set()

    worker = Thread(target=blocked_writer)
    worker.start()
    assert writer_blocked.wait(timeout=1)

    assert (
        admission.try_admit_command(
            second_authority,
            _command_snapshot(m, create, second_authority, 4),
        )
        is m.CommandDisposition.ADMITTED
    )
    assert not worker_done.is_set()
    release_writer.set()
    assert worker_done.wait(timeout=1)
    worker.join(timeout=1)
    second_item = writer.get_nowait()
    admission.complete_ordered_item(second_item)
    assert admission.diagnostics().queue_record_count == 3


def test_capacity_failure_after_sequence_allocation_taints_without_fabricating_record() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    writer = a.BoundedEvidenceWriterQueueV1()
    original_put_nowait = writer.put_nowait
    fail = False

    def controlled_put(item: Any) -> None:
        if fail:
            raise Full
        original_put_nowait(item)

    writer.put_nowait = controlled_put
    admission, _, _, create = _active_admission(
        a,
        m,
        owner_generation=241,
        writer=writer,
    )
    authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=241,
    )
    fail = True

    disposition = admission.try_admit_command(
        authority,
        _command_snapshot(m, create, authority, 3),
    )
    assert disposition is m.CommandDisposition.SESSION_TAINTED
    diagnostics = admission.diagnostics()
    assert diagnostics.capture_state is m.CaptureState.FAULTED
    assert diagnostics.sticky_fault is None
    assert diagnostics.queue_record_count == 3


def test_admission_detaches_queued_snapshot_from_caller_owned_payload() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=242)
    authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=242,
    )
    snapshot = _command_snapshot(m, create, authority, 3)
    assert (
        admission.try_admit_command(authority, snapshot)
        is m.CommandDisposition.ADMITTED
    )
    object.__setattr__(snapshot.payload, "routing_disposition", "mutated-after-admit")

    queued = writer.get_nowait()
    assert queued.payload.snapshot.payload.routing_disposition == "command"


def test_deny_quota_and_internal_exception_fail_closed_before_sequence_allocation() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    def denied(snapshot: Any) -> bool:
        del snapshot
        return True

    denied_admission, _, _, create = _active_admission(
        a,
        m,
        owner_generation=251,
        deny_filter=denied,
    )
    denied_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=251,
    )
    assert (
        denied_admission.try_admit_command(
            denied_authority,
            _command_snapshot(m, create, denied_authority, 3),
        )
        is m.CommandDisposition.SESSION_TAINTED
    )

    def quota_rejected(snapshot: Any) -> bool:
        del snapshot
        return False

    quota_admission, _, _, quota_create = _active_admission(
        a,
        m,
        owner_generation=252,
        quota_check=quota_rejected,
    )
    quota_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        quota_create,
        owner_generation=252,
    )
    assert (
        quota_admission.try_admit_command(
            quota_authority,
            _command_snapshot(m, quota_create, quota_authority, 3),
        )
        is m.CommandDisposition.SESSION_TAINTED
    )

    def broken_filter(snapshot: Any) -> bool:
        del snapshot
        raise RuntimeError("private detail must not escape")

    broken_admission, _, _, broken_create = _active_admission(
        a,
        m,
        owner_generation=253,
        deny_filter=broken_filter,
    )
    broken_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        broken_create,
        owner_generation=253,
    )
    assert (
        broken_admission.try_admit_command(
            broken_authority,
            _command_snapshot(m, broken_create, broken_authority, 3),
        )
        is m.CommandDisposition.WRITER_FAULT
    )
    assert broken_admission.diagnostics().sticky_fault is m.WriterFault.SQLITE_FAULT


def test_user_final_batch_publication_fault_rolls_back_shared_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2601,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2601,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    before_ordinal = admission._ordinary_scheduler.next_admission_ordinal
    before = admission.diagnostics()
    before_credits = admission._ordinary_scheduler.credits

    def reject_batch(_items: object) -> None:
        raise RuntimeError("deterministic publication fault")

    monkeypatch.setattr(writer, "put_ordered_batch_nowait", reject_batch)
    assert (
        admission.try_admit_user_final(
            lease,
            authority,
            "Scheduler-backed batch.",
        )
        is m.AppendDisposition.SESSION_TAINTED
    )
    after = admission.diagnostics()
    assert admission._ordinary_scheduler.next_admission_ordinal == before_ordinal
    assert admission._ordinary_scheduler.credits == before_credits
    assert after.queue_record_count == before.queue_record_count
    assert after.queue_canonical_bytes == before.queue_canonical_bytes


def test_user_turn_uses_ordinary_then_reserved_terminal_fifo_and_retires_once() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=261,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=261,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease

    opened = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.USER_RESPONSE,
            utterance_id=authority.utterance_id,
            replay_of_evidence_turn_id=None,
        ),
    )
    user_final = _turn_snapshot(
        m,
        create,
        lease,
        4,
        m.EventKind.USER_FINAL_ACCEPTED,
        m.UserFinalAcceptedPayloadV1(
            utterance_id=authority.utterance_id,
            evidence_turn_id=lease.evidence_turn_id,
            source=authority.source,
            routing_disposition="response",
            text="Exact accepted text.",
        ),
    )
    terminal_snapshot = _turn_snapshot(
        m,
        create,
        lease,
        5,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.USER_RESPONSE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=True,
            assistant_delivery_context_recorded=True,
        ),
    )
    settled = _turn_snapshot(
        m,
        create,
        lease,
        6,
        m.EventKind.TURN_SETTLED,
        m.TurnSettledPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            terminal_disposition=m.TerminalDisposition.COMPLETED,
            terminal_reason=m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
            context_committed=True,
            generated_segment_count=0,
            transport_confirmed_full_count=0,
        ),
    )

    assert admission.try_append_turn(lease, opened) is m.AppendDisposition.ADMITTED
    assert admission.try_append_turn(lease, user_final) is m.AppendDisposition.ADMITTED
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
        )
        is m.CauseDisposition.RECORDED
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
        )
        is m.CauseDisposition.ALREADY_RECORDED
    )
    resolution = admission.freeze_and_resolve_terminal_causes(
        lease,
        context_committed=True,
    )
    assert resolution.terminal_disposition is m.TerminalDisposition.COMPLETED
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.PROVIDER_FAILED,
        )
        is m.CauseDisposition.CAUSE_SET_FROZEN
    )
    assert (
        admission.try_append_turn(lease, terminal_snapshot)
        is m.AppendDisposition.ADMITTED
    )
    assert admission.try_append_turn(lease, settled) is m.AppendDisposition.ADMITTED
    assert admission.diagnostics().active_lease_count == 0

    queued = [writer.get_nowait() for _ in range(4)]
    assert [item.payload.snapshot.event_sequence for item in queued] == [3, 4, 5, 6]
    assert [item.payload.reservation_class for item in queued] == [
        m.QueueReservationClass.ORDINARY,
        m.QueueReservationClass.ORDINARY,
        m.QueueReservationClass.TERMINAL,
        m.QueueReservationClass.TERMINAL,
    ]
    assert queued[2].payload.lease_open_ordinal == queued[0].admission_ordinal
    assert queued[3].payload.lease_open_ordinal == queued[0].admission_ordinal
    for item in queued:
        admission.complete_ordered_item(item)
    assert admission.diagnostics().queue_record_count == 3
    assert admission.diagnostics().queue_canonical_bytes == 98_304
    assert (
        admission.try_append_turn(lease, settled)
        is m.AppendDisposition.INVALID_AUTHORITY
    )


def test_terminal_cause_resolution_is_exception_order_independent() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, _, create = _active_admission(a, m, owner_generation=271)
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=271,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease

    for cause in (
        m.TerminalReason.CALLER_CANCELLED,
        m.TerminalReason.PROVIDER_FAILED,
        m.TerminalReason.CONSENT_REVOKED,
        m.TerminalReason.BARGE_IN,
    ):
        assert (
            admission.record_terminal_cause(lease.terminal_cause, cause)
            is m.CauseDisposition.RECORDED
        )
    resolution = admission.freeze_and_resolve_terminal_causes(
        lease,
        context_committed=False,
    )
    assert resolution.terminal_disposition is m.TerminalDisposition.REVOKED
    assert resolution.terminal_reason is m.TerminalReason.CONSENT_REVOKED


def test_terminal_capability_and_settlement_must_match_the_frozen_resolution() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=269,
    )
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=269,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    lease_result = admission.try_reserve_proactive_turn(authority, operation)
    assert lease_result.lease is not None
    lease = lease_result.lease

    opened = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            utterance_id=None,
            replay_of_evidence_turn_id=None,
        ),
    )
    snapshot = _turn_snapshot(
        m,
        create,
        lease,
        4,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=False,
            assistant_delivery_context_recorded=False,
        ),
    )
    assert admission.try_append_turn(lease, opened) is m.AppendDisposition.ADMITTED
    assert admission.try_append_turn(lease, snapshot) is m.AppendDisposition.ADMITTED

    sink_token = lease.terminal_cause.sink_token
    object.__setattr__(lease.terminal_cause, "sink_token", b"x" * 16)
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.CONSENT_REVOKED,
        )
        is m.CauseDisposition.INVALID_AUTHORITY
    )
    object.__setattr__(lease.terminal_cause, "sink_token", sink_token)
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.CONSENT_REVOKED,
        )
        is m.CauseDisposition.RECORDED
    )
    resolution = admission.freeze_and_resolve_terminal_causes(
        lease,
        context_committed=False,
    )
    assert resolution.terminal_disposition is m.TerminalDisposition.REVOKED

    contradictory = _turn_snapshot(
        m,
        create,
        lease,
        5,
        m.EventKind.TURN_SETTLED,
        m.TurnSettledPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            terminal_disposition=m.TerminalDisposition.CANCELLED,
            terminal_reason=m.TerminalReason.CALLER_CANCELLED,
            context_committed=False,
            generated_segment_count=0,
            transport_confirmed_full_count=0,
        ),
    )
    assert (
        admission.try_append_turn(lease, contradictory)
        is m.AppendDisposition.SESSION_TAINTED
    )
    assert admission.diagnostics().active_lease_count == 0
    assert admission.diagnostics().capture_state is m.CaptureState.FAULTED
    queued = [writer.get_nowait(), writer.get_nowait()]
    assert [item.payload.snapshot.event_sequence for item in queued] == [3, 4]
    for item in queued:
        admission.complete_ordered_item(item)
    operations.release(operation)


def test_turn_event_kind_and_order_are_closed_and_fail_without_writer_exception() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, _, create = _active_admission(a, m, owner_generation=281)
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=281,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    premature = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=False,
            assistant_delivery_context_recorded=False,
        ),
    )

    assert (
        admission.try_append_turn(lease, premature)
        is m.AppendDisposition.INVALID_AUTHORITY
    )
    assert admission.diagnostics().queue_record_count == 5


def test_user_final_capture_is_atomic_and_enforces_exact_text_bound() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2811,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2811,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None

    assert (
        admission.try_admit_user_final(result.lease, authority, "x" * 4_096)
        is m.AppendDisposition.ADMITTED
    )
    queued = [writer.get_nowait(), writer.get_nowait()]
    assert [item.payload.snapshot.event_kind for item in queued] == [
        m.EventKind.TURN_OPENED,
        m.EventKind.USER_FINAL_ACCEPTED,
    ]
    assert queued[1].payload.snapshot.payload.text == "x" * 4_096

    second_authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2811,
        serial=2,
    )
    second_operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert second_operation is not None
    second_result = admission.try_reserve_user_turn(second_authority, second_operation)
    assert second_result.lease is not None
    assert (
        admission.try_admit_user_final(
            second_result.lease,
            second_authority,
            "y" * 4_097,
        )
        is m.AppendDisposition.REJECTED_OVERSIZE
    )
    assert writer.ordered_count == 0
    assert admission.diagnostics().capture_state is m.CaptureState.FAULTED


def test_spawn_failure_atomically_settles_open_user_turn_and_taints() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2812,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2812,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Captured before spawn.")
        is m.AppendDisposition.ADMITTED
    )

    assert admission.settle_spawn_failed(lease) is m.AppendDisposition.ADMITTED
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.capture_state is m.CaptureState.FAULTED

    queued = [writer.get_nowait() for _ in range(4)]
    assert [item.payload.snapshot.event_kind for item in queued] == [
        m.EventKind.TURN_OPENED,
        m.EventKind.USER_FINAL_ACCEPTED,
        m.EventKind.TURN_SNAPSHOT,
        m.EventKind.TURN_SETTLED,
    ]
    terminal_snapshot = queued[2].payload.snapshot.payload
    settled = queued[3].payload.snapshot.payload
    assert terminal_snapshot.model_context_admitted is False
    assert terminal_snapshot.assistant_delivery_context_recorded is False
    assert settled.terminal_disposition is m.TerminalDisposition.FAILED
    assert settled.terminal_reason is m.TerminalReason.TASK_SPAWN_FAILED
    assert settled.context_committed is False
    assert [item.payload.reservation_class for item in queued] == [
        m.QueueReservationClass.ORDINARY,
        m.QueueReservationClass.ORDINARY,
        m.QueueReservationClass.TERMINAL,
        m.QueueReservationClass.TERMINAL,
    ]


def test_generated_segment_admission_mints_canonical_identity_and_enforces_bound() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2813,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2813,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )

    assert (
        admission.try_admit_generated(lease, "g" * 4_096)
        is m.AppendDisposition.ADMITTED
    )
    generated = [writer.get_nowait() for _ in range(3)][2].payload.snapshot
    assert generated.event_kind is m.EventKind.ASSISTANT_SEGMENT_GENERATED
    assert generated.payload.evidence_turn_id == lease.evidence_turn_id
    assert generated.payload.segment_ordinal == 1
    assert generated.payload.text == "g" * 4_096
    m.validate_canonical_uuid4(
        generated.payload.evidence_segment_id,
        field_name="evidence_segment_id",
    )

    assert (
        admission.try_admit_generated(lease, "h" * 4_097)
        is m.AppendDisposition.REJECTED_OVERSIZE
    )
    assert writer.ordered_count == 0
    assert admission.diagnostics().capture_state is m.CaptureState.FAULTED


def test_transport_confirmation_resolves_generated_segment_and_exact_attempt_lineage() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2814,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2814,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.try_admit_generated(lease, "Answer.")
        is m.AppendDisposition.ADMITTED
    )
    synthesis_attempt_id = "00000000-0000-4000-8000-000000000171"
    transport_attempt_id = "00000000-0000-4000-8000-000000000172"

    assert admission.try_admit_transport_confirmed_full(
        lease,
        segment_ordinal=1,
        synthesis_attempt_id=synthesis_attempt_id,
        transport_attempt_id=transport_attempt_id,
        text="Answer.",
    ) is m.AppendDisposition.ADMITTED

    queued = [writer.get_nowait() for _ in range(4)]
    generated = queued[2].payload.snapshot.payload
    confirmed = queued[3].payload.snapshot
    assert confirmed.event_kind is m.EventKind.ASSISTANT_CHUNK_TRANSPORT_CONFIRMED_FULL
    assert confirmed.payload.evidence_segment_id == generated.evidence_segment_id
    assert confirmed.payload.synthesis_attempt_id == synthesis_attempt_id
    assert confirmed.payload.transport_attempt_id == transport_attempt_id
    assert confirmed.payload.chunk_ordinal == 1
    assert confirmed.payload.text == "Answer."
    m.validate_canonical_uuid4(
        confirmed.payload.evidence_chunk_id,
        field_name="evidence_chunk_id",
    )


def test_completed_turn_settlement_uses_exact_counts_and_retires_lease() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2815,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2815,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.try_admit_generated(lease, "Answer.")
        is m.AppendDisposition.ADMITTED
    )

    assert admission.settle_completed(
        lease,
        queued_chunk_count=1,
        started_chunk_count=1,
        transport_confirmed_full_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED

    queued = [writer.get_nowait() for _ in range(5)]
    terminal = [item.payload.snapshot for item in queued[-2:]]
    assert [record.event_kind for record in terminal] == [
        m.EventKind.TURN_SNAPSHOT,
        m.EventKind.TURN_SETTLED,
    ]
    assert terminal[0].payload == m.TurnSnapshotPayloadV1(
        evidence_turn_id=lease.evidence_turn_id,
        turn_kind=m.TurnKind.USER_RESPONSE,
        generated_segment_count=1,
        queued_chunk_count=1,
        started_chunk_count=1,
        transport_confirmed_full_count=0,
        model_context_admitted=True,
        assistant_delivery_context_recorded=False,
    )
    assert terminal[1].payload.terminal_disposition is m.TerminalDisposition.COMPLETED
    assert terminal[1].payload.terminal_reason is m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED
    assert terminal[1].payload.context_committed is True
    outcome = admission.settled_terminal_outcome(lease)
    assert outcome is not None
    assert outcome.terminal_disposition is m.TerminalDisposition.COMPLETED
    assert outcome.terminal_reason is m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED
    assert outcome.context_committed is True
    assert admission.diagnostics().active_lease_count == 0


@pytest.mark.parametrize(("queued_chunk_count", "started_chunk_count"), [(0, 0), (10, 10)])
def test_terminal_queue_charges_exact_detached_canonical_bytes_and_releases_once(
    queued_chunk_count: int,
    started_chunk_count: int,
) -> None:
    """Terminal reservations retain capacity, but queued items hold only their bytes."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=28_155 + queued_chunk_count,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=28_155 + queued_chunk_count,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )

    # Leave the ordinary records charged and consume all but one spare credit.
    # Terminal enqueue must still be capacity-authorized by its earlier maximum
    # reservation, then reconcile to its actual canonical byte representation.
    edge_reservations = [admission.try_reserve_terminal() for _ in range(28)]
    assert all(reservation is not None for reservation in edge_reservations)
    before_terminal = admission.diagnostics()
    assert before_terminal.queue_record_count == 63
    assert before_terminal.queue_canonical_bytes <= a.MAX_QUEUE_CANONICAL_BYTES

    assert admission.settle_completed(
        lease,
        queued_chunk_count=queued_chunk_count,
        started_chunk_count=started_chunk_count,
        transport_confirmed_full_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED

    queued = [writer.get_nowait() for _ in range(4)]
    terminal_items = queued[-2:]
    terminal_sizes = tuple(
        a.canonical_record_bytes(item.payload.snapshot) for item in terminal_items
    )
    assert terminal_sizes[0] != terminal_sizes[1]
    after_terminal = admission.diagnostics()
    assert after_terminal.queue_record_count == before_terminal.queue_record_count
    assert after_terminal.queue_canonical_bytes == (
        before_terminal.queue_canonical_bytes
        - a.TERMINAL_CANONICAL_BYTE_CREDITS
        + sum(terminal_sizes)
    )
    assert after_terminal.queue_canonical_bytes <= a.MAX_QUEUE_CANONICAL_BYTES

    for item in queued[:2]:
        admission.complete_ordered_item(item)
    admission.complete_ordered_item(terminal_items[0])
    assert admission.diagnostics().queue_canonical_bytes == (
        a.SESSION_CONTROL_CANONICAL_BYTE_CREDITS
        + terminal_sizes[1]
        + 28 * a.TERMINAL_CANONICAL_BYTE_CREDITS
    )
    with pytest.raises(a.ReservationError, match="stale"):
        admission.complete_ordered_item(terminal_items[0])
    admission.complete_ordered_item(terminal_items[1])
    for reservation in edge_reservations:
        assert reservation is not None
        admission.release_terminal_reservation(reservation)
    assert admission.diagnostics().queue_record_count == 3
    assert admission.diagnostics().queue_canonical_bytes == a.SESSION_CONTROL_CANONICAL_BYTE_CREDITS


def test_full_single_terminal_enqueue_restores_credits_and_keeps_lease_retryable() -> None:
    """A sink refusal must not consume the terminal reservation or sequence."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    writer = a.BoundedEvidenceWriterQueueV1()
    original_put_nowait = writer.put_nowait
    reject_terminal = False

    def put_nowait(item: Any) -> None:
        if reject_terminal:
            raise Full
        original_put_nowait(item)

    writer.put_nowait = put_nowait
    admission, operations, _, create = _active_admission(
        a,
        m,
        owner_generation=28_156,
        writer=writer,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=28_156,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
        )
        is m.CauseDisposition.RECORDED
    )
    resolution = admission.freeze_and_resolve_terminal_causes(lease, context_committed=True)
    assert resolution.terminal_reason is m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED
    terminal = _turn_snapshot(
        m,
        create,
        lease,
        5,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.USER_RESPONSE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=True,
            assistant_delivery_context_recorded=True,
        ),
    )
    settled = _turn_snapshot(
        m,
        create,
        lease,
        6,
        m.EventKind.TURN_SETTLED,
        m.TurnSettledPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            terminal_disposition=m.TerminalDisposition.COMPLETED,
            terminal_reason=m.TerminalReason.AUTHORITATIVE_CLOSE_COMPLETED,
            context_committed=True,
            generated_segment_count=0,
            transport_confirmed_full_count=0,
        ),
    )
    before = admission.diagnostics()
    terminal_bytes = a.canonical_record_bytes(terminal)

    reject_terminal = True
    assert admission.try_append_turn(lease, terminal) is m.AppendDisposition.DROPPED_CAPACITY
    assert admission.diagnostics() == before
    assert writer.ordered_count == 2
    assert admission.diagnostics().active_lease_count == 1

    reject_terminal = False
    assert admission.try_append_turn(lease, terminal) is m.AppendDisposition.ADMITTED
    after_retry = admission.diagnostics()
    assert after_retry.queue_record_count == before.queue_record_count
    assert after_retry.queue_canonical_bytes == (
        before.queue_canonical_bytes - a.MAX_CANONICAL_RECORD_BYTES + terminal_bytes
    )
    assert admission.try_append_turn(lease, settled) is m.AppendDisposition.ADMITTED
    assert admission.diagnostics().active_lease_count == 0


def test_terminal_outcome_is_absent_when_terminal_admission_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=28151,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=28151,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )

    def reject_terminal_batch(_self: object, _items: object) -> None:
        raise RuntimeError("injected terminal enqueue failure")

    monkeypatch.setattr(
        a.BoundedEvidenceWriterQueueV1,
        "put_ordered_batch_nowait",
        reject_terminal_batch,
    )
    assert admission.settle_completed(
        lease,
        queued_chunk_count=0,
        started_chunk_count=0,
        transport_confirmed_full_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.SESSION_TAINTED
    assert admission.settled_terminal_outcome(lease) is None
    assert writer.ordered_count == 2
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 5
    assert diagnostics.queue_canonical_bytes == (
        a.SESSION_CONTROL_CANONICAL_BYTE_CREDITS
        + sum(
            a.canonical_record_bytes(item.payload.snapshot)
            for item in (writer.get_nowait(), writer.get_nowait())
        )
    )


def test_failed_turn_settlement_resolves_all_causes_before_freeze() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2816,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2816,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.CALLER_CANCELLED,
        )
        is m.CauseDisposition.RECORDED
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.TRANSPORT_FAILED,
        )
        is m.CauseDisposition.RECORDED
    )

    assert admission.settle_terminal(
        lease,
        queued_chunk_count=1,
        started_chunk_count=1,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED

    queued = [writer.get_nowait() for _ in range(4)]
    snapshot, settled = [item.payload.snapshot for item in queued[-2:]]
    assert snapshot.payload.queued_chunk_count == 1
    assert snapshot.payload.started_chunk_count == 1
    assert snapshot.payload.model_context_admitted is False
    assert settled.payload.terminal_disposition is m.TerminalDisposition.FAILED
    assert settled.payload.terminal_reason is m.TerminalReason.TRANSPORT_FAILED
    assert settled.payload.context_committed is False
    assert admission.diagnostics().active_lease_count == 0


def test_generated_but_undelivered_turn_settles_with_context_uncommitted() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2817,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2817,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.try_admit_generated(lease, "Generated but never delivered.")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.PROVIDER_FAILED,
        )
        is m.CauseDisposition.RECORDED
    )

    assert admission.settle_terminal(
        lease,
        queued_chunk_count=0,
        started_chunk_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED

    queued = [writer.get_nowait() for _ in range(5)]
    snapshot, settled = [item.payload.snapshot for item in queued[-2:]]
    assert snapshot.payload.generated_segment_count == 1
    assert snapshot.payload.model_context_admitted is False
    assert snapshot.payload.assistant_delivery_context_recorded is False
    assert settled.payload.context_committed is False
    assert settled.payload.terminal_reason is m.TerminalReason.PROVIDER_FAILED


def test_controller_settlement_freezes_causes_and_is_exactly_once() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=2818,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=2818,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    assert (
        admission.try_admit_user_final(lease, authority, "Question?")
        is m.AppendDisposition.ADMITTED
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.CALLER_CANCELLED,
        )
        is m.CauseDisposition.RECORDED
    )
    assert admission.settle_terminal(
        lease,
        queued_chunk_count=0,
        started_chunk_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED
    assert writer.ordered_count == 4
    queued_before = tuple(writer.get_nowait() for _ in range(4))
    assert writer.ordered_count == 0

    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.PROVIDER_FAILED,
        )
        is m.CauseDisposition.CAUSE_SET_FROZEN
    )
    assert admission.settle_terminal(
        lease,
        queued_chunk_count=0,
        started_chunk_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.INVALID_AUTHORITY
    assert writer.ordered_count == 0
    terminal = [item.payload.snapshot for item in queued_before[-2:]]
    assert terminal[-1].payload.terminal_reason is m.TerminalReason.CALLER_CANCELLED


def test_hostile_4097_code_point_source_is_rejected_before_model_reconstruction() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(a, m, owner_generation=282)
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=282,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    opened = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.USER_RESPONSE,
            utterance_id=authority.utterance_id,
            replay_of_evidence_turn_id=None,
        ),
    )
    assert admission.try_append_turn(lease, opened) is m.AppendDisposition.ADMITTED

    payload = object.__new__(m.UserFinalAcceptedPayloadV1)
    object.__setattr__(payload, "utterance_id", authority.utterance_id)
    object.__setattr__(payload, "evidence_turn_id", lease.evidence_turn_id)
    object.__setattr__(payload, "source", authority.source)
    object.__setattr__(payload, "routing_disposition", "response")
    object.__setattr__(payload, "text", "x" * 4_097)
    hostile = object.__new__(m.EvidenceSnapshotV1)
    object.__setattr__(hostile, "schema_version", 1)
    object.__setattr__(hostile, "installation_id", create.installation_id)
    object.__setattr__(hostile, "producer_instance_id", create.producer_instance_id)
    object.__setattr__(hostile, "logical_session_id", create.logical_session_id)
    object.__setattr__(hostile, "event_id", _uuid(6_050))
    object.__setattr__(hostile, "event_sequence", 4)
    object.__setattr__(hostile, "event_kind", m.EventKind.USER_FINAL_ACCEPTED)
    object.__setattr__(hostile, "payload", payload)

    assert (
        admission.try_append_turn(lease, hostile)
        is m.AppendDisposition.REJECTED_OVERSIZE
    )
    diagnostics = admission.diagnostics()
    assert diagnostics.capture_state is m.CaptureState.FAULTED
    assert diagnostics.sticky_fault is None
    assert writer.ordered_count == 1


def test_revoke_and_owner_drain_use_independent_coalescing_priority_lanes() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=291)
    revoke_authority = _revoke_authority(m, create)
    revoke_ticket = admission.begin_revoke(revoke_authority)
    assert revoke_ticket.disposition is m.RevokeDisposition.CLOSED_NOT_DURABLE
    assert revoke_ticket.durability_event is not revoke_ticket.terminal_event
    assert admission.begin_revoke(revoke_authority) is revoke_ticket
    assert admission.diagnostics().pending_revoke is True
    assert admission.diagnostics().capture_state is m.CaptureState.REVOKED_PURGING

    drain_authority = _owner_drain_authority(
        m,
        create,
        admission,
        owner_generation=291,
    )
    drain_ticket = admission.request_drain(drain_authority)
    assert drain_ticket.disposition is m.DrainDisposition.DRAIN_QUEUED
    assert admission.request_drain(drain_authority) is drain_ticket
    assert drain_ticket.disposition is m.DrainDisposition.DRAIN_QUEUED

    revoke_item = writer.get_nowait()
    drain_item = writer.get_nowait()
    assert revoke_item.lane is a.WriterQueueLane.REVOKE
    assert drain_item.lane is a.WriterQueueLane.DRAIN
    admission.complete_revoke_request(
        revoke_item,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )
    assert revoke_ticket.durability_event.is_set()
    assert not revoke_ticket.terminal_event.is_set()
    with pytest.raises(a.ReservationError, match="drain prerequisites"):
        admission.complete_drain(drain_item, m.DrainDisposition.STOPPED)
    finalize = admission.try_enqueue_revoke_finalize(revoke_ticket)
    assert finalize is not None
    assert writer.get_nowait() is finalize
    admission.complete_revoke_finalize(finalize, m.RevokeDisposition.PURGE_COMPLETED)
    admission.complete_drain(drain_item, m.DrainDisposition.STOPPED)
    assert drain_ticket.terminal_event.is_set()
    with pytest.raises(a.ReservationError):
        admission.request_drain(object())


def test_drain_enqueue_failure_is_a_writer_fault_not_a_durable_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed drain enqueue cannot claim the owner reached durable STOPPED."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=292)
    authority = _owner_drain_authority(
        m,
        create,
        admission,
        owner_generation=292,
    )

    def reject_drain(_item: object) -> None:
        raise Full

    monkeypatch.setattr(writer, "put_nowait", reject_drain)
    ticket = admission.request_drain(authority)

    assert ticket.disposition is m.DrainDisposition.WRITER_FAULT
    assert ticket.terminal_event.is_set()
    assert admission.diagnostics().owner_state is m.OwnerState.FAULTED
    assert admission.request_drain(authority) is ticket
    assert ticket.disposition is m.DrainDisposition.WRITER_FAULT


@pytest.mark.parametrize(
    ("terminal_disposition", "capture_state", "purge_required"),
    [
        ("purge_completed", "idle", False),
        ("purge_failed", "purge_failed", True),
    ],
)
def test_revoke_terminal_outcome_releases_epoch_credits_and_closes_owner(
    terminal_disposition: str,
    capture_state: str,
    purge_required: bool,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=292)
    ticket = admission.begin_revoke(_revoke_authority(m, create))
    request = writer.get_nowait()
    admission.complete_revoke_request(
        request,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )
    finalize = admission.try_enqueue_revoke_finalize(ticket)
    assert finalize is not None
    assert writer.get_nowait() is finalize

    outcome = m.RevokeDisposition(terminal_disposition)
    admission.complete_revoke_finalize(finalize, outcome)

    assert ticket.disposition is outcome
    assert ticket.durability_event.is_set()
    assert ticket.terminal_event.is_set()
    diagnostics = admission.diagnostics()
    assert diagnostics.pending_revoke is False
    assert diagnostics.purge_required is purge_required
    assert diagnostics.capture_state is m.CaptureState(capture_state)
    assert diagnostics.owner_state is m.OwnerState.STOPPED
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0


def test_revoke_dispatch_exception_is_a_closed_terminal_pre_durability_failure() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=293)

    def fail_dispatch(item: Any) -> None:
        del item
        raise RuntimeError("injected revoke dispatch failure")

    writer.put_nowait = fail_dispatch
    ticket = admission.begin_revoke(_revoke_authority(m, create))

    assert ticket.disposition is m.RevokeDisposition.WRITER_FAULT
    assert ticket.durability_event.is_set()
    assert ticket.terminal_event.is_set()
    diagnostics = admission.diagnostics()
    assert diagnostics.pending_revoke is False
    assert diagnostics.purge_required is True
    assert diagnostics.owner_state is m.OwnerState.FAULTED
    assert diagnostics.capture_state is m.CaptureState.FAULTED


def test_revoke_finalize_enqueue_failure_terminally_faults_the_durable_ticket() -> None:
    """A finalizer publication failure cannot leave a durable revoke pending."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=294)
    ticket = admission.begin_revoke(_revoke_authority(m, create))
    request = writer.get_nowait()
    original_put_nowait = writer.put_nowait
    attempted: list[object] = []

    def fail_revoke_finalizer(item: Any) -> None:
        if type(item.payload) is m.RevokeFinalizeV1:
            attempted.append(item)
            raise RuntimeError("injected revoke finalizer sink failure")
        original_put_nowait(item)

    writer.put_nowait = fail_revoke_finalizer
    admission.complete_revoke_request(
        request,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )

    assert len(attempted) == 1
    assert ticket.disposition is m.RevokeDisposition.WRITER_FAULT
    assert ticket.durability_event.is_set()
    assert ticket.terminal_event.is_set()
    diagnostics = admission.diagnostics()
    assert diagnostics.owner_state is m.OwnerState.FAULTED
    assert diagnostics.capture_state is m.CaptureState.FAULTED
    assert diagnostics.pending_revoke is False
    assert diagnostics.purge_required is True
    assert admission._revoke_item is None
    assert admission.try_enqueue_revoke_finalize(ticket) is None
    assert admission.try_enqueue_revoke_finalize(ticket) is None
    with pytest.raises(Empty):
        writer.get_nowait()


def test_automatic_and_manual_revoke_finalizers_enqueue_one_live_item() -> None:
    """The finalizer lock returns the same item to both race participants."""

    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=295)
    ticket = admission.begin_revoke(_revoke_authority(m, create))
    request = writer.get_nowait()
    automatic_entered = Event()
    manual_attempted = Event()
    release_automatic = Event()
    original_enqueue = admission._try_enqueue_revoke_finalize_locked

    def gate_automatic(self: Any, supplied_ticket: Any) -> Any:
        del self
        automatic_entered.set()
        assert manual_attempted.wait(timeout=1)
        assert release_automatic.wait(timeout=1)
        return original_enqueue(supplied_ticket)

    admission._try_enqueue_revoke_finalize_locked = MethodType(  # type: ignore[method-assign]
        gate_automatic,
        admission,
    )
    automatic_errors: list[BaseException] = []
    manual_results: list[Any] = []

    def complete_request() -> None:
        try:
            admission.complete_revoke_request(
                request,
                m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
            )
        except BaseException as error:
            automatic_errors.append(error)

    def manually_enqueue() -> None:
        manual_attempted.set()
        manual_results.append(admission.try_enqueue_revoke_finalize(ticket))

    automatic = Thread(target=complete_request)
    automatic.start()
    assert automatic_entered.wait(timeout=1)
    manual = Thread(target=manually_enqueue)
    manual.start()
    assert manual_attempted.wait(timeout=1)
    release_automatic.set()
    automatic.join(timeout=1)
    manual.join(timeout=1)

    assert not automatic.is_alive()
    assert not manual.is_alive()
    assert automatic_errors == []
    [finalizer] = manual_results
    assert type(finalizer.payload) is m.RevokeFinalizeV1
    assert admission._revoke_item is finalizer
    assert writer.get_nowait() is finalizer
    with pytest.raises(Empty):
        writer.get_nowait()
    assert admission.try_enqueue_revoke_finalize(ticket) is finalizer
    assert admission._revoke_item is finalizer


def test_writer_queue_has_exact_ordered_capacity_and_finite_maintenance_priority() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    writer = a.BoundedEvidenceWriterQueueV1()
    create = _create_epoch(m, seed=700)
    ordered = [
        a.EvidenceWriterQueueItemV1(
            protocol_version=1,
            lane=a.WriterQueueLane.ORDERED,
            payload=create,
            admission_ordinal=index + 1,
        )
        for index in range(65)
    ]
    for item in ordered[:64]:
        writer.put_nowait(item)
    with pytest.raises(Full):
        writer.put_nowait(ordered[64])

    revoke = a.EvidenceWriterQueueItemV1(
        protocol_version=1,
        lane=a.WriterQueueLane.REVOKE,
        payload=m.RevokeRequestV1(
            protocol_version=1,
            erasure_request_id=_uuid(800),
            consent_epoch_id=create.consent_epoch_id,
            control_sequence=2,
            control_fingerprint_hash=HASH,
            last_admission_ordinal=1,
        ),
        admission_ordinal=66,
    )
    drain = a.EvidenceWriterQueueItemV1(
        protocol_version=1,
        lane=a.WriterQueueLane.DRAIN,
        payload=m.DrainAndStopV1(
            protocol_version=1,
            owner_generation=1,
            final_admission_ordinal=66,
        ),
        admission_ordinal=67,
    )
    writer.put_nowait(drain)
    writer.put_nowait(revoke)

    assert writer.get_nowait() is revoke
    for item in ordered[:64]:
        assert writer.get_nowait() is item
    assert writer.get_nowait() is drain


def test_writer_runtime_owner_latches_dispatch_failure_without_retry() -> None:
    from threading import Event

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterRuntimeOwnerV1

    writer = a.BoundedEvidenceWriterQueueV1()
    create = _create_epoch(m, seed=703)
    item = a.EvidenceWriterQueueItemV1(
        protocol_version=1,
        lane=a.WriterQueueLane.ORDERED,
        payload=create,
        admission_ordinal=1,
    )
    failure = RuntimeError("injected dispatcher failure")
    calls: list[object] = []
    attempted = Event()

    def dispatch(exact_item: object) -> None:
        calls.append(exact_item)
        attempted.set()
        raise failure

    owner = EvidenceWriterRuntimeOwnerV1(source=writer, dispatch=dispatch)
    writer.put_nowait(item)

    assert attempted.wait(timeout=1)
    assert owner.close(timeout=1) is True
    assert calls == [item]
    assert owner.failure is failure
    with pytest.raises(a.ReservationError):
        writer.put_nowait(item)


def test_writer_runtime_owner_dispatches_on_owned_thread_and_closes_idempotently() -> None:
    from threading import Event, get_ident

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterRuntimeOwnerV1

    writer = a.BoundedEvidenceWriterQueueV1()
    create = _create_epoch(m, seed=702)
    item = a.EvidenceWriterQueueItemV1(
        protocol_version=1,
        lane=a.WriterQueueLane.ORDERED,
        payload=create,
        admission_ordinal=1,
    )
    caller_thread_id = get_ident()
    dispatched: list[tuple[object, int]] = []
    completed = Event()

    def dispatch(exact_item: object) -> None:
        dispatched.append((exact_item, get_ident()))
        completed.set()

    owner = EvidenceWriterRuntimeOwnerV1(source=writer, dispatch=dispatch)
    writer.put_nowait(item)

    assert completed.wait(timeout=1)
    assert dispatched[0][0] is item
    assert dispatched[0][1] != caller_thread_id
    assert owner.close(timeout=1) is True
    assert owner.close(timeout=1) is True
    assert owner.is_running is False
    assert owner.failure is None


def test_writer_queue_blocking_consumer_wakes_for_item_then_stop() -> None:
    from threading import Event, Thread

    a = _admission()
    from hermes_realtime.evidence import models as m

    writer = a.BoundedEvidenceWriterQueueV1()
    create = _create_epoch(m, seed=701)
    ordered = a.EvidenceWriterQueueItemV1(
        protocol_version=1,
        lane=a.WriterQueueLane.ORDERED,
        payload=create,
        admission_ordinal=1,
    )
    received: list[object | None] = []
    errors: list[BaseException] = []
    ready = Event()
    item_received = Event()

    def consume() -> None:
        ready.set()
        try:
            received.append(writer.get_blocking())
            item_received.set()
            received.append(writer.get_blocking())
        except BaseException as error:
            errors.append(error)
            item_received.set()

    consumer = Thread(target=consume)
    consumer.start()
    assert ready.wait(timeout=1)
    writer.put_nowait(ordered)
    assert item_received.wait(timeout=1)
    if errors:
        raise errors[0]
    writer.stop_consumer()
    consumer.join(timeout=1)

    assert not consumer.is_alive()
    assert received == [ordered, None]
    with pytest.raises(a.ReservationError):
        writer.put_nowait(ordered)


@pytest.mark.parametrize(
    ("payload_type_name", "transport_method", "completion_method", "disposition"),
    (
        (
            "RevokeRequestV1",
            "commit_revoke_request",
            "complete_revoke_request",
            "revoke_durably_scheduled",
        ),
        (
            "RevokeFinalizeV1",
            "finalize_revoke",
            "complete_revoke_finalize",
            "purge_completed",
        ),
        (
            "DrainAndStopV1",
            "drain_and_close",
            "complete_drain",
            "stopped",
        ),
    ),
)
def test_writer_dispatcher_maps_owner_only_lanes_exactly(
    payload_type_name: str,
    transport_method: str,
    completion_method: str,
    disposition: str,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    payload = object.__new__(getattr(m, payload_type_name))
    item = object.__new__(a.EvidenceWriterQueueItemV1)
    object.__setattr__(item, "protocol_version", 1)
    object.__setattr__(item, "lane", a.WriterQueueLane.REVOKE)
    object.__setattr__(item, "payload", payload)
    object.__setattr__(item, "admission_ordinal", 1)
    if payload_type_name == "DrainAndStopV1":
        object.__setattr__(item, "lane", a.WriterQueueLane.DRAIN)
    calls: list[tuple[str, object]] = []
    completions: list[tuple[str, object, object]] = []
    typed_disposition: object
    if payload_type_name == "DrainAndStopV1":
        typed_disposition = m.DrainDisposition(disposition)
    else:
        typed_disposition = m.RevokeDisposition(disposition)

    class Source:
        def get_nowait(self) -> object:
            return item

    class Admission:
        def __getattr__(self, name: str) -> object:
            if name == completion_method:
                return lambda queued, result: completions.append((name, queued, result))
            raise AttributeError(name)

    class Transport:
        def __getattr__(self, name: str) -> object:
            if name == transport_method:
                return lambda exact_payload: (
                    calls.append((name, exact_payload)),
                    typed_disposition,
                )[1]
            raise AttributeError(name)

    dispatcher = object.__new__(EvidenceWriterDispatcherV1)
    dispatcher._source = Source()
    dispatcher._admission = Admission()
    dispatcher._transport = Transport()
    dispatcher._binding_is_current = lambda _command: True

    assert dispatcher.dispatch_one() is True
    assert calls == [(transport_method, payload)]
    assert completions == [(completion_method, item, typed_disposition)]


@pytest.mark.parametrize(
    ("payload_kind", "transport_method"),
    (
        ("binding_close", "append_binding_close"),
        ("rollover", "rollover_session"),
        ("expiry", "expire_session"),
        ("seal", "seal_epoch"),
    ),
)
def test_writer_dispatcher_maps_every_ordered_control_payload(
    payload_kind: str,
    transport_method: str,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    class Source:
        def get_nowait(self) -> object:
            return item

    class Admission:
        def complete_ordered_item(self, queued: object, *, writer_succeeded: bool = True) -> None:
            completions.append(("ordered", queued, writer_succeeded))

        def complete_rollover(self, queued: object, disposition: object) -> None:
            completions.append(("rollover", queued, disposition))

    class Transport:
        def __getattr__(self, name: str) -> object:
            if name in {
                "create_epoch",
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: dispatch(name, payload)
            raise AttributeError(name)

    payload = object.__new__(
        {
            "binding_close": m.BindingCloseV1,
            "rollover": m.RolloverSessionV1,
            "expiry": m.ExpireSessionV1,
            "seal": m.SealEpochV1,
        }[payload_kind]
    )
    item = object.__new__(a.EvidenceWriterQueueItemV1)
    object.__setattr__(item, "protocol_version", 1)
    object.__setattr__(item, "lane", a.WriterQueueLane.ORDERED)
    object.__setattr__(item, "payload", payload)
    object.__setattr__(item, "admission_ordinal", 1)
    calls: list[tuple[str, object]] = []
    completions: list[tuple[object, ...]] = []

    def dispatch(name: str, exact_payload: object) -> object:
        calls.append((name, exact_payload))
        if payload_kind == "expiry":
            return m.PurgeDisposition.PURGE_COMPLETED
        return m.StoreDisposition.COMMITTED

    dispatcher = object.__new__(EvidenceWriterDispatcherV1)
    dispatcher._source = Source()
    dispatcher._admission = Admission()
    dispatcher._transport = Transport()
    dispatcher._binding_is_current = lambda _command: True

    assert dispatcher.dispatch_one() is True
    assert calls == [(transport_method, payload)]
    if payload_kind == "rollover":
        assert completions == [("rollover", item, m.StoreDisposition.COMMITTED)]
    else:
        assert completions == [("ordered", item, True)]


def test_writer_dispatcher_completes_consent_only_for_current_binding() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    writer = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=304,
        writer_sink=writer,
    )
    create = _create_epoch(m, seed=404)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    ticket = admission.begin_consent(_consent_authority(m, create, reservation))
    calls: list[object] = []

    class Transport:
        def __getattr__(self, name: str) -> object:
            if name in {
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

        def create_epoch(self, command: object) -> m.StoreDisposition:
            calls.append(command)
            return m.StoreDisposition.COMMITTED

        def append_record(self, payload: object) -> m.StoreDisposition:
            raise AssertionError(payload)

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda command: command is create,
    )

    assert dispatcher.dispatch_one() is True
    assert calls == [create]
    assert ticket.durability_event.is_set()
    assert ticket.disposition is m.ConsentDisposition.CONSENT_ACTIVATED
    assert admission.diagnostics().capture_state is m.CaptureState.ACTIVE


def test_writer_dispatcher_faults_a_raised_create_and_releases_exact_credits() -> None:
    """A transport exception is a completed failed create, never a stranded ticket."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    writer = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=305,
        writer_sink=writer,
    )
    create = _create_epoch(m, seed=405)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    ticket = admission.begin_consent(_consent_authority(m, create, reservation))

    class Transport:
        def create_epoch(self, _command: object) -> m.StoreDisposition:
            raise RuntimeError("synthetic create failure")

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda _command: True,
    )

    assert dispatcher.dispatch_one() is True
    assert ticket.durability_event.is_set()
    assert ticket.disposition is m.ConsentDisposition.CREATE_FAILED
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert admission.operation_scheduler.active_count == 0


@pytest.mark.parametrize(
    ("callback_case", "system_error"),
    (
        ("raised", None),
        ("non_bool", None),
        ("keyboard_interrupt", KeyboardInterrupt),
        ("system_exit", SystemExit),
    ),
)
def test_writer_dispatcher_fault_completes_malformed_create_binding_callback_once(
    callback_case: str,
    system_error: type[BaseException] | None,
) -> None:
    """The post-create binding boundary cannot strand its exact ticket."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    writer = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=306,
        writer_sink=writer,
    )
    create = _create_epoch(m, seed=406)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    ticket = admission.begin_consent(_consent_authority(m, create, reservation))

    class Transport:
        def create_epoch(self, received: object) -> m.StoreDisposition:
            assert received is create
            return m.StoreDisposition.COMMITTED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    def binding_is_current(received: object) -> bool:
        assert received is create
        if callback_case == "raised":
            raise RuntimeError("synthetic binding callback failure")
        if callback_case == "non_bool":
            return 1  # type: ignore[return-value]
        if callback_case == "keyboard_interrupt":
            raise KeyboardInterrupt
        raise SystemExit

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=binding_is_current,
    )

    if system_error is None:
        assert dispatcher.dispatch_one() is True
    else:
        with pytest.raises(system_error):
            dispatcher.dispatch_one()

    assert ticket.durability_event.is_set()
    assert ticket.disposition is m.ConsentDisposition.CREATE_FAILED
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert admission.operation_scheduler.active_count == 0


@pytest.mark.parametrize(
    "invalid_request_disposition",
    (
        "purge_completed",
        "purge_failed",
    ),
)
def test_writer_dispatcher_fault_completes_live_lease_revoke_request_purge_return(
    invalid_request_disposition: str,
) -> None:
    """A request-stage purge return cannot consume a still-live turn's authority."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=3071,
    )
    before_lease = admission.diagnostics()
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=3071,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    live_lease = admission.diagnostics()
    ticket = admission.begin_revoke(_revoke_authority(m, create))

    class Transport:
        def commit_revoke_request(self, _command: m.RevokeRequestV1) -> m.RevokeDisposition:
            return m.RevokeDisposition(invalid_request_disposition)

        def __getattr__(self, name: str) -> object:
            if name in {
                "create_epoch",
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda _command: True,
    )

    assert dispatcher.dispatch_one() is True

    assert ticket.disposition is m.RevokeDisposition.WRITER_FAULT
    assert ticket.durability_event.is_set()
    assert ticket.terminal_event.is_set()
    diagnostics = admission.diagnostics()
    assert diagnostics.owner_state is m.OwnerState.FAULTED
    assert diagnostics.capture_state is m.CaptureState.FAULTED
    assert diagnostics.pending_revoke is False
    assert diagnostics.purge_required is True
    assert diagnostics.active_lease_count == 1
    assert diagnostics.queue_record_count == live_lease.queue_record_count
    assert diagnostics.queue_canonical_bytes == live_lease.queue_canonical_bytes
    assert operations.active_count == 1

    assert admission.discard_unopened_user_turn(lease, authority) is True
    cleaned = admission.diagnostics()
    assert cleaned.active_lease_count == 0
    assert cleaned.queue_record_count == before_lease.queue_record_count
    assert cleaned.queue_canonical_bytes == before_lease.queue_canonical_bytes
    operations.release(operation)
    assert operations.active_count == 0


def test_turn_reservation_consume_failure_retires_lease_without_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, _, create = _active_admission(a, m, owner_generation=425)
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=425,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None

    def fail_consume(self: Any, supplied: Any, kind: Any) -> None:
        del self, supplied, kind
        raise a.ReservationError("injected consume failure")

    monkeypatch.setattr(type(operations), "consume", fail_consume)
    results: list[Any] = []
    worker = Thread(
        target=lambda: results.append(
            admission.try_reserve_proactive_turn(authority, operation)
        )
    )
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive()
    [result] = results
    assert result.lease is None
    assert result.disposition is m.AppendDisposition.INVALID_AUTHORITY
    assert admission.diagnostics().active_lease_count == 0
    operations.release(operation)


@pytest.mark.parametrize(
    "disallowed_disposition",
    (
        "closed_not_durable",
        "revoke_durably_scheduled",
        "already_scheduled",
        "control_timed_out",
        "writer_fault",
    ),
)
def test_writer_dispatcher_maps_each_exact_disallowed_revoke_finalize_to_purge_failure(
    disallowed_disposition: str,
) -> None:
    """Finalize only accepts purge-terminal outcomes from its transport boundary."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=307,
    )
    ticket = admission.begin_revoke(_revoke_authority(m, create))
    calls: list[str] = []

    class Transport:
        def commit_revoke_request(self, _command: object) -> m.RevokeDisposition:
            calls.append("request")
            return m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        def finalize_revoke(self, _command: object) -> m.RevokeDisposition:
            calls.append("finalize")
            return m.RevokeDisposition(disallowed_disposition)

        def __getattr__(self, name: str) -> object:
            if name in {
                "create_epoch",
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda _command: True,
    )

    assert dispatcher.dispatch_one() is True
    assert ticket.disposition is m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED
    assert ticket.durability_event.is_set()
    assert admission.try_enqueue_revoke_finalize(ticket) is not None
    assert dispatcher.dispatch_one() is True

    assert calls == ["request", "finalize"]
    assert ticket.disposition is m.RevokeDisposition.PURGE_FAILED
    assert ticket.terminal_event.is_set()
    diagnostics = admission.diagnostics()
    assert diagnostics.pending_revoke is False
    assert diagnostics.purge_required is True
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert operations.active_count == 0


def test_writer_runtime_owner_cleans_up_after_malformed_create_binding_callback() -> None:
    """A completed create callback fault leaves the owned consumer reusable for close."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import (
        EvidenceWriterDispatcherV1,
        EvidenceWriterRuntimeOwnerV1,
    )

    writer = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=308,
        writer_sink=writer,
    )
    create = _create_epoch(m, seed=408)
    reservation = admission.try_reserve_create_epoch(create)
    assert reservation is not None
    ticket = admission.begin_consent(_consent_authority(m, create, reservation))

    class Transport:
        def create_epoch(self, _command: object) -> m.StoreDisposition:
            return m.StoreDisposition.IDEMPOTENT

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda _command: 1,  # type: ignore[arg-type]
    )
    owner = EvidenceWriterRuntimeOwnerV1(source=writer, dispatch=dispatcher.dispatch_item)

    try:
        assert ticket.durability_event.wait(timeout=1)
    finally:
        assert owner.close(timeout=1) is True

    assert ticket.disposition is m.ConsentDisposition.CREATE_FAILED
    assert owner.failure is None
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0


def test_writer_runtime_owner_cleans_up_after_disallowed_revoke_finalize_member() -> None:
    """A finalized revoke fault-completes before the owned consumer can stop."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import (
        EvidenceWriterDispatcherV1,
        EvidenceWriterRuntimeOwnerV1,
    )

    admission, _, writer, create = _active_admission(a, m, owner_generation=309)
    ticket = admission.begin_revoke(_revoke_authority(m, create))

    class Transport:
        def commit_revoke_request(self, _command: object) -> m.RevokeDisposition:
            return m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        def finalize_revoke(self, _command: object) -> m.RevokeDisposition:
            return m.RevokeDisposition.WRITER_FAULT

        def __getattr__(self, name: str) -> object:
            if name in {
                "create_epoch",
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda _command: True,
    )
    owner = EvidenceWriterRuntimeOwnerV1(source=writer, dispatch=dispatcher.dispatch_item)

    try:
        assert ticket.durability_event.wait(timeout=1)
        assert ticket.terminal_event.wait(timeout=1)
    finally:
        assert owner.close(timeout=1) is True

    assert ticket.disposition is m.RevokeDisposition.PURGE_FAILED
    assert owner.failure is None
    diagnostics = admission.diagnostics()
    assert diagnostics.pending_revoke is False
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0


@pytest.mark.parametrize(
    ("payload_kind", "transport_method"),
    (
        ("create", "create_epoch"),
        ("binding_close", "append_binding_close"),
        ("rollover", "rollover_session"),
        ("expiry", "expire_session"),
        ("seal", "seal_epoch"),
        ("revoke_request", "commit_revoke_request"),
        ("revoke_finalize", "finalize_revoke"),
        ("drain", "drain_and_close"),
        ("record", "append_record"),
    ),
)
def test_writer_dispatcher_fault_completes_each_malformed_transport_return_once(
    payload_kind: str,
    transport_method: str,
) -> None:
    """A wrong transport result is the same one-shot fault as an exception."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    payload_type = {
        "create": m.CreateEpochV1,
        "binding_close": m.BindingCloseV1,
        "rollover": m.RolloverSessionV1,
        "expiry": m.ExpireSessionV1,
        "seal": m.SealEpochV1,
        "revoke_request": m.RevokeRequestV1,
        "revoke_finalize": m.RevokeFinalizeV1,
        "drain": m.DrainAndStopV1,
        "record": m.QueuedEvidenceRecordV1,
    }[payload_kind]
    payload = object.__new__(payload_type)
    item = object.__new__(a.EvidenceWriterQueueItemV1)
    object.__setattr__(item, "protocol_version", 1)
    object.__setattr__(item, "lane", a.WriterQueueLane.ORDERED)
    object.__setattr__(item, "payload", payload)
    object.__setattr__(item, "admission_ordinal", 1)
    completions: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    class Admission:
        def __getattr__(self, name: str) -> object:
            if name not in {
                "complete_create_epoch",
                "complete_ordered_item",
                "complete_rollover",
                "complete_revoke_request",
                "complete_revoke_finalize",
                "complete_drain",
            }:
                raise AttributeError(name)

            def complete(*args: object, **kwargs: object) -> None:
                completions.append((name, args, kwargs))

            return complete

    class Transport:
        def __getattr__(self, name: str) -> object:
            if name == transport_method:
                return lambda _payload: object()
            raise AttributeError(name)

    dispatcher = object.__new__(EvidenceWriterDispatcherV1)
    dispatcher._admission = Admission()
    dispatcher._transport = Transport()
    dispatcher._binding_is_current = lambda _command: True

    dispatcher.dispatch_item(item)

    if payload_kind == "create":
        expected = (
            "complete_create_epoch",
            (item,),
            {"disposition": m.StoreDisposition.FAULTED, "binding_current": False},
        )
    elif payload_kind == "rollover":
        expected = ("complete_rollover", (item, m.StoreDisposition.FAULTED), {})
    elif payload_kind == "revoke_request":
        expected = (
            "complete_revoke_request",
            (item, m.RevokeDisposition.WRITER_FAULT),
            {},
        )
    elif payload_kind == "revoke_finalize":
        expected = (
            "complete_revoke_finalize",
            (item, m.RevokeDisposition.PURGE_FAILED),
            {},
        )
    elif payload_kind == "drain":
        expected = ("complete_drain", (item, m.DrainDisposition.WRITER_FAULT), {})
    else:
        expected = ("complete_ordered_item", (item,), {"writer_succeeded": False})
    assert completions == [expected]


@pytest.mark.parametrize(
    ("payload_kind", "transport_method", "completion_method", "fault_disposition"),
    (
        ("binding_close", "append_binding_close", "complete_ordered_item", False),
        ("rollover", "rollover_session", "complete_rollover", "faulted"),
        ("expiry", "expire_session", "complete_ordered_item", False),
        ("seal", "seal_epoch", "complete_ordered_item", False),
        ("revoke_request", "commit_revoke_request", "complete_revoke_request", "writer_fault"),
        (
            "revoke_finalize",
            "finalize_revoke",
            "complete_revoke_finalize",
            "purge_failed",
        ),
        ("drain", "drain_and_close", "complete_drain", "writer_fault"),
        ("record", "append_record", "complete_ordered_item", False),
    ),
)
def test_writer_dispatcher_faults_every_dequeued_noncreate_transport_exception(
    payload_kind: str,
    transport_method: str,
    completion_method: str,
    fault_disposition: object,
) -> None:
    """Each transport boundary has one exact admission-owned fault completion."""

    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    payload_type = {
        "binding_close": m.BindingCloseV1,
        "rollover": m.RolloverSessionV1,
        "expiry": m.ExpireSessionV1,
        "seal": m.SealEpochV1,
        "revoke_request": m.RevokeRequestV1,
        "revoke_finalize": m.RevokeFinalizeV1,
        "drain": m.DrainAndStopV1,
        "record": m.QueuedEvidenceRecordV1,
    }[payload_kind]
    payload = object.__new__(payload_type)
    item = object.__new__(a.EvidenceWriterQueueItemV1)
    object.__setattr__(item, "protocol_version", 1)
    object.__setattr__(item, "lane", a.WriterQueueLane.ORDERED)
    object.__setattr__(item, "payload", payload)
    object.__setattr__(item, "admission_ordinal", 1)
    completions: list[tuple[object, ...]] = []

    class Admission:
        def complete_ordered_item(self, queued: object, *, writer_succeeded: bool) -> None:
            completions.append(("complete_ordered_item", queued, writer_succeeded))

        def complete_rollover(self, queued: object, disposition: object) -> None:
            completions.append(("complete_rollover", queued, disposition))

        def complete_revoke_request(self, queued: object, disposition: object) -> None:
            completions.append(("complete_revoke_request", queued, disposition))

        def complete_revoke_finalize(self, queued: object, disposition: object) -> None:
            completions.append(("complete_revoke_finalize", queued, disposition))

        def complete_drain(self, queued: object, disposition: object) -> None:
            completions.append(("complete_drain", queued, disposition))

    class Transport:
        def __getattr__(self, name: str) -> object:
            if name == transport_method:
                return lambda _payload: (_ for _ in ()).throw(RuntimeError(name))
            raise AttributeError(name)

    dispatcher = object.__new__(EvidenceWriterDispatcherV1)
    dispatcher._admission = Admission()
    dispatcher._transport = Transport()
    dispatcher._binding_is_current = lambda _command: True

    dispatcher.dispatch_item(item)

    expected_disposition: object = fault_disposition
    if fault_disposition == "faulted":
        expected_disposition = m.StoreDisposition.FAULTED
    elif fault_disposition == "writer_fault":
        expected_disposition = (
            m.DrainDisposition.WRITER_FAULT
            if payload_kind == "drain"
            else m.RevokeDisposition.WRITER_FAULT
        )
    elif fault_disposition == "purge_failed":
        expected_disposition = m.RevokeDisposition.PURGE_FAILED
    assert completions == [(completion_method, item, expected_disposition)]


def test_writer_dispatcher_persists_and_completes_one_ordinary_item() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    admission, _, writer, create = _active_admission(a, m, owner_generation=303)
    authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=303,
    )
    snapshot = _command_snapshot(m, create, authority, 3)
    assert admission.try_admit_command(authority, snapshot) is m.CommandDisposition.ADMITTED
    queued = writer.get_nowait()
    writer.put_nowait(queued)
    calls: list[object] = []

    class Transport:
        def __getattr__(self, name: str) -> object:
            if name in {
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
                "drain_and_close",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

        def create_epoch(self, command: object) -> m.StoreDisposition:
            raise AssertionError(command)

        def append_record(self, payload: object) -> m.StoreDisposition:
            calls.append(payload)
            return m.StoreDisposition.COMMITTED

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),
        binding_is_current=lambda _command: True,
    )

    assert dispatcher.dispatch_one() is True
    assert calls == [queued.payload]
    assert writer.ordered_count == 0
    assert admission.diagnostics().queue_record_count == 3
    assert dispatcher.dispatch_one() is False


def test_writer_dispatcher_real_ordinary_fault_releases_exact_credits_once() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import EvidenceWriterDispatcherV1

    admission, _, writer, create = _active_admission(a, m, owner_generation=3031)
    authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=3031,
    )
    snapshot = _command_snapshot(m, create, authority, 3)
    assert admission.try_admit_command(authority, snapshot) is m.CommandDisposition.ADMITTED
    charged = admission.diagnostics()

    class Transport:
        def append_record(self, payload: object) -> m.StoreDisposition:
            raise RuntimeError(f"writer failed for {type(payload).__name__}")

        def __getattr__(self, name: str) -> object:
            return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))

    dispatcher = EvidenceWriterDispatcherV1(
        source=writer,
        admission=admission,
        transport=Transport(),  # type: ignore[arg-type]
        binding_is_current=lambda _command: True,
    )

    assert dispatcher.dispatch_one() is True
    after = admission.diagnostics()
    assert after.queue_record_count == charged.queue_record_count - 1
    assert after.queue_canonical_bytes == (
        charged.queue_canonical_bytes - a.canonical_record_bytes(snapshot)
    )
    assert after.sticky_fault is m.WriterFault.SQLITE_FAULT
    assert dispatcher.dispatch_one() is False


def test_failed_ordered_writer_completion_latches_fault_without_success_transition() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(
        a,
        m,
        owner_generation=302,
    )
    close_authority = _binding_close_authority(m, create, 302)
    close_snapshot = _binding_close_snapshot(m, create)
    assert (
        admission.try_close_binding(close_authority, close_snapshot)
        is m.AppendDisposition.ADMITTED
    )
    admission.complete_ordered_item(writer.get_nowait())
    seal_snapshot = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(6_101),
        event_sequence=4,
        event_kind=m.EventKind.SESSION_SEAL_REQUESTED,
        payload=m.SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=create.consent_epoch_id,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest=HASH,
        ),
    )
    seal_authority = _capability(
        m.LifecycleSealAuthorityV1,
        protocol_version=1,
        owner_generation=302,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        final_event_sequence=4,
        close_epoch=True,
    )
    seal = m.SealEpochV1(
        protocol_version=1,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        binding_id=create.binding_id,
        final_event_sequence=4,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        close_epoch=True,
        admission_ordinal=4,
        snapshot=seal_snapshot,
    )
    assert admission.request_seal(seal_authority, seal) is m.SealDisposition.SEAL_QUEUED
    item = writer.get_nowait()

    admission.complete_ordered_item(item, writer_succeeded=False)

    diagnostics = admission.diagnostics()
    assert diagnostics.sticky_fault is m.WriterFault.SQLITE_FAULT
    assert diagnostics.capture_state is m.CaptureState.FAULTED
    assert diagnostics.owner_state is not m.OwnerState.STOPPED
    assert diagnostics.queue_record_count == 1
    assert diagnostics.queue_canonical_bytes == 32_768
    assert writer.ordered_count == 0
    with pytest.raises(a.ReservationError):
        admission.complete_ordered_item(item, writer_succeeded=False)


def test_binding_close_and_owner_seal_consume_then_release_retained_control_credits() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=301)
    close_authority = _binding_close_authority(m, create, 301)
    close_snapshot = _binding_close_snapshot(m, create)
    assert (
        admission.try_close_binding(close_authority, close_snapshot)
        is m.AppendDisposition.ADMITTED
    )
    assert admission.diagnostics().queue_record_count == 3
    close_item = writer.get_nowait()
    assert type(close_item.payload) is m.BindingCloseV1
    assert close_item.payload.admission_ordinal == 3
    admission.complete_ordered_item(close_item)
    assert admission.diagnostics().queue_record_count == 2
    assert admission.diagnostics().queue_canonical_bytes == 65_536

    seal_snapshot = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(6_100),
        event_sequence=4,
        event_kind=m.EventKind.SESSION_SEAL_REQUESTED,
        payload=m.SessionSealRequestedPayloadV1(
            final_event_sequence=4,
            consent_epoch_id=create.consent_epoch_id,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest=HASH,
        ),
    )
    seal_authority = _capability(
        m.LifecycleSealAuthorityV1,
        protocol_version=1,
        owner_generation=301,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        final_event_sequence=4,
        close_epoch=True,
    )
    seal = m.SealEpochV1(
        protocol_version=1,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        binding_id=create.binding_id,
        final_event_sequence=4,
        close_reason=m.BindingCloseReason.CLIENT_CLOSED,
        close_epoch=True,
        admission_ordinal=4,
        snapshot=seal_snapshot,
    )
    assert admission.request_seal(seal_authority, seal) is m.SealDisposition.SEAL_QUEUED
    assert (
        admission.request_seal(seal_authority, seal)
        is m.SealDisposition.ALREADY_QUEUED
    )
    seal_item = writer.get_nowait()
    admission.complete_ordered_item(seal_item)
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert diagnostics.owner_state is m.OwnerState.STOPPED
    with pytest.raises(a.ReservationError):
        admission.request_seal("raw-session-id", seal)  # type: ignore[arg-type]


def test_expiry_revalidates_its_watermark_and_releases_erased_session_credits() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(
        a,
        m,
        owner_generation=311,
    )
    watermark = admission.final_admission_ordinal

    def authority(deadline: int, mode: Any) -> Any:
        return _capability(
            m.SessionExpiryAuthorityV1,
            protocol_version=1,
            owner_generation=311,
            consent_epoch_id=create.consent_epoch_id,
            logical_session_id=create.logical_session_id,
            expires_at_utc="2030-01-02T03:04:05.000000Z",
            deadline_admission_ordinal=deadline,
            mode=mode,
        )

    wrong_watermark = authority(watermark + 1, m.ExpiryMode.ERASE_STUCK)
    assert admission.begin_expiry(wrong_watermark) is m.ExpiryDisposition.WRITER_FAULT
    assert writer.ordered_count == 0

    exact = authority(watermark, m.ExpiryMode.ERASE_STUCK)
    assert (
        admission.begin_expiry(exact)
        is m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
    )
    assert admission.begin_expiry(exact) is m.ExpiryDisposition.ALREADY_EXPIRING
    item = writer.get_nowait()
    assert type(item.payload) is m.ExpireSessionV1
    assert item.payload.last_admission_ordinal == watermark
    assert item.admission_ordinal == watermark + 1

    admission.complete_ordered_item(item)
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert diagnostics.owner_state is m.OwnerState.STOPPED
    assert diagnostics.capture_state is m.CaptureState.IDLE

    rollover_admission, _, rollover_writer, rollover_create = _active_admission(
        a,
        m,
        owner_generation=312,
    )
    rollover_watermark = rollover_admission.final_admission_ordinal
    rollover_authority = _capability(
        m.SessionExpiryAuthorityV1,
        protocol_version=1,
        owner_generation=312,
        consent_epoch_id=rollover_create.consent_epoch_id,
        logical_session_id=rollover_create.logical_session_id,
        expires_at_utc="2030-01-02T03:04:05.000000Z",
        deadline_admission_ordinal=rollover_watermark,
        mode=m.ExpiryMode.ROLLOVER,
    )
    assert (
        rollover_admission.begin_expiry(rollover_authority)
        is m.ExpiryDisposition.ROLLOVER_QUEUED
    )
    assert (
        rollover_admission.begin_expiry(rollover_authority)
        is m.ExpiryDisposition.ALREADY_EXPIRING
    )
    assert rollover_writer.ordered_count == 0


def test_retention_owner_closes_admission_before_terminal_settlement_and_erasure() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=313,
    )
    admission.close_for_retention_expiry()
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=313,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is None
    assert result.disposition is m.AppendDisposition.SESSION_CLOSING

    expiry = _capability(
        m.SessionExpiryAuthorityV1,
        protocol_version=1,
        owner_generation=313,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        expires_at_utc="2030-01-02T03:04:05.000000Z",
        deadline_admission_ordinal=admission.final_admission_ordinal,
        mode=m.ExpiryMode.ERASE_STUCK,
    )
    assert admission.begin_expiry(expiry) is m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
    assert type(writer.get_nowait().payload) is m.ExpireSessionV1


def test_rollover_temporarily_holds_seven_credits_then_transfers_exactly_three() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=311)
    authority, command = _rollover_pair(m, create, owner_generation=311)
    assert (
        admission.try_rollover(authority, command)
        is m.RolloverDisposition.ROLLOVER_QUEUED
    )
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 7
    assert diagnostics.queue_canonical_bytes == 229_376
    item = writer.get_nowait()
    assert item.payload is command
    successor = admission.complete_rollover(item, m.StoreDisposition.COMMITTED)
    assert successor is not None
    assert successor.logical_session_id == command.successor_logical_session_id
    assert successor.cleanup_only is False
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 3
    assert diagnostics.queue_canonical_bytes == 98_304
    assert diagnostics.capture_state is m.CaptureState.ACTIVE


def test_rollover_capacity_failure_is_atomic_and_taints_without_session_switch() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=321)
    blockers = [admission.try_reserve_terminal() for _ in range(29)]
    assert all(item is not None for item in blockers)
    assert admission.diagnostics().queue_record_count == 61
    authority, command = _rollover_pair(m, create, owner_generation=321)

    assert (
        admission.try_rollover(authority, command)
        is m.RolloverDisposition.INSUFFICIENT_CAPACITY
    )
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 61
    assert diagnostics.queue_canonical_bytes == 1_998_848
    assert diagnostics.capture_state is m.CaptureState.FAULTED
    assert writer.ordered_count == 0


@pytest.mark.parametrize("value", [True, 0, 65])
def test_operation_scheduler_rejects_non_exact_or_out_of_range_limits(value: object) -> None:
    a = _admission()

    with pytest.raises((TypeError, ValueError)):
        a.ConversationOperationScheduler(owner_generation=1, max_operations=value)


def test_operation_scheduler_admits_exactly_64_concurrent_callers() -> None:
    a = _admission()
    from hermes_realtime.evidence import ConversationOperationKind

    owner = a.ConversationOperationScheduler(owner_generation=401, max_operations=64)
    start = Barrier(81)
    results_lock = Lock()
    admitted: list[Any] = []
    rejected = 0

    def reserve() -> None:
        nonlocal rejected
        start.wait()
        reservation = owner.try_reserve(ConversationOperationKind.RESPONSE)
        with results_lock:
            if reservation is None:
                rejected += 1
            else:
                admitted.append(reservation)

    threads = [Thread(target=reserve) for _ in range(80)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert len(admitted) == 64
    assert rejected == 16
    assert {item.operation_serial for item in admitted} == set(range(1, 65))
    for reservation in admitted:
        owner.release(reservation)
    assert owner.active_count == 0


def test_terminal_records_reject_foreign_installation_and_producer_lineage() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=331,
    )
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=331,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    opened = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            utterance_id=None,
            replay_of_evidence_turn_id=None,
        ),
    )
    terminal = _turn_snapshot(
        m,
        create,
        lease,
        4,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=False,
            assistant_delivery_context_recorded=False,
        ),
    )
    assert admission.try_append_turn(lease, opened) is m.AppendDisposition.ADMITTED

    foreign_installation = dataclasses.replace(
        terminal,
        installation_id=_uuid(6_200),
    )
    foreign_producer = dataclasses.replace(
        terminal,
        producer_instance_id=_uuid(6_201),
    )
    assert (
        admission.try_append_turn(lease, foreign_installation)
        is m.AppendDisposition.INVALID_AUTHORITY
    )
    assert (
        admission.try_append_turn(lease, foreign_producer)
        is m.AppendDisposition.INVALID_AUTHORITY
    )
    assert writer.ordered_count == 1


def test_mutated_live_lease_identity_is_rejected_before_enqueue() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=332,
    )
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=332,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    object.__setattr__(lease, "evidence_turn_id", _uuid(6_202))
    forged_open = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            utterance_id=None,
            replay_of_evidence_turn_id=None,
        ),
    )

    assert (
        admission.try_append_turn(lease, forged_open)
        is m.AppendDisposition.INVALID_AUTHORITY
    )
    assert writer.ordered_count == 0


def test_mutated_turn_authority_is_rejected_after_lease_mint() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=334,
    )
    authority = _turn_authority(
        m,
        "UserTurnAuthorityV1",
        create,
        owner_generation=334,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    result = admission.try_reserve_user_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    object.__setattr__(authority, "utterance_id", _uuid(6_203))
    forged_open = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.USER_RESPONSE,
            utterance_id=authority.utterance_id,
            replay_of_evidence_turn_id=None,
        ),
    )

    assert (
        admission.try_append_turn(lease, forged_open)
        is m.AppendDisposition.INVALID_AUTHORITY
    )
    assert writer.ordered_count == 0


def test_expiry_completion_cannot_race_a_terminal_append_into_stopped_session() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=333,
    )
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=333,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    opened = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            utterance_id=None,
            replay_of_evidence_turn_id=None,
        ),
    )
    assert admission.try_append_turn(lease, opened) is m.AppendDisposition.ADMITTED
    expiry_authority = _capability(
        m.SessionExpiryAuthorityV1,
        protocol_version=1,
        owner_generation=333,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        expires_at_utc="2030-01-02T03:04:05.000000Z",
        deadline_admission_ordinal=admission.final_admission_ordinal,
        mode=m.ExpiryMode.ERASE_STUCK,
    )
    assert (
        admission.begin_expiry(expiry_authority)
        is m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
    )
    ordinary_item = writer.get_nowait()
    expiry_item = writer.get_nowait()
    admission.complete_ordered_item(ordinary_item)
    terminal = _turn_snapshot(
        m,
        create,
        lease,
        4,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=False,
            assistant_delivery_context_recorded=False,
        ),
    )
    entered = Event()
    proceed = Event()
    original_enqueue = admission._try_enqueue_terminal_locked

    def gated_enqueue(self: Any, *args: Any) -> Any:
        del self
        entered.set()
        assert proceed.wait(timeout=1)
        return original_enqueue(*args)

    admission._try_enqueue_terminal_locked = MethodType(  # type: ignore[method-assign]
        gated_enqueue,
        admission,
    )
    dispositions: list[Any] = []
    append_thread = Thread(
        target=lambda: dispositions.append(admission.try_append_turn(lease, terminal))
    )
    append_thread.start()
    assert entered.wait(timeout=1)
    admission.complete_ordered_item(expiry_item)
    proceed.set()
    append_thread.join(timeout=1)
    assert not append_thread.is_alive()

    assert dispositions == [m.AppendDisposition.SESSION_CLOSING]
    assert writer.ordered_count == 0


def test_preexisting_lease_settles_through_revoke_before_finalize() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, operations, writer, create = _active_admission(
        a,
        m,
        owner_generation=336,
    )
    authority = _turn_authority(
        m,
        "ProactiveTurnAuthorityV1",
        create,
        owner_generation=336,
    )
    operation = operations.try_reserve(m.ConversationOperationKind.PROACTIVE)
    assert operation is not None
    result = admission.try_reserve_proactive_turn(authority, operation)
    assert result.lease is not None
    lease = result.lease
    opened = _turn_snapshot(
        m,
        create,
        lease,
        3,
        m.EventKind.TURN_OPENED,
        m.TurnOpenedPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            utterance_id=None,
            replay_of_evidence_turn_id=None,
        ),
    )
    assert admission.try_append_turn(lease, opened) is m.AppendDisposition.ADMITTED
    admission.complete_ordered_item(writer.get_nowait())

    ticket = admission.begin_revoke(_revoke_authority(m, create))
    revoke_request = writer.get_nowait()
    admission.complete_revoke_request(
        revoke_request,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )
    assert (
        admission.record_terminal_cause(
            lease.terminal_cause,
            m.TerminalReason.CONSENT_REVOKED,
        )
        is m.CauseDisposition.RECORDED
    )
    resolution = admission.freeze_and_resolve_terminal_causes(
        lease,
        context_committed=False,
    )
    assert resolution.terminal_disposition is m.TerminalDisposition.REVOKED

    terminal_snapshot = _turn_snapshot(
        m,
        create,
        lease,
        4,
        m.EventKind.TURN_SNAPSHOT,
        m.TurnSnapshotPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            turn_kind=m.TurnKind.PROACTIVE_UPDATE,
            generated_segment_count=0,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            model_context_admitted=False,
            assistant_delivery_context_recorded=False,
        ),
    )
    settled = _turn_snapshot(
        m,
        create,
        lease,
        5,
        m.EventKind.TURN_SETTLED,
        m.TurnSettledPayloadV1(
            evidence_turn_id=lease.evidence_turn_id,
            terminal_disposition=m.TerminalDisposition.REVOKED,
            terminal_reason=m.TerminalReason.CONSENT_REVOKED,
            context_committed=False,
            generated_segment_count=0,
            transport_confirmed_full_count=0,
        ),
    )
    assert (
        admission.try_append_turn(lease, terminal_snapshot)
        is m.AppendDisposition.ADMITTED
    )
    assert admission.try_append_turn(lease, settled) is m.AppendDisposition.ADMITTED
    assert admission.diagnostics().active_lease_count == 0
    assert admission.try_enqueue_revoke_finalize(ticket) is None

    first_terminal = writer.get_nowait()
    second_terminal = writer.get_nowait()
    admission.complete_ordered_item(first_terminal)
    assert admission.try_enqueue_revoke_finalize(ticket) is None
    admission.complete_ordered_item(second_terminal)
    finalize = admission.try_enqueue_revoke_finalize(ticket)
    assert finalize is not None
    assert finalize.payload.final_admission_ordinal == second_terminal.admission_ordinal


def test_drain_completion_waits_for_ordered_records_and_revoke_finalize() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(
        a,
        m,
        owner_generation=337,
    )
    command_authority = _turn_authority(
        m,
        "CommandAdmissionAuthorityV1",
        create,
        owner_generation=337,
    )
    assert (
        admission.try_admit_command(
            command_authority,
            _command_snapshot(m, create, command_authority, 3),
        )
        is m.CommandDisposition.ADMITTED
    )
    admission.begin_revoke(_revoke_authority(m, create))
    drain_ticket = admission.request_drain(
        _owner_drain_authority(
            m,
            create,
            admission,
            owner_generation=337,
        )
    )
    revoke_item = writer.get_nowait()
    ordered_item = writer.get_nowait()
    admission.complete_revoke_request(
        revoke_item,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )

    assert not drain_ticket.terminal_event.is_set()
    admission.complete_ordered_item(ordered_item)
    finalize = writer.get_nowait()
    assert type(finalize.payload) is m.RevokeFinalizeV1
    admission.complete_revoke_finalize(finalize, m.RevokeDisposition.PURGE_COMPLETED)
    drain_item = writer.get_nowait()
    assert type(drain_item.payload) is m.DrainAndStopV1
    admission.complete_drain(drain_item, m.DrainDisposition.STOPPED)
    assert drain_ticket.terminal_event.is_set()


def test_revoke_finalize_is_published_before_writer_can_complete_it() -> None:
    a = _admission()
    from hermes_realtime.evidence import models as m

    admission, _, writer, create = _active_admission(a, m, owner_generation=424)
    ticket = admission.begin_revoke(_revoke_authority(m, create))
    request = writer.get_nowait()
    original_put_nowait = writer.put_nowait

    def complete_immediately(item: Any) -> None:
        original_put_nowait(item)
        dequeued = writer.get_nowait()
        assert dequeued is item
        admission.complete_revoke_finalize(
            dequeued,
            m.RevokeDisposition.PURGE_COMPLETED,
        )

    writer.put_nowait = complete_immediately
    admission.complete_revoke_request(
        request,
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
    )

    assert ticket.disposition is m.RevokeDisposition.PURGE_COMPLETED
    assert ticket.terminal_event.is_set()

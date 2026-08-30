from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from queue import Full
from threading import Barrier, Event, Thread
from typing import Any

import pytest

HASH = "a" * 64


def _uuid(value: int) -> str:
    from uuid import UUID

    return str(UUID(int=value, version=4))


def _create_epoch(m: Any, *, seed: int = 90_000) -> Any:
    installation_id = _uuid(seed)
    producer_instance_id = _uuid(seed + 1)
    consent_epoch_id = _uuid(seed + 2)
    logical_session_id = _uuid(seed + 3)
    binding_id = _uuid(seed + 4)
    opened = m.EvidenceSnapshotV1(
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
    binding = m.EvidenceSnapshotV1(
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
        session_opened=opened,
        binding_opened=binding,
    )


def _active_admission(a: Any, m: Any, *, owner_generation: int) -> tuple[Any, Any]:
    sink = a.BoundedEvidenceWriterQueueV1()
    admission = a.EvidenceAdmissionControllerV1(
        enabled=True,
        owner_generation=owner_generation,
        writer_sink=sink,
    )
    reservation = admission.try_reserve_create_epoch(_create_epoch(m))
    assert reservation is not None
    item = admission.try_enqueue_create_epoch(reservation)
    assert item is not None and sink.get_nowait() is item
    retained = admission.complete_create_epoch(
        item,
        disposition=m.StoreDisposition.COMMITTED,
        binding_current=True,
    )
    assert retained is not None
    return admission, sink


def _snapshot(*, seed: int = 90_000) -> Any:
    from hermes_realtime.evidence import models as m

    return _create_epoch(m, seed=seed).session_opened


def _capability(cls: Any, **values: Any) -> Any:
    assert {field.name for field in dataclasses.fields(cls)} == set(values)
    instance = object.__new__(cls)
    for name, value in values.items():
        object.__setattr__(instance, name, value)
    cls._validate(instance)
    return instance


def test_preparation_is_exact_detached_and_canonical_once() -> None:
    from hermes_realtime.evidence import admission as a

    scheduler = a._new_production_evidence_scheduler_v1()
    original = _snapshot()
    prepared = scheduler.prepare(original)
    before = prepared.canonical_bytes

    assert prepared.snapshot is not original
    assert prepared.canonical_byte_charge == len(before)
    assert prepared.canonical_sha256 == hashlib.sha256(before).hexdigest()
    assert a.canonical_record_bytes(prepared.snapshot) == len(before)

    original_payload: Any = original.payload
    retained_payload: Any = prepared.snapshot.payload
    object.__setattr__(original, "event_id", "00000000-0000-4000-8000-000000000001")
    object.__setattr__(original_payload, "retention_hours", 1)
    assert prepared.canonical_bytes == before
    assert prepared.snapshot.event_id != original.event_id
    assert prepared.snapshot.payload is retained_payload
    assert retained_payload is not original_payload
    assert retained_payload.retention_hours == 24

    with pytest.raises(TypeError, match="exact EvidenceSnapshotV1"):
        scheduler.prepare(object())


@pytest.mark.parametrize(
    "changes",
    (
        {"canonical_byte_charge": 0},
        {"canonical_bytes": b"{}"},
        {"canonical_sha256": "0" * 64},
        {"snapshot": _snapshot(seed=91_000)},
    ),
)
def test_forged_prepared_product_is_rejected_before_accounting(
    changes: dict[str, Any],
) -> None:
    from hermes_realtime.evidence import admission as a

    scheduler = a._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())
    forged = dataclasses.replace(prepared, **changes)
    before = (scheduler.credits, scheduler.next_admission_ordinal)

    with pytest.raises(a.ReservationError, match="prepared"):
        scheduler.try_admit(forged)

    assert (scheduler.credits, scheduler.next_admission_ordinal) == before
    genuine = scheduler.try_admit(prepared)
    assert genuine is not None and genuine.admission_ordinal == 1
    scheduler.complete(scheduler.dequeue_nowait())
    assert scheduler.credits == (0, 0, 0)


def test_prepared_product_is_scheduler_issued_and_batch_validation_is_atomic() -> None:
    from hermes_realtime.evidence import admission as a

    first = a._new_production_evidence_scheduler_v1()
    second = a._new_production_evidence_scheduler_v1()
    prepared = first.prepare(_snapshot())
    forged = dataclasses.replace(prepared)
    direct = a._PreparedEvidenceRecordV1(
        snapshot=prepared.snapshot,
        canonical_bytes=prepared.canonical_bytes,
        canonical_byte_charge=prepared.canonical_byte_charge,
        canonical_sha256=prepared.canonical_sha256,
    )

    with pytest.raises(a.ReservationError, match="prepared"):
        second.try_admit(prepared)
    with pytest.raises(a.ReservationError, match="prepared"):
        first.try_admit_batch((prepared, forged))
    with pytest.raises(a.ReservationError, match="prepared"):
        first.try_admit(direct)

    assert first.credits == second.credits == (0, 0, 0)
    assert first.next_admission_ordinal == second.next_admission_ordinal == 1
    item = first.try_admit(prepared)
    assert item is not None and item.admission_ordinal == 1
    first.complete(first.dequeue_nowait())


def test_same_prepared_item_repeats_with_contiguous_ordinals_and_exact_completion() -> None:
    from hermes_realtime.evidence import admission as a

    scheduler = a._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())

    admitted = tuple(scheduler.try_admit(prepared) for _ in range(8))
    assert all(item is not None for item in admitted)
    exact = tuple(item for item in admitted if item is not None)
    assert [item.admission_ordinal for item in exact] == list(range(1, 9))
    assert all(item.payload.snapshot == prepared.snapshot for item in exact)
    assert all(item.payload.snapshot is not prepared.snapshot for item in exact)
    assert scheduler.credits == (8, 8 * prepared.canonical_byte_charge, 8)

    dequeued = tuple(scheduler.dequeue_nowait() for _ in range(8))
    assert dequeued == exact
    assert scheduler.credits == (8, 8 * prepared.canonical_byte_charge, 8)
    for item in dequeued:
        scheduler.complete(item)
    assert scheduler.credits == (0, 0, 0)
    assert scheduler.next_admission_ordinal == 9

    with pytest.raises(a.ReservationError, match="stale|owner|completed"):
        scheduler.complete(dequeued[0])


def test_single_record_admission_does_not_route_through_batch_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import admission as a

    scheduler = a._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())

    def forbidden_batch(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("single-record admission used the batch wrapper")

    monkeypatch.setattr(
        a._ProductionEvidenceSchedulerV1,
        "try_admit_batch",
        forbidden_batch,
    )

    item = scheduler.try_admit(prepared)
    assert item is not None
    assert scheduler.dequeue_nowait() is item
    scheduler.complete(item)
    assert scheduler.credits == (0, 0, 0)


def test_atomic_batch_repeats_prepared_records_with_one_capacity_domain() -> None:
    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence import models as m

    scheduler = a._new_production_evidence_scheduler_v1()
    first = scheduler.prepare(_snapshot())
    second = scheduler.prepare(_create_epoch(m, seed=91_000).session_opened)

    admitted = scheduler.try_admit_batch((first, second))
    assert admitted is not None
    assert tuple(item.admission_ordinal for item in admitted) == (1, 2)
    assert scheduler.credits == (
        2,
        first.canonical_byte_charge + second.canonical_byte_charge,
        2,
    )
    assert tuple(scheduler.dequeue_nowait() for _ in range(2)) == admitted
    for item in admitted:
        scheduler.complete(item)
    assert scheduler.credits == (0, 0, 0)
    assert scheduler.next_admission_ordinal == 3


def test_capacity_drop_consumes_no_credits_or_ordinal() -> None:
    from hermes_realtime.evidence import admission as a

    scheduler = a._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())
    items = tuple(scheduler.try_admit(prepared) for _ in range(64))
    assert all(item is not None for item in items)
    assert scheduler.try_admit(prepared) is None
    assert scheduler.next_admission_ordinal == 65
    assert scheduler.credits == (64, 64 * prepared.canonical_byte_charge, 64)

    scheduler.complete(scheduler.dequeue_nowait())
    next_item = scheduler.try_admit(prepared)
    assert next_item is not None
    assert next_item.admission_ordinal == 65


def test_capacity_ledger_observes_byte_only_rejection_without_charging() -> None:
    from hermes_realtime.evidence import admission as a

    capacity = a._AdmissionCapacityV1(
        max_records=4,
        max_canonical_bytes=10,
        max_physical_items=4,
    )
    assert capacity.try_charge(1, 7, physical_items=1) is True

    with capacity.lock:
        source = capacity._try_charge_observed(1, 4, physical_items=1)

    assert source == "canonical_byte_capacity"
    assert (capacity.charged_records, capacity.charged_bytes, capacity.ordered_physical_items) == (
        1,
        7,
        1,
    )


def test_capacity_ledger_observes_physical_only_rejection_without_charging() -> None:
    from hermes_realtime.evidence import admission as a

    capacity = a._AdmissionCapacityV1(
        max_records=4,
        max_canonical_bytes=10,
        max_physical_items=1,
    )
    assert capacity.try_charge(1, 1, physical_items=1) is True

    with capacity.lock:
        source = capacity._try_charge_observed(1, 1, physical_items=1)

    assert source == "physical_capacity"
    assert (capacity.charged_records, capacity.charged_bytes, capacity.ordered_physical_items) == (
        1,
        1,
        1,
    )


@pytest.mark.parametrize(
    ("records", "canonical_bytes", "physical_items", "expected"),
    (
        (2, 2, 2, "record_capacity"),
        (1, 2, 2, "canonical_byte_capacity"),
        (1, 1, 2, "physical_capacity"),
    ),
)
def test_capacity_ledger_observed_rejection_has_frozen_precedence(
    records: int,
    canonical_bytes: int,
    physical_items: int,
    expected: str,
) -> None:
    from hermes_realtime.evidence import admission as a

    capacity = a._AdmissionCapacityV1(
        max_records=2,
        max_canonical_bytes=2,
        max_physical_items=2,
    )
    assert capacity.try_charge(1, 1, physical_items=1) is True

    with capacity.lock:
        source = capacity._try_charge_observed(
            records,
            canonical_bytes,
            physical_items=physical_items,
        )

    assert source == expected
    assert (capacity.charged_records, capacity.charged_bytes, capacity.ordered_physical_items) == (
        1,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_real_controller_attributes_attempted_user_batch_to_record_capacity(
    tmp_path: Any,
) -> None:
    """A lawful two-record final batch reports its attempted record overflow."""

    from hermes_realtime._qualification import (
        _new_qualification_full_host_dependencies,
        _qualification_capacity_probe_scope,
    )
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    identity = object()
    dependencies = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=lambda: object(),
        speech_presence_factory=lambda: object(),
        synthesizer_factory=lambda: object(),
        transcriber_factory=lambda: object(),
        vad_factory=lambda: object(),
        identity_factory=lambda: "qualification_identity",
    )
    with _qualification_capacity_probe_scope(dependencies._capacity_probe):
        runtime = HostEvidenceRuntimeV1(
            database=tmp_path / "capture-v1.sqlite3",
            owner_generation=911,
            retention_hours=24,
        )

    class BlockFirstOrdinaryAppend:
        def __init__(self, delegate: object) -> None:
            self._delegate = delegate
            self.entered = Event()
            self.release = Event()
            self._blocked = False

        def __getattr__(self, name: str) -> object:
            return getattr(self._delegate, name)

        def append_record(self, item: object) -> object:
            if not self._blocked:
                self._blocked = True
                self.entered.set()
                assert self.release.wait(5.0), "ordinary writer completion was not released"
            return self._delegate.append_record(item)  # type: ignore[union-attr]

    command = _create_epoch(m, seed=95_000)
    projection = BrowserEventProjection()
    consent = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    transport = BlockFirstOrdinaryAppend(runtime.create_sqlite_transport())
    assert await runtime.activate_consent(
        consent,
        transport=transport,
        binding_is_current=lambda candidate: candidate is command,
        timeout_seconds=2.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED
    lifecycle, admission = runtime.resolve_evidence_pair()
    assert lifecycle is not None and admission is not None
    assert admission.diagnostics().queue_record_count == 3

    operation = runtime.operation_scheduler.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    user_authority = lifecycle.decline_to_user(
        lifecycle.mint_final_input(
            source=m.InputSource.TYPED,
            input_incarnation=1,
            media_incarnation=None,
            typed_sequence=1,
        )
    )
    user = admission.try_reserve_user_turn(user_authority, operation)
    assert user.disposition is m.AppendDisposition.ADMITTED and user.lease is not None
    assert admission.diagnostics().queue_record_count == 5

    for index in range(58):
        authority = lifecycle.accept_command(
            lifecycle.mint_final_input(
                source=m.InputSource.TYPED,
                input_incarnation=index + 2,
                media_incarnation=None,
                typed_sequence=index + 2,
            )
        )
        snapshot = m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=command.installation_id,
            producer_instance_id=command.producer_instance_id,
            logical_session_id=command.logical_session_id,
            event_id=_uuid(95_100 + index),
            event_sequence=index + 3,
            event_kind=m.EventKind.COMMAND_ROUTED,
            payload=m.CommandRoutedPayloadV1(
                utterance_id=authority.utterance_id,
                source=authority.source,
                routing_disposition="command",
            ),
        )
        assert admission.try_admit_command(authority, snapshot) is m.CommandDisposition.ADMITTED

    assert await asyncio.to_thread(transport.entered.wait, 2.0)
    before = admission.diagnostics()
    controller = runtime._admission
    assert controller is not None
    before_ordinal = controller.final_admission_ordinal
    assert before.queue_record_count == 63
    observations = dependencies._capacity_probe.observations()
    assert observations[-1].queue_record_count == 63
    assert observations[-1].queue_physical_count == 58
    assert observations[-1].queue_canonical_bytes < 2_097_152

    assert (
        admission.try_admit_user_final(user.lease, user_authority, "lawful final input")
        is m.AppendDisposition.DROPPED_CAPACITY
    )
    rejected = [
        observation
        for observation in dependencies._capacity_probe.observations()
        if observation.kind == "ordinary_rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].rejection_source == "record_capacity"
    assert admission.diagnostics() == before
    assert controller.final_admission_ordinal == before_ordinal

    transport.release.set()
    async with asyncio.timeout(10.0):
        while sum(
            observation.kind == "ordinary_completed"
            for observation in dependencies._capacity_probe.observations()
        ) < 58:
            await asyncio.sleep(0.01)
    assert admission.discard_unopened_user_turn(user.lease, user_authority) is True
    runtime.operation_scheduler.release(operation)
    await runtime.close()
    final = admission.diagnostics()
    assert final.queue_record_count == final.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0
    assert dependencies._capacity_probe.observations()[-1].all_capacity_released is True


def test_publication_full_rolls_back_exact_charge_and_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import admission as a

    def reject(self: Any, item: Any) -> None:
        del self, item
        raise Full

    monkeypatch.setattr(a.BoundedEvidenceWriterQueueV1, "put_nowait", reject)
    scheduler = a._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())

    with pytest.raises(a._SchedulerPublicationFull):
        scheduler.try_admit(prepared)
    assert scheduler.credits == (0, 0, 0)
    assert scheduler.next_admission_ordinal == 1


def test_completion_is_identity_and_owner_bound() -> None:
    from hermes_realtime.evidence import admission as a

    first = a._new_production_evidence_scheduler_v1()
    second = a._new_production_evidence_scheduler_v1()
    item = first.try_admit(first.prepare(_snapshot()))
    assert item is not None
    forged = dataclasses.replace(item)

    with pytest.raises(a.ReservationError, match="stale|owner"):
        first.complete(forged)
    with pytest.raises(a.ReservationError, match="stale|owner"):
        second.complete(item)
    assert first.credits[0] == 1
    first.complete(first.dequeue_nowait())
    assert first.credits == (0, 0, 0)


def test_concurrent_admissions_share_one_contiguous_ordinal_domain() -> None:
    from hermes_realtime.evidence import admission as a

    scheduler = a._new_production_evidence_scheduler_v1()
    prepared = scheduler.prepare(_snapshot())
    barrier = Barrier(9)
    results: list[Any] = []

    def producer() -> None:
        barrier.wait()
        results.extend(scheduler.try_admit(prepared) for _ in range(8))

    threads = [Thread(target=producer) for _ in range(8)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    exact = [item for item in results if item is not None]
    assert len(exact) == 64
    assert sorted(item.admission_ordinal for item in exact) == list(range(1, 65))
    assert scheduler.try_admit(prepared) is None
    for _ in exact:
        scheduler.complete(scheduler.dequeue_nowait())
    assert scheduler.credits == (0, 0, 0)


def test_real_controller_ordinary_scheduler_uses_the_shared_capacity_ledger() -> None:
    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence import models as m

    admission, sink = _active_admission(a, m, owner_generation=909)
    create = _create_epoch(m)
    terminal = admission.try_reserve_terminal()
    assert terminal is not None
    authority = _capability(
        m.CommandAdmissionAuthorityV1,
        protocol_version=1,
        owner_generation=909,
        binding_id=create.binding_id,
        binding_generation=create.binding_generation,
        consent_epoch_id=create.consent_epoch_id,
        logical_session_id=create.logical_session_id,
        utterance_id=_uuid(93_001),
        source=m.InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
        routing_serial=1,
        routing_disposition="command",
    )
    command = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=create.installation_id,
        producer_instance_id=create.producer_instance_id,
        logical_session_id=create.logical_session_id,
        event_id=_uuid(93_002),
        event_sequence=3,
        event_kind=m.EventKind.COMMAND_ROUTED,
        payload=m.CommandRoutedPayloadV1(
            utterance_id=authority.utterance_id,
            source=authority.source,
            routing_disposition="command",
        ),
    )
    before = admission.diagnostics()

    assert admission.try_admit_command(authority, command) is m.CommandDisposition.ADMITTED
    during = admission.diagnostics()
    assert during.queue_record_count == before.queue_record_count + 1
    assert during.queue_canonical_bytes == (
        before.queue_canonical_bytes + a.canonical_record_bytes(command)
    )
    admission.complete_ordered_item(sink.get_nowait())
    admission.release_terminal_reservation(terminal)
    after = admission.diagnostics()
    assert after.queue_record_count == 3
    assert after.queue_canonical_bytes == 98_304
    assert admission._ordinary_scheduler.next_admission_ordinal == 4


def test_live_controller_capacity_and_sink_cannot_be_attached_twice() -> None:
    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence import models as m

    admission, sink = _active_admission(a, m, owner_generation=910)
    binding = admission._capacity._ordinary_scheduler_binding
    assert binding is sink._ordinary_scheduler_binding

    with pytest.raises(a.ReservationError, match="already.*bound"):
        a.EvidenceAdmissionControllerV1(
            enabled=True,
            owner_generation=911,
            writer_sink=sink,
        )

    with pytest.raises(a.ReservationError, match="already.*bound|already.*claimed"):
        a._ProductionEvidenceSchedulerV1(binding=binding)

    fresh_capacity = a._AdmissionCapacityV1(
        max_records=a.MAX_QUEUE_RECORDS,
        max_canonical_bytes=a.MAX_QUEUE_CANONICAL_BYTES,
        max_physical_items=a.MAX_QUEUE_PHYSICAL_ITEMS,
    )
    with pytest.raises(a.ReservationError, match="already.*bound"):
        a._new_controller_ordinary_scheduler_v1(
            capacity=fresh_capacity,
            writer_sink=sink,
        )
    with pytest.raises(a.ReservationError, match="already.*bound"):
        a._new_controller_ordinary_scheduler_v1(
            capacity=admission._capacity,
            writer_sink=a.BoundedEvidenceWriterQueueV1(),
        )

    assert admission.diagnostics().queue_record_count == 3
    assert admission._ordinary_scheduler.next_admission_ordinal == 3


def test_scheduler_remains_private_and_has_no_live_attachment_surface() -> None:
    import inspect

    import hermes_realtime.evidence as evidence
    from hermes_realtime.evidence import admission as a

    assert "_ProductionEvidenceSchedulerV1" not in evidence.__all__
    assert "_new_production_evidence_scheduler_v1" not in evidence.__all__
    constructor_parameters = inspect.signature(a._ProductionEvidenceSchedulerV1).parameters
    assert "capacity" not in constructor_parameters
    assert "writer_sink" not in constructor_parameters
    assert not hasattr(a._ProductionEvidenceSchedulerV1, "attach")
    assert not hasattr(a.EvidenceAdmissionControllerV1, "benchmark")
    assert not hasattr(a.EvidenceAdmissionControllerV1, "qualification_scheduler")

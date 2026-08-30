from __future__ import annotations

import asyncio
import base64
import inspect
import json
import pickle
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest


def _uid(value: int) -> str:
    return str(UUID(int=value, version=4))


def _create_epoch():  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import models as m

    return m.CreateEpochV1(
        protocol_version=1,
        installation_id=_uid(1),
        producer_instance_id=_uid(2),
        consent_epoch_id=_uid(3),
        logical_session_id=_uid(4),
        binding_id=_uid(5),
        binding_generation=11,
        consent_version=m.CONSENT_VERSION,
        disclosure_digest="a" * 64,
        retention_hours=24,
        microphone_accepted=True,
        typed_accepted=True,
        control_sequence=1,
        control_fingerprint_hash="b" * 64,
        session_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=_uid(1),
            producer_instance_id=_uid(2),
            logical_session_id=_uid(4),
            event_id=_uid(6),
            event_sequence=1,
            event_kind=m.EventKind.SESSION_OPENED,
            payload=m.SessionOpenedPayloadV1(
                consent_epoch_id=_uid(3),
                binding_id=_uid(5),
                consent_version=m.CONSENT_VERSION,
                disclosure_digest="a" * 64,
                retention_hours=24,
                microphone_accepted=True,
                typed_accepted=True,
                predecessor_session_id=None,
            ),
        ),
        binding_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=_uid(1),
            producer_instance_id=_uid(2),
            logical_session_id=_uid(4),
            event_id=_uid(7),
            event_sequence=2,
            event_kind=m.EventKind.BINDING_OPENED,
            payload=m.BindingOpenedPayloadV1(
                binding_id=_uid(5),
                binding_generation=11,
                microphone_available=True,
                typed_available=True,
            ),
        ),
    )


_REAL_SQLITE_ACTIVATION_SETTLEMENT_TIMEOUT_SECONDS = 30.0


async def _activate(runtime, command, *, transport=None):  # type: ignore[no-untyped-def]
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m

    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    disposition = await runtime.activate_consent(
        authority,
        transport=transport or runtime.create_sqlite_transport(),
        binding_is_current=lambda candidate: candidate is command,
        timeout_seconds=2.0,
    )
    if disposition is m.ConsentDisposition.CONTROL_TIMED_OUT:
        settlement = runtime.claim_consent_settlement_task(authority)
        disposition = await asyncio.wait_for(
            asyncio.shield(settlement),
            _REAL_SQLITE_ACTIVATION_SETTLEMENT_TIMEOUT_SECONDS,
        )
    assert disposition is m.ConsentDisposition.CONSENT_ACTIVATED


@pytest.mark.asyncio
async def test_real_sqlite_activation_helper_claims_retained_timeout_settlement(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=19,
        retention_hours=24,
    )
    delegate = runtime.create_sqlite_transport()
    entered = threading.Event()
    release = threading.Event()

    class DelayedCreateTransport:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(delegate, name)

        def create_epoch(self, command):  # type: ignore[no-untyped-def]
            entered.set()
            if not release.wait(10.0):
                raise RuntimeError("test did not release delayed real SQLite create")
            return delegate.create_epoch(command)

    activation = asyncio.create_task(
        _activate(runtime, _create_epoch(), transport=DelayedCreateTransport())
    )
    try:
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5.0), 6.0)
        await asyncio.sleep(2.1)
        release.set()
        await activation
    finally:
        release.set()
        await runtime.close()


def _reserve_proactive(runtime):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import models as m

    lifecycle = runtime.evidence_lifecycle
    admission = runtime.evidence_admission
    assert lifecycle is not None and admission is not None
    operation = runtime.operation_scheduler.try_reserve(
        m.ConversationOperationKind.PROACTIVE
    )
    assert operation is not None
    authority = lifecycle.mint_proactive_turn(operation)
    result = admission.try_reserve_proactive_turn(authority, operation)
    return admission, operation, authority, result


def _reserve_replay(runtime, *, replay_id: int = 700):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import models as m

    lifecycle = runtime.evidence_lifecycle
    admission = runtime.evidence_admission
    assert lifecycle is not None and admission is not None
    operation = runtime.operation_scheduler.try_reserve(m.ConversationOperationKind.REPLAY)
    assert operation is not None
    command = _create_epoch()
    authority = lifecycle.mint_replay_turn(
        operation,
        replay_of_evidence_turn_id=_uid(replay_id),
        replay_generation=1,
        source_binding_id=command.binding_id,
        source_binding_generation=command.binding_generation,
        source_logical_session_id=command.logical_session_id,
    )
    result = admission.try_reserve_replay_turn(authority, operation)
    return admission, operation, authority, result


def _reserve_user(runtime):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import models as m

    lifecycle = runtime.evidence_lifecycle
    admission = runtime.evidence_admission
    assert lifecycle is not None and admission is not None
    operation = runtime.operation_scheduler.try_reserve(m.ConversationOperationKind.RESPONSE)
    assert operation is not None
    final_input = lifecycle.mint_final_input(
        source=m.InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )
    authority = lifecycle.decline_to_user(final_input)
    result = admission.try_reserve_user_turn(authority, operation)
    return admission, operation, authority, result


async def _wait_for_rollover(runtime) -> None:  # type: ignore[no-untyped-def]
    from hermes_realtime.production_observation import (
        RolloverObservationV1,
        RolloverStageV1,
    )

    async with asyncio.timeout(5.0):
        while True:
            records = runtime.production_observations.records()
            if any(
                type(record) is RolloverObservationV1
                and record.stage is RolloverStageV1.TERMINAL
                for record in records
            ):
                return
            await asyncio.sleep(0.01)


def _rollover_records(runtime):  # type: ignore[no-untyped-def]
    from hermes_realtime.production_observation import RolloverObservationV1

    return tuple(
        record
        for record in runtime.production_observations.records()
        if type(record) is RolloverObservationV1
    )


def _settle_maximum_proactive(runtime):  # type: ignore[no-untyped-def]
    from hermes_realtime.evidence import models as m

    admission, operation, authority, result = _reserve_proactive(runtime)
    assert result.disposition is m.AppendDisposition.ADMITTED
    assert result.lease is not None
    assert admission.try_open_non_user_turn(result.lease) is m.AppendDisposition.ADMITTED
    assert admission.settle_completed(
        result.lease,
        queued_chunk_count=0,
        started_chunk_count=0,
        transport_confirmed_full_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED
    runtime.operation_scheduler.release(operation)
    return admission, authority, result.lease


def _checkpointed_runtime(database: Path, *, owner_generation: int, publication_fault: bool):  # type: ignore[no-untyped-def]
    from hermes_realtime._qualification import (
        _new_qualification_full_host_dependencies,
        _new_qualification_owner_context_capability,
    )
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    identity = object()
    entered = threading.Event()
    resume = threading.Event()

    class Collector:
        def _qualification_owner_identity_matches(self, candidate: object) -> bool:
            return candidate is identity

        def __getattr__(self, name: str):
            if name.startswith("record_"):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    capability = _new_qualification_owner_context_capability(identity)
    bundle = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=lambda: object(),
        speech_presence_factory=lambda: object(),
        synthesizer_factory=lambda: object(),
        transcriber_factory=lambda: object(),
        vad_factory=lambda: object(),
        identity_factory=lambda: "qualification_identity",
    )
    with capability.wire(Collector(), full_host_dependencies=bundle):
        if publication_fault:
            bundle._capacity_probe.arm_rollover_publication_fault(identity)
        bundle._capacity_probe.arm_rollover_failure_checkpoint(
            identity,
            entered=entered,
            resume=resume,
        )
        runtime = HostEvidenceRuntimeV1(
            database=database,
            owner_generation=owner_generation,
            retention_hours=24,
        )
    return runtime, bundle, entered, resume


def test_closed_activation_loop_aborts_sync_rollover_handoff_fail_closed(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=20,
        retention_hours=24,
    )
    with asyncio.Runner() as runner:
        runner.run(_activate(runtime, _create_epoch()))
        _settle_maximum_proactive(runtime)

    admission, operation, _authority, result = _reserve_proactive(runtime)
    assert result.lease is not None
    assert admission.try_open_non_user_turn(result.lease) is m.AppendDisposition.ADMITTED
    assert admission.settle_completed(
        result.lease,
        queued_chunk_count=0,
        started_chunk_count=0,
        transport_confirmed_full_count=0,
        assistant_delivery_context_recorded=False,
    ) is m.AppendDisposition.ADMITTED
    runtime.operation_scheduler.release(operation)

    diagnostics = admission.diagnostics()
    assert diagnostics.capture_state is m.CaptureState.ACTIVE
    assert diagnostics.sticky_fault is None
    assert diagnostics.active_lease_count == 0
    from hermes_realtime.production_observation import RolloverResultV1, RolloverStageV1

    expected_rollover = [
        (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.TERMINAL, RolloverResultV1.REJECTED),
    ]
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == (
        expected_rollover
    )
    with sqlite3.connect(tmp_path / "capture-v1.sqlite3") as connection:
        assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (1,)

    asyncio.run(runtime.close())
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == (
        expected_rollover
    )
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


def _activate_on_secondary_owner_loop(runtime, command):  # type: ignore[no-untyped-def]
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=run, name="task11b-owner-loop")
    thread.start()
    assert started.wait(2.0)
    asyncio.run_coroutine_threadsafe(_activate(runtime, command), loop).result(timeout=5.0)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert not loop.is_closed()
    return loop


def _run_queued_owner_callbacks(loop) -> None:  # type: ignore[no-untyped-def]
    stopped = threading.Event()

    def stop() -> None:
        loop.stop()
        stopped.set()

    loop.call_soon_threadsafe(stop)
    thread = threading.Thread(target=loop.run_forever, name="task11b-late-owner-loop")
    thread.start()
    assert stopped.wait(2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()


@pytest.mark.asyncio
async def test_stopped_open_owner_loop_handoff_times_out_and_late_callback_is_noop(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverResultV1, RolloverStageV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=41,
        retention_hours=24,
    )
    loop = _activate_on_secondary_owner_loop(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    _settle_maximum_proactive(runtime)
    held, operation, authority, result = _reserve_proactive(runtime)
    assert result.lease is not None
    assert held.discard_unopened_proactive_turn(result.lease, authority) is True
    runtime.operation_scheduler.release(operation)

    async with asyncio.timeout(3.0):
        await runtime.close()
    expected = [
        (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.TERMINAL, RolloverResultV1.CLOSED),
    ]
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == expected
    _run_queued_owner_callbacks(loop)
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == expected
    loop.close()
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_cancelled_close_does_not_cancel_pending_handoff_and_retry_settles(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverResultV1, RolloverStageV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=42,
        retention_hours=24,
    )
    loop = _activate_on_secondary_owner_loop(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    _settle_maximum_proactive(runtime)
    held, operation, authority, result = _reserve_replay(runtime)
    assert result.lease is not None
    assert held.discard_unopened_replay_turn(result.lease, authority) is True
    runtime.operation_scheduler.release(operation)

    close_observer = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    close_observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_observer
    async with asyncio.timeout(3.0):
        await runtime.close()
    expected = [
        (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.TERMINAL, RolloverResultV1.CLOSED),
    ]
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == expected
    _run_queued_owner_callbacks(loop)
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == expected
    loop.close()
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_sync_worker_terminal_settlement_hands_rollover_to_activation_loop(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=21,
        retention_hours=24,
    )
    command = _create_epoch()
    await _activate(runtime, command)
    admission = runtime.evidence_admission
    assert admission is not None
    try:
        _settle_maximum_proactive(runtime)
        held_admission, operation, _authority, result = _reserve_proactive(runtime)
        assert held_admission is admission and result.lease is not None
        assert admission.try_open_non_user_turn(result.lease) is m.AppendDisposition.ADMITTED

        completed = threading.Event()
        outcome: list[m.AppendDisposition] = []
        worker_had_loop: list[bool] = []

        def settle_without_loop() -> None:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                worker_had_loop.append(False)
            else:
                worker_had_loop.append(True)
            outcome.append(
                admission.settle_completed(
                    result.lease,
                    queued_chunk_count=0,
                    started_chunk_count=0,
                    transport_confirmed_full_count=0,
                    assistant_delivery_context_recorded=False,
                )
            )
            completed.set()

        thread = threading.Thread(target=settle_without_loop)
        thread.start()
        assert await asyncio.wait_for(asyncio.to_thread(completed.wait, 2.0), 3.0)
        thread.join(timeout=0.1)
        assert not thread.is_alive()
        assert worker_had_loop == [False]
        assert outcome == [m.AppendDisposition.ADMITTED]
        runtime.operation_scheduler.release(operation)
        await _wait_for_rollover(runtime)

        with sqlite3.connect(tmp_path / "capture-v1.sqlite3") as connection:
            sessions = connection.execute(
                "SELECT logical_session_id, state FROM evidence_sessions ORDER BY rowid"
            ).fetchall()
        assert len(sessions) == 2
        assert sessions[0] == (command.logical_session_id, "sealed")
        assert sessions[1][1] == "open"

        successor_admission, successor_operation, _authority, successor = (
            _reserve_proactive(runtime)
        )
        assert successor_admission is admission
        assert successor.disposition is m.AppendDisposition.ADMITTED
        assert successor.lease is not None
        assert successor.lease.logical_session_id == sessions[1][0]
        assert admission.discard_unopened_proactive_turn(successor.lease, _authority) is True
        runtime.operation_scheduler.release(successor_operation)

        diagnostics = admission.diagnostics()
        assert diagnostics.active_lease_count == 0
        assert diagnostics.capture_state is m.CaptureState.ACTIVE
        assert diagnostics.sticky_fault is None
    finally:
        await runtime.close()

    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert diagnostics.active_lease_count == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "replay", "proactive"])
async def test_lawful_unopened_discard_reaches_real_runtime_owned_rollover(
    tmp_path: Path,
    kind: str,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverResultV1, RolloverStageV1

    case = tmp_path / kind
    case.mkdir()
    runtime = HostEvidenceRuntimeV1(
        database=case / "capture-v1.sqlite3",
        owner_generation=21,
        retention_hours=24,
    )
    await _activate(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    try:
        _settle_maximum_proactive(runtime)
        reserve = {
            "user": _reserve_user,
            "replay": _reserve_replay,
            "proactive": _reserve_proactive,
        }[kind]
        held_admission, operation, authority, result = reserve(runtime)
        assert held_admission is admission and result.lease is not None
        discard = getattr(admission, f"discard_unopened_{kind}_turn")
        assert discard(result.lease, authority) is True
        runtime.operation_scheduler.release(operation)
        await _wait_for_rollover(runtime)
        assert [(record.stage, record.result) for record in _rollover_records(runtime)] == [
            (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
            (RolloverStageV1.QUEUED, RolloverResultV1.ACCEPTED),
            (RolloverStageV1.DURABLE, RolloverResultV1.COMMITTED),
            (RolloverStageV1.PUBLISHED, RolloverResultV1.COMMITTED),
            (RolloverStageV1.TERMINAL, RolloverResultV1.COMMITTED),
        ]
        with sqlite3.connect(case / "capture-v1.sqlite3") as connection:
            sessions = connection.execute(
                "SELECT logical_session_id, state FROM evidence_sessions ORDER BY rowid"
            ).fetchall()
        assert len(sessions) == 2
        assert sessions[0][1] == "sealed"
        assert sessions[1][1] == "open"
        diagnostics = admission.diagnostics()
        assert diagnostics.capture_state is m.CaptureState.ACTIVE
        assert diagnostics.sticky_fault is None
        assert diagnostics.active_lease_count == 0
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_active_real_lease_defers_claim_until_terminal_settlement(tmp_path: Path) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(database=database, owner_generation=21, retention_hours=24)
    await _activate(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    try:
        _settle_maximum_proactive(runtime)
        held_admission, operation, _authority, result = _reserve_proactive(runtime)
        assert held_admission is admission
        assert result.disposition is m.AppendDisposition.ADMITTED
        assert result.lease is not None
        assert admission.try_open_non_user_turn(result.lease) is m.AppendDisposition.ADMITTED
        await asyncio.sleep(0)
        assert _rollover_records(runtime) == ()
        assert admission.settle_completed(
            result.lease,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            assistant_delivery_context_recorded=False,
        ) is m.AppendDisposition.ADMITTED
        runtime.operation_scheduler.release(operation)
        await _wait_for_rollover(runtime)
        assert len(_rollover_records(runtime)) == 5
        next_admission, next_operation, next_authority, next_result = _reserve_proactive(runtime)
        assert next_result.disposition is m.AppendDisposition.ADMITTED
        assert next_result.lease is not None
        assert next_admission.discard_unopened_proactive_turn(
            next_result.lease, next_authority
        ) is True
        runtime.operation_scheduler.release(next_operation)
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (2,)
    finally:
        await runtime.close()
    assert admission.diagnostics().active_lease_count == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_real_replayable_turn_defers_claim_until_lawful_discard(tmp_path: Path) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(database=database, owner_generation=22, retention_hours=24)
    await _activate(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    try:
        _settle_maximum_proactive(runtime)
        replay_admission, operation, authority, result = _reserve_replay(runtime)
        assert replay_admission is admission
        assert result.disposition is m.AppendDisposition.ADMITTED
        assert result.lease is not None
        await asyncio.sleep(0)
        assert _rollover_records(runtime) == ()
        assert admission.discard_unopened_replay_turn(result.lease, authority) is True
        runtime.operation_scheduler.release(operation)
        await _wait_for_rollover(runtime)
        assert len(_rollover_records(runtime)) == 5
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (2,)
    finally:
        await runtime.close()
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_third_real_reservation_loses_terminal_claim_without_quota_taint(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(database=database, owner_generation=23, retention_hours=24)
    await _activate(runtime, _create_epoch())
    admission = runtime.evidence_admission
    assert admission is not None
    try:
        _settle_maximum_proactive(runtime)
        _settle_maximum_proactive(runtime)
        losing_admission, losing_operation, _authority, losing = _reserve_proactive(runtime)
        assert losing_admission is admission
        assert losing.lease is None
        assert losing.disposition is m.AppendDisposition.SESSION_CLOSING
        runtime.operation_scheduler.release(losing_operation)
        diagnostics = admission.diagnostics()
        assert diagnostics.sticky_fault is None
        assert diagnostics.capture_state is m.CaptureState.ACTIVE
        await _wait_for_rollover(runtime)
        assert len(_rollover_records(runtime)) == 5
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (2,)
    finally:
        await runtime.close()
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_conflicting_real_terminal_owners_cannot_split_rollover_lineage(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverResultV1, RolloverStageV1

    database = tmp_path / "capture-v1.sqlite3"
    command = _create_epoch()
    runtime = HostEvidenceRuntimeV1(database=database, owner_generation=24, retention_hours=24)
    await _activate(runtime, command)
    admission = runtime.evidence_admission
    assert admission is not None

    _settle_maximum_proactive(runtime)
    _settle_maximum_proactive(runtime)

    projection = BrowserEventProjection()
    revoke_authority = runtime.reserve_browser_revoke(
        binding_generation=command.binding_generation,
        request=m.EvidenceRevokeRequestV1(sequence=2),
        projection_reservation=projection.reserve_capture_status(),
        validate_projection_reservation=projection.validate_capture_status_reservation,
    )

    revoke = asyncio.create_task(runtime.activate_revoke(revoke_authority, timeout_seconds=2.0))
    lifecycle_retirement = asyncio.create_task(
        runtime.invalidate_active_binding(m.BindingCloseReason.CLIENT_CLOSED)
    )
    host_close = asyncio.create_task(runtime.close())
    revoke_result, retirement_result, close_result = await asyncio.gather(
        revoke,
        lifecycle_retirement,
        host_close,
        return_exceptions=True,
    )
    assert revoke_result in {
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
        m.RevokeDisposition.PURGE_COMPLETED,
    }
    assert retirement_result is None
    assert close_result is None
    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == [
        (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.TERMINAL, RolloverResultV1.CLOSED),
    ]
    with sqlite3.connect(database) as connection:
        sessions = connection.execute(
            "SELECT logical_session_id FROM evidence_sessions ORDER BY rowid"
        ).fetchall()
        successor_links = connection.execute(
            "SELECT canonical_payload FROM evidence_events WHERE event_kind = 'session_opened' "
            "ORDER BY rowid"
        ).fetchall()
    assert sessions == []
    assert successor_links == []
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_two_real_maximum_turns_roll_over_through_runtime_writer_and_sqlite(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.admission import (
        TURN_SESSION_CANONICAL_BYTE_RESERVATION,
        TURN_SESSION_EVENT_RESERVATION,
    )
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverObservationV1

    database = tmp_path / "capture-v1.sqlite3"
    command = _create_epoch()
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=11,
        retention_hours=24,
    )
    await _activate(runtime, command)
    predecessor_lifecycle = runtime.evidence_lifecycle
    predecessor_admission = runtime.evidence_admission
    assert predecessor_lifecycle is not None and predecessor_admission is not None
    assert not hasattr(runtime, "lifecycle_owner")
    assert not hasattr(predecessor_lifecycle, "prepare_rollover")
    assert not hasattr(predecessor_admission, "try_rollover")

    try:
        for _ in range(2):
            _settle_maximum_proactive(runtime)

        closing_admission, closing_operation, _closing_authority, closing_result = (
            _reserve_proactive(runtime)
        )
        assert closing_result.lease is None
        assert closing_result.disposition is m.AppendDisposition.SESSION_CLOSING
        runtime.operation_scheduler.release(closing_operation)

        await _wait_for_rollover(runtime)
        successor_lifecycle = runtime.evidence_lifecycle
        successor_admission = runtime.evidence_admission
        assert successor_lifecycle is predecessor_lifecycle
        assert successor_admission is predecessor_admission
        diagnostics = successor_admission.diagnostics()
        assert diagnostics.capture_state is m.CaptureState.ACTIVE
        assert diagnostics.sticky_fault is None

        records = tuple(
            record
            for record in runtime.production_observations.records()
            if type(record) is RolloverObservationV1
        )
        assert len(records) == 5

        next_admission, next_operation, next_authority, next_result = _reserve_proactive(runtime)
        assert next_result.disposition is m.AppendDisposition.ADMITTED
        assert next_result.lease is not None
        assert next_result.lease.logical_session_id != command.logical_session_id
        assert next_admission.discard_unopened_proactive_turn(
            next_result.lease, next_authority
        ) is True
        runtime.operation_scheduler.release(next_operation)

        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            sessions = connection.execute(
                "SELECT logical_session_id, state FROM evidence_sessions ORDER BY rowid"
            ).fetchall()
            rollover_events = connection.execute(
                "SELECT event_kind FROM evidence_events WHERE event_kind IN "
                "('binding_closed', 'session_seal_requested', 'session_opened', "
                    "'binding_opened') ORDER BY rowid DESC LIMIT 4"
            ).fetchall()
            epoch = connection.execute(
                "SELECT state FROM consent_epochs WHERE consent_epoch_id = ?",
                (command.consent_epoch_id,),
            ).fetchone()
            successor_open = connection.execute(
                "SELECT canonical_payload FROM evidence_events "
                "WHERE logical_session_id = ? AND event_kind = 'session_opened'",
                (sessions[1][0],),
            ).fetchone()
        assert len(sessions) == 2
        assert sessions[0] == (command.logical_session_id, "sealed")
        assert sessions[1][1] == "open"
        assert [row[0] for row in reversed(rollover_events)] == [
            "binding_closed",
            "session_seal_requested",
            "session_opened",
            "binding_opened",
        ]
        assert epoch == ("active",)
        assert json.loads(successor_open[0])["predecessor_session_id"] == (
            command.logical_session_id
        )
        assert 2 * TURN_SESSION_EVENT_RESERVATION == 8_712
        assert 2 * TURN_SESSION_CANONICAL_BYTE_RESERVATION == 32 * 1024 * 1024
    finally:
        await runtime.close()

    diagnostics = predecessor_admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert diagnostics.active_lease_count == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_close_accepts_real_rollover_completed_before_deadline_during_loop_stall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence import runtime as runtime_module
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=12,
        retention_hours=24,
    )
    sqlite_transport = runtime.create_sqlite_transport()
    rollover_entered = threading.Event()
    release_rollover = threading.Event()

    class DelayedRolloverTransport:
        def rollover_session(self, command):  # type: ignore[no-untyped-def]
            rollover_entered.set()
            if not release_rollover.wait(2.0):
                raise RuntimeError("test did not release the real rollover")
            return sqlite_transport.rollover_session(command)

        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(sqlite_transport, name)

    await _activate(runtime, _create_epoch(), transport=DelayedRolloverTransport())
    _settle_maximum_proactive(runtime)
    _settle_maximum_proactive(runtime)
    _admission, operation, _authority, closing = _reserve_proactive(runtime)
    assert closing.lease is None
    assert closing.disposition is m.AppendDisposition.SESSION_CLOSING
    runtime.operation_scheduler.release(operation)

    async with asyncio.timeout(2.0):
        while not rollover_entered.is_set():
            await asyncio.sleep(0.001)

    original_wait = runtime_module._wait_for_thread_signal
    close_wait_entered = asyncio.Event()

    async def observe_close_wait(event, *, timeout_seconds):  # type: ignore[no-untyped-def]
        close_wait_entered.set()
        return await original_wait(event, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(runtime_module, "_wait_for_thread_signal", observe_close_wait)
    close_task = asyncio.create_task(runtime.close())
    try:
        await asyncio.wait_for(close_wait_entered.wait(), timeout=2.0)
        release_rollover.set()
        time.sleep(2.1)
        await close_task
    finally:
        release_rollover.set()

    assert close_task.result() is None


@pytest.mark.asyncio
async def test_duplicate_terminal_trigger_is_idempotent_and_publishes_one_successor(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverObservationV1

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=12,
        retention_hours=24,
    )
    await _activate(runtime, _create_epoch())
    try:
        _settle_maximum_proactive(runtime)
        admission, _authority, second_lease = _settle_maximum_proactive(runtime)
        assert admission.settle_completed(
            second_lease,
            queued_chunk_count=0,
            started_chunk_count=0,
            transport_confirmed_full_count=0,
            assistant_delivery_context_recorded=False,
        ) is m.AppendDisposition.INVALID_AUTHORITY
        await _wait_for_rollover(runtime)
        assert len(
            tuple(
                record
                for record in runtime.production_observations.records()
                if type(record) is RolloverObservationV1
            )
        ) == 5
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (2,)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_real_writer_rollover_fault_never_publishes_successor(tmp_path: Path) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import (
        RolloverObservationV1,
        RolloverResultV1,
        RolloverStageV1,
    )

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=13,
        retention_hours=24,
    )
    delegate = runtime.create_sqlite_transport()

    class FaultRolloverTransport:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(delegate, name)

        def rollover_session(self, command):  # type: ignore[no-untyped-def]
            del command
            return m.StoreDisposition.FAULTED

    await _activate(runtime, _create_epoch(), transport=FaultRolloverTransport())
    admission = runtime.evidence_admission
    assert admission is not None
    try:
        _settle_maximum_proactive(runtime)
        _settle_maximum_proactive(runtime)
        await _wait_for_rollover(runtime)
        rollover = tuple(
            record
            for record in runtime.production_observations.records()
            if type(record) is RolloverObservationV1
        )
        assert rollover[-1].stage is RolloverStageV1.TERMINAL
        assert rollover[-1].result is RolloverResultV1.FAULTED
        assert admission.diagnostics().capture_state is m.CaptureState.FAULTED
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (1,)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_close_waits_for_real_faulted_rollover_lifecycle_abort(tmp_path: Path) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.production_observation import (
        CloseResultV1,
        CloseStageObservationV1,
        CloseStageV1,
        RolloverObservationV1,
        RolloverResultV1,
        RolloverStageV1,
    )

    database = tmp_path / "capture-v1.sqlite3"
    runtime, bundle, failure_entered, resume_failure = _checkpointed_runtime(
        database,
        owner_generation=42,
        publication_fault=False,
    )
    delegate = runtime.create_sqlite_transport()
    rollover_entered = threading.Event()
    release_rollover = threading.Event()

    class BlockedFaultRolloverTransport:
        def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
            return getattr(delegate, name)

        def rollover_session(self, command):  # type: ignore[no-untyped-def]
            del command
            rollover_entered.set()
            assert release_rollover.wait(2.0)
            return m.StoreDisposition.FAULTED

    await _activate(
        runtime,
        _create_epoch(),
        transport=BlockedFaultRolloverTransport(),
    )
    admission = runtime.evidence_admission
    assert admission is not None
    _settle_maximum_proactive(runtime)
    _settle_maximum_proactive(runtime)
    assert await asyncio.to_thread(rollover_entered.wait, 2.0)

    close_task = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert not close_task.done()
    release_rollover.set()
    assert await asyncio.to_thread(failure_entered.wait, 2.0)
    assert not close_task.done()
    resume_failure.set()
    async with asyncio.timeout(5.0):
        await close_task

    records = runtime.production_observations.records()
    rollover_terminals = tuple(
        record
        for record in records
        if type(record) is RolloverObservationV1
        and record.stage is RolloverStageV1.TERMINAL
    )
    assert tuple(record.result for record in rollover_terminals) == (
        RolloverResultV1.FAULTED,
    )
    close_terminals = tuple(
        record
        for record in records
        if type(record) is CloseStageObservationV1
        and record.stage is CloseStageV1.EVIDENCE_RUNTIME
    )
    assert tuple(record.result for record in close_terminals) == (CloseResultV1.SUCCEEDED,)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (1,)
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert diagnostics.active_lease_count == 0
    assert runtime.operation_scheduler.active_count == 0
    assert not runtime.writer_running
    assert bundle._capacity_probe.rollover_failure_checkpoint_consumed() is True


@pytest.mark.asyncio
async def test_pre_enqueue_publication_fault_aborts_real_rollover_without_successor(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.production_observation import (
        RolloverObservationV1,
        RolloverResultV1,
        RolloverStageV1,
    )

    database = tmp_path / "capture-v1.sqlite3"
    runtime, bundle, failure_entered, resume_failure = _checkpointed_runtime(
        database,
        owner_generation=14,
        publication_fault=True,
    )

    command = _create_epoch()
    await _activate(runtime, command)
    admission = runtime.evidence_admission
    lifecycle = runtime.evidence_lifecycle
    assert admission is not None and lifecycle is not None
    try:
        _settle_maximum_proactive(runtime)
        _settle_maximum_proactive(runtime)
        assert await asyncio.to_thread(failure_entered.wait, 2.0)
        close_task = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        assert not close_task.done()
        resume_failure.set()
        async with asyncio.timeout(5.0):
            await close_task

        rollover = tuple(
            record
            for record in runtime.production_observations.records()
            if type(record) is RolloverObservationV1
        )
        assert [(record.stage, record.result) for record in rollover] == [
            (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
            (RolloverStageV1.TERMINAL, RolloverResultV1.REJECTED),
        ]
        assert bundle._capacity_probe.rollover_publication_fault_consumed() is True
        assert bundle._capacity_probe.rollover_failure_checkpoint_consumed() is True
        diagnostics = admission.diagnostics()
        assert diagnostics.capture_state is m.CaptureState.ACTIVE
        assert diagnostics.sticky_fault is None

        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT logical_session_id, state FROM evidence_sessions"
            ).fetchall() == [(command.logical_session_id, "open")]
            assert connection.execute(
                "SELECT count(*) FROM evidence_events WHERE event_kind IN "
                "('binding_closed', 'session_seal_requested')"
            ).fetchone() == (0,)
    finally:
        resume_failure.set()
        await runtime.close()

    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert diagnostics.active_lease_count == 0
    assert bundle._capacity_probe.observations()[-1].all_capacity_released is True
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_rollover_failure_checkpoint_requires_owner_context_and_cannot_rearm() -> None:
    from hermes_realtime._qualification import (
        _new_qualification_full_host_dependencies,
        _new_qualification_owner_context_capability,
    )

    identity = object()

    class Collector:
        def _qualification_owner_identity_matches(self, candidate: object) -> bool:
            return candidate is identity

        def __getattr__(self, name: str):
            if name.startswith("record_"):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    capability = _new_qualification_owner_context_capability(identity)
    bundle = _new_qualification_full_host_dependencies(
        identity=identity,
        inference_factory=lambda: object(),
        speech_presence_factory=lambda: object(),
        synthesizer_factory=lambda: object(),
        transcriber_factory=lambda: object(),
        vad_factory=lambda: object(),
        identity_factory=lambda: "qualification_identity",
    )
    entered = threading.Event()
    resume = threading.Event()
    resume.set()
    with pytest.raises(RuntimeError, match="active owner context"):
        bundle._capacity_probe.arm_rollover_failure_checkpoint(
            identity,
            entered=entered,
            resume=resume,
        )
    with capability.wire(Collector(), full_host_dependencies=bundle):
        bundle._capacity_probe.arm_rollover_failure_checkpoint(
            identity,
            entered=entered,
            resume=resume,
        )
        await bundle._capacity_probe.pause_after_rollover_rejection()
        assert entered.is_set()
        assert bundle._capacity_probe.rollover_failure_checkpoint_consumed() is True
        with pytest.raises(RuntimeError, match="single-use"):
            bundle._capacity_probe.arm_rollover_failure_checkpoint(
                identity,
                entered=threading.Event(),
                resume=threading.Event(),
            )


class _BlockingRolloverTransport:
    def __init__(self, delegate, entered: threading.Event, release: threading.Event) -> None:  # type: ignore[no-untyped-def]
        self._delegate = delegate
        self._entered = entered
        self._release = release

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        return getattr(self._delegate, name)

    def rollover_session(self, command):  # type: ignore[no-untyped-def]
        self._entered.set()
        while not self._release.is_set():
            time.sleep(0.005)
        return self._delegate.rollover_session(command)


@pytest.mark.asyncio
async def test_cancelled_close_observer_does_not_cancel_admitted_real_rollover(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import RolloverStageV1

    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(database=database, owner_generation=31, retention_hours=24)
    entered = threading.Event()
    release = threading.Event()
    transport = _BlockingRolloverTransport(runtime.create_sqlite_transport(), entered, release)
    await _activate(runtime, _create_epoch(), transport=transport)
    admission = runtime.evidence_admission
    assert admission is not None

    _settle_maximum_proactive(runtime)
    _settle_maximum_proactive(runtime)
    assert await asyncio.wait_for(asyncio.to_thread(entered.wait, 5.0), 6.0)
    close_observer = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    close_observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_observer
    release.set()
    await _wait_for_rollover(runtime)
    assert (
        sum(record.stage is RolloverStageV1.TERMINAL for record in _rollover_records(runtime))
        == 1
    )
    await runtime.close()
    await runtime.close()
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0


@pytest.mark.asyncio
async def test_host_close_before_rollover_queue_publication_is_exact_closed_rejection(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool
    from hermes_realtime.production_observation import RolloverResultV1, RolloverStageV1

    database = tmp_path / "capture-v1.sqlite3"
    command = _create_epoch()
    runtime = HostEvidenceRuntimeV1(database=database, owner_generation=32, retention_hours=24)
    await _activate(runtime, command)
    admission = runtime.evidence_admission
    assert admission is not None
    _settle_maximum_proactive(runtime)
    _settle_maximum_proactive(runtime)
    await runtime.close()

    assert [(record.stage, record.result) for record in _rollover_records(runtime)] == [
        (RolloverStageV1.CLAIMED, RolloverResultV1.ACCEPTED),
        (RolloverStageV1.TERMINAL, RolloverResultV1.CLOSED),
    ]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM evidence_sessions WHERE logical_session_id != ?",
            (command.logical_session_id,),
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT state FROM evidence_sessions WHERE logical_session_id = ?",
            (command.logical_session_id,),
        ).fetchone() == ("open",)
    diagnostics = admission.diagnostics()
    assert diagnostics.active_lease_count == 0
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0
    assert runtime.operation_scheduler.active_count == 0
    recovered = SQLiteEvidenceSpool(
        database,
        clock=lambda: datetime.now(UTC),
        owner_generation=32,
    )
    try:
        assert recovered.recover_existing() is RecoveryDisposition.RECOVERED
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM consent_epochs").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (0,)
            assert connection.execute(
                "SELECT reason_code FROM erasure_tombstones"
            ).fetchall() == [("unclean_epoch",)]
    finally:
        recovered.close()


@pytest.mark.asyncio
async def test_rollover_succeeds_when_bounded_production_trace_overflows(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.production_observation import (
        RolloverStageV1,
        _new_observation_channel,
    )

    observations, recorder = _new_observation_channel(max_records=4)
    database = tmp_path / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=33,
        retention_hours=24,
        production_observations=observations,
        production_observation_recorder=recorder,
    )
    await _activate(runtime, _create_epoch())
    try:
        _settle_maximum_proactive(runtime)
        _settle_maximum_proactive(runtime)
        async with asyncio.timeout(5.0):
            while observations.status().trace_complete:
                await asyncio.sleep(0.01)
        assert [record.stage for record in _rollover_records(runtime)] == [
            RolloverStageV1.CLAIMED,
            RolloverStageV1.QUEUED,
            RolloverStageV1.DURABLE,
            RolloverStageV1.PUBLISHED,
        ]
        assert observations.status().trace_complete is False
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (2,)

        def task12_style_trace_is_valid() -> bool:
            return observations.status().trace_complete and any(
                record.stage is RolloverStageV1.TERMINAL
                for record in _rollover_records(runtime)
            )

        assert task12_style_trace_is_valid() is False
    finally:
        await runtime.close()


def _crash_child_source(database: Path, marker: Path, release: Path, *, after_commit: bool) -> str:
    command_image = base64.b64encode(pickle.dumps(_create_epoch())).decode("ascii")
    return textwrap.dedent(
        f"""
        import asyncio, base64, pickle, time
        from datetime import UTC, datetime
        from pathlib import Path
        from hermes_realtime.client import BrowserEventProjection
        from hermes_realtime.evidence import models as m
        from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
        from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

        class Blocking:
            def __init__(self, delegate): self.delegate = delegate
            def __getattr__(self, name): return getattr(self.delegate, name)
            def rollover_session(self, command):
                if {after_commit!r}:
                    result = self.delegate.rollover_session(command)
                Path({str(marker)!r}).write_text('ready')
                while not Path({str(release)!r}).exists(): time.sleep(0.01)
                if not {after_commit!r}:
                    result = self.delegate.rollover_session(command)
                return result
        def settle(runtime):
            operation = runtime.operation_scheduler.try_reserve(
                m.ConversationOperationKind.PROACTIVE
            )
            lifecycle, admission = runtime.resolve_evidence_pair()
            authority = lifecycle.mint_proactive_turn(operation)
            result = admission.try_reserve_proactive_turn(authority, operation)
            assert admission.try_open_non_user_turn(result.lease) is m.AppendDisposition.ADMITTED
            assert admission.settle_completed(
                result.lease,
                queued_chunk_count=0,
                started_chunk_count=0,
                transport_confirmed_full_count=0,
                assistant_delivery_context_recorded=False,
            ) is m.AppendDisposition.ADMITTED
            runtime.operation_scheduler.release(operation)
        async def main():
            runtime = HostEvidenceRuntimeV1(
                database=Path({str(database)!r}), owner_generation=41, retention_hours=24
            )
            command = pickle.loads(base64.b64decode({command_image!r}))
            projection = BrowserEventProjection()
            authority = runtime.reserve_consent_authority(
                command,
                projection.reserve_capture_status(),
                projection.validate_capture_status_reservation,
            )
            transport = Blocking(
                SQLiteEvidenceSpool(
                    Path({str(database)!r}),
                    clock=lambda: datetime.now(UTC),
                    owner_generation=41,
                )
            )
            assert await runtime.activate_consent(
                authority,
                transport=transport,
                binding_is_current=lambda candidate: candidate is command,
                timeout_seconds=2.0,
            ) is m.ConsentDisposition.CONSENT_ACTIVATED
            settle(runtime)
            settle(runtime)
            while True: await asyncio.sleep(1)
        asyncio.run(main())
        """
    )


@pytest.mark.parametrize(
    "after_commit", [False, True], ids=["before_sqlite_commit", "after_sqlite_commit"]
)
def test_process_crash_rollover_is_purged_before_fresh_availability(
    tmp_path: Path,
    after_commit: bool,
) -> None:
    from hermes_realtime.evidence.models import RecoveryDisposition
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

    case = tmp_path / ("after" if after_commit else "before")
    case.mkdir()
    database = case / "capture-v1.sqlite3"
    marker = case / "rollover.checkpoint"
    release = case / "rollover.release"
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _crash_child_source(database, marker, release, after_commit=after_commit),
        ],
        cwd=Path.cwd(),
    )
    try:
        deadline = time.monotonic() + 15.0
        while not marker.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists(), f"child exited before checkpoint: {child.poll()}"
        child.kill()
        child.wait(timeout=10.0)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10.0)

    recovered = None
    recovery_disposition = None
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        candidate = SQLiteEvidenceSpool(
            database,
            clock=lambda: datetime.now(UTC),
            owner_generation=41,
        )
        recovery_disposition = candidate.recover_existing()
        if recovery_disposition is RecoveryDisposition.OWNERSHIP_UNAVAILABLE:
            candidate.close()
            time.sleep(0.02)
            continue
        recovered = candidate
        break
    assert recovered is not None
    try:
        assert recovery_disposition is RecoveryDisposition.RECOVERED
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT count(*) FROM consent_epochs").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM evidence_sessions").fetchone() == (0,)
            assert connection.execute("SELECT count(*) FROM evidence_events").fetchone() == (0,)
            assert connection.execute(
                "SELECT reason_code FROM erasure_tombstones"
            ).fetchall() == [("unclean_epoch",)]
    finally:
        recovered.close()


def test_public_runtime_authority_is_exactly_confined() -> None:
    from hermes_realtime.evidence.admission import EvidenceAdmissionViewV1
    from hermes_realtime.evidence.lifecycle import EvidenceConversationAuthorityV1
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    forbidden = {
        "abort_prepared_rollover",
        "commit_prepared_rollover",
        "complete_rollover",
        "enqueue_prepared_rollover",
        "finalize_failed_rollover_completion",
        "prepare_rollover",
        "prepare_rollover_completion",
        "publish_prepared_rollover",
        "rollover_authority",
        "rollover_binding",
        "try_rollover",
    }
    assert forbidden.isdisjoint(dir(EvidenceConversationAuthorityV1))
    assert forbidden.isdisjoint(dir(EvidenceAdmissionViewV1))
    assert "lifecycle_owner" not in dir(HostEvidenceRuntimeV1)
    parameters = inspect.signature(HostEvidenceRuntimeV1).parameters
    assert not any("rollover" in name or "authority" in name for name in parameters)

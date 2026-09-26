from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from threading import Lock, local
from time import monotonic_ns

import pytest


def _create_epoch(retention_hours: int = 24) -> object:
    from uuid import UUID

    from hermes_realtime.evidence import models as m

    def uid(value: int) -> str:
        return str(UUID(int=value, version=4))

    installation_id = uid(1)
    producer_instance_id = uid(2)
    consent_epoch_id = uid(3)
    logical_session_id = uid(4)
    binding_id = uid(5)
    digest = "a" * 64
    return m.CreateEpochV1(
        protocol_version=1,
        installation_id=installation_id,
        producer_instance_id=producer_instance_id,
        consent_epoch_id=consent_epoch_id,
        logical_session_id=logical_session_id,
        binding_id=binding_id,
        binding_generation=11,
        consent_version=m.CONSENT_VERSION,
        disclosure_digest=digest,
        retention_hours=retention_hours,
        microphone_accepted=True,
        typed_accepted=True,
        control_sequence=1,
        control_fingerprint_hash="b" * 64,
        session_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            logical_session_id=logical_session_id,
            event_id=uid(6),
            event_sequence=1,
            event_kind=m.EventKind.SESSION_OPENED,
            payload=m.SessionOpenedPayloadV1(
                consent_epoch_id=consent_epoch_id,
                binding_id=binding_id,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest=digest,
                retention_hours=retention_hours,
                microphone_accepted=True,
                typed_accepted=True,
                predecessor_session_id=None,
            ),
        ),
        binding_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            logical_session_id=logical_session_id,
            event_id=uid(7),
            event_sequence=2,
            event_kind=m.EventKind.BINDING_OPENED,
            payload=m.BindingOpenedPayloadV1(
                binding_id=binding_id,
                binding_generation=11,
                microphone_available=True,
                typed_available=True,
            ),
        ),
    )


def _run_runtime_subprocess(script: str, *, timeout_message: str) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source_root = repository_root / "src"
    expected_runtime = source_root / "hermes_realtime" / "evidence" / "runtime.py"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), environment.get("PYTHONPATH")) if part
    )
    environment["HERMES_REALTIME_EXPECTED_RUNTIME"] = str(expected_runtime)
    source_assertion = """
import os
from pathlib import Path

from hermes_realtime.evidence import runtime as runtime_module

if Path(runtime_module.__file__).resolve() != Path(
    os.environ["HERMES_REALTIME_EXPECTED_RUNTIME"]
).resolve():
    raise RuntimeError("subprocess did not import the exact candidate runtime")
"""
    try:
        completed = subprocess.run(
            [sys.executable, "-B", "-c", source_assertion + script],
            check=False,
            capture_output=True,
            cwd=repository_root,
            env=environment,
            text=True,
            timeout=5.0,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(timeout_message)

    if completed.returncode != 0:
        pytest.fail(completed.stdout + completed.stderr)


def test_thread_signal_waiter_does_not_block_asyncio_shutdown() -> None:
    script = """
import asyncio
from threading import Event

from hermes_realtime.evidence.runtime import _wait_for_thread_signal


async def main() -> None:
    asyncio.create_task(
        _wait_for_thread_signal(Event(), timeout_seconds=None),
        name="shutdown-signal-probe",
    )
    await asyncio.sleep(0.1)


asyncio.run(main())
"""
    _run_runtime_subprocess(
        script,
        timeout_message="thread-signal waiter blocked asyncio shutdown",
    )


def test_thread_signal_timeout_survives_default_executor_starvation() -> None:
    script = """
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Event

from hermes_realtime.evidence.runtime import _wait_for_thread_signal


async def main() -> None:
    release_worker = Event()
    worker_started = Event()
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)

    def occupy_worker() -> None:
        worker_started.set()
        release_worker.wait()

    worker = loop.run_in_executor(None, occupy_worker)
    while not worker_started.is_set():
        await asyncio.sleep(0)
    try:
        if await _wait_for_thread_signal(Event(), timeout_seconds=0.02):
            raise RuntimeError("unset signal completed successfully")
    finally:
        release_worker.set()
        await worker


asyncio.run(main())
"""
    _run_runtime_subprocess(
        script,
        timeout_message="thread-signal timeout was defeated by executor starvation",
    )


def test_thread_signal_fanout_preserves_deadline_and_shutdown() -> None:
    script = """
import asyncio
from threading import Event
from time import monotonic

from hermes_realtime.evidence.runtime import _wait_for_thread_signal


async def main() -> None:
    pending = [
        asyncio.create_task(_wait_for_thread_signal(Event(), timeout_seconds=None))
        for _ in range(80)
    ]
    await asyncio.sleep(0.01)
    started = monotonic()
    if await _wait_for_thread_signal(Event(), timeout_seconds=0.02):
        raise RuntimeError("unset signal completed successfully under fanout")
    if monotonic() - started >= 0.2:
        raise RuntimeError("signal deadline was defeated by fanout")
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


asyncio.run(main())
"""
    _run_runtime_subprocess(
        script,
        timeout_message="thread-signal fanout defeated deadline or shutdown",
    )


def test_thread_signal_rejects_signal_observed_after_deadline() -> None:
    script = """
import asyncio
from threading import Event, Thread
from time import sleep

from hermes_realtime.evidence.admission import _set_ticket_signal
from hermes_realtime.evidence.runtime import _wait_for_thread_signal


async def main() -> None:
    event = Event()

    def signal_late() -> None:
        sleep(0.1)
        _set_ticket_signal(event)

    setter = Thread(target=signal_late)
    setter.start()
    waiter = asyncio.create_task(
        _wait_for_thread_signal(event, timeout_seconds=0.05)
    )
    await asyncio.sleep(0)
    sleep(0.15)
    result = await waiter
    setter.join()
    if result:
        raise RuntimeError("signal observed after its deadline was accepted")


asyncio.run(main())
"""
    _run_runtime_subprocess(
        script,
        timeout_message="late thread signal defeated the observation deadline",
    )


def test_thread_signal_accepts_production_signal_set_before_deadline() -> None:
    script = """
import asyncio
from threading import Event, Thread
from time import sleep

from hermes_realtime.evidence.admission import _set_ticket_signal
from hermes_realtime.evidence.runtime import _wait_for_thread_signal


async def main() -> None:
    event = Event()

    def signal_early() -> None:
        sleep(0.01)
        _set_ticket_signal(event)
        sleep(0.1)
        _set_ticket_signal(event)

    setter = Thread(target=signal_early)
    setter.start()
    waiter = asyncio.create_task(
        _wait_for_thread_signal(event, timeout_seconds=0.1)
    )
    await asyncio.sleep(0)
    sleep(0.15)
    result = await waiter
    setter.join()
    if not result:
        raise RuntimeError("pre-deadline production signal was refused")


asyncio.run(main())
"""
    _run_runtime_subprocess(
        script,
        timeout_message="pre-deadline thread signal was lost during loop delay",
    )


def test_thread_signal_rechecks_coherently_before_deadline_timeout() -> None:
    script = """
import asyncio
from threading import Event, Thread
from time import sleep

from hermes_realtime.evidence import runtime as runtime_module
from hermes_realtime.evidence.admission import _set_ticket_signal


async def main() -> None:
    event = Event()
    publish = Event()
    published = Event()
    original_status = runtime_module._ticket_signal_status
    first_call = True

    def coordinated_status(candidate, *, deadline):
        nonlocal first_call
        if first_call:
            first_call = False
            status = original_status(candidate, deadline=deadline)
            if status is not None:
                raise RuntimeError("atomic-race event was already classified")
            publish.set()
            if not published.wait(1.0):
                raise RuntimeError("atomic-race producer did not publish")
            sleep(0.6)
            return None
        return original_status(candidate, deadline=deadline)

    def signal_during_observation_gap() -> None:
        if not publish.wait(1.0):
            return
        _set_ticket_signal(event)
        published.set()

    runtime_module._ticket_signal_status = coordinated_status
    setter = Thread(target=signal_during_observation_gap)
    setter.start()
    result = await runtime_module._wait_for_thread_signal(
        event,
        timeout_seconds=0.5,
    )
    setter.join()
    if not result:
        raise RuntimeError("timely signal was lost after coherent-read gap")


asyncio.run(main())
"""
    _run_runtime_subprocess(
        script,
        timeout_message="coherent deadline recheck did not settle",
    )


@pytest.mark.asyncio
async def test_unconsented_host_evidence_runtime_is_artifact_free_and_injects_null_capture(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    root = tmp_path / "absent-evidence"
    database = root / "capture-v1.sqlite3"
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=7,
        retention_hours=24,
    )

    assert not hasattr(runtime, "lifecycle_owner")
    assert runtime.evidence_lifecycle is None
    assert runtime.evidence_admission is None
    assert runtime.writer_running is False
    assert root.exists() is False

    await runtime.close()
    await runtime.close()

    assert root.exists() is False


@pytest.mark.parametrize("retention_hours", [0, 169])
def test_host_evidence_runtime_rejects_retention_outside_slice_zero_bounds(
    tmp_path: Path,
    retention_hours: int,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    with pytest.raises(ValueError, match="1 through 168"):
        HostEvidenceRuntimeV1(
            database=tmp_path / "capture-v1.sqlite3",
            owner_generation=7,
            retention_hours=retention_hours,
        )


@pytest.mark.asyncio
async def test_host_evidence_runtime_publishes_capture_only_after_durable_create(
    tmp_path: Path,
) -> None:
    from uuid import UUID

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    def uid(value: int) -> str:
        return str(UUID(int=value, version=4))

    installation_id = uid(1)
    producer_instance_id = uid(2)
    consent_epoch_id = uid(3)
    logical_session_id = uid(4)
    binding_id = uid(5)
    disclosure_digest = "a" * 64
    create = m.CreateEpochV1(
        protocol_version=1,
        installation_id=installation_id,
        producer_instance_id=producer_instance_id,
        consent_epoch_id=consent_epoch_id,
        logical_session_id=logical_session_id,
        binding_id=binding_id,
        binding_generation=11,
        consent_version=m.CONSENT_VERSION,
        disclosure_digest=disclosure_digest,
        retention_hours=24,
        microphone_accepted=True,
        typed_accepted=True,
        control_sequence=1,
        control_fingerprint_hash="b" * 64,
        session_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            logical_session_id=logical_session_id,
            event_id=uid(6),
            event_sequence=1,
            event_kind=m.EventKind.SESSION_OPENED,
            payload=m.SessionOpenedPayloadV1(
                consent_epoch_id=consent_epoch_id,
                binding_id=binding_id,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest=disclosure_digest,
                retention_hours=24,
                microphone_accepted=True,
                typed_accepted=True,
                predecessor_session_id=None,
            ),
        ),
        binding_opened=m.EvidenceSnapshotV1(
            schema_version=1,
            installation_id=installation_id,
            producer_instance_id=producer_instance_id,
            logical_session_id=logical_session_id,
            event_id=uid(7),
            event_sequence=2,
            event_kind=m.EventKind.BINDING_OPENED,
            payload=m.BindingOpenedPayloadV1(
                binding_id=binding_id,
                binding_generation=11,
                microphone_available=True,
                typed_available=True,
            ),
        ),
    )
    calls: list[object] = []

    class Transport:
        def create_epoch(self, command: m.CreateEpochV1) -> m.StoreDisposition:
            calls.append(command)
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        create,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert calls == []
    assert runtime.evidence_lifecycle is None
    assert runtime.evidence_admission is None


    result = await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda command: command is create,
        timeout_seconds=1.0,
    )

    assert result is m.ConsentDisposition.CONSENT_ACTIVATED
    assert calls == [create]
    assert runtime.evidence_lifecycle is not None
    assert not hasattr(runtime.conversation_authority, "prepare_rollover")
    assert not hasattr(runtime.conversation_authority, "rollover_binding")
    assert not hasattr(runtime.conversation_authority, "rollover_authority")
    assert runtime.evidence_admission is not None
    assert runtime.evidence_admission.operation_scheduler is runtime.operation_scheduler
    assert runtime.writer_running is True

    await runtime.close()


@pytest.mark.asyncio
async def test_host_runtime_injects_only_the_conversation_admission_view(
    tmp_path: Path,
) -> None:
    """Conversation receives ordinary admission, never lifecycle control methods."""

    from uuid import UUID

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import InputSource
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    appended: list[m.QueuedEvidenceRecordV1] = []

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            return m.StoreDisposition.COMMITTED

        def append_record(self, received: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            appended.append(received)
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _received: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=71,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda received: received is command,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED

    admission = runtime.evidence_admission
    assert admission is not None
    conversation_authority = runtime.evidence_lifecycle
    assert conversation_authority is not None
    for lifecycle_method in (
        "activate_binding",
        "close_binding",
        "rollover_binding",
        "seal_lifecycle",
        "mint_drain_authority",
    ):
        assert not hasattr(conversation_authority, lifecycle_method)
    for lifecycle_method in (
        "mint_drain_authority",
        "begin_revoke",
        "request_seal",
        "try_rollover",
        "begin_expiry",
        "request_drain",
    ):
        assert not hasattr(admission, lifecycle_method)

    final_input = conversation_authority.mint_final_input(
        source=InputSource.TYPED,
        input_incarnation=1,
        media_incarnation=None,
        typed_sequence=1,
    )
    command_authority = conversation_authority.accept_command(final_input)
    snapshot = m.EvidenceSnapshotV1(
        schema_version=1,
        installation_id=command.installation_id,
        producer_instance_id=command.producer_instance_id,
        logical_session_id=command.logical_session_id,
        event_id=str(UUID(int=101, version=4)),
        event_sequence=3,
        event_kind=m.EventKind.COMMAND_ROUTED,
        payload=m.CommandRoutedPayloadV1(
            utterance_id=command_authority.utterance_id,
            source=command_authority.source,
            routing_disposition="command",
        ),
    )
    assert (
        admission.try_admit_command(command_authority, snapshot)
        is m.CommandDisposition.ADMITTED
    )

    await runtime.close()
    assert len(appended) == 1


def test_host_evidence_runtime_is_the_trusted_consent_authority_issuer(tmp_path: Path) -> None:
    import dataclasses

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    command = _create_epoch()
    projection = BrowserEventProjection()
    status_reservation = projection.reserve_capture_status()

    authority = runtime.reserve_consent_authority(
        command,
        status_reservation,
        projection.validate_capture_status_reservation,
    )

    assert type(authority) is m.ConsentCreateAuthorityV1
    assert authority.binding_id == command.binding_id
    assert authority.binding_generation == command.binding_generation
    assert authority.control_sequence == command.control_sequence
    assert authority.projection_reservation is status_reservation
    m.validate_authority_composition(authority, command)
    forged = object.__new__(m.ConsentCreateAuthorityV1)
    for field in dataclasses.fields(m.ConsentCreateAuthorityV1):
        object.__setattr__(forged, field.name, getattr(authority, field.name))
    forged._validate()
    with pytest.raises(RuntimeError, match="exact pending authority"):
        runtime.validate_pending_consent_authority(forged)
    runtime.validate_pending_consent_authority(authority)
    with pytest.raises(RuntimeError, match="pending|reserved"):
        runtime.reserve_consent_authority(
            command,
            projection.reserve_capture_status(),
            projection.validate_capture_status_reservation,
        )


def test_host_evidence_runtime_rejects_forged_projection_before_durable_reservation(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    command = _create_epoch()
    forged = object.__new__(m.ProjectionReservation)

    with pytest.raises(RuntimeError, match="stale or foreign"):
        runtime.reserve_consent_authority(
            command,
            forged,
            projection.validate_capture_status_reservation,
        )

    real = projection.reserve_capture_status()
    authority = runtime.reserve_consent_authority(
        command,
        real,
        projection.validate_capture_status_reservation,
    )
    assert authority.projection_reservation is real


def test_host_evidence_runtime_reports_authoritative_initial_capture_status(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=4,
        retention_hours=24,
    )

    status = runtime.capture_status(disclosure_digest="a" * 64)

    assert status == m.CaptureStatusV1(
        available=True,
        capture_state=m.CaptureState.IDLE,
        retention_hours=24,
        consent_version=m.CONSENT_VERSION,
        disclosure_digest="a" * 64,
    )


@pytest.mark.asyncio
async def test_host_evidence_runtime_mints_and_dispatches_trusted_revoke_authority(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def commit_revoke_request(
            self, _command: m.RevokeRequestV1
        ) -> m.RevokeDisposition:
            return m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        def finalize_revoke(self, _command: m.RevokeFinalizeV1) -> m.RevokeDisposition:
            return m.RevokeDisposition.PURGE_COMPLETED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
            }:
                return lambda *_args: m.StoreDisposition.COMMITTED
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection(capacity=4)
    create_authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert await runtime.activate_consent(
        create_authority,
        transport=Transport(),
        binding_is_current=lambda _command: True,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED
    request = m.EvidenceRevokeRequestV1(sequence=2)

    revoke_authority = runtime.reserve_browser_revoke(
        binding_generation=command.binding_generation,
        request=request,
        projection_reservation=projection.reserve_capture_status(),
        validate_projection_reservation=projection.validate_capture_status_reservation,
    )
    disposition = await runtime.activate_revoke(revoke_authority, timeout_seconds=1.0)

    assert type(revoke_authority) is m.ConsentRevokeAuthorityV1
    assert disposition in (
        m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
        m.RevokeDisposition.PURGE_COMPLETED,
    )
    assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
        m.CaptureState.IDLE
        if disposition is m.RevokeDisposition.PURGE_COMPLETED
        else m.CaptureState.REVOKED_PURGING
    )

    await runtime.close()


@pytest.mark.asyncio
async def test_reserving_revoke_synchronously_closes_final_input_and_turn_admission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A blocked browser submit cannot acquire evidence authority after revoke ownership."""

    from threading import Event

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import (
        AppendDisposition,
        ConversationOperationKind,
        EvidenceAdmissionControllerV1,
        InputSource,
        ReservationError,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def commit_revoke_request(
            self, _command: m.RevokeRequestV1
        ) -> m.RevokeDisposition:
            return m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        def finalize_revoke(self, _command: m.RevokeFinalizeV1) -> m.RevokeDisposition:
            return m.RevokeDisposition.PURGE_COMPLETED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
            }:
                return lambda *_args: m.StoreDisposition.COMMITTED
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=72,
        retention_hours=24,
    )
    projection = BrowserEventProjection(capacity=4)
    entered_begin_revoke = Event()
    release_begin_revoke = Event()
    try:
        consent = runtime.reserve_consent_authority(
            command,
            projection.reserve_capture_status(),
            projection.validate_capture_status_reservation,
        )
        assert await runtime.activate_consent(
            consent,
            transport=Transport(),
            binding_is_current=lambda _command: True,
            timeout_seconds=1.0,
        ) is m.ConsentDisposition.CONSENT_ACTIVATED
        admission = runtime.evidence_admission
        assert admission is not None
        lifecycle = runtime.evidence_lifecycle
        assert lifecycle is not None
        pre_revoke_authority = lifecycle.decline_to_user(
            lifecycle.mint_final_input(
                source=InputSource.TYPED,
                input_incarnation=1,
                media_incarnation=None,
                typed_sequence=1,
            )
        )

        original_begin_revoke = EvidenceAdmissionControllerV1.begin_revoke

        def block_begin_revoke(
            self: EvidenceAdmissionControllerV1,
            authority: m.ConsentRevokeAuthorityV1,
        ) -> object:
            entered_begin_revoke.set()
            assert release_begin_revoke.wait(timeout=2.0)
            return original_begin_revoke(self, authority)

        monkeypatch.setattr(
            EvidenceAdmissionControllerV1,
            "begin_revoke",
            block_begin_revoke,
        )
        reserving = asyncio.create_task(
            asyncio.to_thread(
                runtime.reserve_browser_revoke,
                binding_generation=command.binding_generation,
                request=m.EvidenceRevokeRequestV1(sequence=2),
                projection_reservation=projection.reserve_capture_status(),
                validate_projection_reservation=projection.validate_capture_status_reservation,
            )
        )
        assert await asyncio.to_thread(entered_begin_revoke.wait, 1.0)

        with pytest.raises(ReservationError, match="no active evidence binding"):
            lifecycle.mint_final_input(
                source=InputSource.TYPED,
                input_incarnation=2,
                media_incarnation=None,
                typed_sequence=2,
            )
        reservation = runtime.operation_scheduler.try_reserve(
            ConversationOperationKind.RESPONSE
        )
        assert reservation is not None
        result = admission.try_reserve_user_turn(pre_revoke_authority, reservation)
        if result.lease is not None:
            assert admission.discard_unopened_user_turn(
                result.lease,
                pre_revoke_authority,
            )
        else:
            runtime.operation_scheduler.release(reservation)
        assert result.disposition is not AppendDisposition.ADMITTED
        release_begin_revoke.set()
        await reserving
    finally:
        release_begin_revoke.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_revoke_requires_owning_projection_reservation_before_state_mutation(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection(capacity=4)
    create = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert await runtime.activate_consent(
        create,
        transport=Transport(),
        binding_is_current=lambda _command: True,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED

    forged = object.__new__(m.ProjectionReservation)
    with pytest.raises(RuntimeError, match="stale or foreign"):
        runtime.reserve_browser_revoke(
            binding_generation=command.binding_generation,
            request=m.EvidenceRevokeRequestV1(sequence=2),
            projection_reservation=forged,
            validate_projection_reservation=projection.validate_capture_status_reservation,
        )

    assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is m.CaptureState.ACTIVE
    assert runtime._pending_revoke_authority is None
    reservation = projection.reserve_capture_status()
    authority = runtime.reserve_browser_revoke(
        binding_generation=command.binding_generation,
        request=m.EvidenceRevokeRequestV1(sequence=2),
        projection_reservation=reservation,
        validate_projection_reservation=projection.validate_capture_status_reservation,
    )
    assert authority.projection_reservation is reservation

    await runtime.close()


@pytest.mark.asyncio
async def test_revoke_terminal_observer_retries_timeouts_and_publishes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    runtime._pending_revoke_ticket = object()  # type: ignore[assignment]
    observations: list[float] = []
    published: list[m.RevokeDisposition] = []
    released: list[None] = []
    done = asyncio.Event()

    async def wait_revoke_terminal(*, timeout_seconds: float) -> m.RevokeDisposition:
        observations.append(timeout_seconds)
        if len(observations) == 1:
            return m.RevokeDisposition.CONTROL_TIMED_OUT
        done.set()
        return m.RevokeDisposition.PURGE_COMPLETED

    monkeypatch.setattr(runtime, "wait_revoke_terminal", wait_revoke_terminal)
    runtime.observe_revoke_terminal(
        lambda disposition: (published.append(disposition), True)[1],
        lambda: released.append(None),
    )
    await asyncio.wait_for(done.wait(), timeout=1.0)
    await asyncio.sleep(0)

    assert observations == [2.0, 2.0]
    assert published == [m.RevokeDisposition.PURGE_COMPLETED]
    assert released == []
    await runtime.close()


@pytest.mark.asyncio
async def test_host_evidence_close_rejects_orphan_writer_without_unowned_stop(
    tmp_path: Path,
) -> None:
    from threading import Event

    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence.runtime import (
        EvidenceWriterRuntimeOwnerV1,
        HostEvidenceRuntimeV1,
    )

    entered = Event()
    release = Event()
    queue = a.BoundedEvidenceWriterQueueV1()
    item = a.EvidenceWriterQueueItemV1(
        protocol_version=1,
        lane=a.WriterQueueLane.ORDERED,
        payload=_create_epoch(),
        admission_ordinal=1,
    )

    def dispatch(_item: object) -> None:
        entered.set()
        assert release.wait(timeout=1.0)

    writer = EvidenceWriterRuntimeOwnerV1(source=queue, dispatch=dispatch)
    queue.put_nowait(item)
    assert entered.wait(timeout=1.0)

    closes: list[str] = []

    class Transport:
        def close(self) -> None:
            closes.append("transport")

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    runtime._writer = writer
    runtime._transport = Transport()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="writer.*no admission"):
        await runtime.close()

    assert writer.is_running is True
    assert closes == []
    assert runtime._closed is False
    assert runtime._transport is not None

    release.set()
    assert await asyncio.to_thread(writer.close, 1.0)

    assert writer.is_running is False
    assert closes == []
    assert runtime._closed is False


@pytest.mark.asyncio
async def test_host_consent_checkpoint_is_emitted_after_durable_active_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence import runtime as runtime_module

    command = _create_epoch()
    checkpoints: list[tuple[str, m.CaptureState]] = []
    runtime: runtime_module.HostEvidenceRuntimeV1

    class CheckpointChannel:
        async def emit(self, checkpoint: str) -> None:
            checkpoints.append(
                (checkpoint, runtime.capture_status(disclosure_digest="a" * 64).capture_state)
            )

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def close(self) -> bool:
            return True

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda *_args: m.StoreDisposition.COMMITTED
            raise AttributeError(name)

    monkeypatch.setattr(
        runtime_module,
        "_current_qualification_checkpoint_channel",
        lambda: CheckpointChannel(),
        raising=False,
    )
    runtime = runtime_module.HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )

    assert await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda _command: True,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED
    assert checkpoints == [("host_consent_active", m.CaptureState.ACTIVE)]

    await runtime.close()


@pytest.mark.asyncio
async def test_host_evidence_close_drains_before_stopping_writer_and_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host close owns the terminal drain rather than stopping admission's consumer."""

    from threading import Event

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence import runtime as runtime_module

    command = _create_epoch()
    events: list[str] = []
    drain_started = Event()
    release_drain = Event()

    class CheckpointChannel:
        async def emit(self, checkpoint: str) -> None:
            events.append(f"checkpoint:{checkpoint}")

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            events.append("create")
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, received: m.DrainAndStopV1) -> m.DrainDisposition:
            assert received.owner_generation == 7
            assert received.final_admission_ordinal == 2
            events.append("drain")
            drain_started.set()
            assert release_drain.wait(timeout=1.0)
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            events.append("transport")

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    monkeypatch.setattr(
        runtime_module,
        "_current_qualification_checkpoint_channel",
        lambda: CheckpointChannel(),
    )
    runtime = runtime_module.HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda received: received is command,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED
    assert events == ["create", "checkpoint:host_consent_active"]
    writer = runtime._writer
    assert writer is not None
    writer_close = writer.close

    def close_writer(timeout: float | None = None) -> bool:
        events.append("writer")
        return writer_close(timeout)

    monkeypatch.setattr(writer, "close", close_writer)

    closing = asyncio.create_task(runtime.close())
    assert await asyncio.wait_for(asyncio.to_thread(drain_started.wait), timeout=1.0)
    await asyncio.sleep(0)
    observed_before_release = list(events)

    release_drain.set()
    await closing

    assert observed_before_release == [
        "create",
        "checkpoint:host_consent_active",
        "checkpoint:host_drain_started",
        "drain",
    ]
    assert events == [
        "create",
        "checkpoint:host_consent_active",
        "checkpoint:host_drain_started",
        "drain",
        "writer",
        "transport",
    ]
    assert runtime._admission is None
    assert runtime._queue is None
    assert runtime._writer is None
    assert runtime._pending_create is None
    assert runtime._pending_consent_authority is None
    assert runtime._pending_consent_ticket is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("reported", "raised"))
async def test_host_evidence_close_retains_transport_authority_until_transport_close_succeeds(
    tmp_path: Path,
    failure: str,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    class Transport:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> bool:
            self.close_calls += 1
            if self.close_calls == 1:
                if failure == "raised":
                    raise RuntimeError("transport close raised")
                return False
            return True

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    observations = runtime.production_observations
    transport = Transport()
    runtime._transport = transport  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="transport close"):
        await runtime.close()

    assert transport.close_calls == 1
    assert runtime._transport is transport
    assert runtime._closed is False

    await runtime.close()

    assert transport.close_calls == 2
    assert runtime._transport is None
    assert runtime._closed is True
    assert [(item.stage.value, item.result.value) for item in observations.records()] == [
        ("transport_close", "failed"),
        ("evidence_runtime", "failed"),
        ("transport_close", "succeeded"),
        ("evidence_runtime", "succeeded"),
    ]


@pytest.mark.asyncio
async def test_host_evidence_close_is_shielded_coalesced_and_retry_safe_after_cancellation(
    tmp_path: Path,
) -> None:
    from threading import Event

    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    entered = Event()
    release = Event()

    class Transport:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> bool:
            self.close_calls += 1
            entered.set()
            assert release.wait(timeout=1.0)
            return self.close_calls > 1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    transport = Transport()
    runtime._transport = transport  # type: ignore[assignment]

    caller = asyncio.create_task(runtime.close())
    assert await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.0)
    coalesced = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert transport.close_calls == 1
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    release.set()
    with pytest.raises(RuntimeError, match="transport close"):
        await coalesced
    assert transport.close_calls == 1
    assert runtime._transport is transport

    await runtime.close()
    assert transport.close_calls == 2
    assert runtime._closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("writer_failure", "expected_error"),
    (("timeout", "writer.*stop"), ("failure", "writer join failed")),
)
async def test_host_evidence_close_retains_owned_drain_after_writer_join_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    writer_failure: str,
    expected_error: str,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    events: list[str] = []

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _received: m.DrainAndStopV1) -> m.DrainDisposition:
            events.append("drain")
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            events.append("transport")

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda received: received is command,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED
    admission = runtime._admission
    writer = runtime._writer
    assert admission is not None
    assert writer is not None
    close_writer = writer.close
    writer_calls = 0

    def close(timeout: float | None = None) -> bool:
        nonlocal writer_calls
        writer_calls += 1
        events.append(f"writer:{writer_calls}")
        if writer_calls == 1:
            if writer_failure == "timeout":
                return False
            raise RuntimeError("writer join failed")
        return close_writer(timeout)

    monkeypatch.setattr(writer, "close", close)

    with pytest.raises(RuntimeError, match=expected_error):
        await runtime.close()

    assert events == ["drain", "writer:1"]
    assert runtime._close_drain_authority is not None
    assert runtime._close_drain_ticket is not None
    assert runtime._writer is writer
    assert runtime._transport is not None
    assert runtime._closed is False

    await runtime.close()

    assert events == ["drain", "writer:1", "writer:2", "transport"]
    assert runtime._closed is True


# Bounds a hung real SQLite daemon, not latency.  These tests assert correctness
# through the real daemon on runner disks they do not control: a healthy run
# finishes each step in milliseconds, yet hosted Windows runners have stalled a
# single first file open for more than 3.5 s (#73, #13).  Sixty seconds is an
# order of magnitude past that, so expiry means the daemon stopped making
# progress.  The product's two-second consent control bound stays in force and
# is still passed; its timeout is settled through the runtime's retained
# settlement, and the bound itself is covered deterministically by
# test_consent_control_timeout_names_its_reached_durability_milestone.
_REAL_DAEMON_HANG_BOUND_SECONDS = 60.0


async def _activate_real_consent(runtime, authority, *, binding_is_current):  # type: ignore[no-untyped-def]
    """Activate through the real daemon under the product's two-second bound.

    A control timeout is a lawful product outcome: the runtime keeps owning the
    in-flight create, so the durable result is its retained settlement.
    """

    from hermes_realtime.evidence import models as m

    disposition = await runtime.activate_consent(
        authority,
        transport=runtime.create_sqlite_transport(),
        binding_is_current=binding_is_current,
        timeout_seconds=2.0,
    )
    if disposition is m.ConsentDisposition.CONTROL_TIMED_OUT:
        settlement = runtime.claim_consent_settlement_task(authority)
        disposition = await asyncio.wait_for(
            asyncio.shield(settlement),
            _REAL_DAEMON_HANG_BOUND_SECONDS,
        )
    return disposition


_SQLITE_CONSENT_OPEN_PHASES = (
    "root_validation",
    "root_occupancy",
    "root_marker",
    "sentinel_ensure",
    "sentinel_read",
    "sentinel_decode",
    "maintenance_headroom",
    "sentinel_transition",
    "database_create",
)


def _expected_fresh_sqlite_open_phase_counts() -> dict[str, int]:
    counts = {phase: 1 for phase in _SQLITE_CONSENT_OPEN_PHASES}
    # A newly created sentinel already carries FIRST_CREATE_PENDING.
    counts["sentinel_transition"] = 0
    return counts


def _observe_sqlite_consent(scenario):
    """Observe the real methods without retaining arguments, errors or identities."""
    @wraps(scenario)
    async def observed(*args, **kwargs):
        from hermes_realtime.evidence import models as m
        from hermes_realtime.evidence import sqlite_spool
        from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
        from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

        try:
            start = monotonic_ns()
        except Exception:
            start = None
        lock = Lock()
        offsets: dict[str, int] = {}
        open_phase_counts = {phase: 0 for phase in _SQLITE_CONSENT_OPEN_PHASES}
        open_scope = local()
        activation = None
        store = None
        closed = False

        def mark(stage):
            try:
                if start is not None:
                    offset = (monotonic_ns() - start) // 1_000_000
                    with lock:
                        offsets.setdefault(stage, offset)
            except Exception:
                pass

        def begin_phase(phase):
            try:
                with lock:
                    # Two is a fixed "repeated" bucket, not an unbounded call count.
                    open_phase_counts[phase] = min(open_phase_counts[phase] + 1, 2)
            except Exception:
                pass
            mark(f"{phase}_enter")

        def observe_method_phase(phase, method):
            def phase_method(spool, *values, **options):
                record = getattr(open_scope, "spool", None) is spool
                if record:
                    begin_phase(phase)
                try:
                    return method(spool, *values, **options)
                finally:
                    if record:
                        mark(f"{phase}_exit")

            return phase_method

        def observe_global_phase(phase, function):
            def phase_function(*values, **options):
                record = getattr(open_scope, "spool", None) is not None
                if record:
                    begin_phase(phase)
                try:
                    return function(*values, **options)
                finally:
                    if record:
                        mark(f"{phase}_exit")

            return phase_function

        activate = HostEvidenceRuntimeV1.activate_consent
        close = HostEvidenceRuntimeV1.close
        create = SQLiteEvidenceSpool.create_epoch
        open_store = SQLiteEvidenceSpool._open_owned_store
        method_phases = (
            ("root_validation", "_ensure_root", SQLiteEvidenceSpool._ensure_root),
            ("root_marker", "_ensure_root_marker", SQLiteEvidenceSpool._ensure_root_marker),
            ("sentinel_ensure", "_ensure_sentinel", SQLiteEvidenceSpool._ensure_sentinel),
            ("sentinel_read", "_read_sentinel", SQLiteEvidenceSpool._read_sentinel),
            (
                "sentinel_transition",
                "_transition_sentinel",
                SQLiteEvidenceSpool._transition_sentinel,
            ),
        )
        global_phases = (
            ("root_occupancy", "audit_root_occupancy", sqlite_spool.audit_root_occupancy),
            ("sentinel_decode", "decode_sentinel_image", sqlite_spool.decode_sentinel_image),
            (
                "maintenance_headroom",
                "check_maintenance_headroom",
                sqlite_spool.check_maintenance_headroom,
            ),
            ("database_create", "create_store_database", sqlite_spool.create_store_database),
        )

        async def observe_activation(runtime, *values, **options):
            nonlocal activation
            mark("activation_enter")
            try:
                result = await activate(runtime, *values, **options)
                if type(result) is m.ConsentDisposition:
                    activation = result.value
                return result
            finally:
                mark("activation_exit")

        async def observe_close(runtime, *values, **options):
            nonlocal closed
            mark("close_enter")
            try:
                result = await close(runtime, *values, **options)
                closed = True
                return result
            finally:
                mark("close_exit")

        def observe_create(spool, *values, **options):
            nonlocal store
            mark("create_enter")
            try:
                result = create(spool, *values, **options)
                if type(result) is m.StoreDisposition:
                    store = result.value
                return result
            finally:
                mark("create_exit")

        def observe_open(spool, *values, **options):
            previous = getattr(open_scope, "spool", None)
            open_scope.spool = spool
            mark("open_enter")
            try:
                return open_store(spool, *values, **options)
            finally:
                mark("open_exit")
                if previous is None:
                    del open_scope.spool
                else:
                    open_scope.spool = previous

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(HostEvidenceRuntimeV1, "activate_consent", observe_activation)
            patch.setattr(HostEvidenceRuntimeV1, "close", observe_close)
            patch.setattr(SQLiteEvidenceSpool, "create_epoch", observe_create)
            patch.setattr(SQLiteEvidenceSpool, "_open_owned_store", observe_open)
            for phase, name, method in method_phases:
                patch.setattr(
                    SQLiteEvidenceSpool,
                    name,
                    observe_method_phase(phase, method),
                )
            for phase, name, function in global_phases:
                patch.setattr(
                    sqlite_spool,
                    name,
                    observe_global_phase(phase, function),
                )
            try:
                return await scenario(*args, **kwargs)
            finally:
                try:
                    with lock:
                        observation = {
                            "version": 2,
                            "offset_ms": dict(offsets),
                            "open_phase_counts": dict(open_phase_counts),
                            "activation": activation,
                            "store": store,
                            "closed": closed,
                        }
                    print(
                        "[sqlite-consent] " + json.dumps(observation, sort_keys=True),
                        flush=True,
                    )
                except Exception:
                    pass

    return observed


@pytest.mark.asyncio
@_observe_sqlite_consent
async def test_host_evidence_runtime_activates_consent_through_real_sqlite_daemon(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    database = tmp_path / "evidence" / "capture-v1.sqlite3"
    database.parent.mkdir()
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=17,
        retention_hours=24,
        close_timeout_seconds=_REAL_DAEMON_HANG_BOUND_SECONDS,
    )
    try:
        request = m.parse_evidence_consent_request(
            b'{"accepted":true,"consentVersion":"realtime-evidence-consent-v1",'
            b'"disclosureDigest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            b'"retentionHours":24,"sequence":1,'
            b'"sources":{"microphone":true,"typed":true}}'
        )
        projection = BrowserEventProjection()
        authority = runtime.reserve_browser_consent(
            binding_generation=3,
            request=request,
            projection_reservation=projection.reserve_capture_status(),
            validate_projection_reservation=projection.validate_capture_status_reservation,
            microphone_available=True,
            typed_available=True,
        )

        disposition = await _activate_real_consent(
            runtime,
            authority,
            binding_is_current=lambda command: command.binding_generation == 3,
        )

        assert disposition is m.ConsentDisposition.CONSENT_ACTIVATED
        assert database.is_file()
        assert database.stat().st_size > 0
        assert runtime.writer_running is True

        revoke_projection = BrowserEventProjection()
        revoke = runtime.reserve_browser_revoke(
            binding_generation=3,
            request=m.EvidenceRevokeRequestV1(sequence=2),
            projection_reservation=revoke_projection.reserve_capture_status(),
            validate_projection_reservation=revoke_projection.validate_capture_status_reservation,
        )
        revoke_disposition = await runtime.activate_revoke(
            revoke,
            timeout_seconds=_REAL_DAEMON_HANG_BOUND_SECONDS,
        )
        assert revoke_disposition in {
            m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED,
            m.RevokeDisposition.PURGE_COMPLETED,
        }
        terminal: list[m.RevokeDisposition] = []
        terminal_published = asyncio.Event()

        def publish_terminal(disposition: m.RevokeDisposition) -> bool:
            terminal.append(disposition)
            terminal_published.set()
            return True

        runtime.observe_revoke_terminal(
            publish_terminal,
            lambda: None,
        )
        ticket = runtime._pending_revoke_ticket
        assert ticket is not None
        await asyncio.wait_for(
            terminal_published.wait(),
            timeout=_REAL_DAEMON_HANG_BOUND_SECONDS,
        )
        assert terminal == [m.RevokeDisposition.PURGE_COMPLETED], (
            ticket.disposition if ticket is not None else None,
            ticket.durability_event.is_set() if ticket is not None else None,
            ticket.terminal_event.is_set() if ticket is not None else None,
            runtime._revoke_observer_failures,
            runtime._revoke_observers,
            getattr(runtime._writer, "_failure", None),
        )
        assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
            m.CaptureState.IDLE
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    ("binding_replaced", "media_incarnation_replaced", "projection_resync", "client_closed"),
)
async def test_binding_invalidation_retires_the_epoch_before_replacement_media_is_accepted(
    tmp_path: Path,
    reason: str,
) -> None:
    """A stale binding cannot retain capture authority through a usable conversation."""

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda *_args: m.StoreDisposition.COMMITTED
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    assert await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda _command: True,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED

    await runtime.invalidate_active_binding(m.BindingCloseReason(reason))

    assert runtime.evidence_lifecycle is None
    assert runtime.evidence_admission is None
    assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is m.CaptureState.IDLE
    assert not hasattr(runtime, "lifecycle_owner")

    fresh_projection = BrowserEventProjection()
    fresh = runtime.reserve_browser_consent(
        binding_generation=12,
        request=m.parse_evidence_consent_request(
            b'{"accepted":true,"consentVersion":"realtime-evidence-consent-v1",'
            b'"disclosureDigest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            b'"retentionHours":24,"sequence":1,'
            b'"sources":{"microphone":true,"typed":true}}'
        ),
        projection_reservation=fresh_projection.reserve_capture_status(),
        validate_projection_reservation=fresh_projection.validate_capture_status_reservation,
        microphone_available=True,
        typed_available=True,
    )
    assert await runtime.activate_consent(
        fresh,
        transport=Transport(),
        binding_is_current=lambda _command: True,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED
    assert runtime.evidence_admission is not None

    await runtime.invalidate_active_binding(m.BindingCloseReason(reason))
    assert runtime.evidence_admission is None
    assert runtime.writer_running is False
    await runtime.close()


@pytest.mark.asyncio
async def test_host_retention_owner_cancels_then_expires_real_sqlite_session(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime, timedelta

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    database = tmp_path / "evidence" / "capture-v1.sqlite3"
    database.parent.mkdir()
    runtime = HostEvidenceRuntimeV1(
        database=database,
        owner_generation=19,
        retention_hours=24,
        close_timeout_seconds=_REAL_DAEMON_HANG_BOUND_SECONDS,
    )
    try:
        cancelled: list[str] = []
        wall = [datetime(2026, 8, 12, tzinfo=UTC)]

        async def cancel() -> None:
            cancelled.append("retention_expired")

        async def advance(seconds: float) -> None:
            wall[0] += timedelta(seconds=seconds)

        runtime.configure_retention_owner(
            cancel=cancel,
            wall_clock=lambda: wall[0],
            sleep=advance,
        )
        request = m.parse_evidence_consent_request(
            b'{"accepted":true,"consentVersion":"realtime-evidence-consent-v1",'
            b'"disclosureDigest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
            b'"retentionHours":24,"sequence":1,'
            b'"sources":{"microphone":true,"typed":true}}'
        )
        projection = BrowserEventProjection()
        authority = runtime.reserve_browser_consent(
            binding_generation=3,
            request=request,
            projection_reservation=projection.reserve_capture_status(),
            validate_projection_reservation=projection.validate_capture_status_reservation,
            microphone_available=True,
            typed_available=True,
        )
        assert await _activate_real_consent(
            runtime,
            authority,
            binding_is_current=lambda _command: True,
        ) is m.ConsentDisposition.CONSENT_ACTIVATED

        assert await runtime.wait_retention_terminal(
            timeout_seconds=_REAL_DAEMON_HANG_BOUND_SECONDS,
        ) is m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
        assert cancelled == ["retention_expired"]
        assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
            m.CaptureState.IDLE
        )
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        test_host_evidence_runtime_activates_consent_through_real_sqlite_daemon,
        test_host_retention_owner_cancels_then_expires_real_sqlite_session,
    ],
    ids=["consent", "retention"],
)
@pytest.mark.parametrize("failure", ["failed_disposition", "exception", "pending_timeout"])
async def test_consent_scenarios_close_their_real_writer_after_activation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, scenario, failure: str,
) -> None:
    import threading

    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    class InjectedActivationFailure(RuntimeError):
        pass

    original = HostEvidenceRuntimeV1.activate_consent
    runtimes: list[HostEvidenceRuntimeV1] = []
    owned_threads: set[threading.Thread] = set()
    injected = False

    async def activate(runtime, *args, **kwargs):
        nonlocal injected
        runtimes.append(runtime)
        release = threading.Event()
        entered = threading.Event()
        if failure == "pending_timeout":
            transport = kwargs["transport"]
            create_epoch = transport.create_epoch

            def hold_create_epoch(command):
                entered.set()
                assert release.wait(timeout=5.0)
                return create_epoch(command)

            monkeypatch.setattr(transport, "create_epoch", hold_create_epoch)
        try:
            result = await original(runtime, *args, **kwargs)
        finally:
            release.set()
        if failure == "pending_timeout":
            assert result is m.ConsentDisposition.CONTROL_TIMED_OUT
            assert entered.is_set()
        else:
            if result is m.ConsentDisposition.CONTROL_TIMED_OUT:
                # A slow runner disk may reach the real bound; the runtime's
                # retained settlement still owns the in-flight create.
                result = await asyncio.wait_for(
                    asyncio.shield(runtime.claim_consent_settlement_task(args[0])),
                    _REAL_DAEMON_HANG_BOUND_SECONDS,
                )
            assert result is m.ConsentDisposition.CONSENT_ACTIVATED
        assert runtime._writer is not None and runtime._transport is not None
        owned_threads.update((runtime._writer._thread, runtime._transport._thread))
        assert runtime.writer_running and len(owned_threads) == 2
        assert kwargs["timeout_seconds"] == 2.0
        injected = True
        if failure == "exception":
            raise InjectedActivationFailure("synthetic activation failure after writer startup")
        if failure == "failed_disposition":
            return m.ConsentDisposition.CREATE_FAILED
        return result

    monkeypatch.setattr(HostEvidenceRuntimeV1, "activate_consent", activate)
    try:
        if failure == "pending_timeout":
            # A real control timeout is not a failure: the scenario claims the
            # runtime's retained settlement, completes, and still closes.
            await scenario(tmp_path)
        elif failure == "exception":
            with pytest.raises(InjectedActivationFailure):
                await scenario(tmp_path)
        else:
            with pytest.raises(AssertionError):
                await scenario(tmp_path)
        assert injected and len(runtimes) == 1
        assert runtimes[0]._closed
        assert not runtimes[0].writer_running
        assert all(not thread.is_alive() for thread in owned_threads)
        if scenario is test_host_evidence_runtime_activates_consent_through_real_sqlite_daemon:
            output = capsys.readouterr().out
            lines = [line for line in output.splitlines() if line.startswith("[sqlite-consent] ")]
            assert len(lines) == 1
            observation = json.loads(lines[0].removeprefix("[sqlite-consent] "))
            assert observation["version"] == 2
            assert observation["closed"] is True
            offsets = observation["offset_ms"]
            assert all(type(value) is int and value >= 0 for value in offsets.values())
            assert offsets["activation_enter"] <= offsets["create_enter"]
            assert offsets["create_enter"] <= offsets["open_enter"] <= offsets["open_exit"]
            assert offsets["open_exit"] <= offsets["create_exit"] <= offsets["close_exit"]
            assert offsets["activation_exit"] <= offsets["close_enter"] <= offsets["close_exit"]
            assert observation["open_phase_counts"] == (
                _expected_fresh_sqlite_open_phase_counts()
            )
            assert observation["store"] == "committed"
            assert observation["activation"] == {
                "exception": None,
                "failed_disposition": "create_failed",
                "pending_timeout": "control_timed_out",
            }[failure]
            assert "synthetic activation failure" not in lines[0]
            assert str(tmp_path) not in lines[0]
            # The held create always reaches the real control bound without a
            # durable milestone.  Other cases reach it only on a slow disk, and
            # then report the same content-free observation.
            timeout_lines = [
                line
                for line in output.splitlines()
                if line.startswith("[consent-activation] ")
            ]
            assert len(timeout_lines) == 1 if failure == "pending_timeout" else (
                len(timeout_lines) <= 1
            )
            for timeout_line in timeout_lines:
                timed_out = json.loads(timeout_line.removeprefix("[consent-activation] "))
                assert timed_out["version"] == 1
                assert timed_out["operation"] == "consent"
                assert timed_out["bound_seconds"] == 2.0
                assert set(timed_out["offset_ms"]) == {"create_durable"}
                assert str(tmp_path) not in timeout_line
            if failure == "pending_timeout":
                timed_out = json.loads(timeout_lines[0].removeprefix("[consent-activation] "))
                assert timed_out["offset_ms"] == {"create_durable": None}
    finally:
        # A RED regression must not itself contaminate the remaining test process.
        for runtime in runtimes:
            await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("milestone", ["unsignalled", "late", "early"])
async def test_consent_control_timeout_names_its_reached_durability_milestone(
    tmp_path: Path, capsys, milestone: str,
) -> None:
    """One control timeout disposition hides materially different failures.

    Admission already records every production milestone's monotonic time, so a
    timeout can say whether the writer never reached the milestone or reached it
    late, and by how much.  A milestone recorded before the bound is rescued by
    the wait itself and must never reach the timeout path at all.
    """

    import threading
    from time import monotonic, sleep

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence import runtime as runtime_module
    from hermes_realtime.evidence.admission import _set_ticket_signal
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    # The rescued case must clear a real dispatch, so it keeps the production bound.
    bound = 2.0 if milestone == "early" else 0.05
    release = threading.Event()

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            if milestone != "early":
                assert release.wait(timeout=5.0)
            return m.StoreDisposition.COMMITTED

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda *_args: m.StoreDisposition.COMMITTED
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        _create_epoch(),
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )

    original_status = runtime_module._ticket_signal_status
    signalled_after_deadline = False

    def coordinated_status(candidate, *, deadline):
        # Publish the milestone exactly once, strictly after its own deadline,
        # so the late case is a measured margin instead of a race.
        nonlocal signalled_after_deadline
        ticket = runtime._pending_consent_ticket
        if (
            milestone == "late"
            and not signalled_after_deadline
            and ticket is not None
            and candidate is ticket.durability_event
            and deadline is not None
            and monotonic() >= deadline
        ):
            signalled_after_deadline = True
            sleep(0.02)
            _set_ticket_signal(candidate)
        return original_status(candidate, deadline=deadline)

    runtime_module._ticket_signal_status = coordinated_status
    try:
        disposition = await runtime.activate_consent(
            authority,
            transport=Transport(),
            binding_is_current=lambda _command: True,
            timeout_seconds=bound,
        )
    finally:
        runtime_module._ticket_signal_status = original_status
        release.set()
        await runtime.close()

    observations = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[consent-activation] ")
    ]
    if milestone == "early":
        assert disposition is m.ConsentDisposition.CONSENT_ACTIVATED
        assert observations == []
        return
    assert milestone != "late" or signalled_after_deadline
    assert disposition is m.ConsentDisposition.CONTROL_TIMED_OUT
    assert len(observations) == 1
    observation = json.loads(observations[0].removeprefix("[consent-activation] "))
    assert set(observation) == {"bound_seconds", "offset_ms", "operation", "version"}
    assert observation["version"] == 1
    assert observation["operation"] == "consent"
    assert observation["bound_seconds"] == bound
    offsets = observation["offset_ms"]
    assert set(offsets) == {"create_durable"}
    if milestone == "unsignalled":
        assert offsets["create_durable"] is None
    else:
        assert type(offsets["create_durable"]) is float
        assert offsets["create_durable"] > 5.0
    assert str(tmp_path) not in observations[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "store", "close"])
async def test_sqlite_consent_observation_preserves_outcomes_without_private_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, failure,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

    marker = "synthetic-private-diagnostic-payload"
    if failure == "store":
        original_create = SQLiteEvidenceSpool.create_epoch

        def fail_after_create(self, command):
            original_create(self, command)
            raise RuntimeError(marker)

        monkeypatch.setattr(SQLiteEvidenceSpool, "create_epoch", fail_after_create)
    if failure == "close":
        original_close = HostEvidenceRuntimeV1.close

        async def fail_after_close(self):
            await original_close(self)
            raise RuntimeError(marker)

        monkeypatch.setattr(HostEvidenceRuntimeV1, "close", fail_after_close)
    if failure:
        with pytest.raises(AssertionError if failure == "store" else RuntimeError):
            await test_host_evidence_runtime_activates_consent_through_real_sqlite_daemon(tmp_path)
    else:
        await test_host_evidence_runtime_activates_consent_through_real_sqlite_daemon(tmp_path)
    output = capsys.readouterr().out
    assert output.startswith("[sqlite-consent] ") and len(output.splitlines()) == 1
    assert marker not in output and str(tmp_path) not in output
    observation = json.loads(output.removeprefix("[sqlite-consent] "))
    assert set(observation) == {
        "version", "offset_ms", "open_phase_counts", "activation", "store", "closed",
    }
    assert observation["version"] == 2
    assert observation["closed"] is (failure != "close")
    assert observation["activation"] == (
        "create_failed" if failure == "store" else "consent_activated"
    )
    assert observation["store"] == (None if failure == "store" else "committed")
    expected_offsets = {
        "activation_enter", "activation_exit", "create_enter", "create_exit",
        "open_enter", "open_exit", "close_enter", "close_exit",
    }
    phase_counts = _expected_fresh_sqlite_open_phase_counts()
    expected_offsets.update(
        f"{phase}_{edge}"
        for phase in _SQLITE_CONSENT_OPEN_PHASES
        if phase_counts[phase]
        for edge in ("enter", "exit")
    )
    assert set(observation["offset_ms"]) == expected_offsets
    assert observation["open_phase_counts"] == phase_counts
    assert all(type(value) is int and value >= 0 for value in observation["offset_ms"].values())
    phase_edges = ["open_enter"]
    phase_edges.extend(
        f"{phase}_{edge}"
        for phase in _SQLITE_CONSENT_OPEN_PHASES
        if phase_counts[phase]
        for edge in ("enter", "exit")
    )
    phase_edges.append("open_exit")
    assert [observation["offset_ms"][edge] for edge in phase_edges] == sorted(
        observation["offset_ms"][edge] for edge in phase_edges
    )


@pytest.mark.asyncio
async def test_sqlite_consent_open_phases_exclude_foreign_thread_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from threading import Thread

    from hermes_realtime.evidence import sqlite_spool
    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

    calls: list[str] = []

    def create_store_database(_path):
        calls.append("database")

    def open_store(_spool, *, for_create):
        assert for_create is True
        foreign = Thread(
            target=lambda: sqlite_spool.create_store_database(Path("private-foreign"))
        )
        foreign.start()
        foreign.join()
        sqlite_spool.create_store_database(Path("private-owner"))

    monkeypatch.setattr(sqlite_spool, "create_store_database", create_store_database)
    monkeypatch.setattr(SQLiteEvidenceSpool, "_open_owned_store", open_store)

    async def scenario():
        SQLiteEvidenceSpool._open_owned_store(object(), for_create=True)

    await _observe_sqlite_consent(scenario)()

    output = capsys.readouterr().out
    assert "private-owner" not in output and "private-foreign" not in output
    observation = json.loads(output.removeprefix("[sqlite-consent] "))
    assert calls == ["database", "database"]
    assert observation["open_phase_counts"] == {
        phase: int(phase == "database_create") for phase in _SQLITE_CONSENT_OPEN_PHASES
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario_raises", [False, True])
async def test_sqlite_consent_observer_bookkeeping_cannot_replace_scenario_outcome(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
    scenario_raises: bool,
) -> None:
    marker = object()
    error = RuntimeError("private original scenario error")

    def fail_clock():
        raise RuntimeError("private observer bookkeeping error")

    async def scenario():
        if scenario_raises:
            raise error
        return marker

    monkeypatch.setattr(sys.modules[__name__], "monotonic_ns", fail_clock)
    if scenario_raises:
        with pytest.raises(RuntimeError) as raised:
            await _observe_sqlite_consent(scenario)()
        assert raised.value is error
    else:
        assert await _observe_sqlite_consent(scenario)() is marker

    output = capsys.readouterr().out
    assert "private original" not in output and "private observer" not in output
    observation = json.loads(output.removeprefix("[sqlite-consent] "))
    assert observation["version"] == 2
    assert observation["offset_ms"] == {}
    assert observation["open_phase_counts"] == {
        phase: 0 for phase in _SQLITE_CONSENT_OPEN_PHASES
    }


@pytest.mark.asyncio
async def test_sqlite_consent_diagnostic_write_failure_preserves_scenario_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    marker = RuntimeError("private original scenario error")

    async def scenario():
        raise marker

    def fail_write(*_values, **_options):
        raise OSError("private diagnostic write error")

    monkeypatch.setattr(builtins, "print", fail_write)
    with pytest.raises(RuntimeError) as raised:
        await _observe_sqlite_consent(scenario)()
    assert raised.value is marker


@pytest.mark.asyncio
async def test_sqlite_consent_observer_locates_an_injected_root_validation_delay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    from threading import Event

    from hermes_realtime.evidence.sqlite_spool import SQLiteEvidenceSpool

    entered = Event()
    release = Event()
    ensure_root = SQLiteEvidenceSpool._ensure_root

    def delayed_ensure_root(spool):
        entered.set()
        assert release.wait(timeout=5.0)
        return ensure_root(spool)

    async def release_after_control_timeout() -> None:
        deadline = asyncio.get_running_loop().time() + 1.0
        while not entered.is_set():
            assert asyncio.get_running_loop().time() < deadline
            await asyncio.sleep(0.001)
        await asyncio.sleep(2.1)
        release.set()

    monkeypatch.setattr(SQLiteEvidenceSpool, "_ensure_root", delayed_ensure_root)
    releaser = asyncio.create_task(release_after_control_timeout())
    try:
        # A store step that outlives the two-second control bound times out the
        # control and is still settled to a correct activation, revoke and close.
        await test_host_evidence_runtime_activates_consent_through_real_sqlite_daemon(tmp_path)
    finally:
        release.set()
        await releaser

    output = capsys.readouterr().out
    lines = [line for line in output.splitlines() if line.startswith("[sqlite-consent] ")]
    assert len(lines) == 1
    observation = json.loads(lines[0].removeprefix("[sqlite-consent] "))
    assert observation["activation"] == "control_timed_out"
    assert observation["store"] == "committed"
    assert observation["closed"] is True
    assert observation["open_phase_counts"] == _expected_fresh_sqlite_open_phase_counts()
    offsets = observation["offset_ms"]
    assert offsets["root_validation_exit"] - offsets["root_validation_enter"] >= 2_000
    assert offsets["activation_exit"] <= offsets["root_validation_exit"]
    assert offsets["root_validation_exit"] <= offsets["root_occupancy_enter"]
    # A store that never finished opening cannot have reached the durable
    # milestone, and the timeout must say so rather than only that it expired.
    timeout_lines = [
        line for line in output.splitlines() if line.startswith("[consent-activation] ")
    ]
    assert len(timeout_lines) == 1
    timed_out = json.loads(timeout_lines[0].removeprefix("[consent-activation] "))
    assert timed_out["operation"] == "consent"
    assert timed_out["bound_seconds"] == 2.0
    assert timed_out["offset_ms"] == {"create_durable": None}


@pytest.mark.asyncio
async def test_host_evidence_runtime_exact_retry_observes_timed_out_consent_ticket(
    tmp_path: Path,
) -> None:
    from threading import Event

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    entered = Event()
    release = Event()
    calls: list[object] = []

    class Transport:
        def create_epoch(self, value: m.CreateEpochV1) -> m.StoreDisposition:
            calls.append(value)
            entered.set()
            assert release.wait(timeout=1)
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )

    first = await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda value: value is command,
        timeout_seconds=0.01,
    )
    assert first is m.ConsentDisposition.CONTROL_TIMED_OUT
    assert entered.is_set()
    assert runtime.evidence_admission is None
    assert runtime.evidence_lifecycle is None

    settlement = runtime.claim_consent_settlement_task(authority)
    release.set()
    assert await settlement is m.ConsentDisposition.CONSENT_ACTIVATED
    assert calls == [command]
    assert runtime.evidence_admission is not None
    await runtime.close()


@pytest.mark.asyncio
async def test_host_evidence_runtime_durable_create_failure_never_publishes_capture(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()

    class Transport:
        def create_epoch(self, value: m.CreateEpochV1) -> m.StoreDisposition:
            assert value is command
            return m.StoreDisposition.FAULTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    result = await runtime.activate_consent(
        authority,
        transport=Transport(),
        binding_is_current=lambda value: value is command,
        timeout_seconds=1.0,
    )

    assert result is m.ConsentDisposition.CREATE_FAILED
    assert runtime.evidence_admission is None
    assert runtime.evidence_lifecycle is None
    assert runtime.writer_running is False
    await runtime.close()
    assert runtime.writer_running is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second_callback", "process_error"),
    (
        ("non_bool", None),
        ("raises", None),
        ("keyboard_interrupt", KeyboardInterrupt),
        ("system_exit", SystemExit),
    ),
)
async def test_public_runtime_rolls_back_second_binding_callback_fault_after_exact_accounting(
    tmp_path: Path,
    second_callback: str,
    process_error: type[BaseException] | None,
) -> None:
    """Both binding checks must fail closed before public browser capacity can leak."""

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError((name, payload)))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=71,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    primary = projection.reserve_capture_status()
    authority = runtime.reserve_consent_authority(
        command,
        primary,
        projection.validate_capture_status_reservation,
    )
    admission = runtime._admission
    assert admission is not None
    calls: list[str] = []

    def binding_is_current(received: m.CreateEpochV1) -> bool:
        assert received is command
        calls.append("binding")
        if len(calls) == 1:
            return True
        if second_callback == "non_bool":
            return 1  # type: ignore[return-value]
        if second_callback == "raises":
            raise RuntimeError("synthetic second binding callback failure")
        if second_callback == "keyboard_interrupt":
            raise KeyboardInterrupt
        raise SystemExit

    try:
        if process_error is None:
            assert await runtime.activate_consent(
                authority,
                transport=Transport(),
                binding_is_current=binding_is_current,
                timeout_seconds=1.0,
            ) is m.ConsentDisposition.CREATE_FAILED
        else:
            with pytest.raises(process_error):
                await runtime.activate_consent(
                    authority,
                    transport=Transport(),
                    binding_is_current=binding_is_current,
                    timeout_seconds=1.0,
                )

        assert calls == ["binding", "binding"]
        assert runtime.evidence_admission is None
        assert runtime.evidence_lifecycle is None
        assert runtime.writer_running is False
        assert runtime.operation_scheduler.active_count == 0
        assert runtime._admission is None
        assert runtime._queue is None
        assert runtime._writer is None
        assert runtime._transport is None
        assert runtime._pending_create is None
        assert runtime._pending_consent_authority is None
        assert runtime._pending_consent_ticket is None
        assert runtime._lifecycle_status_reservations == []
        diagnostics = admission.diagnostics()
        assert diagnostics.queue_record_count == 0
        assert diagnostics.queue_canonical_bytes == 0
    finally:
        await runtime.close()

    projection.release_capture_status_reservations((primary,))
    assert not projection._capture_status_reservations


@pytest.mark.asyncio
async def test_host_evidence_runtime_rejected_consent_never_starts_a_writer_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admission rejection happens before the runtime owns a named consumer thread."""

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import admission as a
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    command = _create_epoch()
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    admission = runtime._admission
    assert admission is not None

    class Transport:
        def create_epoch(self, _command: object) -> object:
            raise AssertionError("admission rejection must precede dispatch")

        def drain_and_close(self, _command: object) -> object:
            raise AssertionError("admission rejection must precede dispatch")

        def __getattr__(self, _name: str) -> object:
            return lambda _payload: (_ for _ in ()).throw(
                AssertionError("admission rejection must precede dispatch")
            )

    def reject(_authority: object) -> object:
        raise a.ReservationError("rejected before dispatch")

    monkeypatch.setattr(admission, "begin_consent", reject)

    with pytest.raises(a.ReservationError, match="rejected before dispatch"):
        await runtime.activate_consent(
            authority,
            transport=Transport(),  # type: ignore[arg-type]
            binding_is_current=lambda _command: True,
            timeout_seconds=1.0,
        )

    assert runtime._writer is None
    assert runtime.writer_running is False


@pytest.mark.asyncio
async def test_direct_consent_activation_rejects_malformed_transport_before_ticket_or_enqueue(
    tmp_path: Path,
) -> None:
    """Direct runtime callers keep the exact reserved authority retryable."""

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()

    class MalformedTransport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            raise AssertionError("malformed transport must not dispatch")

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda _payload: (_ for _ in ()).throw(AssertionError(name))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    queue = runtime._queue
    admission = runtime._admission
    assert queue is not None
    assert admission is not None
    before_activation = admission.diagnostics()

    with pytest.raises(TypeError, match="transport must provide append_record\\(\\)"):
        await runtime.activate_consent(
            authority,
            transport=MalformedTransport(),  # type: ignore[arg-type]
            binding_is_current=lambda _command: True,
            timeout_seconds=1.0,
        )

    assert runtime._pending_consent_ticket is None
    assert runtime._writer is None
    assert runtime._transport is None
    assert queue.ordered_count == 0
    assert admission.diagnostics() == before_activation

    try:
        assert await runtime.activate_consent(
            authority,
            transport=Transport(),
            binding_is_current=lambda received: received is command,
            timeout_seconds=1.0,
        ) is m.ConsentDisposition.CONSENT_ACTIVATED
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_close_wins_over_pending_consent_before_lifecycle_publication(
    tmp_path: Path,
) -> None:
    """A close begun during create durability leaves no published lifecycle or writer."""

    from threading import Event

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    entered = Event()
    release = Event()
    events: list[str] = []

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            events.append("create")
            entered.set()
            assert release.wait(timeout=1.0)
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            events.append("drain")
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            events.append("transport")

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    admission = runtime._admission
    assert admission is not None
    activation = asyncio.create_task(
        runtime.activate_consent(
            authority,
            transport=Transport(),
            binding_is_current=lambda received: received is command,
            timeout_seconds=1.0,
        )
    )
    assert await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.0)

    closing = asyncio.create_task(runtime.close())
    await asyncio.sleep(0)
    assert runtime._close_operation is not None

    release.set()
    assert await activation is m.ConsentDisposition.CREATE_FAILED
    await closing

    assert events == ["create", "drain", "transport"]
    assert runtime.evidence_lifecycle is None
    assert runtime.evidence_admission is None
    assert runtime.writer_running is False
    assert runtime._retention_task is None
    assert runtime._admission is None
    assert runtime._pending_create is None
    assert runtime._pending_consent_authority is None
    assert runtime._pending_consent_ticket is None
    assert runtime.operation_scheduler.active_count == 0
    diagnostics = admission.diagnostics()
    assert diagnostics.queue_record_count == 0
    assert diagnostics.queue_canonical_bytes == 0

    fresh = HostEvidenceRuntimeV1(
        database=tmp_path / "fresh-capture-v1.sqlite3",
        owner_generation=8,
        retention_hours=24,
    )
    fresh_projection = BrowserEventProjection()
    fresh_authority = fresh.reserve_consent_authority(
        command,
        fresh_projection.reserve_capture_status(),
        fresh_projection.validate_capture_status_reservation,
    )
    try:
        assert await fresh.activate_consent(
            fresh_authority,
            transport=Transport(),
            binding_is_current=lambda received: received is command,
            timeout_seconds=1.0,
        ) is m.ConsentDisposition.CONSENT_ACTIVATED
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_binding_invalidation_retries_drain_before_stopping_writer_or_transport(
    tmp_path: Path,
) -> None:
    """An invalidated epoch retains its owner until its drain truthfully stops."""

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    events: list[str] = []

    class Transport:
        def __init__(self) -> None:
            self.drain_calls = 0

        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            self.drain_calls += 1
            events.append(f"drain:{self.drain_calls}")
            if self.drain_calls == 1:
                return m.DrainDisposition.TIMED_OUT
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            events.append("transport")

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    transport = Transport()
    assert await runtime.activate_consent(
        authority,
        transport=transport,
        binding_is_current=lambda _command: True,
        timeout_seconds=1.0,
    ) is m.ConsentDisposition.CONSENT_ACTIVATED

    with pytest.raises(RuntimeError, match="drain did not reach terminal stopped"):
        await runtime.invalidate_active_binding(m.BindingCloseReason.BINDING_REPLACED)

    assert events == ["drain:1"]
    assert runtime._transport is transport
    assert runtime._admission is not None
    assert runtime.writer_running is True

    # Close owns bounded recovery of the exact failed epoch. It must not preserve
    # the previous failure after this retry reaches a definitive stop.
    await runtime.close()

    assert events == ["drain:1", "drain:2", "transport"]
    assert runtime._transport is None
    assert runtime._admission is None
    assert runtime.writer_running is False
    assert runtime._invalidation_authority is None
    assert runtime._invalidation_operation is None


@pytest.mark.asyncio
async def test_retention_owner_retries_live_lease_drain_before_durable_expiry(
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.admission import ExpiryTicketV1
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=29,
        retention_hours=24,
    )
    command = _create_epoch()
    active_leases = [1]
    cancellations: list[str] = []
    sleeps: list[float] = []
    began: list[object] = []

    import threading

    class AdmissionProbe:
        final_admission_ordinal = 1

        def __init__(self) -> None:
            self.expiry_terminal_event = threading.Event()
            self.expiry_terminal_event.set()

        def close_for_retention_expiry(self) -> None:
            return None

        def diagnostics(self) -> object:
            return SimpleNamespace(
                active_lease_count=active_leases[0],
                owner_state=m.OwnerState.STOPPED,
            )

        def begin_expiry(self, authority: object) -> ExpiryTicketV1:
            began.append(authority)
            return ExpiryTicketV1(
                m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED,
                self.expiry_terminal_event,
            )

    class TransportProbe:
        def active_session_expiry(self, logical_session_id: str) -> str:
            assert logical_session_id == command.logical_session_id
            return "2026-08-12T00:00:00.000000Z"

    async def cancel() -> None:
        cancellations.append("retention_expired")
        if len(cancellations) == 2:
            active_leases[0] = 0

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    runtime._admission = AdmissionProbe()  # type: ignore[assignment]
    runtime._pending_create = command  # type: ignore[assignment]
    runtime._transport = TransportProbe()  # type: ignore[assignment]
    runtime._retention_cancel = cancel
    runtime._retention_wall_clock = lambda: datetime(2026, 8, 12, tzinfo=UTC)
    runtime._retention_sleep = sleep

    assert await runtime._run_retention_owner() is (
        m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
    )
    assert cancellations == ["retention_expired", "retention_expired"]
    assert sleeps == [0.0, 0.01]
    assert len(began) == 1

    await runtime.close()


class _InstantSleepAsyncio:
    """The runtime module's ``asyncio`` whose sleeps only yield to the loop.

    It lets a fixed real-time polling window elapse in a few loop turns, so a
    test can hold the store past that window without waiting for it.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(_delay: float, result: object = None) -> object:
        return await asyncio.sleep(0, result)


def _retention_probe_runtime(tmp_path: Path, owner_state: object):  # type: ignore[no-untyped-def]
    """A runtime whose retention owner reaches expiry against a controllable store."""

    import threading
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.admission import ExpiryTicketV1
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=31,
        retention_hours=24,
    )
    command = _create_epoch()

    class AdmissionProbe:
        final_admission_ordinal = 1

        def __init__(self) -> None:
            self.expiry_terminal_event = threading.Event()
            self.owner_state = owner_state
            self.began = threading.Event()

        def close_for_retention_expiry(self) -> None:
            return None

        def diagnostics(self) -> object:
            return SimpleNamespace(active_lease_count=0, owner_state=self.owner_state)

        def begin_expiry(self, _authority: object) -> ExpiryTicketV1:
            self.began.set()
            return ExpiryTicketV1(
                m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED,
                self.expiry_terminal_event,
            )

    class TransportProbe:
        def active_session_expiry(self, _logical_session_id: str) -> str:
            return "2026-08-12T00:00:00.000000Z"

    async def no_wait(_seconds: float = 0.0) -> None:
        return None

    admission = AdmissionProbe()
    runtime._admission = admission  # type: ignore[assignment]
    runtime._pending_create = command  # type: ignore[assignment]
    runtime._transport = TransportProbe()  # type: ignore[assignment]
    runtime._retention_cancel = no_wait
    runtime._retention_wall_clock = lambda: datetime(2026, 8, 12, tzinfo=UTC)
    runtime._retention_sleep = no_wait
    runtime._capture_state = m.CaptureState.ACTIVE
    return runtime, admission


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal", "capture", "disposition"),
    (
        ("stopped", "idle", "erasure_durably_scheduled"),
        ("faulted", "faulted", "writer_fault"),
    ),
)
async def test_retention_owner_capture_follows_the_store_after_a_slow_erasure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: str,
    capture: str,
    disposition: str,
) -> None:
    """Capture state follows the store's terminal outcome, however long it takes."""

    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence import runtime as runtime_module

    runtime, admission = _retention_probe_runtime(tmp_path, m.OwnerState.RUNNING)
    runtime._published = True
    monkeypatch.setattr(runtime_module, "asyncio", _InstantSleepAsyncio())
    task = asyncio.create_task(runtime._run_retention_owner())
    try:
        assert await asyncio.wait_for(asyncio.to_thread(admission.began.wait, 5.0), 6.0)
        # The erasure is slow: the store is still running long after scheduling.
        for _ in range(2_000):
            if task.done():
                break
            await asyncio.sleep(0)
        assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
            m.CaptureState.ACTIVE
        )

        admission.owner_state = m.OwnerState(terminal)
        admission.expiry_terminal_event.set()

        assert await asyncio.wait_for(task, 5.0) is m.ExpiryDisposition(disposition)
        assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
            m.CaptureState(capture)
        )
        assert runtime._published is False
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # The probe never activated a lifecycle binding for close to revoke.
        runtime._published = False
        await runtime.close()


@pytest.mark.asyncio
async def test_retention_owner_terminal_wait_is_cancelled_by_close(tmp_path: Path) -> None:
    from hermes_realtime.evidence import models as m

    runtime, admission = _retention_probe_runtime(tmp_path, m.OwnerState.RUNNING)
    task = asyncio.create_task(runtime._run_retention_owner())
    runtime._retention_task = task
    assert await asyncio.wait_for(asyncio.to_thread(admission.began.wait, 5.0), 6.0)

    # The store never settles; close alone must end the owned wait.
    await asyncio.wait_for(runtime.close(), 5.0)

    assert task.cancelled()
    assert runtime._closed is True


@pytest.mark.asyncio
async def test_wait_retention_terminal_reports_its_own_timeout(tmp_path: Path) -> None:
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=37,
        retention_hours=24,
    )
    never = asyncio.Event()
    pending = asyncio.create_task(never.wait())
    runtime._retention_task = pending  # type: ignore[assignment]
    try:
        # A timeout is its own outcome, never a writer fault.
        with pytest.raises(TimeoutError):
            await runtime.wait_retention_terminal(timeout_seconds=0.05)
        assert not pending.done()
        for invalid in (0, -1.0, True, "1", float("nan"), float("inf"), 301.0):
            with pytest.raises(ValueError, match="timeout_seconds"):
                await runtime.wait_retention_terminal(timeout_seconds=invalid)  # type: ignore[arg-type]
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)

    async def settled() -> m.ExpiryDisposition:
        return m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED

    runtime._retention_task = asyncio.create_task(settled())
    # A generous observation bound is lawful: it bounds a hang, not latency.
    assert await runtime.wait_retention_terminal(timeout_seconds=60.0) is (
        m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
    )
    await runtime.close()


@pytest.mark.parametrize(
    "invalid", (0, 0.0, -1.0, True, "2", None, float("nan"), float("inf"), 300.5)
)
def test_host_evidence_runtime_validates_its_close_timeout(tmp_path: Path, invalid: object) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    with pytest.raises(ValueError, match="close_timeout_seconds"):
        HostEvidenceRuntimeV1(
            database=tmp_path / "capture-v1.sqlite3",
            owner_generation=41,
            retention_hours=24,
            close_timeout_seconds=invalid,  # type: ignore[arg-type]
        )


def test_host_evidence_runtime_close_timeout_defaults_to_two_seconds(tmp_path: Path) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=43,
        retention_hours=24,
    )
    assert runtime._close_timeout_seconds == 2.0
    for accepted in (1, 0.05, 300):
        bounded = HostEvidenceRuntimeV1(
            database=tmp_path / "capture-v1.sqlite3",
            owner_generation=43,
            retention_hours=24,
            close_timeout_seconds=accepted,
        )
        assert bounded._close_timeout_seconds == float(accepted)


@pytest.mark.asyncio
async def test_host_evidence_close_honours_its_close_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from hermes_realtime.client import BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    command = _create_epoch()
    release = threading.Event()

    class Transport:
        def create_epoch(self, received: m.CreateEpochV1) -> m.StoreDisposition:
            assert received is command
            return m.StoreDisposition.COMMITTED

        def drain_and_close(self, _received: m.DrainAndStopV1) -> m.DrainDisposition:
            if not release.wait(10.0):
                raise RuntimeError("test did not release the held drain")
            return m.DrainDisposition.STOPPED

        def __getattr__(self, name: str) -> object:
            if name in {
                "append_record",
                "append_binding_close",
                "rollover_session",
                "expire_session",
                "seal_epoch",
                "commit_revoke_request",
                "finalize_revoke",
            }:
                return lambda payload: (_ for _ in ()).throw(AssertionError(payload))
            raise AttributeError(name)

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=47,
        retention_hours=24,
        close_timeout_seconds=0.05,
    )
    projection = BrowserEventProjection()
    authority = runtime.reserve_consent_authority(
        command,
        projection.reserve_capture_status(),
        projection.validate_capture_status_reservation,
    )
    try:
        assert await runtime.activate_consent(
            authority,
            transport=Transport(),
            binding_is_current=lambda received: received is command,
            timeout_seconds=1.0,
        ) is m.ConsentDisposition.CONSENT_ACTIVATED
        writer = runtime._writer
        assert writer is not None
        close_writer = writer.close
        writer_timeouts: list[float | None] = []

        def close(timeout: float | None = None) -> bool:
            writer_timeouts.append(timeout)
            return close_writer(timeout)

        monkeypatch.setattr(writer, "close", close)

        # The held drain outlives the configured bound, not the two-second default.
        with pytest.raises(RuntimeError, match="evidence drain did not stop"):
            await asyncio.wait_for(runtime.close(), 1.0)
        assert runtime._closed is False

        release.set()
        await asyncio.wait_for(runtime.close(), 5.0)
        assert writer_timeouts == [0.05]
        assert runtime._closed is True
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_each_epoch_retention_waits_for_its_own_slow_erasure(tmp_path: Path) -> None:
    """A reused admission owner cannot hand a later expiry an earlier outcome."""

    from datetime import UTC, datetime
    from queue import Empty

    from test_admission import _activate_next_epoch, _active_admission, _admission

    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1

    admission, _operations, writer, first = _active_admission(
        _admission(), m, owner_generation=53
    )
    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=53,
        retention_hours=24,
    )

    class TransportProbe:
        def active_session_expiry(self, _logical_session_id: str) -> str:
            return "2026-08-12T00:00:00.000000Z"

    async def no_wait(_seconds: float = 0.0) -> None:
        return None

    runtime._admission = admission
    runtime._transport = TransportProbe()  # type: ignore[assignment]
    runtime._retention_cancel = no_wait
    runtime._retention_wall_clock = lambda: datetime(2026, 8, 12, tzinfo=UTC)
    runtime._retention_sleep = no_wait
    try:
        for epoch, seed in enumerate((None, 610)):
            command = first if seed is None else _activate_next_epoch(
                m, admission, writer, seed=seed
            )
            runtime._pending_create = command
            runtime._capture_state = m.CaptureState.ACTIVE
            task = asyncio.create_task(runtime._run_retention_owner())
            try:
                deadline = asyncio.get_running_loop().time() + 5.0
                while True:
                    try:
                        item = writer.get_nowait()
                        break
                    except Empty:
                        assert not task.done(), (epoch, task)
                        assert asyncio.get_running_loop().time() < deadline
                        await asyncio.sleep(0.001)
                assert type(item.payload) is m.ExpireSessionV1
                # The erasure is slow: this epoch's owner must still be waiting.
                for _ in range(200):
                    await asyncio.sleep(0)
                assert not task.done(), (epoch, task)
                assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
                    m.CaptureState.ACTIVE
                )

                admission.complete_ordered_item(item)

                assert await asyncio.wait_for(task, 5.0) is (
                    m.ExpiryDisposition.ERASURE_DURABLY_SCHEDULED
                )
                assert runtime.capture_status(disclosure_digest="a" * 64).capture_state is (
                    m.CaptureState.IDLE
                )
                assert admission.diagnostics().owner_state is m.OwnerState.STOPPED
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
    finally:
        runtime._admission = None
        await runtime.close()

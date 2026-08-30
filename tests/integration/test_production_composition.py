"""Production-only composition checks for deterministic evidence dependencies."""

import asyncio
from pathlib import Path
from uuid import UUID

import pytest


def _uuid(value: int) -> str:
    return str(UUID(int=value, version=4))


@pytest.mark.asyncio
async def test_hostile_writer_factory_result_is_rejected_before_consent_authority_or_projection(
    tmp_path: Path,
) -> None:
    """A malformed injected transport cannot reserve or strand any owner state."""

    from hermes_realtime.client import BrowserBindingSnapshot, BrowserEventProjection
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import (
        _HostEvidenceConsentDependenciesV1,
        _HostEvidenceConsentGatewayV1,
    )

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            return m.StoreDisposition.COMMITTED

        def append_record(self, _item: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            raise AssertionError("no records are admitted")

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            raise AssertionError("no binding is closed")

        def rollover_session(self, _command: m.RolloverSessionV1) -> m.StoreDisposition:
            raise AssertionError("no rollover is requested")

        def expire_session(self, _command: m.ExpireSessionV1) -> m.PurgeDisposition:
            raise AssertionError("no expiry is requested")

        def seal_epoch(self, _command: m.SealEpochV1) -> m.StoreDisposition:
            raise AssertionError("no seal is requested")

        def commit_revoke_request(
            self, _command: m.RevokeRequestV1
        ) -> m.RevokeDisposition:
            raise AssertionError("no revoke is requested")

        def finalize_revoke(
            self, _command: m.RevokeFinalizeV1
        ) -> m.RevokeDisposition:
            raise AssertionError("no revoke finalization is requested")

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

    calls = 0
    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=79,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    try:
        def factory(_runtime: HostEvidenceRuntimeV1) -> object:
            nonlocal calls
            calls += 1
            return object() if calls == 1 else Transport()

        gateway = _HostEvidenceConsentGatewayV1(
            runtime=runtime,
            projection=projection,
            live_generation=lambda: 1,
            dependencies=_HostEvidenceConsentDependenciesV1(
                writer_transport_factory=factory
            ),
        )
        request = m.EvidenceConsentRequestV1(
            accepted=True,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
            retention_hours=24,
            sequence=1,
            sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
        )
        binding = BrowserBindingSnapshot(
            participant_identity="browser_0123456789abcdef",
            binding_generation=1,
        )

        with pytest.raises(TypeError, match="transport must provide"):
            gateway.reserve(binding, request)

        assert not projection._capture_status_reservations
        assert runtime._admission is None
        assert runtime._queue is None
        assert runtime._pending_create is None
        assert runtime._pending_consent_authority is None
        assert runtime.operation_scheduler.active_count == 0

        response = await gateway.reserve(binding, request).complete()
        assert response.status == 200
        assert calls == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_browser_consent_close_observes_active_owner_before_primary_status_release(
    tmp_path: Path,
) -> None:
    """A close racing a blocked create publishes one settled 503, never stale status."""

    from threading import Event

    from hermes_realtime.client import (
        BrowserEventProjection,
        BrowserSessionDirector,
        BrowserTokenIssuer,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import (
        _HostEvidenceConsentDependenciesV1,
        _HostEvidenceConsentGatewayV1,
    )
    from hermes_realtime.livekit import LiveKitConnection

    entered = Event()
    release = Event()

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            entered.set()
            assert release.wait(timeout=2.0)
            return m.StoreDisposition.COMMITTED

        def append_record(self, _item: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            raise AssertionError("no records are admitted")

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            raise AssertionError("no binding is closed")

        def rollover_session(self, _command: m.RolloverSessionV1) -> m.StoreDisposition:
            raise AssertionError("no rollover is requested")

        def expire_session(self, _command: m.ExpireSessionV1) -> m.PurgeDisposition:
            raise AssertionError("no expiry is requested")

        def seal_epoch(self, _command: m.SealEpochV1) -> m.StoreDisposition:
            raise AssertionError("no seal is requested")

        def commit_revoke_request(
            self, _command: m.RevokeRequestV1
        ) -> m.RevokeDisposition:
            raise AssertionError("no revoke is requested")

        def finalize_revoke(
            self, _command: m.RevokeFinalizeV1
        ) -> m.RevokeDisposition:
            raise AssertionError("no revoke finalization is requested")

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            return None

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=80,
        retention_hours=24,
    )
    projection = BrowserEventProjection()
    try:
        gateway = _HostEvidenceConsentGatewayV1(
            runtime=runtime,
            projection=projection,
            live_generation=lambda: 1,
            dependencies=_HostEvidenceConsentDependenciesV1(
                writer_transport_factory=lambda _runtime: Transport()
            ),
        )

        async def provision(_identity: str) -> int:
            return 1

        async def submit(
            _identity: str, _generation: int, _sequence: int, _text: str
        ) -> None:
            return None

        async def stop(_identity: str, _generation: int) -> None:
            return None

        async def approval(
            _identity: str,
            _generation: int,
            _sequence: int,
            _approval_id: str,
            _decision: str,
        ) -> None:
            return None

        director = BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=LiveKitConnection(
                    "wss://livekit.test",
                    "test-key",
                    "synthetic-browser-close-race-secret-32-bytes",
                ),
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=submit,
            stop=stop,
            approval=approval,
            projection=projection,
            evidence_consent=gateway.reserve,
            evidence_status=lambda: runtime.capture_status(disclosure_digest="a" * 64),
        )
        credential = await director.start()
        request = m.EvidenceConsentRequestV1(
            accepted=True,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
            retention_hours=24,
            sequence=1,
            sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
        )
        activation = asyncio.create_task(
            director.consent_to_evidence(
                participant_identity=credential.participant_identity,
                request=request,
            )
        )
        assert await asyncio.wait_for(asyncio.to_thread(entered.wait), timeout=1.0)

        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0)
        release.set()
        response = await asyncio.wait_for(activation, timeout=2.0)
        await asyncio.wait_for(closing, timeout=2.0)

        assert response.status == 503
        assert response.payload == {
            "captureState": "faulted",
            "error": "writer_unavailable",
            "sequence": 1,
        }
        assert len(
            [
                event
                for event in projection._events
                if event.kind == "capture_status"
                and event.data["captureState"] == "faulted"
            ]
        ) == 1
        assert not projection._capture_status_reservations
        assert runtime.evidence_admission is None
        assert runtime.evidence_lifecycle is None
        assert runtime.operation_scheduler.active_count == 0
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_browser_consent_path_uses_the_injected_writer_transport_before_publish(
    tmp_path: Path,
) -> None:
    """The browser director reaches the real host owner through the consent seam."""

    from hermes_realtime.client import (
        BrowserEventProjection,
        BrowserSessionDirector,
        BrowserTokenIssuer,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import (
        _HostEvidenceConsentDependenciesV1,
        _HostEvidenceConsentGatewayV1,
    )
    from hermes_realtime.livekit import LiveKitConnection

    calls: list[object] = []

    class Transport:
        def create_epoch(self, command: m.CreateEpochV1) -> m.StoreDisposition:
            calls.append(command)
            return m.StoreDisposition.COMMITTED

        def append_record(self, _item: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            raise AssertionError("the composition has not admitted a record")

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            raise AssertionError("the composition has not closed a binding")

        def rollover_session(self, _command: m.RolloverSessionV1) -> m.StoreDisposition:
            raise AssertionError("the composition has not rolled over")

        def expire_session(self, _command: m.ExpireSessionV1) -> m.PurgeDisposition:
            raise AssertionError("the composition has not expired")

        def seal_epoch(self, _command: m.SealEpochV1) -> m.StoreDisposition:
            raise AssertionError("the composition has not sealed")

        def commit_revoke_request(
            self,
            _command: m.RevokeRequestV1,
        ) -> m.RevokeDisposition:
            raise AssertionError("the composition has not revoked")

        def finalize_revoke(
            self,
            _command: m.RevokeFinalizeV1,
        ) -> m.RevokeDisposition:
            raise AssertionError("the composition has not finalized revocation")

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=7,
        retention_hours=24,
    )
    try:
        connection = LiveKitConnection(
            "wss://livekit.test",
            "test-key",
            "synthetic-browser-bootstrap-secret-32-bytes",
        )
        projection = BrowserEventProjection()
        dependencies = _HostEvidenceConsentDependenciesV1(
            writer_transport_factory=lambda supplied: (
                Transport() if supplied is runtime else (_ for _ in ()).throw(AssertionError())
            )
        )
        gateway = _HostEvidenceConsentGatewayV1(
            runtime=runtime,
            projection=projection,
            live_generation=lambda: 1,
            dependencies=dependencies,
        )

        async def provision(_identity: str) -> int:
            return 1

        async def submit(_identity: str, _generation: int, _sequence: int, _text: str) -> None:
            return None

        async def stop(_identity: str, _generation: int) -> None:
            return None

        async def approval(
            _identity: str,
            _generation: int,
            _sequence: int,
            _approval_id: str,
            _decision: str,
        ) -> None:
            return None

        director = BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=connection,
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=submit,
            stop=stop,
            approval=approval,
            projection=projection,
            evidence_consent=gateway.reserve,
            evidence_status=lambda: runtime.capture_status(disclosure_digest="a" * 64),
        )
        credential = await director.start()

        response = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=m.EvidenceConsentRequestV1(
                accepted=True,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest="a" * 64,
                retention_hours=24,
                sequence=1,
                sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
            ),
        )

        assert response.status == 200
        assert response.payload["result"] == "consent_activated"
        assert len(calls) == 1
        assert runtime.evidence_admission is not None
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_host_evidence_owner_observes_its_actual_close_through_constructor_capability(
    tmp_path: Path,
) -> None:
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=8,
        retention_hours=24,
    )
    try:
        observations = runtime.production_observations
        await runtime.close()

        assert [(record.stage.value, record.result.value) for record in observations.records()] == [
            ("evidence_runtime", "succeeded")
        ]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_browser_malformed_create_return_releases_real_consent_reservations_for_retry(
    tmp_path: Path,
) -> None:
    from hermes_realtime.client import (
        BrowserEventProjection,
        BrowserSessionDirector,
        BrowserTokenIssuer,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import (
        _HostEvidenceConsentDependenciesV1,
        _HostEvidenceConsentGatewayV1,
    )
    from hermes_realtime.livekit import LiveKitConnection

    class Transport:
        def __init__(
            self,
            *,
            runtime: HostEvidenceRuntimeV1,
            create_succeeds: bool,
        ) -> None:
            self._runtime = runtime
            self._create_succeeds = create_succeeds

        def create_epoch(self, _command: m.CreateEpochV1) -> object:
            if not self._create_succeeds:
                failed_admissions.append(self._runtime._admission)
                return object()
            return m.StoreDisposition.COMMITTED

        def append_record(self, _item: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            raise AssertionError("the retry does not admit a record")

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            raise AssertionError("the retry does not close a binding")

        def rollover_session(self, _command: m.RolloverSessionV1) -> m.StoreDisposition:
            raise AssertionError("the retry does not roll over")

        def expire_session(self, _command: m.ExpireSessionV1) -> m.PurgeDisposition:
            raise AssertionError("the retry does not expire")

        def seal_epoch(self, _command: m.SealEpochV1) -> m.StoreDisposition:
            raise AssertionError("the retry does not seal")

        def commit_revoke_request(
            self,
            _command: m.RevokeRequestV1,
        ) -> m.RevokeDisposition:
            raise AssertionError("the retry does not revoke")

        def finalize_revoke(
            self,
            _command: m.RevokeFinalizeV1,
        ) -> m.RevokeDisposition:
            raise AssertionError("the retry does not finalize revocation")

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

    factory_calls = 0
    failed_admissions: list[object | None] = []
    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=81,
        retention_hours=24,
    )
    try:
        # Start publishes two events. Consent fills the remaining four. After
        # the fault status is acknowledged, retry fits only when all three
        # lifecycle reservations have been returned.
        projection = BrowserEventProjection(capacity=6)

        def writer_factory(_runtime: HostEvidenceRuntimeV1) -> Transport:
            nonlocal factory_calls
            factory_calls += 1
            return Transport(runtime=_runtime, create_succeeds=factory_calls > 1)

        gateway = _HostEvidenceConsentGatewayV1(
            runtime=runtime,
            projection=projection,
            live_generation=lambda: 1,
            dependencies=_HostEvidenceConsentDependenciesV1(
                writer_transport_factory=writer_factory
            ),
        )

        async def provision(_identity: str) -> int:
            return 1

        async def submit(_identity: str, _generation: int, _sequence: int, _text: str) -> None:
            return None

        async def stop(_identity: str, _generation: int) -> None:
            return None

        async def approval(
            _identity: str,
            _generation: int,
            _sequence: int,
            _approval_id: str,
            _decision: str,
        ) -> None:
            return None

        director = BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=LiveKitConnection(
                    "wss://livekit.test",
                    "test-key",
                    "synthetic-browser-bootstrap-secret-32-bytes",
                ),
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=submit,
            stop=stop,
            approval=approval,
            projection=projection,
            evidence_consent=gateway.reserve,
            evidence_status=lambda: runtime.capture_status(disclosure_digest="a" * 64),
        )
        credential = await director.start()
        request = m.EvidenceConsentRequestV1(
            accepted=True,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
            retention_hours=24,
            sequence=1,
            sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
        )

        failed = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )

        assert failed.status == 503
        assert failed.payload == {
            "captureState": "faulted",
            "error": "writer_unavailable",
            "sequence": 1,
        }
        assert len(failed_admissions) == 1
        failed_admission = failed_admissions[0]
        assert failed_admission is not None
        diagnostics = failed_admission.diagnostics()  # type: ignore[union-attr]
        assert diagnostics.queue_record_count == 0
        assert diagnostics.queue_canonical_bytes == 0
        assert runtime.operation_scheduler.active_count == 0
        assert runtime.evidence_admission is None
        assert runtime._admission is None
        assert runtime._writer is None
        assert runtime._transport is None
        assert runtime._pending_create is None
        assert runtime._pending_consent_authority is None
        assert runtime._pending_consent_ticket is None
        assert runtime._consent_settlement_operation is None
        await director.public_events_after(
            participant_identity=credential.participant_identity,
            sequence=3,
        )
        settled = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )
        assert settled.status == 503
        assert settled.payload == {
            "captureState": "faulted",
            "error": "writer_unavailable",
            "sequence": 1,
        }
        retry = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=m.EvidenceConsentRequestV1(
                accepted=True,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest="a" * 64,
                retention_hours=24,
                sequence=2,
                sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
            ),
        )
        assert retry.status == 200
        assert factory_calls == 2
        await runtime.close()
        await runtime.close()
        assert runtime.operation_scheduler.active_count == 0
        assert runtime.evidence_admission is None
        assert runtime.evidence_lifecycle is None
        assert runtime.writer_running is False
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_public_consent_timeout_close_cancels_the_retained_settlement_without_leaks(
    tmp_path: Path,
) -> None:
    """Close owns a timed-out consent settlement after browser timeout returns."""

    import time
    from threading import Event

    from hermes_realtime.client import (
        BrowserEventProjection,
        BrowserSessionDirector,
        BrowserTokenIssuer,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import (
        _HostEvidenceConsentDependenciesV1,
        _HostEvidenceConsentGatewayV1,
    )
    from hermes_realtime.livekit import LiveKitConnection

    entered = Event()
    release = Event()
    transports: list[object] = []

    class Transport:
        def __init__(self, *, block_create: bool, create_succeeds: bool) -> None:
            self._block_create = block_create
            self._create_succeeds = create_succeeds
            self._release = release

        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            if self._block_create:
                entered.set()
                assert self._release.wait(timeout=5.0)
            return (
                m.StoreDisposition.COMMITTED
                if self._create_succeeds
                else m.StoreDisposition.FAULTED
            )

        def append_record(self, _item: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            raise AssertionError("the timeout test does not admit a record")

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            raise AssertionError("the timeout test does not close a binding")

        def rollover_session(self, _command: m.RolloverSessionV1) -> m.StoreDisposition:
            raise AssertionError("the timeout test does not roll over")

        def expire_session(self, _command: m.ExpireSessionV1) -> m.PurgeDisposition:
            raise AssertionError("the timeout test does not expire")

        def seal_epoch(self, _command: m.SealEpochV1) -> m.StoreDisposition:
            raise AssertionError("the timeout test does not seal")

        def commit_revoke_request(
            self,
            _command: m.RevokeRequestV1,
        ) -> m.RevokeDisposition:
            raise AssertionError("the timeout test does not revoke")

        def finalize_revoke(
            self,
            _command: m.RevokeFinalizeV1,
        ) -> m.RevokeDisposition:
            raise AssertionError("the timeout test does not finalize revocation")

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            return None

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=82,
        retention_hours=24,
    )
    try:
        projection = BrowserEventProjection(capacity=6)

        def writer_factory(_runtime: HostEvidenceRuntimeV1) -> Transport:
            transport = Transport(
                block_create=not transports,
                create_succeeds=bool(transports),
            )
            transports.append(transport)
            return transport

        gateway = _HostEvidenceConsentGatewayV1(
            runtime=runtime,
            projection=projection,
            live_generation=lambda: 1,
            dependencies=_HostEvidenceConsentDependenciesV1(
                writer_transport_factory=writer_factory
            ),
        )

        async def provision(_identity: str) -> int:
            return 1

        async def submit(
            _identity: str,
            _generation: int,
            _sequence: int,
            _text: str,
        ) -> None:
            return None

        async def stop(_identity: str, _generation: int) -> None:
            return None

        async def approval(
            _identity: str,
            _generation: int,
            _sequence: int,
            _approval_id: str,
            _decision: str,
        ) -> None:
            return None

        director = BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=LiveKitConnection(
                    "wss://livekit.test",
                    "test-key",
                    "synthetic-browser-timeout-secret-32-bytes",
                ),
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=submit,
            stop=stop,
            approval=approval,
            projection=projection,
            evidence_consent=gateway.reserve,
            evidence_status=lambda: runtime.capture_status(disclosure_digest="a" * 64),
        )
        credential = await director.start()
        request = m.EvidenceConsentRequestV1(
            accepted=True,
            consent_version=m.CONSENT_VERSION,
            disclosure_digest="a" * 64,
            retention_hours=24,
            sequence=1,
            sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
        )

        timed_out = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=request,
        )

        assert entered.is_set()
        assert timed_out.status == 503
        assert timed_out.payload == {
            "captureState": "idle",
            "error": "control_timeout",
            "sequence": 1,
        }
        assert (
            await director.consent_to_evidence(
                participant_identity=credential.participant_identity,
                request=request,
            )
        ) is timed_out

        closing = asyncio.create_task(runtime.close())
        # Runtime cancellation must win before the blocked writer can complete.
        await asyncio.to_thread(time.sleep, 0.1)
        settlement = runtime._consent_settlement_operation
        assert settlement is not None and settlement.cancelled()
        release.set()
        await closing
        await runtime.close()

        assert len(transports) == 1
        assert runtime.capture_status(disclosure_digest="a" * 64).available is False
        assert runtime.evidence_admission is None
        assert runtime.evidence_lifecycle is None
        assert runtime.operation_scheduler.active_count == 0
        assert not projection._capture_status_reservations
        assert director._pending_evidence_consent_settlement is None
        leaked = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name()
            in {
                "host-evidence-consent-timeout-settlement",
                "browser-evidence-consent-settlement-observer",
            }
        ]
        assert leaked == []
    finally:
        release.set()
        await runtime.close()


@pytest.mark.asyncio
async def test_public_durable_revoke_close_before_purge_releases_terminal_projection(
    tmp_path: Path,
) -> None:
    """Runtime close, not a browser observer, releases unpublished revoke status."""

    import time
    from threading import Event

    from hermes_realtime.client import (
        BrowserEventProjection,
        BrowserSessionDirector,
        BrowserTokenIssuer,
    )
    from hermes_realtime.evidence import models as m
    from hermes_realtime.evidence.runtime import HostEvidenceRuntimeV1
    from hermes_realtime.host_launcher import (
        _HostEvidenceConsentDependenciesV1,
        _HostEvidenceConsentGatewayV1,
        _reserve_host_evidence_revoke,
    )
    from hermes_realtime.livekit import LiveKitConnection

    events: list[str] = []
    finalize_entered = Event()
    finalize_release = Event()

    class Transport:
        def create_epoch(self, _command: m.CreateEpochV1) -> m.StoreDisposition:
            events.append("create")
            return m.StoreDisposition.COMMITTED

        def append_record(self, _item: m.QueuedEvidenceRecordV1) -> m.StoreDisposition:
            raise AssertionError("the close test does not admit a record")

        def append_binding_close(self, _command: m.BindingCloseV1) -> m.StoreDisposition:
            raise AssertionError("the close test does not close a binding")

        def rollover_session(self, _command: m.RolloverSessionV1) -> m.StoreDisposition:
            raise AssertionError("the close test does not roll over")

        def expire_session(self, _command: m.ExpireSessionV1) -> m.PurgeDisposition:
            raise AssertionError("the close test does not expire")

        def seal_epoch(self, _command: m.SealEpochV1) -> m.StoreDisposition:
            raise AssertionError("the close test does not seal")

        def commit_revoke_request(
            self,
            _command: m.RevokeRequestV1,
        ) -> m.RevokeDisposition:
            events.append("revoke_request")
            return m.RevokeDisposition.REVOKE_DURABLY_SCHEDULED

        def finalize_revoke(
            self,
            _command: m.RevokeFinalizeV1,
        ) -> m.RevokeDisposition:
            events.append("revoke_finalize")
            finalize_entered.set()
            assert finalize_release.wait(timeout=5.0)
            return m.RevokeDisposition.PURGE_COMPLETED

        def drain_and_close(self, _command: m.DrainAndStopV1) -> m.DrainDisposition:
            events.append("drain")
            return m.DrainDisposition.STOPPED

        def close(self) -> None:
            events.append("transport")

    runtime = HostEvidenceRuntimeV1(
        database=tmp_path / "capture-v1.sqlite3",
        owner_generation=83,
        retention_hours=24,
    )
    try:
        projection = BrowserEventProjection(capacity=6)
        gateway = _HostEvidenceConsentGatewayV1(
            runtime=runtime,
            projection=projection,
            live_generation=lambda: 1,
            dependencies=_HostEvidenceConsentDependenciesV1(
                writer_transport_factory=lambda _runtime: Transport()
            ),
        )

        async def provision(_identity: str) -> int:
            return 1

        async def submit(
            _identity: str,
            _generation: int,
            _sequence: int,
            _text: str,
        ) -> None:
            return None

        async def stop(_identity: str, _generation: int) -> None:
            return None

        async def approval(
            _identity: str,
            _generation: int,
            _sequence: int,
            _approval_id: str,
            _decision: str,
        ) -> None:
            return None

        director = BrowserSessionDirector(
            issuer=BrowserTokenIssuer(
                connection=LiveKitConnection(
                    "wss://livekit.test",
                    "test-key",
                    "synthetic-browser-close-secret-32-bytes",
                ),
                room_name="hermes-local",
                identity_factory=lambda: "browser_0123456789abcdef",
            ),
            provision=provision,
            submit=submit,
            stop=stop,
            approval=approval,
            projection=projection,
            evidence_consent=gateway.reserve,
            evidence_revoke=lambda binding, request: _reserve_host_evidence_revoke(
                runtime=runtime,
                projection=projection,
                binding=binding,
                request=request,
            ),
            evidence_status=lambda: runtime.capture_status(disclosure_digest="a" * 64),
        )
        credential = await director.start()
        response = await director.consent_to_evidence(
            participant_identity=credential.participant_identity,
            request=m.EvidenceConsentRequestV1(
                accepted=True,
                consent_version=m.CONSENT_VERSION,
                disclosure_digest="a" * 64,
                retention_hours=24,
                sequence=1,
                sources=m.EvidenceConsentSourcesV1(microphone=True, typed=False),
            ),
        )
        assert response.status == 200

        durable_revoke = await director.revoke_evidence(
            participant_identity=credential.participant_identity,
            request=m.EvidenceRevokeRequestV1(sequence=2),
        )
        assert durable_revoke.status == 202
        assert await asyncio.to_thread(finalize_entered.wait, 2.0)

        closing = asyncio.create_task(runtime.close())
        # The observer owns the transferred terminal reservation.  Give its
        # cancellation-aware bounded bridge one wall-clock slice to run, then
        # release the blocked purge only after close has detached it.
        await asyncio.to_thread(time.sleep, 0.1)
        assert not runtime._revoke_observers
        finalize_release.set()
        await closing
        await runtime.close()

        assert events == ["create", "revoke_request", "revoke_finalize", "drain", "transport"]
        assert runtime.evidence_admission is None
        assert runtime.evidence_lifecycle is None
        assert runtime.writer_running is False
        assert runtime._admission is None
        assert runtime._queue is None
        assert runtime._writer is None
        assert runtime._transport is None
        assert runtime._pending_create is None
        assert runtime._pending_consent_authority is None
        assert runtime._pending_consent_ticket is None
        assert runtime._pending_revoke_authority is None
        assert runtime._pending_revoke_ticket is None
        assert runtime._invalidation_authority is None
        assert runtime._invalidation_operation is None
        assert runtime._lifecycle_owner._binding is None
        assert runtime._lifecycle_status_reservations == []
        assert runtime.operation_scheduler.active_count == 0
        assert not projection._capture_status_reservations
        leaked = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name()
            in {
                "evidence-revoke-terminal",
                "host-evidence-consent-timeout-settlement",
                "browser-evidence-consent-settlement-observer",
            }
        ]
        assert leaked == []
    finally:
        finalize_release.set()
        await runtime.close()
